import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

ti.reset()
real = ti.f32
ti.init(arch=ti.cuda, default_fp=real, device_memory_GB=12)

# init parameters
size = 1
dim = 2
N = 80  # reduce to 30 if run out of GPU memory
n_particles = N * N
n_grid = 40
dx = 1 / n_grid
inv_dx = 1 / dx
dt_scale = 1e0
dt = 2e-2 * dx / size * dt_scale
dt = 1e-4
p_mass = 1
p_vol = 1

nu = 0.2

max_steps = 1024
steps = max_steps
gravity = 0


x = ti.Vector.field(dim,
                    dtype=real,
                    shape=(max_steps, n_particles),
                    needs_grad=True)
x_avg = ti.Vector.field(dim, dtype=real, shape=(), needs_grad=True)
v = ti.Vector.field(dim,
                    dtype=real,
                    shape=(max_steps, n_particles),
                    needs_grad=True)
grid_v_in = ti.Vector.field(dim,
                            dtype=real,
                            shape=(max_steps, n_grid, n_grid),
                            needs_grad=True)
grid_v_out = ti.Vector.field(dim,
                             dtype=real,
                             shape=(max_steps, n_grid, n_grid),
                             needs_grad=True)
f_ext = ti.Vector.field(dim,
                        dtype=real,
                        shape=(max_steps, n_grid, n_grid),
                        needs_grad=True)
grid_m_in = ti.field(dtype=real,
                     shape=(max_steps, n_grid, n_grid),
                     needs_grad=True)
C = ti.Matrix.field(dim,
                    dim,
                    dtype=real,
                    shape=(max_steps, n_particles),
                    needs_grad=True)
F = ti.Matrix.field(dim,
                    dim,
                    dtype=real,
                    shape=(max_steps, n_particles),
                    needs_grad=True)
strain = ti.Matrix.field(dim,
                         dim,
                         dtype=real,
                         shape=(max_steps, n_particles),
                         needs_grad=True)
strain2 = ti.Matrix.field(dim,
                         dim,
                         dtype=real,
                         shape=(max_steps, n_particles),
                         needs_grad=True)
init_v = ti.Vector.field(dim, dtype=real, shape=(), needs_grad=True)
loss = ti.field(dtype=real, shape=(), needs_grad=True)
init_g = ti.field(dtype=real, shape=(), needs_grad=True)
force = ti.field(dtype=real, shape=(), needs_grad=True)
E = ti.field(dtype=real, shape=(), needs_grad=True)


@ti.kernel
def p2g(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
        F[f + 1, p] = new_F
        J = (new_F).determinant()
        r, s = ti.polar_decompose(new_F)
        cauchy = 2 * E[None] / (2 * (1 + nu)) * (new_F - r) @ new_F.transpose() + \
                 ti.Matrix.diag(2, E[None] * nu / ((1 + nu) * (1 - 2 * nu)) * (J - 1) * J)
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + p_mass * C[f, p]
        strain[f, p] += 0.5 * (new_F.transpose() @ new_F - ti.math.eye(dim))
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v_in[f, base + offset] += weight * (p_mass * v[f, p] +
                                                         affine @ dpos)
                grid_m_in[f, base + offset] += weight * p_mass

@ti.kernel
def grid_op(f: ti.i32):
    for i, j in ti.ndrange(n_grid, n_grid):     
        inv_m = 1 / (grid_m_in[f, i, j] + 1e-10) 
        v_out = inv_m * grid_v_in[f, i, j] + dt * f_ext[f, i, j]
        # v_out[1] -= dt * init_g[None]
        if i == 5 and j == 5:
            v_out[0] = 0
            v_out[1] = 0
        if i == 14 and j == 5:
            v_out[1] = 0
        if i == 15 and j == 15:
            v_out[0] += dt * force[None]
        grid_v_out[f, i, j] = v_out

# bound = 6
# @ti.kernel
# def grid_op(f: ti.i32):
#     for i, j in ti.ndrange(n_grid, n_grid):     
#         inv_m = 1 / (grid_m_in[f, i, j] + 1e-10) # + dt * grid_v_ext[f, i, j])
#         v_out = inv_m * grid_v_in[f, i, j] 
#         v_out[1] -= dt * init_g[None]
#         if i < bound and v_out[0] < 0:
#             v_out[0] = 0
#         if i > n_grid - bound and v_out[0] > 0:
#             v_out[0] = 0
#         if j < bound and v_out[1] < 0:
#             v_out[1] = 0
#         if j > n_grid - bound and v_out[1] > 0:
#             v_out[1] = 0
#         grid_v_out[f, i, j] = v_out


@ti.kernel
def g2p(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = ti.cast(ti.Vector([i, j]), real) - fx
                g_v = grid_v_out[f, base[0] + i, base[1] + j]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx
        ### stress and strain from nodal velocity
        # shape function gradient
        grad = ti.Matrix([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 
                          [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 
                          [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        vi = ti.Matrix([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 
                        [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 
                        [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        strain_rate = ti.Vector([0.0, 0.0, 0.0])


        fx = fx * dx 
        grad[0, 0] = 0.25 * fx[1] * (fx[1] - 1.) * (2 * fx[0] - 1.)
        grad[1, 0] = 0.25 * fx[1] * (fx[1] - 1.) * (2 * fx[0] + 1.)
        grad[2, 0] = 0.25 * fx[1] * (fx[1] + 1.) * (2 * fx[0] + 1.)
        grad[3, 0] = 0.25 * fx[1] * (fx[1] + 1.) * (2 * fx[0] - 1.)
        grad[4, 0] = -fx[0] * fx[1] * (fx[1] - 1.)
        grad[5, 0] = -0.5 * (2. * fx[0] + 1.) * ((fx[1] * fx[1]) - 1.)
        grad[6, 0] = -fx[0] * fx[1] * (fx[1] + 1.)
        grad[7, 0] = -0.5 * (2. * fx[0] - 1.) * ((fx[1] * fx[1]) - 1.)
        grad[8, 0] = 2. * fx[0] * ((fx[1] * fx[1]) - 1.)
        grad[0, 1] = 0.25 * fx[0] * (fx[0] - 1.) * (2. * fx[1] - 1.)
        grad[1, 1] = 0.25 * fx[0] * (fx[0] + 1.) * (2. * fx[1] - 1.)
        grad[2, 1] = 0.25 * fx[0] * (fx[0] + 1.) * (2. * fx[1] + 1.)
        grad[3, 1] = 0.25 * fx[0] * (fx[0] - 1.) * (2. * fx[1] + 1.)
        grad[4, 1] = -0.5 * (2. * fx[1] - 1.) * ((fx[0] * fx[0]) - 1.)
        grad[5, 1] = -fx[0] * fx[1] * (fx[0] + 1.)
        grad[6, 1] = -0.5 * (2. * fx[1] + 1.) * ((fx[0] * fx[0]) - 1.)
        grad[7, 1] = -fx[0] * fx[1] * (fx[0] - 1.)
        grad[8, 1] = 2. * fx[1] * ((fx[0] * fx[0]) - 1.)
        
        vi[0, 0] = grid_v_out[f, base[0], base[1]][0]
        vi[1, 0] = grid_v_out[f, base[0] + 2, base[1]][0]
        vi[2, 0] = grid_v_out[f, base[0] + 2, base[1] + 2][0]
        vi[3, 0] = grid_v_out[f, base[0], base[1] + 2][0]
        vi[4, 0] = grid_v_out[f, base[0] + 1, base[1]][0]
        vi[5, 0] = grid_v_out[f, base[0] + 2, base[1] + 1][0]
        vi[6, 0] = grid_v_out[f, base[0] + 1, base[1] + 2][0]
        vi[7, 0] = grid_v_out[f, base[0], base[1] + 1][0]
        vi[8, 0] = grid_v_out[f, base[0] + 1, base[1] + 1][0]
        vi[0, 1] = grid_v_out[f, base[0], base[1]][1]
        vi[1, 1] = grid_v_out[f, base[0] + 2, base[1]][1]
        vi[2, 1] = grid_v_out[f, base[0] + 2, base[1] + 2][1]
        vi[3, 1] = grid_v_out[f, base[0], base[1] + 2][1]
        vi[4, 1] = grid_v_out[f, base[0] + 1, base[1]][1]
        vi[5, 1] = grid_v_out[f, base[0] + 2, base[1] + 1][1]
        vi[6, 1] = grid_v_out[f, base[0] + 1, base[1] + 2][1]
        vi[7, 1] = grid_v_out[f, base[0], base[1] + 1][1]
        vi[8, 1] = grid_v_out[f, base[0] + 1, base[1] + 1][1]

        # calc strain rate
        for k in ti.static(range(9)):
            strain_rate[0] += grad[k, 0] * vi[k, 0]
            strain_rate[1] += grad[k, 1] * vi[k, 1]
            strain_rate[2] += grad[k, 0] * vi[k, 1] + grad[k, 1] * vi[k, 0]

        # update stress and strain
        # G = E[None] / (2 * 1 + nu)
        # bulk_modulus = E[None] / (3 * 1 - 2 * nu)
        # a1 = bulk_modulus + (4 * G / 3)
        # a2 = bulk_modulus - (2 * G / 3)
        # de = ti.Matrix([
        #             [a1, a2, 0],
        #             [a2, a1, 0],
        #             [0, 0, G]
        #         ])
        dstrain = strain_rate * dt
        strain2[f + 1, p][0, 0] = strain2[f, p][0, 0] + dstrain[0]
        strain2[f + 1, p][1, 1] = strain2[f, p][1, 1] + dstrain[1]
        strain2[f + 1, p][0, 1] = strain2[f, p][0, 1] + dstrain[2]
        strain2[f + 1, p][1, 1] = strain2[f, p][1, 0] + dstrain[2]
        # grad = ti.Matrix([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        # vi = ti.Matrix([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        # dstrain = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])

        # grad[0, 0] = -0.25 * (1 - fx[1])
        # grad[1, 0] = 0.25 * (1 - fx[1])
        # grad[2, 0] = 0.25 * (1 + fx[1])
        # grad[3, 0] = -0.25 * (1 + fx[1])

        # grad[0, 1] = -0.25 * (1 - fx[0])
        # grad[1, 1] = -0.25 * (1 + fx[0])
        # grad[2, 1] = 0.25 * (1 + fx[0])
        # grad[3, 1] = 0.25 * (1 - fx[0])

        # vi[0, 0] = grid_v_out[f, base[0], base[1]][0]
        # vi[1, 0] = grid_v_out[f, base[0] + 1, base[1]][0]
        # vi[2, 0] = grid_v_out[f, base[0] + 1, base[1] + 1][0]
        # vi[3, 0] = grid_v_out[f, base[0], base[1] + 1][0]

        # vi[0, 1] = grid_v_out[f, base[0], base[1]][1]
        # vi[1, 1] = grid_v_out[f, base[0] + 1, base[1]][1]
        # vi[2, 1] = grid_v_out[f, base[0] + 1, base[1] + 1][1]
        # vi[3, 1] = grid_v_out[f, base[0], base[1] + 1][1]
        
        # for i in ti.static(range(2)):
        #     for j in ti.static(range(2)):
        #         for k in ti.static(range(4)):
        #             dstrain[i, j] += 0.5 * (grad[k, i] * vi[k, j] + grad[k, j] * vi[k, i])
   
        # strain2[f + 1, p] = strain2[f, p] + dstrain
        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C

@ti.kernel
def compute_x_avg():
    for i in range(n_particles):
        x_avg[None] += (1 / n_particles) * x[steps - 1, i]

@ti.kernel
def compute_loss():
    for i in range(steps - 1):
        for j in range(n_particles):
            dist = (target_strain[i, j] - strain2[i, j]) ** 2
            # dist = (1 / ((steps - 1) * n_particles)) * \
            #     (target_strain[i, j] - strain2[i, j]) ** 2
            loss[None] += 0.5 * (dist[0, 0] + dist[1, 1])

def substep(s):
    p2g(s)
    grid_op(s)
    g2p(s)


# f_ext_scale = 1   
# velocity = 4
# frequency = 5
# node_x_locs = np.arange(0, 1, 1 / n_grid)
# time_to_center = node_x_locs / velocity
# t_steps = np.arange(max_steps) * dt
# t_steps_n = np.array([t_steps - time for time in time_to_center])
# t_steps_n = np.stack(t_steps_n, axis=1)
# node_ids_fext_x = range(n_grid)
# _, _, e, = utils.gausspulse(t_steps_n)
# grid_v_ext = ti.Vector.field(dim,
#                             dtype=real,
#                             shape=(max_steps, n_grid, n_grid),
#                             needs_grad=True)
# print('assigning external loads')
# for t in range(max_steps):
#     for node in node_ids_fext_x:
#         grid_v_ext[t, node, node] = [0, f_ext_scale * e[t, node]]

# @ti.kernel
# def assign_ext_load():
#     for t in range(max_steps):
#             f_ext[t, 15, 15] = [force[None], 0]







for i in range(n_particles):
    F[0, i] = [[1, 0], [0, 1]]




for i in range(N):
    for j in range(N):
        x[0, i * N + j] = [(i)/(4*N) + 0.125, (j)/(4*N) + 0.125]




print('loading target')

target_strain_np = np.load('strain2_f.npy')
target_strain = ti.Matrix.field(dim,
                            dim,
                           dtype=real,
                           shape=(max_steps, n_particles),
                           needs_grad=True)

@ti.kernel
def load_target(target_np: ti.types.ndarray()):
    for i, j, k, l in ti.ndrange(steps, n_particles, dim, dim):
        target_strain[i, j][k, l] = target_np[i, j, k, l]

load_target(target_strain_np)


# ADAM parameters
lr = 1e1
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
n_params = 2
m_adam = [0 for _ in range(n_params)]
v_adam = [0 for _ in range(n_params)]
v_hat = [0 for _ in range(n_params)]

init_g[None] = 0
force[None] = -4.5 * 1e3
E[None] = 0.9 * 1e4
grad_iterations = 400

losses = []
es = np.zeros((grad_iterations))
fs = np.zeros((grad_iterations))


print('running grad iterations')
optim = 'grad'
if optim == 'grad':

    for j in range(grad_iterations):
        grid_v_in.fill(0)
        grid_m_in.fill(0)
        loss[None] = 0

        with ti.ad.Tape(loss=loss):
            for s in range(steps - 1):
                substep(s)
            compute_loss()

        l = loss[None]
        losses.append(l)

        params = [E, force]
        param_vals = [E[None], force[None]]
        for i in range(n_params):
            gradient = params[i].grad[None]
            m_adam[i] = beta1 * m_adam[i] + (1 - beta1) * gradient
            v_adam[i] = beta2 * v_adam[i] + (1 - beta2) * gradient**2
            # m_hat = m_adam[i] / (1 - beta1**(j + 1))
            # v_hat = v_adam[i] / (1 - beta2**(j + 1))
            v_hat[i] = ti.max(v_hat[i], v_adam[i])
            param_vals[i] -= lr * m_adam[i] / (ti.sqrt(v_hat[i]) + epsilon)

        E[None] = param_vals[0]
        force[None] = param_vals[1]

        es[j] = np.array([E[None]])
        fs[j] = np.array([force[None]])
        print(j, 
            'loss=', l, 
            '   grad=', params[0].grad[None], params[1].grad[None],
            '   E=', E[None],
            '   F=', force[None])

    print(es)
    plt.title("Optimization of Block Subject to Constant Force via $\epsilon (t)$")
    plt.ylabel("Loss")
    plt.xlabel("Gradient Descent Iterations")
    plt.plot(losses)
    plt.yscale('log')
    plt.show()

    plt.title("Force Learning Curve")
    plt.ylabel("$F$")
    plt.xlabel("Iterations")
    plt.hlines(-5e3, 0, grad_iterations, color='r', label='True Value')
    plt.plot(fs, color='b', label='Estimated Value')
    plt.legend()
    plt.show()

    plt.title("Young's Modulus Learning Curve")
    plt.ylabel("$E$")
    plt.xlabel("Iterations")
    plt.hlines(1e4, 0, grad_iterations, color='r', label='True Value')
    plt.plot(es, color='b', label='Estimated Value')
    plt.legend()
    plt.show()


elif optim == 'lbfgs':
    from scipy.optimize import minimize
    def compute_loss_and_grad(params):
        E[None] = params[0]
        force[None] = params[1]

        grid_v_in.fill(0)
        grid_m_in.fill(0)
        loss[None] = 0
        
        with ti.ad.Tape(loss=loss):
            for s in range(steps - 1):
                substep(s)
            compute_loss()

        loss_val = loss[None]
        grad_val = [E.grad[None], force.grad[None]]

        return loss_val, grad_val


    initial_params = [0.9e4, -4.5e3]
    tol = 1e-36
    result = minimize(compute_loss_and_grad, 
                    initial_params, 
                    method='L-BFGS-B', 
                    jac=True, 
                    hess='2-point',
                    options={
                        'disp': 1, 
                        'xatol': tol, 
                        'fatol': tol, 
                        'xtol': tol, 
                        'ftol': tol, 
                        'gtol': tol,
                        'tol': tol,
                        'catol': tol,
                        'barrier_tol': tol,
                        'maxCGit': 1000,
                        'maxfun': 1000, 
                        'maxiter': 1000,
                        'verbose': 2,
                        'adaptive': True
                        })

    print(result)