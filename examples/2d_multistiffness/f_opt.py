import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import json 

ti.reset()
real = ti.f32
ti.init(arch=ti.cuda, default_fp=real, device_memory_GB=12)

# init parameters
size = 1
dim = 2
Nx = 80  # reduce to 30 if run out of GPU memory
Ny = 40
n_particles = Nx * Ny
n_grid = 20
dx = 1 / n_grid
inv_dx = 1 / dx
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
# strain = ti.Matrix.field(dim,
#                          dim,
#                          dtype=real,
#                          shape=(max_steps, n_particles),
#                          needs_grad=True)
strain2 = ti.Matrix.field(dim,
                         dim,
                         dtype=real,
                         shape=(max_steps, n_particles),
                         needs_grad=True)
loss = ti.field(dtype=real, shape=(), needs_grad=True)
init_g = ti.field(dtype=real, shape=(), needs_grad=True)
force = ti.field(dtype=real, shape=(), needs_grad=True)
E = ti.field(dtype=real, shape=(n_particles), needs_grad=True)


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
        cauchy = 2 * E[p] / (2 * (1 + nu)) * (new_F - r) @ new_F.transpose() + \
                 ti.Matrix.diag(2, E[None] * nu / ((1 + nu) * (1 - 2 * nu)) * (J - 1) * J)
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + p_mass * C[f, p]
        # strain[f, p] += 0.5 * (new_F.transpose() @ new_F - ti.math.eye(dim))
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
    for t, node in ti.ndrange(max_steps, (2, 19)):
            f_ext[t, node, 14] = [0, -force[None] * e[t, node - 2]]
    for i, j in ti.ndrange(n_grid, n_grid):     
        inv_m = 1 / (grid_m_in[f, i, j] + 1e-10) 
        v_out = inv_m * grid_v_in[f, i, j] + dt * f_ext[f, i, j]
        # v_out[1] -= dt * gravity
        if i == 2 and j == 6:
            v_out[0] = 0
            v_out[1] = 0
        if i == 18 and j == 6:
            v_out[1] = 0
        grid_v_out[f, i, j] = v_out



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


        dstrain = strain_rate * dt
        strain2[f + 1, p][0, 0] = strain2[f, p][0, 0] + dstrain[0]
        strain2[f + 1, p][1, 1] = strain2[f, p][1, 1] + dstrain[1]
        strain2[f + 1, p][0, 1] = strain2[f, p][0, 1] + dstrain[2]
        strain2[f + 1, p][1, 1] = strain2[f, p][1, 0] + dstrain[2]

        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C



@ti.kernel
def compute_loss():
    for i in range(steps - 1):
        for j in range(Nx):
            dist = (target_strain[i, j] - strain2[i, j]) ** 2
            # dist = (1 / ((steps - 1) * n_particles)) * \
            #     (target_strain[i, j] - strain2[i, j]) ** 2
            loss[None] += 0.5 * (dist[1, 1])# + dist[0, 0])

def substep(s):
    p2g(s)
    grid_op(s)
    g2p(s)



f_ext_scale = 5e4   
velocity = 16 / 20 / 0.1
node_x_locs = ti.Vector(np.arange(0, 17 / n_grid, 1 / n_grid))
time_to_center = node_x_locs / velocity
t_steps = ti.Vector(np.arange(max_steps)) * dt
t_steps_n = np.array([t_steps - time for time in time_to_center])
t_steps_n = np.stack(t_steps_n, axis=1)

t = np.asarray(t_steps_n)
fc, bw, bwr = 100, 0.5, -6
ref = np.power(10.0, bwr / 20.0)
a = -(np.pi * fc * bw) ** 2 / (4.0 * np.log(ref))
e_np = np.exp(-a * t * t)
e = ti.field(ti.f32, (1024, 17))
e.from_numpy(e_np)

@ti.kernel
def assign_ext_load():
    for t, node in ti.ndrange(max_steps, (2, 19)):
            f_ext[t, node, 14] = [0, -f_ext_scale * e[t, node - 2]]

@ti.kernel
def assign_E():
    for i in range(n_particles):
        col = i % Nx
        if col % 20 < 20 or col % 20 >= 60:
            E[i] = 1100
        else:
            if i < n_particles * 0.5:
                E[i] = 800
            else:
                E[i] = 900




for i in range(n_particles):
    F[0, i] = [[1, 0], [0, 1]]



for i in range(Nx):
    for j in range(Ny):
        x[0, j * Nx + i] = [(i)/(Nx) * 0.8 + 0.1, (j)/(Ny) * 0.4 + 0.3]




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
lr = 1e3
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
n_params = 4
m_adam = [0 for _ in range(n_params)]
v_adam = [0 for _ in range(n_params)]
v_hat = [0 for _ in range(n_params)]

init_g[None] = 0
force[None] = 5 * 1e4
E[None] = 0.9 * 1e4
grad_iterations = 250

losses = []
es = []
fs = []


print('running grad iterations')
optim = 'grad'
if optim == 'grad':

    for j in range(grad_iterations):
        grid_v_in.fill(0)
        grid_m_in.fill(0)
        loss[None] = 0
        # assign_ext_load()
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

        es.append(E[None])
        fs.append(force[None])
        print(j, 
            'loss=', l, 
            '   grad=', params[0].grad[None], # params[1].grad[None],
            '   E=', E[None],
            '   F=', force[None])

    plt.title("Optimization of Block Subject to Dynamic Rolling Force via $\epsilon (t)$")
    plt.ylabel("Loss")
    plt.xlabel("Gradient Descent Iterations")
    plt.plot(losses)
    plt.yscale('log')
    plt.show()

    plt.title("Force Learning Curve")
    plt.ylabel("$F$")
    plt.xlabel("Iterations")
    plt.hlines(5e4, 0, grad_iterations, color='r', label='True Value')
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
        # force[None] = params[1]

        grid_v_in.fill(0)
        grid_m_in.fill(0)
        loss[None] = 0
        
        with ti.ad.Tape(loss=loss):
            for s in range(steps - 1):
                substep(s)
            compute_loss()

        loss_val = loss[None]
        grad_val = [E.grad[None]]

        return loss_val, grad_val
    
    def callback_fn(intermediate_result):
        params = intermediate_result.x
        loss, grad = compute_loss_and_grad(params)
        losses.append(loss)
        es.append(params[0])
        # fs.append(params[1])
        print(j, 
            'loss=', loss, 
            '   grad=', grad[0],
            '   E=', params[0])

    initial_params = [0.9e4]
    tol = 1e-3600
    result = minimize(compute_loss_and_grad, 
                    initial_params, 
                    method='L-BFGS-B', 
                    jac=True, 
                    hess='2-point',
                    callback=callback_fn,
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

    es.insert(0, initial_params[0])

    plt.title("Optimization of Block Subject to Dynamic Rolling Force via $\epsilon (t)$")
    plt.ylabel("Loss")
    plt.xlabel("LBFGS-B Iterations")
    plt.plot(losses)
    plt.yscale('log')
    plt.show()

    plt.title("Force Learning Curve")
    plt.ylabel("$F$")
    plt.xlabel("Iterations")
    plt.hlines(5e4, 0, result.nit - 1, color='r', label='True Value')
    plt.plot(fs, color='b', label='Estimated Value')
    plt.legend()
    plt.show()

    plt.title("Young's Modulus Learning Curve")
    plt.ylabel("$E$")
    plt.xlabel("Iterations")
    plt.hlines(1e4, 0, result.nit - 1, color='r', label='True Value')
    plt.plot(es, color='b', label='Estimated Value')
    plt.legend()
    plt.show()

result_dict = {
    "losses" : losses,
    "fs" : fs,
    "es" : es
}

with open("result_bot_y_grad.json", "w") as outfile: 
    json.dump(result_dict, outfile)