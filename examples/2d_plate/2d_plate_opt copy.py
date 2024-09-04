import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import json

ti.reset()
real = ti.f32
ti.init(arch=ti.cuda, default_fp=real, device_memory_GB=12)

# init parameters
size = 2
dim = 2
n_particles = 64
n_grid = 2
dx = 1 / (n_grid)
inv_dx = 1 / dx
dt = 1e-2
p_mass = 1
p_vol = 1

nu = 0

max_steps = 100
steps = max_steps
gravity = 10





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
                          shape=(max_steps + 1, n_particles),
                          needs_grad=True)
# stresses = ti.Vector.field(6,
#                            dtype=real,
#                            shape=(max_steps+1, n_particles))
# stresses2 = ti.Vector.field(3,
#                            dtype=real,
#                            shape=(max_steps+1, n_particles))

loss = ti.field(dtype=real, shape=(), needs_grad=True)
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
        # stresses[f, p] = [-stress[0, 0], -stress[1, 1], 0, -stress[0, 1], 0, 0]
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
                



# @ti.kernel
# def grid_op(f: ti.i32):
#     for i, j in ti.ndrange(n_grid, n_grid):
#         inv_m = 1 / (grid_m_in[f, i, j] + 1e-10)
#         # external force implementation
#         v_out = inv_m * grid_v_in[f, i, j] + dt * f_ext[f, i, j]
#         v_out[1] -= dt * gravity
#         # boundary condition
#         if (i == 0 and j == 0):
#             v_out = [0, 0]
#             m_out = grid_m_in[f, i, j]
#             m_out = 0
#             grid_m_in[f, i, j] = m_out
#         if (i == 0 and j == 2):
#             v_out[1] = 0
#             m_out = grid_m_in[f, i, j]
#             m_out = 0
#             grid_m_in[f, i, j] = m_out
#         grid_v_out[f, i, j] = v_out

bound=1
@ti.kernel
def grid_op(f: ti.i32):
    for i, j in ti.ndrange(n_grid, n_grid):     
        inv_m = 1 / (grid_m_in[f, i, j] + 1e-10)
        v_out = inv_m * grid_v_in[f, i, j] + dt * f_ext[f, i, j]
        v_out[1] -= dt * gravity
        if i < bound and v_out[0] < 0:
            v_out[0] = 0
        if i > n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
        if j < bound and v_out[1] < 0:
            v_out[1] = 0
        if j > n_grid - bound and v_out[1] > 0:
            v_out[1] = 0
        grid_v_out[f, i, j] = v_out


@ti.kernel
def g2p(f: ti.i32):
    # ti.loop_config(serialize=True)
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
        
        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C
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
        # stresses2[f + 1, p] = stresses2[f, p] + de @ dstrain

       
@ti.kernel
def compute_loss():
    for i in range(steps - 1):
        for j in range(n_particles):
            dist = (1 / ((steps - 1) * n_particles)) * \
                (target_strain[i, j] - strain2[i, j]) ** 2
            loss[None] += 0.5 * (dist[0, 0] + dist[1, 1])


def substep(s):
    p2g(s)
    grid_op(s)
    g2p(s)

@ti.kernel
def assign_ext_load():
    for t in range(max_steps):
        f_ext[t, 2, 2] = [-10, 0]



print('assigning external loads')
assign_ext_load()


for i in range(n_particles):
    F[0, i] = [[1, 0], [0, 1]]



with open("particles.json", "r") as f:
    ploc = np.asarray(json.load(f))

for i in range(n_particles):
    x[0, i] = ploc[i, 0]

print('loading target')
target_strain_np = np.load('strain2_matrix_oldbound_grav.npy')
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
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
m_adam = 0
v_adam = 0


E[None] = 0.98 * 1e3
grad_iterations = 1000

losses = []
es = np.zeros((grad_iterations))

print('running opt')
optim = 'grad'
if optim == 'grad':

    for i in range(grad_iterations):
        grid_v_in.fill(0)
        grid_m_in.fill(0)
        loss[None] = 0

        with ti.ad.Tape(loss=loss):
            for s in range(steps - 1):
                substep(s)
            compute_loss()

        l = loss[None]
        losses.append(l)
        e = E[None]
        grad = E.grad[None]
        learning_rate = 1e-2
        m_adam = beta1 * m_adam + (1 - beta1) * grad
        v_adam = beta2 * v_adam + (1 - beta2) * grad**2
        m_hat = m_adam / (1 - beta1**(i + 1))
        v_hat = v_adam / (1 - beta2**(i + 1))
        E[None] -= learning_rate * m_hat / (ti.sqrt(v_hat) + epsilon)
        es[i] = E[None]
        print(i, 
            'loss=', l, 
            '   grad=', grad,
            '   E=', E[None])

    print(es)
    plt.title("Optimization of $E$ via $\epsilon (t)$")
    plt.ylabel("Loss")
    plt.xlabel("Gradient Descent Iterations")
    plt.plot(losses)
    plt.yscale('log')
    plt.show()

    plt.title("Learning Curve via $\epsilon (t)$")
    plt.ylabel("$E$")
    plt.xlabel("Iterations")
    plt.hlines(1e3, 0, grad_iterations, color='r', label='True Value')
    plt.plot(es, color='b', label='Estimated Value')
    plt.legend()
    plt.show()


elif optim == 'lbfgs':
    from scipy.optimize import minimize
    def compute_loss_and_grad(params):
        E[None] = params

        grid_v_in.fill(0)
        grid_m_in.fill(0)
        loss[None] = 0
        
        with ti.ad.Tape(loss=loss):
            for s in range(steps - 1):
                substep(s)
            compute_loss()

        loss_val = loss[None]
        grad_val = E.grad[None]

        return loss_val, grad_val


    initial_params = 1.05 * 1e3
    tol = 1e-18
    result = minimize(compute_loss_and_grad, 
                    initial_params, 
                    method='L-BFGS-B', 
                    jac=True, 
                    options={'disp': 1,'ftol': tol, 'gtol': tol, 'maxiter': 1000})

    print(result)