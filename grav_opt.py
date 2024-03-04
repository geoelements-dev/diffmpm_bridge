import taichi as ti
import numpy as np
import engine.utils as utils
import matplotlib.pyplot as plt

ti.reset()
real = ti.f32
ti.init(arch=ti.cuda, default_fp=real, device_memory_GB=12)

# init parameters
size = 1
dim = 2
N = 60  # reduce to 30 if run out of GPU memory
n_particles = N * N
n_grid = 120
dx = 1 / n_grid
inv_dx = 1 / dx
dt_scale = 1e0
dt = 2e-2 * dx / size * dt_scale
dt = 3e-4
p_mass = 1
p_vol = 1
# E = ti.field(dtype=real, shape=(), needs_grad=True)
# nu = 0.2
# mu = ti.field(dtype=real, shape=(), needs_grad=True)
# la = ti.field(dtype=real, shape=(), needs_grad=True)
# E[None] = 100
E = 100
mu = E
la = E
max_steps = 1024
steps = max_steps
gravity = 0
# target = [0.3, 0.6]

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

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
init_v = ti.Vector.field(dim, dtype=real, shape=(), needs_grad=True)
loss = ti.field(dtype=real, shape=(), needs_grad=True)
init_g = ti.field(dtype=real, shape=(), needs_grad=True)

@ti.kernel
def set_v():
    for i in range(n_particles):
        v[0, i] = init_v[None]

@ti.kernel
def set_g():
    gravity = init_g[None]

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
        # cauchy = 2 * E[None] / (2 * (1 + nu)) * (new_F - r) @ new_F.transpose() + \
        #          ti.Matrix.diag(2, E[None] * nu / ((1 + nu) * (1 - 2 * nu)) * (J - 1) * J)
        cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
                 ti.Matrix.diag(2, la * (J - 1) * J)
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

bound = 3

@ti.kernel
def grid_op(f: ti.i32):
    for i, j in ti.ndrange(n_grid, n_grid):     
        inv_m = 1 / (grid_m_in[f, i, j] + 1e-10) # + dt * grid_v_ext[f, i, j])
        v_out = inv_m * grid_v_in[f, i, j] 
        v_out[1] -= dt * init_g[None]
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

@ti.kernel
def compute_x_avg():
    for i in range(n_particles):
        x_avg[None] += (1 / n_particles) * x[steps - 1, i]

@ti.kernel
def compute_loss():
    for i in range(steps - 1):
        for j in range(n_particles):
            dist = (1 / ((steps - 1) * n_particles)) * \
                (target_x[i, j] - x[i, j]) ** 2
            loss[None] += 0.5 * (dist[0] + dist[1])
    # dist = (x_avg[None] - ti.Vector(target))**2
    # loss[None] = 0.5 * (dist[0] + dist[1])

def substep(s):
    # if s == 0:
    #     print(init_g[None])
    p2g(s)
    grid_op(s)
    g2p(s)

# @ti.kernel
# def set_E():
#     mu[None] = E[None] / (2 * (1 + nu))
#     la[None] = E[None] * nu / ((1 + nu) * (1 - 2 * nu))
    # print(mu, la)

@ti.kernel
def reset_sim():
    strain.fill(0)
    grid_m_in.fill(0)
    grid_v_in.fill(0)

    # for i in range(n_particles):
    #     F[0, i] = [[1, 0], [0, 1]]

    for i in range(N):
        for j in range(N):
            x[0, i * N + j] = [(i)/(2*N), (j)/(2*N)]

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



init_v[None] = [0., 0.]


for i in range(n_particles):
    F[0, i] = [[1, 0], [0, 1]]

# for i in range(N):
#     for j in range(N):
#         x[0, i * N + j] = [dx * (i * 0.7 + 10), dx * (j * 0.7 + 25)]


for i in range(N):
    for j in range(N):
        x[0, i * N + j] = [(i)/(4*N) + 0.125, (j)/(4*N)]





# print('running target sim')
# reset_sim()
# set_v()


# for s in range(steps):
#     substep(s)

print('loading target')
# target_x = x
# target_strain = strain
target_strain_np = np.load('strain_grav.npy')
target_x_np = np.load('x_grav.npy')
target_x = ti.Vector.field(dim,
                           dtype=real,
                           shape=(max_steps, n_particles),
                           needs_grad=True)
target_strain = ti.Matrix.field(dim,
                            dim,
                           dtype=real,
                           shape=(max_steps, n_particles),
                           needs_grad=True)

@ti.kernel
def load_target(target_np: ti.types.ndarray()):
    for i, j, k in ti.ndrange(steps, n_particles, dim):
        target_x[i, j][k] = target_np[i, j, k]
    # for i, j, k, l in ti.ndrange(steps, n_particles, dim, dim):
    #     target_strain[i, j][k, l] = target_np[i, j, k, l]

load_target(target_x_np)


# gui = ti.GUI("Taichi Elements", (640, 640), background_color=0x112F41)
# out_dir = 'out_test'

# frame = 0
# x_np = x.to_numpy()
# for s in range(steps):
#     scale = 4
#     gui.circles(x_np[s], color=0xFFFFFF, radius=1.5)
#     gui.show(f'{out_dir}/{frame:06d}.png')
#     frame += 1

# np.save('x_grav.npy', x.to_numpy())
# np.save('grid_v_in.npy', grid_v_in.to_numpy())
# np.save('grid_v_out.npy', grid_v_out.to_numpy())
# np.save('grid_v_ext.npy', grid_v_ext.to_numpy())
# np.save('strain.npy', strain.to_numpy())
# np.save('target_strain_simple.npy', target_strain.to_numpy())


init_g[None] = 9.5
grad_iterations = 50

losses = []
gs = np.zeros((grad_iterations))
# init_v[None] = [0, 0]
print('running grad iterations')
for i in range(grad_iterations):
    grid_v_in.fill(0)
    grid_m_in.fill(0)
    loss[None] = 0
#     x_avg[None] = [0, 0]
    with ti.ad.Tape(loss=loss):
        # reset_sim()
        # set_v()
        for s in range(steps - 1):
            substep(s)
        compute_x_avg()
        compute_loss()

    l = loss[None]
    losses.append(l)
#     v = init_v[None]
    g = init_g[None]
    grad = init_g.grad[None]
#     grad = init_v.grad[None]
    learning_rate = 1e7
    init_g[None] -= learning_rate * grad
#     learning_rate = 1e1
#     init_v[None][0] -= learning_rate * grad[0]
#     init_v[None][1] -= learning_rate * grad[1]
    gs[i] = np.array([g])
    print('loss=', l, 
          '   grad=', grad,
          '   g=', init_g[None])
#     print('loss=', l, '   grad=', (grad[0], grad[1]), '   v=', init_v[None])
# print('done')
# # vs = np.vstack(np.array(vs))
print(gs)
plt.title("Optimization of $g$ via $x(t)$")
plt.ylabel("Loss")
plt.xlabel("Gradient Descent Iterations")
plt.plot(losses)
plt.yscale('log')
plt.show()

plt.title("Learning Curve via $x(t)$")
plt.ylabel("$g$")
plt.xlabel("Iterations")
plt.hlines(10, 0, grad_iterations, color='r', label='True Value')
plt.plot(gs, color='b', label='Estimated Value')
plt.legend()
plt.show()

# plt.title("$V_{0,y}$ Learning Curve via $x(t)$ (Single Step)")
# plt.ylabel("$V_{0,y}$")
# plt.xlabel("Iterations")
# plt.hlines(0.6, 0, 30, color='r', label='True Value')
# plt.plot(vs[:,1], color='b', label='Estimated Value')
# plt.legend()
# plt.show()