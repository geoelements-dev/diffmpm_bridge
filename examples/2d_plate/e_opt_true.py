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
E = ti.field(dtype=real, shape=(), needs_grad=True)
# nu = 0.2
# mu = ti.field(dtype=real, shape=(), needs_grad=True)
# la = ti.field(dtype=real, shape=(), needs_grad=True)
E[None] = 1e4
# E = 1e3
nu = 0.2
mu = E
la = E
max_steps = 1024
steps = max_steps
gravity = 9.81
# target = [0.3, 0.6]


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
strain2 = ti.Matrix.field(dim,
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
        # cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
        #          ti.Matrix.diag(2, la * (J - 1) * J)
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
        v_out = inv_m * grid_v_in[f, i, j] 
        v_out[1] -= dt * gravity
        if i == 5 and j == 5:
            v_out[0] = 0
            v_out[1] = 0
        if i == 14 and j == 5:
            v_out[1] = 0
        grid_v_out[f, i, j] = v_out

# bound = 6

# @ti.kernel
# def grid_op(f: ti.i32):
#     for i, j in ti.ndrange(n_grid, n_grid):     
#         inv_m = 1 / (grid_m_in[f, i, j] + 1e-10) # + dt * grid_v_ext[f, i, j])
#         v_out = inv_m * grid_v_in[f, i, j] 
#         v_out[1] -= dt * gravity
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
            dist = (1 / ((steps - 1) * n_particles)) * \
                (target_x[i, j] - x[i, j]) ** 2
            loss[None] += 0.5 * (dist[0] + dist[1])
    # dist = (x_avg[None] - ti.Vector(target))**2
    # loss[None] = 0.5 * (dist[0] + dist[1])

def substep(s):
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
init_g[None] = 9.81

for i in range(n_particles):
    F[0, i] = [[1, 0], [0, 1]]

# for i in range(N):
#     for j in range(N):
#         x[0, i * N + j] = [dx * (i * 0.7 + 10), dx * (j * 0.7 + 25)]


for i in range(N):
    for j in range(N):
        x[0, i * N + j] = [(i)/(4*N) + 0.125, (j)/(4*N) + 0.125]





print('running target sim')
# reset_sim()
# set_v()


for s in range(steps):
    # print(s)
    substep(s)


# node_locs = ti.Vector.field(dim,
#                             dtype=real,
#                             shape=(max_steps, n_grid * n_grid))
# @ti.kernel
# def assign_node_locs():
#     for s in range(max_steps):
#         for i in range(n_grid):
#             for j in range(n_grid):
#                 node_locs[s, i * n_grid + j] = [i * dx, j * dx]
# print('assigning node locs')
# assign_node_locs()
# gui = ti.GUI("Taichi Elements", (640, 640), background_color=0x112F41)
# out_dir = 'out_test'

# frame = 0
# x_np = x.to_numpy()
# node_locs_np = node_locs.to_numpy()
# for s in range(0, steps, 10):
#     scale = 4
#     gui.circles(x_np[s], color=0xFFFFFF, radius=1.5)
#     gui.circles(node_locs_np[s], color=0xFFA500, radius=1)
#     gui.show(f'{out_dir}/{frame:06d}.png')
#     frame += 1



# np.save('x_e_realistic.npy', x.to_numpy())
# np.save('grid_v_in.npy', grid_v_in.to_numpy())
# np.save('grid_v_out.npy', grid_v_out.to_numpy())
# np.save('grid_v_ext.npy', grid_v_ext.to_numpy())
# np.save('strain_e_realistic.npy', strain.to_numpy())
np.save('strain2_e_newboundaries.npy', strain2.to_numpy())
# np.save('target_strain_simple.npy', target_strain.to_numpy())
