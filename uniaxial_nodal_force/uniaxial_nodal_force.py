import taichi as ti
import numpy as np
import matplotlib.pyplot as plt

ti.reset()
real = ti.f32
ti.init(arch=ti.cuda, default_fp=real, device_memory_GB=12)

# init parameters
size = 1
dim = 2
N = 12
n_particles = 12
n_grid = 10
dx = 1 / n_grid
inv_dx = 1 / dx
dt = 1e-3
p_mass = 1000
p_vol = 1
E = 1000000.0
nu = 0
mu = E
la = E
max_steps = 1000
steps = max_steps
gravity = 0

G = E / (2 * 1 + nu)
bulk_modulus = E / (3 * 1 - 2 * nu)
a1 = bulk_modulus + (4 * G / 3)
a2 = bulk_modulus - (2 * G / 3)

de = ti.Matrix([
                [a1, a2, 0],
                [a2, a1, 0],
                [0, 0, G]
            ])

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
strain = ti.Matrix.field(dim,
                         dim,
                         dtype=real,
                         shape=(max_steps, n_particles),
                         needs_grad=True)
strain2 = ti.Vector.field(3,
                          dtype=real,
                          shape=(max_steps + 1, n_particles),
                          needs_grad=True)
stresses = ti.Vector.field(6,
                           dtype=real,
                           shape=(max_steps+1, n_particles))
stresses2 = ti.Vector.field(3,
                           dtype=real,
                           shape=(max_steps+1, n_particles))
init_v = ti.Vector.field(dim, dtype=real, shape=(), needs_grad=True)
loss = ti.field(dtype=real, shape=(), needs_grad=True)

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
        cauchy = 2 * E / (2 * (1 + nu)) * (new_F - r) @ new_F.transpose() + \
                 ti.Matrix.diag(2, E * nu / ((1 + nu) * (1 - 2 * nu)) * (J - 1) * J)
        stress = -(dt * p_vol * 4 * inv_dx * inv_dx) * cauchy
        stresses[f, p] = [-stress[0, 0], -stress[1, 1], 0, -stress[0, 1], 0, 0]
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
        # external force implementation
        v_out = inv_m * grid_v_in[f, i, j] + dt * f_ext[f, i, j]
        v_out[1] -= dt * gravity
        # boundary condition
        if i == 0 and j == 0:
            v_out[0] = 0
            m_out = grid_m_in[f, i, j]
            m_out = 0
            grid_m_in[f, i, j] = m_out
        if i == 0 and j == 1:
            v_out[0] = 0
            m_out = grid_m_in[f, i, j]
            m_out = 0
            grid_m_in[f, i, j] = m_out
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
        dstrain = strain_rate * dt
        strain2[f + 1, p] = strain2[f, p] + dstrain
        stresses2[f + 1, p] = stresses2[f, p] + de @ dstrain

       



def substep(s):
    p2g(s)
    grid_op(s)
    g2p(s)


@ti.kernel
def assign_ext_load():
    for t in range(max_steps):
        if t < 500:
            f_ext[t, 3, 0] = [0.05 * t * dt * 2, 0]
            f_ext[t, 3, 1] = [0.05 * t * dt * 2, 0]
        if t >= 500:
            f_ext[t, 3, 0] = [0.05, 0]
            f_ext[t, 3, 1] = [0.05, 0]


print('assigning external loads')
assign_ext_load()


for i in range(n_particles):
    F[0, i] = [[1, 0], [0, 1]]



for i in range(N):
    for j in range(N):
        x[0, 0] = [0.025, 0.025]
        x[0, 1] = [0.075, 0.025]
        x[0, 2] = [0.125, 0.025]
        x[0, 3] = [0.175, 0.025]
        x[0, 4] = [0.225, 0.025]
        x[0, 5] = [0.275, 0.025]
        x[0, 6] = [0.025, 0.075]
        x[0, 7] = [0.075, 0.075]
        x[0, 8] = [0.125, 0.075]
        x[0, 9] = [0.175, 0.075]
        x[0, 10] = [0.225, 0.075]
        x[0, 11] = [0.275, 0.075]
        




print('running target sim')

for s in range(steps):
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
# for s in range(steps):
#     scale = 4
#     gui.circles(x_np[s], color=0xFFFFFF, radius=1.5)
#     gui.circles(node_locs_np[s], color=0xFFA500, radius=1)
#     gui.show(f'{out_dir}/{frame:06d}.png')
#     frame += 1
# np.save('x.npy', x.to_numpy())
# np.save('stresses.npy', stresses.to_numpy())
np.save('stresses2.npy', stresses2.to_numpy())
