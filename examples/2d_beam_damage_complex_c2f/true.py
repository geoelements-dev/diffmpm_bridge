import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import json, einops

ti.reset()
real = ti.f32
ti.init(arch=ti.cuda, default_fp=real, device_memory_GB=12)

# init parameters
size = 1
dim = 2
Nx = 80  # reduce to 30 if run out of GPU memory
Ny = 10
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
    # for i in range(n_particles):
    #     col = i % Nx
    #     if col % 20 < 20 or col % 20 >= 60:
    #         E[i] = E_params[0]
    #     else:
    #         if i < n_particles * 0.5:
    #             E[i] = E_params[2]
    #         else:
    #             E[i] = E_params[1]
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
        F[f + 1, p] = new_F
        J = (new_F).determinant()
        r, s = ti.polar_decompose(new_F)
        cauchy = 2 * E[p] / (2 * (1 + nu)) * (new_F - r) @ new_F.transpose() + \
                 ti.Matrix.diag(2, E[p] * nu / ((1 + nu) * (1 - 2 * nu)) * (J - 1) * J)
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
    # for t, node in ti.ndrange(max_steps, (2, 19)):
    #         f_ext[t, node, 14] = [0, -force[None] * e[t, node - 2]]
    for i, j in ti.ndrange(n_grid, n_grid):     
        inv_m = 1 / (grid_m_in[f, i, j] + 1e-10) 
        v_out = inv_m * grid_v_in[f, i, j] + dt * f_ext[f, i, j]
        # v_out[1] -= dt * gravity
        if i == 2 and j == 6:
            v_out[0] = 0
            v_out[1] = 0
        if i == 1 and j == 6:
            v_out[0] = 0
            v_out[1] = 0
        if i == 2 and j == 5:
            v_out[0] = 0
            v_out[1] = 0
        if i == 1 and j == 5:
            v_out[0] = 0
            v_out[1] = 0
        if i == 18 and j == 6:
            v_out[0] = 0
            v_out[1] = 0
        if i == 19 and j == 6:
            v_out[0] = 0
            v_out[1] = 0
        if i == 18 and j == 5:
            v_out[0] = 0
            v_out[1] = 0
        if i == 19 and j == 5:
            v_out[0] = 0
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
        for j in range(n_particles):
            dist = (target_strain[i, j] - strain2[i, j]) ** 2
            # dist = (1 / ((steps - 1) * n_particles)) * \
            #     (target_strain[i, j] - strain2[i, j]) ** 2
            loss[None] += 0.5 * (dist[0, 0] + dist[1, 1])

def substep(s):
    p2g(s)
    grid_op(s)
    g2p(s)



f_ext_scale = 5
velocity = 100# 16 / 20 / 0.1
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
e = ti.field(ti.f32, (steps, 17))
e.from_numpy(e_np)

@ti.kernel
def assign_ext_load():
    for t, node in ti.ndrange(max_steps, (2, 19)):
            f_ext[t, node, 8] = [0, -f_ext_scale* e[t, node - 2]]


from scipy.stats import multivariate_normal

def gaussian_damage(center, start=9000, cov=[[4, 0], [0, 4]]):
    x, y = np.meshgrid(np.arange(Nx), np.arange(Ny))
    coords = np.column_stack([x.ravel(), y.ravel()])
    dmg = multivariate_normal.pdf(coords, mean=center, cov=cov)
    dmg = dmg / dmg.max() * start
    return dmg

def gradient_damage(E_np, start, width, half_length, horizontal=True, E_start=1000, E_stop=10000):
    interp = np.interp(np.arange(half_length), [0, half_length-1], [E_start, E_stop])
    if horizontal:
        for row in np.arange(width):
            E_np[np.arange(start+Nx*row, start+half_length+Nx*row)] = interp
            E_np[np.arange(start+Nx*row, start-half_length+Nx*row, -1)] = interp
    else:
        for col in np.arange(width):
            E_np[np.arange(start+col, start+col+half_length*Nx, Nx)] = interp
    return E_np
# case = ''

# case = 'd'

# case = 'dm'

# case = 'g'
dmg = gaussian_damage([40, 0])
dmg_ti = ti.field(dtype=real, shape=(n_particles), needs_grad=True)
dmg_ti.from_numpy(dmg)

# case = 'gm'
dmg1 = gaussian_damage([30, 0])
dmg2 = gaussian_damage([50, 0])
dmg1_ti = ti.field(dtype=real, shape=(n_particles), needs_grad=True)
dmg2_ti = ti.field(dtype=real, shape=(n_particles), needs_grad=True)
dmg1_ti.from_numpy(dmg1)
dmg2_ti.from_numpy(dmg2)

case = 'h'
E_np_h = np.zeros(n_particles) + 10000
E_np_h = gradient_damage(E_np_h, 40, 1, 10)
E_h_ti = ti.field(dtype=real, shape=(n_particles), needs_grad=True)
E_h_ti.from_numpy(E_np_h)

case = 'v'
E_np_v = np.zeros(n_particles) + 10000
E_np_v = gradient_damage(E_np_v, 40, 2, 6, horizontal=False)
E_v_ti = ti.field(dtype=real, shape=(n_particles), needs_grad=True)
E_v_ti.from_numpy(E_np_v)

case = 'vm'
E_np_vm = np.zeros(n_particles) + 10000
E_np_vm = gradient_damage(E_np_vm, 30, 2, 6, horizontal=False)
E_np_vm = gradient_damage(E_np_vm, 50, 2, 6, horizontal=False)
E_vm_ti = ti.field(dtype=real, shape=(n_particles), needs_grad=True)
E_vm_ti.from_numpy(E_np_vm)

case = 'gt'
dmg_t = gaussian_damage([40, 9])
dmg_t_ti = ti.field(dtype=real, shape=(n_particles), needs_grad=True)
dmg_t_ti.from_numpy(dmg_t)

case = 'gtm'
dmg1_t = gaussian_damage([30, 9])
dmg2_t = gaussian_damage([50, 9])
dmg1_t_ti = ti.field(dtype=real, shape=(n_particles), needs_grad=True)
dmg2_t_ti = ti.field(dtype=real, shape=(n_particles), needs_grad=True)
dmg1_t_ti.from_numpy(dmg1_t)
dmg2_t_ti.from_numpy(dmg2_t)

case = 'ht'
E_np_ht = np.zeros(n_particles) + 10000
E_np_ht = gradient_damage(E_np_ht, 760, 1, 10)
E_ht_ti = ti.field(dtype=real, shape=(n_particles), needs_grad=True)
E_ht_ti.from_numpy(E_np_ht)


@ti.kernel
def assign_E():
    E.fill(10000)
    if case == 'd':
        for i in range(2):
            for j in range(2):
                E[40 + Nx*i + j] = 1000
    if case == 'dm':
        for i in range(2):
            for j in range(2):
                E[30 + Nx*i + j] = 1000
                E[50 + Nx*i + j] = 1000
    if case == 'g':
        for i in range(n_particles):
            E[i] = E[i] - dmg_ti[i]
    if case == 'gm':
        for i in range(n_particles):
            E[i] = E[i] - dmg1_ti[i] - dmg2_ti[i]
    if case == 'h':
        for i in range(n_particles):
            E[i] = E_h_ti[i]
    if case == 'v':
        for i in range(n_particles):
            E[i] = E_v_ti[i]
    if case == 'vm':
        for i in range(n_particles):
            E[i] = E_vm_ti[i]
    if case == 'gt':
        for i in range(n_particles):
            E[i] = E[i] - dmg_t_ti[i]
    if case == 'gtm':
        for i in range(n_particles):
            E[i] = E[i] - dmg1_t_ti[i] - dmg2_t_ti[i]
    if case == 'ht':
        for i in range(n_particles):
            E[i] = E_ht_ti[i]


for i in range(n_particles):
    F[0, i] = [[1, 0], [0, 1]]



for i in range(Nx):
    for j in range(Ny):
        x[0, j * Nx + i] = [(i)/(Nx) * 0.8 + 0.1, (j)/(Ny) * 0.1 + 0.3]


print('running target sim')
assign_ext_load()
assign_E()
print(E)

for s in range(steps):
    # print(s)
    substep(s)


node_locs = ti.Vector.field(dim,
                            dtype=real,
                            shape=(max_steps, n_grid * n_grid))
load_locs = ti.Vector.field(dim,
                            dtype=real,
                            shape=(max_steps))
pin_locs = ti.Vector.field(dim,
                            dtype=real,
                            shape=(max_steps))
roller_locs = ti.Vector.field(dim,
                            dtype=real,
                            shape=(max_steps))

@ti.kernel
def assign_node_locs():
    for s in range(max_steps):
        load_locs[s] = [2*dx + velocity * s * dt, 15 * dx]
        for i in range(n_grid):
            for j in range(n_grid):
                node_locs[s, i * n_grid + j] = [i * dx, j * dx]
print('assigning node locs')
assign_node_locs()
gui = ti.GUI("Taichi Elements", (640, 640), background_color=0x112F41)
out_dir = 'out_test'

frame = 0
x_np = x.to_numpy()
# node_locs_np = node_locs.to_numpy()
# load_locs_np = load_locs.to_numpy()
# for s in range(0, steps, 1):
#     scale = 4
#     gui.circles(x_np[s, :2*Nx], color=0x198C19, radius=1.5)
#     gui.circles(x_np[s, [42,43,42+80,43+80]], color=0xFF0000, radius=1.5)
#     gui.circles(x_np[s, 2*Nx:], color=0xFFA500, radius=1.5)
#     # x_np_reshape = einops.rearrange(x_np[s], '(w x h y) c -> (h w) x y c', h=4, w=1, x=10, c=2)
#     # gui.circles(x_np_reshape[[0,3]].reshape((-1, dim)), color=0x198C19, radius=1.5)
#     # gui.circles(x_np_reshape[[1,2]].reshape((-1, dim)), color=0xFF4400, radius=1.5)
#     # # gui.circles(x_np_reshape[[2]].reshape((-1, dim)), color=0xFDB100, radius=1.5)
#     gui.circles(node_locs_np[s], color=0xFFFFFF, radius=1)
    
#     # gui.circle(load_locs_np[s], color=0xFF0000, radius=10)
#     gui.arrow(orig=load_locs_np[s], direction = [0, -dx], color=0xFF0000, radius=3)
#     gui.triangle([2 * dx, 6 * dx], [1.5 * dx, 5.5 * dx], [2.5 * dx, 5.5 * dx], color=0x00FF00)
#     gui.triangle([18 * dx, 6 * dx], [17.5 * dx, 5.5 * dx], [18.5 * dx, 5.5 * dx], color=0x00FF00)
#     gui.show(f'{out_dir}/{frame:06d}.png')
#     frame += 1


np.save('x_true_' + case + '.npy', x_np)
np.save('strain2_true_' + case + '.npy', strain2.to_numpy())