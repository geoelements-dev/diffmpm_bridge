import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import json, einops

ti.reset()
real = ti.f32
ti.init(arch=ti.cuda, default_fp=real, device_memory_GB=12)

# init parameters
size = 100 * 12 
span = 60 * 12 # in
depth = 4 * 12 # in
dim = 2
Nx = 90  # reduce to 30 if run out of GPU memory
Ny = 6
n_particles = Nx * Ny
n_grid = 25
dx = size / n_grid
inv_dx = 1 / dx
dt = 1e-3
p_vol = (span / Nx) * (depth / Ny)
assert (span / Nx) == (depth / Ny)
p_mass =  0.15 * p_vol / (12 * 12 * 12) # 0.15 klb/ft3 2400 kg/m3
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
                 ti.Matrix.diag(2, E[p] * nu / ((1 + nu) * (1 - 2 * nu)) * (J - 1) * J)
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
        if i == 5 and j == 9:
            v_out[0] = 0
            v_out[1] = 0
        if i == 20 and j == 9:
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

        xi = fx - ti.cast(ti.Vector([1, 1]), real)
        grad[0, 0] = 0.25 * xi[1] * (xi[1] - 1.) * (2 * xi[0] - 1.)
        grad[1, 0] = 0.25 * xi[1] * (xi[1] - 1.) * (2 * xi[0] + 1.)
        grad[2, 0] = 0.25 * xi[1] * (xi[1] + 1.) * (2 * xi[0] + 1.)
        grad[3, 0] = 0.25 * xi[1] * (xi[1] + 1.) * (2 * xi[0] - 1.)
        grad[4, 0] = -xi[0] * xi[1] * (xi[1] - 1.)
        grad[5, 0] = -0.5 * (2. * xi[0] + 1.) * ((xi[1] * xi[1]) - 1.)
        grad[6, 0] = -xi[0] * xi[1] * (xi[1] + 1.)
        grad[7, 0] = -0.5 * (2. * xi[0] - 1.) * ((xi[1] * xi[1]) - 1.)
        grad[8, 0] = 2. * xi[0] * ((xi[1] * xi[1]) - 1.)
        grad[0, 1] = 0.25 * xi[0] * (xi[0] - 1.) * (2. * xi[1] - 1.)
        grad[1, 1] = 0.25 * xi[0] * (xi[0] + 1.) * (2. * xi[1] - 1.)
        grad[2, 1] = 0.25 * xi[0] * (xi[0] + 1.) * (2. * xi[1] + 1.)
        grad[3, 1] = 0.25 * xi[0] * (xi[0] - 1.) * (2. * xi[1] + 1.)
        grad[4, 1] = -0.5 * (2. * xi[1] - 1.) * ((xi[0] * xi[0]) - 1.)
        grad[5, 1] = -xi[0] * xi[1] * (xi[0] + 1.)
        grad[6, 1] = -0.5 * (2. * xi[1] + 1.) * ((xi[0] * xi[0]) - 1.)
        grad[7, 1] = -xi[0] * xi[1] * (xi[0] - 1.)
        grad[8, 1] = 2. * xi[1] * ((xi[0] * xi[0]) - 1.)
        
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

    
        nodal_coordinates = ti.Matrix([
            [base[0], base[1]],
            [base[0] + 2, base[1]], 
            [base[0] + 2, base[1] + 2], 
            [base[0], base[1] + 2], 
            [base[0] + 1, base[1]], 
            [base[0] + 2, base[1] + 1], 
            [base[0] + 1, base[1] + 2], 
            [base[0], base[1] + 1], 
            [base[0] + 1, base[1] + 1]
            ], dt=ti.f32)
        nodal_coordinates = nodal_coordinates * dx
        J = grad.transpose() @ nodal_coordinates
        dn_dx = grad @ J.inverse().transpose()

        # calc strain rate
        for k in ti.static(range(9)):
            strain_rate[0] += dn_dx[k, 0] * vi[k, 0]
            strain_rate[1] += dn_dx[k, 1] * vi[k, 1]
            strain_rate[2] += dn_dx[k, 0] * vi[k, 1] + dn_dx[k, 1] * vi[k, 0]


        dstrain = strain_rate * dt
        strain2[f + 1, p][0, 0] = strain2[f, p][0, 0] + dstrain[0]
        strain2[f + 1, p][1, 1] = strain2[f, p][1, 1] + dstrain[1]
        strain2[f + 1, p][0, 1] = strain2[f, p][0, 1] + dstrain[2]
        strain2[f + 1, p][1, 1] = strain2[f, p][1, 0] + dstrain[2]

        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C



def substep(s):
    p2g(s)
    grid_op(s)
    g2p(s)



# f_ext_scale = 200
# start = 12
# stop = 13
# linear = f_ext_scale / ((stop - start + 1))

# @ti.kernel
# def assign_ext_load():
#     for t, node in ti.ndrange(max_steps, (start,stop+1)):
#             f_ext[t, node, 10] = [0, -linear]

f_ext_scale = 50
velocity = 100*12 # in/s
node_x_locs = ti.Vector(np.arange(0, 1, 1 / n_grid) * size)
time_to_center = node_x_locs / velocity
t_steps = ti.Vector(np.arange(max_steps)) * dt
t_steps_n = np.array([t_steps - time for time in time_to_center])
t_steps_n = np.stack(t_steps_n, axis=1)

t = np.asarray(t_steps_n)
fc, bw, bwr = 20, 0.5, -6
ref = np.power(10.0, bwr / 20.0)
a = -(np.pi * fc * bw) ** 2 / (4.0 * np.log(ref))
e_np = np.exp(-a * t * t)
e = ti.field(ti.f32, (steps, n_grid))
e.from_numpy(e_np)

@ti.kernel
def assign_ext_load():
    for t, node in ti.ndrange(max_steps, n_grid):
            f_ext[t, node, 10] = [0, -f_ext_scale * e[t, node]]


@ti.kernel
def assign_E():
    for i in range(n_particles):
        col = i % Nx
        if col < Nx*1/4 or col >= Nx*3/4:
            E[i] = 4.4e3
        elif col >= Nx*1/4 and col < Nx*3/4:
            E[i] = 3.6e3
        else:
            E[i] = 3.6e3




for i in range(n_particles):
    F[0, i] = [[1, 0], [0, 1]]



for i in range(Nx):
    for j in range(Ny):
        x[0, j * Nx + i] = [
            ((i)/(Nx) * 0.6 + 0.2 + 0.6 / Nx * 0.5) * size, 
            ((j)/(Ny) * 0.04 + 0.36 + 0.04 / Ny * 0.5) * size
            ]


print('running target sim')
assign_ext_load()
assign_E()

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
        for i in range(n_grid):
            for j in range(n_grid):
                node_locs[s, i * n_grid + j] = [i * dx, j * dx]
print('assigning node locs')
assign_node_locs()
gui = ti.GUI("Taichi Elements", (640, 640), background_color=0x000000)
out_dir = 'out_test'

frame = 0
x_np = x.to_numpy() / size
node_locs_np = node_locs.to_numpy() / size
load_locs_np = load_locs.to_numpy() / size
for s in range(0, 1, 1):
    scale = 4
    # gui.circles(x_np[s], color=0xFFFFFF, radius=1.5)
    x_np_reshape = einops.rearrange(x_np[s], '(w x h y) c -> (h w) x y c', h=3, w=1, x=6, c=2)
    gui.circles(x_np_reshape[[0]].reshape((-1, dim)), color=0x0000FF, radius=1.5)
    gui.circles(x_np_reshape[[1]].reshape((-1, dim)), color=0x008000, radius=1.5)
    gui.circles(x_np_reshape[[2]].reshape((-1, dim)), color=0xFFFF00, radius=1.5)
    # gui.circles(x_np_reshape[[3]].reshape((-1, dim)), color=0xFF0000, radius=1.5)
    # gui.circles(x_np_reshape[[2]].reshape((-1, dim)), color=0xFDB100, radius=1.5)
    gui.circles(node_locs_np[s], color=0xFFA500, radius=1)
    
    # gui.circle(load_locs_np[s], color=0xFF0000, radius=10)
    # gui.triangle([2 * dx, 6 * dx], [1.5 * dx, 5.5 * dx], [2.5 * dx, 5.5 * dx], color=0x00FF00)
    # gui.circle([18 * dx, 5.5 * dx], color=0x00FF00, radius=15)
    gui.show(f'{out_dir}/{frame:06d}.png')
    frame += 1



np.save('x_cs.npy', x.to_numpy())
np.save('s_cs1.npy', strain.to_numpy())
# np.save('strain2_final.npy', strain2.to_numpy())
np.save('s_cs.npy', strain2.to_numpy())
