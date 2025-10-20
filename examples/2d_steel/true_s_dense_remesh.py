import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import json, einops

ti.reset()
real = ti.f32
ti.init(arch=ti.cuda, default_fp=real, device_memory_GB=11)



# init parameters
size = 5.000 # 5 m
span = 4.400 # 4.4 m
depth = 0.400 # 0.4 m
dim = 2
factor = 1/100
Nx = int(span / factor)
Ny = int(depth / factor)
n_particles = int(Nx * Ny)
grid_factor = 4
n_grid = 25*grid_factor

dx = size / n_grid
inv_dx = 1 / dx
dt =  1 / 100000

nu = 0.3

max_steps = int(0.09 / dt) + 1
steps = max_steps
gravity = 0

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(dim, dtype=real)
mat = lambda: ti.Matrix.field(dim, dim, dtype=real)

x, v, a = vec(), vec(), vec()
grid_v_in, grid_m_in = vec(), scalar()
grid_v_out = vec()
grid_a = vec()
f_ext = vec()
C, F, E = mat(), mat(), scalar()
p_mass, p_vol = scalar(), scalar()
# loss = scalar()


# ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x, v, a)
# ti.root.dense(ti.i, n_particles).place(C, F, E, p_vol, p_mass)
# ti.root.dense(ti.ij, n_grid).place(grid_v_in, grid_m_in)
# ti.root.dense(ti.i, max_steps).dense(ti.jk, n_grid).place(grid_v_out, f_ext, grid_a)

ti.root.dense(ti.k, max_steps).dense(ti.l, n_particles).place(x, v, a, C, F)
ti.root.dense(ti.i, n_particles).place(E, p_vol, p_mass)
# ti.root.dense(ti.ij, n_grid).place()
ti.root.dense(ti.i, max_steps).dense(ti.jk, n_grid).place(grid_v_out, f_ext, grid_a, grid_v_in, grid_m_in)

# ti.root.lazy_grad()


# a2 = ti.Vector.field(dim, dtype=real, shape=(max_steps, n_particles), needs_grad=True)

@ti.kernel
def clear_grid():
    for f in range(max_steps):
        for i, j in ti.ndrange(n_grid, n_grid):
            grid_v_in[f, i, j] = [0, 0]
            grid_m_in[f, i, j] = 0


@ti.kernel
def clear_particles():
    for p in C:
        F[p] = 0
        C[p] = 0
        

@ti.kernel
def p2g(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
        F[f+1, p] = new_F
        J = (new_F).determinant()
        r, _ = ti.polar_decompose(new_F)
        cauchy = 2 * E[p] / (2 * (1 + nu)) * (new_F - r) @ new_F.transpose() + \
                 ti.Matrix.diag(2, E[p] * nu / ((1 + nu) * (1 - 2 * nu)) * (J - 1) * J)
        stress = -(dt * p_vol[p] * 4 * inv_dx * inv_dx) * cauchy
        affine = stress + p_mass[p] * C[f, p]
        # strain[f, p] += 0.5 * (new_F.transpose() @ new_F - ti.math.eye(dim))
        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                offset = ti.Vector([i, j])
                dpos = (ti.cast(ti.Vector([i, j]), real) - fx) * dx
                weight = w[i][0] * w[j][1]
                ti.atomic_add(grid_v_in[f, base + offset], weight * (p_mass[p] * v[f, p] +
                                                         affine @ dpos))
                ti.atomic_add(grid_m_in[f, base + offset], weight * p_mass[p])

@ti.kernel
def grid_op(f: ti.i32):
    for i, j in ti.ndrange(n_grid, n_grid):     
        inv_m = 1 / (grid_m_in[f, i, j] + 1e-10) 
        v_out = inv_m * grid_v_in[f, i, j] + dt * f_ext[f, i, j] * inv_m
        if i == 3*grid_factor and j <= 5*grid_factor:
            v_out[0] = 0
            # v_out[1] = 0
        if i ==23*grid_factor and j <= 5*grid_factor:
            v_out[0] = 0
            # v_out[1] = 0
        grid_v_out[f, i, j] = v_out

        if f > 0:
            grid_a[f, i, j] = (1/dt) * (grid_v_out[f, i, j] - grid_v_out[f - 1, i, j])


@ti.kernel
def g2p(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, real)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1.0)**2, 0.5 * (fx - 0.5)**2]
        new_v = ti.Vector([0.0, 0.0])
        new_C = ti.Matrix([[0.0, 0.0], [0.0, 0.0]])
        new_a = ti.Vector([0.0, 0.0])

        for i in ti.static(range(3)):
            for j in ti.static(range(3)):
                dpos = ti.cast(ti.Vector([i, j]), real) - fx
                g_v = grid_v_out[f, base[0] + i, base[1] + j]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

                g_a = grid_a[f, base[0] + i, base[1] + j]
                new_a += weight * g_a

        # ### stress and strain from nodal velocity
        # # shape function gradient
        # grad = ti.Matrix([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 
        #                   [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 
        #                   [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        # vi = ti.Matrix([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 
        #                 [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], 
        #                 [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        # strain_rate = ti.Vector([0.0, 0.0, 0.0])

        # xi = fx - ti.cast(ti.Vector([1, 1]), real)
        # grad[0, 0] = 0.25 * xi[1] * (xi[1] - 1.) * (2 * xi[0] - 1.)
        # grad[1, 0] = 0.25 * xi[1] * (xi[1] - 1.) * (2 * xi[0] + 1.)
        # grad[2, 0] = 0.25 * xi[1] * (xi[1] + 1.) * (2 * xi[0] + 1.)
        # grad[3, 0] = 0.25 * xi[1] * (xi[1] + 1.) * (2 * xi[0] - 1.)
        # grad[4, 0] = -xi[0] * xi[1] * (xi[1] - 1.)
        # grad[5, 0] = -0.5 * (2. * xi[0] + 1.) * ((xi[1] * xi[1]) - 1.)
        # grad[6, 0] = -xi[0] * xi[1] * (xi[1] + 1.)
        # grad[7, 0] = -0.5 * (2. * xi[0] - 1.) * ((xi[1] * xi[1]) - 1.)
        # grad[8, 0] = 2. * xi[0] * ((xi[1] * xi[1]) - 1.)
        # grad[0, 1] = 0.25 * xi[0] * (xi[0] - 1.) * (2. * xi[1] - 1.)
        # grad[1, 1] = 0.25 * xi[0] * (xi[0] + 1.) * (2. * xi[1] - 1.)
        # grad[2, 1] = 0.25 * xi[0] * (xi[0] + 1.) * (2. * xi[1] + 1.)
        # grad[3, 1] = 0.25 * xi[0] * (xi[0] - 1.) * (2. * xi[1] + 1.)
        # grad[4, 1] = -0.5 * (2. * xi[1] - 1.) * ((xi[0] * xi[0]) - 1.)
        # grad[5, 1] = -xi[0] * xi[1] * (xi[0] + 1.)
        # grad[6, 1] = -0.5 * (2. * xi[1] + 1.) * ((xi[0] * xi[0]) - 1.)
        # grad[7, 1] = -xi[0] * xi[1] * (xi[0] - 1.)
        # grad[8, 1] = 2. * xi[1] * ((xi[0] * xi[0]) - 1.)
        
        # vi[0, 0] = grid_v_out[f, base[0], base[1]][0]
        # vi[1, 0] = grid_v_out[f, base[0] + 2, base[1]][0]
        # vi[2, 0] = grid_v_out[f, base[0] + 2, base[1] + 2][0]
        # vi[3, 0] = grid_v_out[f, base[0], base[1] + 2][0]
        # vi[4, 0] = grid_v_out[f, base[0] + 1, base[1]][0]
        # vi[5, 0] = grid_v_out[f, base[0] + 2, base[1] + 1][0]
        # vi[6, 0] = grid_v_out[f, base[0] + 1, base[1] + 2][0]
        # vi[7, 0] = grid_v_out[f, base[0], base[1] + 1][0]
        # vi[8, 0] = grid_v_out[f, base[0] + 1, base[1] + 1][0]
        # vi[0, 1] = grid_v_out[f, base[0], base[1]][1]
        # vi[1, 1] = grid_v_out[f, base[0] + 2, base[1]][1]
        # vi[2, 1] = grid_v_out[f, base[0] + 2, base[1] + 2][1]
        # vi[3, 1] = grid_v_out[f, base[0], base[1] + 2][1]
        # vi[4, 1] = grid_v_out[f, base[0] + 1, base[1]][1]
        # vi[5, 1] = grid_v_out[f, base[0] + 2, base[1] + 1][1]
        # vi[6, 1] = grid_v_out[f, base[0] + 1, base[1] + 2][1]
        # vi[7, 1] = grid_v_out[f, base[0], base[1] + 1][1]
        # vi[8, 1] = grid_v_out[f, base[0] + 1, base[1] + 1][1]

    
        # nodal_coordinates = ti.Matrix([
        #     [base[0], base[1]],
        #     [base[0] + 2, base[1]], 
        #     [base[0] + 2, base[1] + 2], 
        #     [base[0], base[1] + 2], 
        #     [base[0] + 1, base[1]], 
        #     [base[0] + 2, base[1] + 1], 
        #     [base[0] + 1, base[1] + 2], 
        #     [base[0], base[1] + 1], 
        #     [base[0] + 1, base[1] + 1]
        #     ], dt=ti.f32)
        # nodal_coordinates = nodal_coordinates * dx
        # J = grad.transpose() @ nodal_coordinates
        # dn_dx = grad @ J.inverse().transpose()

        # # calc strain rate
        # for k in ti.static(range(9)):
        #     strain_rate[0] += dn_dx[k, 0] * vi[k, 0]
        #     strain_rate[1] += dn_dx[k, 1] * vi[k, 1]
        #     strain_rate[2] += dn_dx[k, 0] * vi[k, 1] + dn_dx[k, 1] * vi[k, 0]


        # dstrain = strain_rate * dt
        # strain2[f + 1, p][0, 0] = strain2[f, p][0, 0] + dstrain[0]
        # strain2[f + 1, p][1, 1] = strain2[f, p][1, 1] + dstrain[1]
        # strain2[f + 1, p][0, 1] = strain2[f, p][0, 1] + dstrain[2]
        # strain2[f + 1, p][1, 1] = strain2[f, p][1, 0] + dstrain[2]

        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C
        a[f + 1, p] = new_a



def substep(s):
    # clear_grid()
    p2g(s)
    grid_op(s)
    g2p(s)


import os
import regex as re

files = os.listdir('impact_data')
file_index = 0
file = files[file_index]
loc = int(re.findall("impactloc(\d)", file)[0])
loads = np.load("loads.npy")[file_index]

true_dt = 1/2000
dt_ratio = true_dt / dt
load_np = np.interp(np.arange(max_steps), np.arange(len(loads))*dt_ratio, loads)

load = ti.field(ti.f32, (max_steps))
load.from_numpy(load_np)

# apply load at nodes [6, 8, 10, 12, 14, 16, 18, 20]
load_locs = np.array([6, 8, 10, 12, 14, 16, 18, 20])*grid_factor



@ti.kernel
def assign_ext_load():
    for t in ti.ndrange(max_steps):
            f_ext[t, load_locs[loc], 7*grid_factor] = [0, -load[t]] # N


flange_height = 0.05
@ti.kernel
def assign_E():
    for i in range(Nx):
        for j in range(Ny):
            x[0, j * Nx + i] = [
                ((i)/(Nx) * 0.88 + 0.06 + 0.88 / Nx * 0.5) * size, 
                # ((j)/(Ny) * 0.078 + 0.202 + 0.078 / Ny * 0.5) * size
                ((j)/(Ny) * 0.08 + 0.2 + 0.08 / Ny * 0.5) * size
                ]
            if j/Ny <= flange_height or (j+1)/Ny >= 1-flange_height:
                E[j * Nx + i] = 200e9 * 0.3
                p_vol[j * Nx + i] = (span / Nx) * (depth / Ny) * 0.3
            else:
                E[j * Nx + i] = 200e9 * 0.011
                p_vol[j * Nx + i] = (span / Nx) * (depth / Ny) * 0.011
    for p in range(n_particles):
        F[0, p] = [[1, 0], [0, 1]]
        p_mass[p] =  7800 * p_vol[p] # 7.8 g/cm3

@ti.kernel
def acc(f: ti.i32):
    for p in range(n_particles):
        a2[f, p] = (1/dt) * (v[f, p] - v[f-1, p])



print('initializing sim')
assign_ext_load()
assign_E()
print('running target sim')
for s in range(steps):
    # print(s)
    substep(s)
    # acc(s + 1)
    # print(a.to_numpy()[s,n_particles-10*19,1])
    # print(a2.to_numpy()[s,n_particles-10*19,1])



np.save(f'a_{loc}_{file_index}5.npy', a.to_numpy())
np.save(f'g_a_{loc}_{file_index}5.npy', grid_a.to_numpy())
np.save(f'mpm_load.npy', f_ext.to_numpy())
# np.save(f'a2_{loc}_{file_index}.npy', a2.to_numpy())
np.save(f'x.npy', x.to_numpy())
# np.save(f's_cs1_{loc}_{f}.npy', strain.to_numpy())
# np.save(f's_cs_{loc}_{f}.npy', strain2.to_numpy())
