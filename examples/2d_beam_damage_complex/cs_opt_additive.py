import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import json 

import argparse
# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("obs", type=str)
parser.add_argument("case", type=str)
parser.add_argument("n_search", type=str)
args = parser.parse_args()

# Extract arguments
obs = args.obs
case = args.case
n_search = int(args.n_search)

n_blocks_x = 180
n_blocks_y = 12

ti.reset()
real = ti.f32
ti.init(arch=ti.cuda, default_fp=real, device_memory_GB=4)

# init parameters
size = 100 * 12
span = 60 * 12
depth = 4 * 12
dim = 2
Nx = 180  # reduce to 30 if run out of GPU memory
Ny = 12
n_particles = Nx * Ny
n_grid = 25
dx = size / n_grid
inv_dx = 1 / dx
dt = 1e-3
p_vol = (span / Nx) * (depth / Ny)
assert (span / Nx) == (depth / Ny)
p_mass = 0.15 * p_vol / (12*12*12) # 0.15 klb/ft3 2400 kg/m3
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
f_scale = ti.field(dtype=real, shape=(), needs_grad=True)
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
        if i <= 5 and j <= 9:
            v_out[0] = 0
            v_out[1] = 0
        if i >= 20 and j <= 9:
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



@ti.kernel
def compute_loss():
    for i in range(steps - 1):
        if obs == "full":
            for j in range(n_particles):
                dist = (target_strain[i, j] - strain2[i, j]) ** 2
                loss[None] += 0.5 * (dist[0, 0])*1e20
        elif obs == "row":
            for j in range(Nx):
                dist = (target_strain[i, j] - strain2[i, j]) ** 2
                loss[None] += 0.5 * (dist[0, 0])*1e20
        elif obs == "sensor":
            for j in range(16):
                dist = (target_strain[i, j*12] - strain2[i, j*12]) ** 2
                loss[None] += 0.5 * (dist[0, 0])*1e20

def substep(s):
    p2g(s)
    grid_op(s)
    g2p(s)


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


n_blocks = n_blocks_y * n_blocks_x
block_nx = int(Nx / n_blocks_x)
block_ny = int(Ny / n_blocks_y)


@ti.kernel
def assign_E():
    for i in range(Nx):
        for j in range(Ny):
            block_index_x = i // block_nx
            block_index_y = j // block_ny
            E[j*Nx+i] = E_block[block_index_x + block_index_y * n_blocks_x]

for i in range(n_particles):
    F[0, i] = [[1, 0], [0, 1]]

for i in range(Nx):
    for j in range(Ny):
        x[0, j * Nx + i] = [
            ((i)/(Nx) * 0.6 + 0.2 + 0.6 / Nx * 0.5) * size, 
            ((j)/(Ny) * 0.04 + 0.36 + 0.04 / Ny * 0.5) * size
            ]


print('loading target')


target_strain_np = np.load(f's_cs_{case}.npy')

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

init_g[None] = 0


E_block = ti.field(dtype=real, shape=(n_blocks), needs_grad=True)

losses = []




print('running grad iterations')

from scipy.optimize import minimize

losses = []
E_hist = []



@ti.kernel
def assign_E_modular():
    for i in range(n_targets):
        E[target_indices[i]] = target_values[i]
    
    for i in range(n_locked):
        E[locked_indices[i]] = locked_values[i]

def compute_loss_and_grad_modular(params):
    grid_v_in.fill(0)
    grid_m_in.fill(0)
    loss[None] = 0
    for i in range(n_targets):
        target_values[i] = params[i]
    with ti.ad.Tape(loss=loss):
        assign_E_modular()
        assign_ext_load()
        for s in range(steps - 1):
            substep(s)
        compute_loss()

    loss_val = loss[None]
    grad_val = [target_values.grad[i] for i in range(n_targets)]

    # losses.append(loss_val)
    # E_hist.append(params.tolist())
    # print(z, 
        # 'loss=', loss, 
        # '   grad=', grad_val,
        # '   params=', params,
        # 'E: ', E.to_numpy()[locked_indices.to_numpy()[:n_locked]],
        # E[0],
        # )
    return loss_val, grad_val

init_e = 4e3

tol = 1e-36
options = {
    # 'disp': 1,
    'disp': 0, 
    'ftol': tol, 
    'gtol': tol,
    }
converged_threshold = 50

# n_search = 5

target_indices = ti.field(dtype=int, shape=(n_particles))
target_values = ti.field(dtype=real, shape=(n_particles), needs_grad=True)

locked_indices = ti.field(dtype=int, shape=(n_particles))
locked_values = ti.field(dtype=real, shape=(n_particles))

E_hist = []

import os
filename = f"r_c_add_{case}_{obs}_{n_search}_p1.json"
if not os.path.exists(filename):

    for z in range(n_search):
        print("searching: " + str(z))
        n_targets = n_particles - z
        n_locked = z

        target_indices_np = np.arange(n_particles)
        target_indices_np = np.delete(target_indices_np, locked_indices.to_numpy()[:n_locked])
        target_indices_np = np.concatenate((target_indices_np, np.zeros(n_locked)))
        if target_indices_np.shape[0] > n_particles:
            excess = target_indices_np.shape[0] - n_particles
            print("shape error: ", target_indices_np.shape[0], target_indices_np[-excess:], len(locked_indices.to_numpy()[:n_locked]))
            target_indices_np = target_indices_np[:-excess]
        target_indices.from_numpy(target_indices_np)
        # target_values.fill(init_e)

        params = np.zeros(n_targets) + init_e

        # run naive
        baseline = np.zeros(n_targets) + init_e
        deviation = 0

        converge_counter = 0
        while deviation < converged_threshold:
            # print("converge counter: ", converge_counter)
            result = minimize(compute_loss_and_grad_modular,
                        params,
                        method='L-BFGS-B',
                        jac=True,
                        bounds = [(1e1, 1e4) for _ in params],
                        options=options)
            # find highest deviation particle
            E_search = np.array(result.x)
            # E_search = np.delete(E_search, locked_indices.to_numpy()[:n_locked])
            deviation = max(E_search - baseline)

            converge_counter += 1
        

        minus_baseline = np.abs(E_search - init_e)
        damage_index = target_indices.to_numpy()[np.where(minus_baseline == max(minus_baseline))[0][0]]
        # lock particle
        curr_locked = locked_indices.to_numpy()[:n_locked]
        locked_indices[z] = damage_index #+ np.sum(curr_locked <= damage_index)
        locked_values[z] = E_search[damage_index]
        print('found: ', locked_values[z], " at ", locked_indices[z])
        # append result
        E_hist.append(E.to_numpy().tolist())


    locked_indices_arr = locked_indices.to_numpy()[:n_search]
    locked_values_arr = locked_values.to_numpy()[:n_search]

    # print(E_hist)
    print("phase 1 results:")
    print(locked_indices_arr)
    print(locked_values_arr)

    result_dict = {
        "E_hist" : E_hist,
        "locked_indices" : locked_indices_arr.tolist(),
        "locked_values" : locked_values_arr.tolist()
    }

    with open(filename, "w") as outfile: 
        json.dump(result_dict, outfile)


# name = f"r_c_add_{case}_{obs}_{n_search}_p1.json"
# with open(name) as json_file:
#         jf = json.load(json_file)

# locked_indices_arr = np.array(jf["locked_indices"])
# locked_values_arr = np.array(jf["locked_values"])

# # reoptimize locked values
# print('Phase 2')
# n_targets = n_search
# n_locked = n_particles - n_targets

# target_indices_np = np.concatenate((locked_indices_arr, np.zeros(n_locked)))
# target_indices.from_numpy(target_indices_np)

# params = np.zeros(n_targets) + init_e


# @ti.kernel
# def assign_E_modular_p2():
#     E.fill(init_e)
#     for i in range(n_targets):
#         E[target_indices[i]] = target_values[i]

# def compute_loss_and_grad_modular_p2(params):
#     grid_v_in.fill(0)
#     grid_m_in.fill(0)
#     loss[None] = 0
#     for i in range(n_targets):
#         target_values[i] = params[i]
#     with ti.ad.Tape(loss=loss):
#         assign_E_modular_p2()
#         assign_ext_load()
#         for s in range(steps - 1):
#             substep(s)
#         compute_loss()

#     loss_val = loss[None]
#     grad_val = [target_values.grad[i] for i in range(n_targets)]

#     return loss_val, grad_val

# deviation = 0
# converge_counter = 0
# baseline = np.zeros(n_targets) + init_e
# init_e_p2 = 100
# params = np.zeros(n_targets) + init_e_p2
# while deviation < converged_threshold:
#     print("converge counter: ", converge_counter)
#     result = minimize(compute_loss_and_grad_modular_p2,
#                 params,
#                 method='L-BFGS-B',
#                 jac=True,
#                 bounds = [(0, 10000) for i in range(len(params))],
#                 options=options)
#     # find highest deviation particle
#     E_search = np.array(result.x)
#     deviation = max(E_search - baseline)
#     print(E_search)
#     converge_counter += 1
# E_hist.append(E.to_numpy().tolist())

# print("final: ", result.x)
# print("found:")
# print(locked_indices_arr)

# result_dict = {
#     "E_hist" : E_hist,
#     "locked_indices" : locked_indices_arr.tolist(),
#     "locked_values" : locked_values_arr.tolist()
# }

# filename = f"r_c_add_{case}_{obs}_{n_search}_p2.json"
# with open(filename, "w") as outfile: 
#     json.dump(result_dict, outfile)

