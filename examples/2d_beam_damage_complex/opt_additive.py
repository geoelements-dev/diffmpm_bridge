import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import json 

import argparse

obs_choices = ["full", "sensor"]
cases = ['d', 'dm', 'g', 'gm', 'h', 'v', 'vm', 'gt', 'gtm', 'ht']
n_search_list = [10]

obs = obs_choices[0]
case = cases[1]
n_search = n_search_list[0]

# # Parse command-line arguments
# parser = argparse.ArgumentParser(description="Run additive optimization.")
# parser.add_argument("obs", type=str, choices=["full", "sensor"], help="Observation choice")
# parser.add_argument("case", type=str, help="Case identifier")
# parser.add_argument("n_search", type=int, help="Number of search iterations")
# args = parser.parse_args()

# # Extract arguments
# obs = args.obs
# case = args.case
# n_search = args.n_search



ti.reset()
real = ti.f32
ti.init(arch=ti.gpu, default_fp=real, device_memory_GB=4, advanced_optimization=True, debug=True)

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
    for i, j in ti.ndrange(n_grid, n_grid):     
        inv_m = 1 / (grid_m_in[f, i, j] + 1e-10) 
        v_out = inv_m * grid_v_in[f, i, j] + dt * f_ext[f, i, j]
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
        if obs == "full":
            for j in range(n_particles):
                dist = (target_strain[i, j] - strain2[i, j]) ** 2
                loss[None] += 0.5 * (dist[0, 0])*1e36# + dist[1, 1]) * 1e16
        elif obs == "row":
            for j in range(Nx):
                dist = (target_strain[i, j] - strain2[i, j]) ** 2
                loss[None] += 0.5 * (dist[0, 0])*1e36# + dist[1, 1]) * 1e16
        elif obs == "sensor":
            for j in range(16):
                dist = (target_strain[i, j*5] - strain2[i, j*5]) ** 2
                loss[None] += 0.5 * (dist[0, 0])*1e36# + dist[1, 1]) * 1e16

@ti.kernel
def loss_weight(i: ti.i32) -> ti.f32:
    return ti.math.pow(i - 40, 2.0) + 1

def substep(s):
    p2g(s)
    grid_op(s)
    g2p(s)

f_ext_scale = 5
velocity = 100 #16 / 20 / 0.1
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
            f_ext[t, node, 8] = [0, -5* e[t, node - 2]]

for i in range(n_particles):
    F[0, i] = [[1, 0], [0, 1]]

for i in range(Nx):
    for j in range(Ny):
        x[0, j * Nx + i] = [(i)/(Nx) * 0.8 + 0.1, (j)/(Ny) * 0.1 + 0.3]

print('loading target')
obs_choices = ["full", "row", "sensor"]
cases = ['', 'd', 'dm', 'g', 'gm', 'h', 'v', 'vm', 'gt', 'gtm' , 'ht']

# obs = obs_choices[2]
# case = 'dm'
# n_search = 10

if case == '':
    target_strain_name = "strain2_true.npy"
else:
    target_strain_name = "strain2_true_"+ case + ".npy"
target_strain_np = np.load(target_strain_name)
target_strain = ti.Matrix.field(dim,
                            dim,
                           dtype=real,
                           shape=(max_steps, n_particles),
                           needs_grad=True)
target_x_np = np.load('x_true.npy')
target_x = ti.Vector.field(dim,
                           dtype=real,
                           shape=(max_steps, n_particles),
                           needs_grad=True)

# inject noise to eps_xx direction
SNR_dB = 1000
SNR_linear = 10 ** (SNR_dB / 10)
# get avg signal power in eps_xx
P_signal = np.mean(target_strain_np[:,:Nx,0,0]**2)
P_noise = P_signal / SNR_linear
# particle-wise noise
noise = np.random.normal(0, P_noise ** 0.5, (steps, Nx))
# target_strain_np[:, :Nx, 0, 0] += noise

@ti.kernel
def load_target(target_np: ti.types.ndarray()):
    for i, j, k, l in ti.ndrange(steps, n_particles, dim, dim):
        target_strain[i, j][k, l] = target_np[i, j, k, l]

load_target(target_strain_np)

init_g[None] = 0


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

init_e = 1e4

tol = 1e-36
options = {
    # 'disp': 1,
    'disp': 0, 
    'ftol': tol, 
    'gtol': tol,
    }
converged_threshold = 100

# n_search = 5

target_indices = ti.field(dtype=int, shape=(n_particles))
target_values = ti.field(dtype=real, shape=(n_particles), needs_grad=True)

locked_indices = ti.field(dtype=int, shape=(n_particles))
locked_values = ti.field(dtype=real, shape=(n_particles))

E_hist = []

for z in range(n_search):
    print("searching: " + str(z))
    n_targets = n_particles - z
    n_locked = z

    target_indices_np = np.arange(n_particles)
    target_indices_np = np.delete(target_indices_np, locked_indices.to_numpy()[:n_locked])
    target_indices_np = np.concatenate((target_indices_np, np.zeros(n_locked)))
    if target_indices_np.shape[0] == n_particles+1:
        target_indices_np = target_indices_np[:-1]
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
                    options=options)
        # find highest deviation particle
        E_search = np.array(result.x)
        # E_search = np.delete(E_search, locked_indices.to_numpy()[:n_locked])
        deviation = max(E_search - baseline)

        converge_counter += 1
    

    minus_baseline = np.abs(E_search - init_e)
    damage_index = np.where(minus_baseline == max(minus_baseline))[0]
    # lock particle
    curr_locked = locked_indices.to_numpy()[:n_locked]
    locked_indices[z] = damage_index + np.sum(curr_locked <= damage_index)
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

filename = f"result_add_{case}_{obs}_{n_search}_p1"
with open(filename + ".json", "w") as outfile: 
    json.dump(result_dict, outfile)

name = f"result_add_{case}_{obs}_{n_search}_p1"
with open(name + '.json') as json_file:
        jf = json.load(json_file)

locked_indices_arr = np.array(jf["locked_indices"])  

# reoptimize locked values
print('Phase 2')
n_targets = n_search
n_locked = n_particles - n_targets

target_indices_np = np.concatenate((locked_indices_arr, np.zeros(n_locked)))
target_indices.from_numpy(target_indices_np)

params = np.zeros(n_targets) + init_e


@ti.kernel
def assign_E_modular_p2():
    E.fill(init_e)
    for i in range(n_targets):
        E[target_indices[i]] = target_values[i]

def compute_loss_and_grad_modular_p2(params):
    grid_v_in.fill(0)
    grid_m_in.fill(0)
    loss[None] = 0
    for i in range(n_targets):
        target_values[i] = params[i]
    with ti.ad.Tape(loss=loss):
        assign_E_modular_p2()
        assign_ext_load()
        for s in range(steps - 1):
            substep(s)
        compute_loss()

    loss_val = loss[None]
    grad_val = [target_values.grad[i] for i in range(n_targets)]

    return loss_val, grad_val

deviation = 0
converge_counter = 0
baseline = np.zeros(n_targets) + init_e
init_e_p2 = 100
params = np.zeros(n_targets) + init_e_p2
while deviation < converged_threshold:
    print("converge counter: ", converge_counter)
    result = minimize(compute_loss_and_grad_modular_p2,
                params,
                method='L-BFGS-B',
                jac=True,
                options=options)
    # find highest deviation particle
    E_search = np.array(result.x)
    deviation = max(E_search - baseline)

    converge_counter += 1
E_hist.append(E.to_numpy().tolist())

print("final: ", result.x)
print("found:")
print(locked_indices_arr)

result_dict = {
    "E_hist" : E_hist,
    "locked_indices" : locked_indices_arr.tolist(),
    "locked_values" : locked_values_arr.tolist()
}

filename = f"result_add_{case}_{obs}_{n_search}_p2"
with open(filename + ".json", "w") as outfile: 
    json.dump(result_dict, outfile)

