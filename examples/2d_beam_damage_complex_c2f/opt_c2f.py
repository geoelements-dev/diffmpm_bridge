import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import json 

import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run additive optimization.")
parser.add_argument("obs", type=str, choices=["full", "sensor"], help="Observation choice")
parser.add_argument("case", type=str, help="Case identifier")
parser.add_argument("deviation_threshold", type=float, help="Number of search iterations")
args = parser.parse_args()

# Extract arguments
obs = args.obs
case = args.case
deviation_threshold = args.deviation_threshold

# obs_choices = ["full", "sensor"]
# cases = ['d', 'dm', 'g', 'gm', 'h', 'v', 'vm', 'gt', 'gtm', 'ht']
# deviation_threshold_list = [0.1, 0.3, 0.5, 0.8]

# obs = obs_choices[1]
# case = 'v'
# deviation_threshold = deviation_threshold_list[1]

ti.reset()
real = ti.f32
ti.init(arch=ti.cuda, default_fp=real, device_memory_GB=12)

# init parameters
size = 1
dim = 2
Nx = 128
Ny = 16
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
                dist = (target_strain[i, j*8] - strain2[i, j*8]) ** 2
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


n_blocks_y = 2
n_blocks_x = 16
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
            (i)/(Nx) * 0.8 + 0.1 + 0.8 / Nx * 0.5, 
            (j)/(Ny) * 0.1 + 0.3 + 0.1 / Ny * 0.5]

print('loading target')
obs_choices = ["full", "row", "sensor"]
# obs = obs_choices[0]

cases = ['', 'd', 'dm', 'g', 'gm', 'h', 'v', 'vm', 'gt', 'gtm' , 'ht']
# case = 'g'
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


def compute_loss_and_grad(params):
    grid_v_in.fill(0)
    grid_m_in.fill(0)
    loss[None] = 0
    for i in range(n_blocks):
        E_block[i] = params[i]
    with ti.ad.Tape(loss=loss):
        assign_E()
        assign_ext_load()
        for s in range(steps - 1):
            substep(s)
        compute_loss()

    loss_val = loss[None]
    grad_val = [E_block.grad[i] for i in range(n_blocks)]

    losses.append(loss_val)
    E_hist.append(params.tolist())
    # print(j, 
    #     'loss=', loss, 
    #     '   grad=', grad_val,
    #     '   params=', params)
    return loss_val, grad_val


def stable_regions(x, reference, n_percent=10, m=2):
    deviation = (x - reference) ** 2
    n_elements = int(len(deviation) * (n_percent / 100))
    
    # Get indices of the lowest n% deviations
    stable_indices = np.argsort(deviation)[:n_elements]
    
    # Compute the standard deviation of these stable elements
    stable_std = np.std(deviation[stable_indices])
    
    # Identify indices where deviation exceeds m times the stable standard deviation
    threshold = m * stable_std
    exceed_indices = np.where(deviation > threshold)[0]

    return stable_indices, stable_std, exceed_indices

@ti.kernel
def assign_E_refine():
    E.fill(background_value)
    for i in range(n_targets):
        for j in range(sub_block_x):
            for k in range(sub_block_y):
                E[sub_block_particle_index[i] + j + k * Nx] = sub_block_values[i]

def compute_loss_and_grad_refine(params):
    grid_v_in.fill(0)
    grid_m_in.fill(0)
    loss[None] = 0
    for i in range(n_targets):
        sub_block_values[i] = params[i]
    with ti.ad.Tape(loss=loss):
        assign_E_refine()
        assign_ext_load()
        for s in range(steps - 1):
            substep(s)
        compute_loss()

    loss_val = loss[None]
    grad_val = [sub_block_values.grad[i] for i in range(n_targets)]

    losses.append(loss_val)
    E_hist.append(params.tolist())
    # print( 
        # 'loss=', loss, 
        # '   grad=', grad_val,)
        # '   params=', params)
    return loss_val, grad_val

def compute_block_indices(host_idx, host_size):
    host_idx = np.array(host_idx)

    sub_size = host_size / 2
    n_host_x = Nx // host_size
    n_host_y = Ny // host_size
    n_sub_x = Nx // sub_size
    n_sub_y = Ny // sub_size
    
    sub_block_indices = np.array([])

    for idx in host_idx:
        host_index_x = idx % n_host_x
        host_index_y = idx // n_host_x

        sub_block_indices_idx = [
            host_index_y * n_sub_x * 2 + host_index_x * 2,
            host_index_y * n_sub_x * 2 + host_index_x * 2 + 1,
            (host_index_y * 2 + 1) * n_sub_x + host_index_x * 2,
            (host_index_y * 2 + 1) * n_sub_x + + host_index_x * 2 + 1,
        ]
        sub_block_indices_idx = np.array([int(i) for i in sub_block_indices_idx])
        sub_block_indices = np.concatenate((sub_block_indices, sub_block_indices_idx))

    particle_idx = []

    for i in sub_block_indices:
        sub_index_x = i % n_sub_x
        sub_index_y = i // n_sub_x

        left_index = sub_index_x * sub_size + sub_index_y * sub_size * Nx

        particle_idx.append(int(left_index))

    return sub_block_indices, np.array(particle_idx)

@ti.kernel
def fill_taichi(arr: ti.types.ndarray()):
    for i in range(n_particles):
        sub_block_particle_index[zti, i] = arr[i]

###

intermediate_results = []

# run naive once
print('First naive')
init_e = 1e4
E_block = ti.field(dtype=real, shape=(n_blocks), needs_grad=True)
E_block.fill(init_e)

initial_params = []
for i in range(n_blocks):
    initial_params.append(init_e)
E_hist.append(E_block.to_numpy().tolist())

tol = 1e-36
options = {
    'disp': 0, 
    'ftol': tol, 
    'gtol': tol,
    }
result = minimize(compute_loss_and_grad,
                    np.array(initial_params),
                    method='L-BFGS-B',
                    jac=True,
                    options=options)
print(result.x)
intermediate_results.append(E.to_numpy().tolist())

background_value = init_e
sub_block_order = [8, 4, 2, 1]
sub_block_tracker = []
# deviation_threshold = 0.5

sub_block_particle_index = ti.field(dtype=int, shape=(n_particles))
sub_block_values = ti.field(dtype=real, shape=(n_particles), needs_grad=True)



for z in range(3):

    print("iteration: ", z, ", block size: ", sub_block_order[z + 1])
    # get blocks that vary
    # stable, std, exceed = stable_regions(np.array(result.x), init_e, n_percent=50, m=1e3)
    deviation = np.abs(np.array(result.x) - background_value)
    max_deviation = np.max(deviation)
    exceed = np.where(deviation >= max_deviation * deviation_threshold)[0]
    exceed_values = np.array(result.x)[exceed]
    if z > 0:
        exceed = sub_block_tracker[z-1][exceed]
    print("found: ", exceed, exceed_values)


    # subdivide block
    n_targets = len(exceed) * 4
    n_sub_blocks = len(exceed) * 4
    sub_block_x = sub_block_order[z + 1]
    sub_block_y = sub_block_order[z + 1]
    if z == 0: # dummy iteration to initialize sub_block_values
        print('dummy iteration')
        n_targets = n_particles
        minimize(compute_loss_and_grad_refine,
                    np.zeros(n_particles) + init_e,
                    method='L-BFGS-B',
                    jac=True,
                    options=options,
                    bounds = [(0, 20000) for i in range(n_targets)]
                    )
    n_targets = len(exceed) * 4
    sub_block_index_np, sub_block_particle_index_np = compute_block_indices(exceed, sub_block_order[z])
    print("next: ", sub_block_index_np)
    sub_block_particle_index_np = np.concatenate((
        sub_block_particle_index_np, 
        np.zeros(n_particles-len(sub_block_particle_index_np))
        ))
    sub_block_particle_index.from_numpy(sub_block_particle_index_np)
    # fill_taichi(sub_block_particle_index_np)
    sub_block_tracker.append(sub_block_index_np)

    # optimize
    result = minimize(compute_loss_and_grad_refine,
                        np.repeat(exceed_values, 4),
                        method='L-BFGS-B',
                        jac=True,
                        options=options,
                        bounds = [(0, 20000) for i in range(n_targets)]
                        )
    # get results
    print(result.x)
    intermediate_results.append(E.to_numpy().tolist())


filename = f"r_{case}_{obs}_{int(deviation_threshold*10)}"

result_dict = {
    # "losses" : losses,
    # "E_hist" : E_hist,
    "results" : intermediate_results
}

with open(filename + ".json", "w") as outfile: 
    json.dump(result_dict, outfile)

