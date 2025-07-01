import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import json 

import argparse
# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("init", type=str)
parser.add_argument("SNR_db", type=str)
args = parser.parse_args()

# Extract arguments
init_e = int(args.init)
SNR_dB = int(args.SNR_db)
print(init_e, SNR_dB)
ti.reset()
real = ti.f32
ti.init(arch=ti.cuda, default_fp=real, device_memory_GB=4, debug=True)

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
                loss[None] += 0.5 * (dist[0, 0])*1e16
        elif obs == "row":
            for j in range(Nx):
                dist = (target_strain[i, j] - strain2[i, j]) ** 2
                loss[None] += 0.5 * (dist[0, 0])*1e16*Ny
        elif obs == "sensor":
            for j in range(16):
                dist = (target_strain[i, j*12] - strain2[i, j*12]) ** 2
                loss[None] += 0.5 * (dist[0, 0])*1e16*12*Ny

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


@ti.kernel
def assign_E():
    for i in range(n_particles):
        col = i % Nx
        if col < Nx*1/4 :
            E[i] = E1[None]
        elif col >= Nx*3/4:
            E[i] = E4[None]
        elif col >= Nx*1/4 and col < Nx*1/2:
            E[i] = E2[None]
        else:
            E[i] = E3[None]

@ti.kernel
def calc_disp():
    for t, p, d in ti.ndrange(steps, n_particles, dim):
        disp[t, p][d] = x[t, p][d] - x[0, p][d]
        target_disp[t, p][d] = target_x[t, p][d] - target_x[0, p][d]



for i in range(n_particles):
    F[0, i] = [[1, 0], [0, 1]]



for i in range(Nx):
    for j in range(Ny):
        x[0, j * Nx + i] = [
            ((i)/(Nx) * 0.6 + 0.2 + 0.6 / Nx * 0.5) * size, 
            ((j)/(Ny) * 0.04 + 0.36 + 0.04 / Ny * 0.5) * size
            ]




print('loading target')

target_strain_np = np.load('s_cs.npy')
target_strain = ti.Matrix.field(dim,
                            dim,
                           dtype=real,
                           shape=(max_steps, n_particles),
                           needs_grad=True)

target_x_np = np.load('x_cs.npy')
target_x = ti.Vector.field(dim,
                    dtype=real,
                    shape=(max_steps, n_particles),
                    needs_grad=True)

# inject noise to eps_xx direction
# SNR_dB = 0
SNR_linear = 10 ** (SNR_dB / 10)
# get avg signal power in eps_xx
P_signal = np.mean(target_strain_np[:,:Nx,0,0]**2)
P_noise = P_signal / SNR_linear
# particle-wise noise
noise = np.random.normal(0, P_noise ** 0.5, (steps, Nx))
target_strain_np[:, :Nx, 0, 0] += noise

@ti.kernel
def load_target(target_np: ti.types.ndarray()):
    for i, j, k, l in ti.ndrange(steps, n_particles, dim, dim):
        target_strain[i, j][k, l] = target_np[i, j, k, l]

@ti.kernel
def load_x(x_np: ti.types.ndarray()):
    for i, j, k in ti.ndrange(steps, n_particles, dim):
        target_x[i, j][k] = x_np[i, j, k]

load_target(target_strain_np)
# load_x(target_x_np)

disp = ti.Vector.field(dim,
                    dtype=real,
                    shape=(max_steps, n_particles),
                    needs_grad=True)
target_disp = ti.Vector.field(dim,
                    dtype=real,
                    shape=(max_steps, n_particles),
                    needs_grad=True)

# ADAM parameters
lr = 1e1
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
n_params = 5
m_adam = [0 for _ in range(n_params)]
v_adam = [0 for _ in range(n_params)]
v_hat = [0 for _ in range(n_params)]

init_g[None] = 0
force[None] = 5

E_params = ti.field(dtype=real, shape=(3), needs_grad=True)
E1 = ti.field(dtype=real, shape=(), needs_grad=True)
E2 = ti.field(dtype=real, shape=(), needs_grad=True)
E3 = ti.field(dtype=real, shape=(), needs_grad=True)
E4 = ti.field(dtype=real, shape=(), needs_grad=True)

E1[None] = 1e4
E2[None] = 1e4
E3[None] = 1e4
E4[None] = 1e4

grad_iterations = 1000

losses = []
param_hist = np.zeros((n_params, grad_iterations))
param_labels = ['E1', 'E2', 'E3', 'E4', 'F']
param_true = [1.1e4, 0.9e4, 0.9e4, 1.1e4, 5]


print('running grad iterations')
optim = 'lbfgs'
if optim == 'lbfgs':
    from scipy.optimize import minimize

    n_ef_it = 1
    E1_hist, E2_hist, E3_hist, E4_hist, F_hist = [], [], [], [], []
    it_hist = []
    
    def compute_loss_and_grad_e(params):
        print("t1: ", params)
        grid_v_in.fill(0)
        grid_m_in.fill(0)
        loss[None] = 0
        f_scale[None] = 0
        E1[None] = params[0]
        E2[None] = params[1]
        E3[None] = params[2]
        E4[None] = params[3]
        with ti.ad.Tape(loss=loss):
            assign_E()
            assign_ext_load()
            for s in range(steps - 1):
                substep(s)
            # calc_disp()
            compute_loss()

        loss_val = loss[None]
        grad_val = [E1.grad[None], E2.grad[None], E3.grad[None], E4.grad[None]]
        return loss_val, grad_val
    

    
        
    def callback_fn_e(intermediate_result):
        params = intermediate_result
        loss, grad = compute_loss_and_grad_e(params)
        losses.append(loss)
        E1_hist.append(params[0])
        E2_hist.append(params[1])
        E3_hist.append(params[2])
        E4_hist.append(params[3])
        print(j, 
            'loss=', loss, 
            '   grad=', grad,
            '   params=', params)
        

    # init_e = 4e3
    initial_params = [init_e, init_e, init_e, init_e, 0]
    params = initial_params
    obs_choices = ["full", "row", "sensor"]
    obs = obs_choices[1]

    E1_hist.append(initial_params[0])
    E2_hist.append(initial_params[1])
    E3_hist.append(initial_params[2])
    E4_hist.append(initial_params[3])
    F_hist.append(initial_params[4])

    tol = 1e-3600
    options = {
        'disp': 1, 
        'xatol': tol, 
        'fatol': tol, 
        'xtol': tol, 
        'ftol': tol, 
        'gtol': tol,
        'tol': tol,
        'catol': tol,
        'barrier_tol': tol,
        'maxCGit': 1000,
        'maxfun': 1000, 
        'maxiter': 1000,
        'verbose': 2,
        'adaptive': True
        }

    init_loss, _ = compute_loss_and_grad_e(initial_params[:4])
    losses.append(init_loss)
    for i in range(n_ef_it):
        
        print("E opt ", i)
        if init_e == 8e3:
            result = minimize(compute_loss_and_grad_e, 
                            initial_params[:4], 
                            method='L-BFGS-B', 
                            jac=True, 
                            hess='2-point',
                            bounds=[(1e2,1e6),(1e2,1e6),(1e2,1e6),(1e2,1e6)],
                            callback=callback_fn_e,
                            options=options)
        else:
            result = minimize(compute_loss_and_grad_e, 
                            initial_params[:4], 
                            method='L-BFGS-B', 
                            jac=True, 
                            hess='2-point',
                            # bounds=[(1e2,1e6),(1e2,1e6),(1e2,1e6),(1e2,1e6)],
                            callback=callback_fn_e,
                            options=options)
        initial_params[:4] = result.x
        it_hist.append(result.nit)
        print(result)
        

    


    result_dict = {
        "losses" : losses,
        "E1" : E1_hist,
        "E2" : E2_hist,
        "E3" : E3_hist,
        "E4" : E4_hist,
        "F" : F_hist,
        "it_hist": it_hist
    }

    with open(f"r_c_{int(SNR_dB)}_{int(init_e)}.json", "w") as outfile: 
        json.dump(result_dict, outfile)
