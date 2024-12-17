import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import json 
from math import pi

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

# diff = ti.field(real, shape=(max_steps, 10))
# target_diff = ti.field(real, shape=(max_steps, 10))
weighed_loss = False
obs_choices = ["full", "row", "sensor"]
obs = obs_choices[0]
@ti.kernel
def compute_loss():
    for i in range(steps - 1):
        # for j in range(n_particles):
        #     # sensor = j * 8
        #     # diff = x[i, sensor + 8] - x[i, sensor]
        #     # target_diff = (target_x[i, sensor + 8] - target_x[i, sensor])

        #     # if j == 9:
        #     #     diff = x[i, sensor + 7] - x[i, sensor]
        #     #     target_diff = target_x[i, sensor + 7] - target_x[i, sensor]
    
        #     dist = (x[i, j] - target_x[i, j]) ** 2
        #     loss[None] += 0.5 * (dist[0]+dist[1])*1e150# + dist[1, 1]) * 1e16
        if obs == "full":
            for j in range(n_particles):
                dist = (target_strain[i, j] - strain2[i, j]) ** 2
                # dist = (1 / ((steps - 1) * n_particles)) * \
                #     (target_strain[i, j] - strain2[i, j]) ** 2
                if weighed_loss:
                    loss[None] += 0.5 * (dist[0, 0])*1e36* (2 - ti.math.sin(i * ti.math.pi / 80))# + dist[1, 1]) * 1e16
                
                else:
                    loss[None] += 0.5 * (dist[0, 0])*1e20# + dist[1, 1]) * 1e16
        elif obs == "row":
            for j in range(Nx):
                dist = (target_strain[i, j] - strain2[i, j]) ** 2
                # dist = (1 / ((steps - 1) * n_particles)) * \
                #     (target_strain[i, j] - strain2[i, j]) ** 2
                if weighed_loss:
                    loss[None] += 0.5 * (dist[0, 0])*1e36*(2 - ti.math.sin(i * ti.math.pi / 80))#(2 - ti.math.sin(i * ti.math.pi / 80))# / dist[0,0]# + dist[1, 1]) * 1e16
                
                else:
                    loss[None] += 0.5 * (dist[0, 0])*1e36# + dist[1, 1]) * 1e16
        elif obs == "sensor":
            for j in range(16):
                dist = (target_strain[i, j*5] - strain2[i, j*5]) ** 2
                # dist = (1 / ((steps - 1) * n_particles)) * \
                #     (target_strain[i, j] - strain2[i, j]) ** 2
                if weighed_loss:
                    loss[None] += 0.5 * (dist[0, 0])*1e36*(2 - ti.math.sin(i * ti.math.pi / 80))# + dist[1, 1]) * 1e16
                
                else:
                    loss[None] += 0.5 * (dist[0, 0])*1e36# + dist[1, 1]) * 1e16



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

n_blocks_y = 1
n_blocks_x = 80
n_blocks = n_blocks_y * n_blocks_x
block_nx = int(Nx / n_blocks_x)
block_ny = int(Ny / n_blocks_y)


@ti.kernel
def assign_E():
    for i in range(n_particles):
        row = i // Nx
        col = i % Nx
        
        if row > 2:
            E[i] = 10000
        else:
            E[i] = 10000 - exp_params[2] * ti.math.clamp( \
                1 / (ti.math.sqrt(2.0 * 3.1415) * exp_params[1]) \
                * ti.math.exp(-ti.math.pow(((col)/(Nx) * 0.8 + 0.1 - exp_params[0]) / exp_params[1], 2.0) / 2),
                0, 1)
        
    # for i in range(Nx):
    #     for j in range(Ny):
    #         block_index_x = i // block_nx
    #         block_index_y = j // block_ny
    #         E[j*Nx+i] = E_block[block_index_x + block_index_y * n_blocks_x]




for i in range(n_particles):
    F[0, i] = [[1, 0], [0, 1]]



for i in range(Nx):
    for j in range(Ny):
        x[0, j * Nx + i] = [(i)/(Nx) * 0.8 + 0.1, (j)/(Ny) * 0.1 + 0.3]




print('loading target')
damaged = True
if damaged: 
    target_strain_name = "strain2_true.npy"
else:
    target_strain_name = "strain2_true_original.npy"
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
# force[None] = 5

n_params = 3
exp_params = ti.field(dtype=real, shape=(3), needs_grad=True)
loc = ti.field(dtype=real, shape=(), needs_grad=True)
std = ti.field(dtype=real, shape=(), needs_grad=True)
height = ti.field(dtype=real, shape=(), needs_grad=True)

losses = []

print('running grad iterations')
optim = 'lbfgs'
if optim == 'lbfgs':
    from scipy.optimize import minimize

    n_ef_it = 1
    E_hist = []
    it_hist = []

    def compute_loss_and_grad(params):
        grid_v_in.fill(0)
        grid_m_in.fill(0)
        loss[None] = 0
        
        # loc[None] = params[0]
        # std[None] = params[1]
        # height[None] = params[2]

        for i in range(3):
            exp_params[i] = params[i]

        with ti.ad.Tape(loss=loss):
            assign_E()
            assign_ext_load()
            for s in range(steps - 1):
                substep(s)
            compute_loss()

        loss_val = loss[None]
        grad_val = [exp_params.grad[i] for i in range(3)]


        return loss_val, grad_val
    
    def callback_fn(intermediate_result):
        params = intermediate_result
        loss, grad = compute_loss_and_grad(params)
        losses.append(loss)
        print(j, 
            'loss=', loss, 
            '   grad=', grad,
            '   params=', params)
   
    initial_params = [0.5, 0.05, 5000]

    tol = 1e-3600
    options = {
        'disp': 1, 
        'ftol': tol, 
        'gtol': tol,
        'tol': tol,
        'verbose': 2,
        'adaptive': True
        }
    result = minimize(compute_loss_and_grad,
                      np.array(initial_params),
                      method='L-BFGS-B',
                      jac=True,
                    # #   hess='2-point',
                      bounds=((0, 1), (0, None), (0, None)),
                      callback=callback_fn,
                      options=options)
    print(result)
    
    E_final = []
    for i in range(n_particles):
        E_final.append(E[i])


    result_dict = {
        "losses" : losses,
        "E_final" : E_final,
        "params" : result.x.tolist(),
        "final_strain" : strain2.to_numpy(),
        "final_x" : x.to_numpy()
    }
    filename = f"result_l_{initial_params[0]}_s_{initial_params[1]}_h_{initial_params[2]}_" + obs
    if weighed_loss:
        filename = filename + "_weighedloss"
    if damaged:
        filename = filename + "_damaged"
    
    with open(filename + ".json", "w") as outfile: 
        json.dump(result_dict, outfile)
