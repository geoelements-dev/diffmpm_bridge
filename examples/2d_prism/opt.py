import taichi as ti
import numpy as np
import matplotlib.pyplot as plt
import json, einops

ti.reset()
real = ti.f32
ti.init(arch=ti.cuda, default_fp=real, device_memory_GB=11)



# init parameters
size = 2.000 # 2 m
span = 1.000 # 4.4 m
depth = 0.120 # 0.4 m
dim = 2
factor = 1/100
Nx = int(span / factor)
Ny = int(depth / factor)
n_particles = int(Nx * Ny)
grid_factor = 1
n_grid = 16*grid_factor

p_vol = (span / Nx) * (depth / Ny)
assert (span / Nx) == (depth / Ny)
p_mass =  2400 * p_vol # 2.4 g/cm3

dx = size / n_grid
inv_dx = 1 / dx
dt =  1 / 200000

nu = 0.3

max_steps = int(0.065 / dt)
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
E = ti.field(dtype=real, shape=(n_particles), needs_grad=True)


# ti.root.lazy_grad()




# @ti.kernel
# def clear_grid():
#     for f in range(max_steps):
#         for i, j in ti.ndrange(n_grid, n_grid):
#             grid_v_in[f, i, j] = [0, 0]
#             grid_m_in[f, i, j] = 0


# @ti.kernel
# def clear_particles():
#     for p in C:
#         F[p] = 0
#         C[p] = 0
        

@ti.kernel
def p2g(f: ti.i32):
    for p in range(n_particles):
        base = ti.cast(x[f, p] * inv_dx - 0.5, ti.i32)
        fx = x[f, p] * inv_dx - ti.cast(base, ti.i32)
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        new_F = (ti.Matrix.diag(dim=2, val=1) + dt * C[f, p]) @ F[f, p]
        F[f + 1, p] = new_F
        J = (new_F).determinant()
        r, _ = ti.polar_decompose(new_F)
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
        v_out = inv_m * grid_v_in[f, i, j] + dt * f_ext[f, i, j] * inv_m
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
        strain2[f + 1, p][1, 0] = strain2[f, p][1, 0] + dstrain[2]

        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C

@ti.kernel
def calc_disp():
    for t, p, d in ti.ndrange(steps, n_particles, dim):
        disp[t, p][d] = x[t, p][d] - x[0, p][d]
        target_disp[t, p][d] = target_x[t, p][d] - target_x[0, p][d]


def substep(s):
    p2g(s)
    grid_op(s)
    g2p(s)


load = 200*1e3 / 2 # N



@ti.kernel
def assign_ext_load():
    for t in ti.ndrange(max_steps):
            f_ext[t, 4*grid_factor, 8*grid_factor] = [-load, 0]
            f_ext[t, 4*grid_factor, 9*grid_factor] = [-load, 0]
            f_ext[t, 13*grid_factor, 8*grid_factor] = [load, 0]
            f_ext[t, 13*grid_factor, 9*grid_factor] = [load, 0]

@ti.kernel
def init_exp():
    for i in range(Nx):
        for j in range(Ny):
            x[0, j * Nx + i] = [
                ((i)/(Nx) * 0.5 + 0.25 + 0.5 / Nx * 0.5 + (1/2 * 1/n_grid)) * size, 
                ((j)/(Ny) * 0.06 + 0.5 + (0.125-0.12)*(1/4) + 0.06 / Ny * 0.5) * size
                ]

    for p in range(n_particles):
        F[0, p] = [[1, 0], [0, 1]]

@ti.kernel
def assign_E():
    for p in range(n_particles):
        E[p] = E_block[p]
    

# @ti.kernel
# def compute_principal_strain():


@ti.kernel
def compute_loss():
    if losstype == "strain":
        if snapshot == "hist":
            for i in range(steps-1):
                if obs == "full":
                    for j in range(n_particles):
                        dist = (target_strain[i, j] - strain2[i, j]) ** 2
                        loss[None] += 0.5 * (dist[0, 0] + dist[1, 1] + dist[1, 0])*1e30
        if snapshot == "snap":
            if obs == "full":
                for j in range(n_particles):
                    dist = (target_strain[steps-1, j] - strain2[steps-1, j]) ** 2
                    loss[None] += 0.5 * (dist[0, 0] + dist[1, 1] + dist[1, 0])*1e30
            # elif obs == "row":
            #     for j in range(Nx):
            #         dist = (target_strain[i, j] - strain2[i, j]) ** 2
            #         loss[None] += 0.5 * (dist[0, 0])*1e20
            # elif obs == "sensor":
            #     for j in range(16):
            #         dist = (target_strain[i, j*12] - strain2[i, j*12]) ** 2
            #         loss[None] += 0.5 * (dist[0, 0])*1e20
    if losstype == "disp":
        if snapshot == "hist":
            if obs == "full":
                for i in range(steps-1):
                    for j in range(n_particles):
                        dist = (target_disp[i, j] - disp[i, j]) ** 2
                        loss[None] += 0.5 * (dist[0] + dist[1])*1e30
        if snapshot == "snap":
            if obs == "full":
                for j in range(n_particles):
                    dist = (target_disp[steps-1, j] - disp[steps-1, j]) ** 2
                    loss[None] += 0.5 * (dist[0] + dist[1])*1e30





def substep(s):
    p2g(s)
    grid_op(s)
    g2p(s)

n_blocks_x = 100
n_blocks_y = 12

losstypes = ['strain', 'disp']

losstype = losstypes[1]

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

print('loading target')

obs = 'full'

widths = [1, 5]
width = widths[0]

snapshots = ["snap", "hist"]
snapshot = snapshots[1]


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

print('loading target')

init_exp()


target_strain_np = np.load(f's_{width}.npy')
target_x_np = np.load(f'x_{width}.npy')
target_x = ti.Vector.field(dim,
                    dtype=real,
                    shape=(max_steps, n_particles),
                    needs_grad=True)

target_strain = ti.Matrix.field(dim,
                            dim,
                           dtype=real,
                           shape=(max_steps, n_particles),
                           needs_grad=True)
# target_p_strain = ti.Vector.field(dim,
#                            dtype=real,
#                            shape=(max_steps, n_particles),
#                            needs_grad=True)
# p_strain = ti.Vector.field(dim,
#                            dtype=real,
#                            shape=(max_steps, n_particles),
#                            needs_grad=True)

@ti.kernel
def load_target(target_np: ti.types.ndarray()):
    for i, j, k, l in ti.ndrange(steps, n_particles, dim, dim):
        target_strain[i, j][k, l] = target_np[i, j, k, l]

@ti.kernel
def load_x(x_np: ti.types.ndarray()):
    for i, j, k in ti.ndrange(steps, n_particles, dim):
        target_x[i, j][k] = x_np[i, j, k]

load_target(target_strain_np)
load_x(target_x_np)

disp = ti.Vector.field(dim,
                    dtype=real,
                    shape=(max_steps, n_particles),
                    needs_grad=True)
target_disp = ti.Vector.field(dim,
                    dtype=real,
                    shape=(max_steps, n_particles),
                    needs_grad=True)



E_block = ti.field(dtype=real, shape=(n_blocks), needs_grad=True)

losses = []



print('running grad iterations')
optim = 'lbfgs'
if optim == 'lbfgs':
    from scipy.optimize import minimize
    import time
    t1 = time.time()
    grad_tracker =[]
    n_ef_it = 1
    E_hist = []
    it_hist = []

    def compute_loss_and_grad(params):
        print(params)
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
            calc_disp()
            compute_loss()

        loss_val = loss[None]
        grad_val = [E_block.grad[i] for i in range(n_blocks)]
        losses.append(loss_val)

        # print(grad_val)
        # print(loss_val)

        return loss_val, grad_val
    

    init_e = 25e9*0.12

    initial_params = []
    for i in range(n_particles):
        initial_params.append(init_e)
    E_hist.append(initial_params)

    tol = 1e-16
    options = { 
        'ftol': tol, 
        'gtol': tol,
        'tol': tol,
        'adaptive': True
        }
    
    
    result = minimize(compute_loss_and_grad,
                      np.array([init_e for b in range(n_blocks)]),
                      method='L-BFGS-B',
                      jac=True,
                      hess='2-point',
                    #   callback=callback_fn,
                    bounds=[(1e8, 1e10) for b in range(n_blocks)],
                      options=options)
    E_hist.append(E.to_numpy().tolist())

    # params = initial_params
    # losses = []
    # lr = 1e17
    # beta1 = 0.9
    # beta2 = 0.999
    # epsilon = 1e-8
    # m_adam = [0 for _ in range(n_particles)]
    # v_adam = [0 for _ in range(n_particles)]
    # v_hat = [0 for _ in range(n_particles)]

    # for _ in range(10):
    #     l, g = compute_loss_and_grad(np.array(params))
    #     for i in range(n_particles):
    #         m_adam[i] = beta1 * m_adam[i] + (1 - beta1) * g[i]
    #         v_adam[i] = beta2 * v_adam[i] + (1 - beta2) * g[i]**2
    #         # m_hat = m_adam[i] / (1 - beta1**(j + 1))
    #         # v_hat = v_adam[i] / (1 - beta2**(j + 1))
    #         v_hat[i] = ti.max(v_hat[i], v_adam[i])
    #         params[i] -= lr * m_adam[i] / (ti.sqrt(v_hat[i]) + epsilon)
    #         # params[j] -= lr * g[j]
    #     E_hist.append(params)
    #     losses.append(l)



    t2 = time.time()
    # print(result)
    print(t2-t1)

    result_dict = {
        "E_hist" : E_hist,
        "losses" : losses,
        "time" : t2-t1
    }

    # with open(f"results/r_{obs}_{losstype}_{snapshot}_{n_blocks_x}_{n_blocks_y}.json", "w") as outfile: 
    #     json.dump(result_dict, outfile)
