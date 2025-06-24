import taichi as ti

ti.reset()
real = ti.f32
ti.init(arch=ti.cuda, default_fp=real, device_memory_GB=1.5)

# init parameters
size = 1
dt_scale = 0.1

dim = 2
grid_size = 4096
t = 0.0
res = (256, 256)
n_grid = res[0]

N = 60
# n_particles = ti.field(ti.i32, shape=())
n_particles = N * N

dx = size / res[0]
inv_dx = 1.0 / dx
dt = 2e-2 * dx / size * dt_scale
p_vol = dx ** dim
p_rho = 1000
p_mass = p_vol * p_rho
max_steps = 1024
steps = 10
max_num_particles = 2 ** 15
gravity = ti.Vector.field(dim, dtype=ti.f32, shape=())

E = 100
mu = 100
la = 100

x = ti.Vector.field(dim, dtype=ti.f32)
v = ti.Vector.field(dim, dtype=ti.f32)

grid_v_in = ti.Vector.field(dim,
                            dtype=real,
                            shape=(max_steps, n_grid, n_grid),
                            needs_grad=True)
grid_v_out = ti.Vector.field(dim,
                             dtype=real,
                             shape=(max_steps, n_grid, n_grid),
                             needs_grad=True)
grid_m_in = ti.field(dtype=real,
                     shape=(max_steps, n_grid, n_grid),
                     needs_grad=True)

C = ti.Matrix.field(dim, dim, dtype=ti.f32)
F = ti.Matrix.field(dim, dim, dtype=ti.f32)


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
        cauchy = 2 * mu * (new_F - r) @ new_F.transpose() + \
                 ti.Matrix.diag(2, la * (J - 1) * J)
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

bound = 3

@ti.kernel
def grid_op(f: ti.i32):
    for i, j in ti.ndrange(n_grid, n_grid):
        inv_m = 1 / (grid_m_in[f, i, j] + 1e-10)
        v_out = inv_m * grid_v_in[f, i, j]
        v_out[1] -= dt * gravity
        if i < bound and v_out[0] < 0:
            v_out[0] = 0
        if i > n_grid - bound and v_out[0] > 0:
            v_out[0] = 0
        if j < bound and v_out[1] < 0:
            v_out[1] = 0
        if j > n_grid - bound and v_out[1] > 0:
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
                print('gv: ', g_v)
                new_C += 4 * weight * g_v.outer_product(dpos) * inv_dx

        v[f + 1, p] = new_v
        x[f + 1, p] = x[f, p] + dt * v[f + 1, p]
        C[f + 1, p] = new_C


def substep(s):
    p2g(s)
    grid_op(s)
    g2p(s)

for i in range(n_particles):
    F[0, i] = [[1, 0], [0, 1]]

for i in range(N):
    for j in range(N):
        x[0, i * N + j] = [dx * (i * 0.7 + 10), dx * (j * 0.7 + 25)]

gui = ti.GUI("Taichi Elements", res=512, background_color=0x112F41)

for s in range(steps - 1):
    substep(s)

x_np = x.to_numpy()

out_dir = 'out_test'

frame = 0
for s in range(15, steps, 16):
    scale = 4
    gui.circles(x_np[s], color=0x112233, radius=1.5)
    gui.show(f'{out_dir}/{frame:06d}.png')
    frame += 1