import taichi as ti
import numpy as np
import math
from engine.mpm_solver import MPMSolver
import argparse
import os




out_dir = 'out_jupyter'
ti.reset()
ti.init(arch=ti.cuda)  # Try to run on GPU

gui = ti.GUI("Taichi Elements", res=512, background_color=0x112F41)

mpm = MPMSolver(res=(256, 256), E_scale=0.001, use_g2p2g=False)
mpm.set_gravity([0, 0])

E1 = 1e6
E2 = 1e3


mpm.add_cube(
    lower_corner=[0.45,0.45],
    cube_size=[0.1,0.1],
    velocity=[0, 0],
    sample_density=0.5,
    material=mpm.material_elastic,
    E=E1,
    nu=0.2
)
for frame in range(1):
    mpm.step(8e-3, print_stat=False)
    particles = mpm.particle_info()
    print(mpm.fan_center, mpm.t, mpm.omega)
    gui.circles(particles['position'], radius=1.5, color=particles['color'])
    gui.show(f'{out_dir}/{frame:06d}.png')