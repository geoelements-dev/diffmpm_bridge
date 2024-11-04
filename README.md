# diffmpm-bridge
Structural identification through differentiable MPM

## Simulation in Taichi MPM
The simulation is built with taichi MPMsolver from [taichi_elements](https://github.com/taichi-dev/taichi_elements).
The project starts with a simple scenario by dropping a wheel with an initial angular velocity $\omega_0$.

To run the mpm simulation you can simply run `python run_mpm.py` with an argument `-o "path/to/output/folder"`. It will output `.png` files for each `dt`. It will run 500 frames for simulation, and store a `.png` file for each frame in output directory. 

To make a .gif of these files, you can run `python make_gif.py -i "path/to/png/folder" -o "path/to/output.gif"`. You can use `--fps` to control the frame rate.


## Experiments
- `2d_plate`: 2D 1:1 plate subject to point force
- `2d_beam_rolling`: 2D 1:2 beam subject to rolling load, E optimized
- `2d_rolling_enlarge`: `2d_beam_rolling`, larger scale to fit frame, E optimized

- `2d_beam_f`: 2D deep beam, testing density compensation
- `2d_beam_p-e_ratio`: 2D deep beam, testing P/E ratios
- `2d_cantilever`: 2D deep cantilever beam subject to point load, nonhomogeneous elastic field

- `2d_multistiffness`: `2d_rolling_enlarge`, nonhomogeneous elastic field, E optimized
- `2d_multistiffness_force`: `2d_multistiffness`, E-F both optimized

- `2d_beam`: 2D 1:8 beam, faster rolling load
- `2d_beam_noise`: `2d_beam` with noise injected into signal
- `2d_beam_noise_avg` : Uniform noise level applied to all particles, determined by avg signal power

- `2d_beam_discretization` : `2d_beam_noise_avg` with modular discretization
- `2d_beam_gradient` : `2d_beam_discretization` but true experiment is a gradient of E's

- `2d_beam_damage` : 2d_beam experiment with damage identification, fill layer + masonry layer

- `2d_beam_damage_uniform` : 2d_beam experiment with damage id, uniform undamaged layer