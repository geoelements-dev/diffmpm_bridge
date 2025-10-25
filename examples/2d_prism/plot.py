import einops, json, itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('text', usetex = True)
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['figure.dpi'] = 300
font = {'size'   : 20}

size = 2.000 # 2 m
span = 1.000 # 4.4 m
depth = 0.120 # 0.4 m
dim = 2
factor = 1/200
Nx = int(span / factor)
Ny = int(depth / factor)
n_particles = int(Nx * Ny)
grid_factor = 8
n_grid = 16*grid_factor

file = "results/E_pf_0.005_gf_8_cw_2.npy"
result = np.load(file, allow_pickle=True)
E_1 = einops.rearrange(result, "(y x) -> x y", y=Ny).transpose()
file = "results/E_pf_0.005_gf_8_cw_10.npy"
result = np.load(file, allow_pickle=True)
E_5 = einops.rearrange(result, "(y x) -> x y", y=Ny).transpose()

obs_choices = ["full"]
snapshots = ["hist", "snap"]
losstypes = [ "disp", "strain"]
n_blocks_xs = ["100", "50", "20", "10"]
n_blocks_ys = ["1"]

widths = ["2"]#, "10"]

sequence = [
    "24 by 200",
    "1 by 200",
    "1 by 100",
    "1 by 50",
    "1 by 20",
    "1 by 10"
]

combinations = list(itertools.product(obs_choices, snapshots, losstypes, widths))
for obs, snapshot, losstype, width in combinations:

    fig, axs = plt.subplots(7,1, sharex=True, sharey=True, figsize=(4,6), layout='constrained')
    plt.suptitle(f"{obs} observability, {snapshot}, {losstype}, {width} crack width")
    if width == "2":
        E_true = E_1
    else:
        E_true = E_5
    im = axs[0].imshow(E_true, origin='lower', cmap='Greys', vmin=0, vmax=4e9)
    filename = [
        f"results/r_8_0.005_{obs}_{losstype}_{snapshot}_200_24_{width}.json",
        f"results/r_8_0.005_{obs}_{losstype}_{snapshot}_200_1_{width}.json",
        f"results/r_8_0.005_{obs}_{losstype}_{snapshot}_100_1_{width}.json",
        f"results/r_8_0.005_{obs}_{losstype}_{snapshot}_50_1_{width}.json",
        f"results/r_8_0.005_{obs}_{losstype}_{snapshot}_20_1_{width}.json",
        f"results/r_8_0.005_{obs}_{losstype}_{snapshot}_10_1_{width}.json",
    ]
    losses = []
    for i, file in enumerate(filename):
        with open(file) as json_file:
            result = json.load(json_file)
        result_final = np.array(result["E_hist"])[-1]
        E = einops.rearrange(result_final, "(y x) -> x y", y=Ny).transpose()
        axs[1+i].imshow(E, origin='lower', cmap='Greys', vmin=0, vmax=4e9)
        losses.append(np.array(result["losses"]))
        print(len(result["losses"]))
    fig.colorbar(im, 
                ax=axs, 
                orientation = 'horizontal', 
                label='Young\'s Modulus (Pa)', 
                aspect=200)
    plt.show()
    plt.savefig(f"results/fig_E_8_0.005_{losstype}_{snapshot}_{width}.png", dpi=300)
    plt.figure(figsize=(3,3))
    for i, loss in enumerate(losses):
        plt.title(f"{snapshot}, {losstype}, {width} crack width")
        plt.plot(loss, label=sequence[i])
        plt.yscale('log')
        plt.legend()
    plt.show()
    plt.savefig(f"results/fig_loss_8_0.005_{losstype}_{snapshot}_{width}.png", dpi=300)