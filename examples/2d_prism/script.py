import subprocess
import time
import os
import itertools
import multiprocessing

# Define the combinations
obs_choices = ["full"]
snapshots = ["hist", "snap"]
losstypes = [ "disp", "strain"]
n_blocks_xs = ["100", "50", "20", "10"]
n_blocks_ys = ["1"]

widths = ["2", "10"]

n_blocks_xs = [str(int(i)*5) for i in n_blocks_xs]
# The target script name
# target_script = "cs_true.py"
target_script_damaged = "script_opt.py"

max_concurrent_processes = 1

# def run_true(case, semaphore):
#     result_filename = f"s_cs_{case}_f.npy"
    
#     with semaphore:
#         while not os.path.exists(result_filename):
#             print(f"[{case} | {result_filename}] Result not found. Running script...")
#             try:
#                 subprocess.run(["python", target_script, case], check=True)
#             except subprocess.CalledProcessError as e:
#                 print(f"Error running script for ({case}): {e}")

#         print(f"[{case}] Result file found. Done.")


def run_until_done(obs, snapshot, losstype, n_blocks_x, n_blocks_y, width, semaphore):
    result_filename = f"results/r_8_0.005_{obs}_{losstype}_{snapshot}_{n_blocks_x}_{n_blocks_y}_{width}.json"
    
    with semaphore:
        while not os.path.exists(result_filename):
            print(f"[{obs} | {snapshot} | {losstype} | {n_blocks_x} | {n_blocks_y} | {width}] Result not found. Running script...")
            try:
                subprocess.run(["python", target_script_damaged, obs, snapshot, losstype, n_blocks_x, n_blocks_y, width], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running script for ({obs} | {snapshot} | {losstype} | {n_blocks_x} | {n_blocks_y} | {width}): {e}")

        print(f"[{obs} | {snapshot} | {losstype} | {n_blocks_x} | {n_blocks_y} | {width}] Result file found. Done.")

if __name__ == "__main__":
    # semaphore = multiprocessing.Semaphore(max_concurrent_processes)
    # processes_true = []
    # for case in cases:
    #     p_true = multiprocessing.Process(target=run_true, args=(case, semaphore))
    #     p_true.start()
    #     processes_true.append(p_true)

    # # Optional: Wait for all processes to finish
    # for p_true in processes_true:
    #     p_true.join()

    combinations = list(itertools.product(obs_choices, snapshots, losstypes, n_blocks_xs,  n_blocks_ys, widths))
    appends = list(itertools.product(['full'], ["hist", "snap"], ["disp", "strain"], ["500"], ["60"], ["1", "5"]))
    for item in appends:
        combinations.append(item)
    semaphore = multiprocessing.Semaphore(max_concurrent_processes)
    # Launch a separate process for each combination
    processes = []
    for obs, snapshot, losstype, n_blocks_x, n_blocks_y, width in combinations:
        p = multiprocessing.Process(target=run_until_done, args=(obs, snapshot, losstype, n_blocks_x, n_blocks_y, width, semaphore))
        p.start()
        processes.append(p)

    # Optional: Wait for all processes to finish
    for p in processes:
        p.join()
