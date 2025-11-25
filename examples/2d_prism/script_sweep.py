import subprocess
import time
import os
import itertools
import multiprocessing

# Define the combinations
obs_choices = ["full"]
snapshots = ["snap"]
losstypes = [ "disp", "strain"]
n_blocks_xs = ["1"]
n_blocks_ys = ["1"]

widths = ["1"]


# The target script name
# target_script = "cs_true.py"
target_script_damaged = "subscript_sweep.py"

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


def run_until_done(obs, snapshot, losstype, width, semaphore):
    result_filename = f"results/r_8_0.005_{obs}_{losstype}_{snapshot}_1_1_sweep.json"
    
    with semaphore:
        while not os.path.exists(result_filename):
            print(f"[{obs} | {snapshot} | {losstype} | {width}] Result not found. Running script...")
            try:
                subprocess.run(["python", target_script_damaged, obs, snapshot, losstype, width], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running script for ({obs} | {snapshot} | {losstype} | {width}): {e}")

        print(f"[{obs} | {snapshot} | {losstype} | {width}] Result file found. Done.")

if __name__ == "__main__":
    combinations = list(itertools.product(obs_choices, snapshots, losstypes, widths))
    semaphore = multiprocessing.Semaphore(max_concurrent_processes)
    # Launch a separate process for each combination
    processes = []
    for obs, snapshot, losstype, width in combinations:
        p = multiprocessing.Process(target=run_until_done, args=(obs, snapshot, losstype, width, semaphore))
        p.start()
        processes.append(p)

    # Optional: Wait for all processes to finish
    for p in processes:
        p.join()
