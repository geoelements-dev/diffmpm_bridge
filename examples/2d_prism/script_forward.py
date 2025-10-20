import subprocess
import time
import os
import itertools
import multiprocessing
import numpy as np

# Define the combinations
init_Es = [str(int(i)) for i in np.linspace(2e8, 4e8, 101)]

widths = ["1", "5"]


# The target script name
# target_script = "cs_true.py"
target_script_damaged = "subscript_forward.py"

max_concurrent_processes = 3

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


def run_until_done(width, init_E, semaphore):
    result_filename = f"results/E_{width}_{int(init_E)}.npy"
    
    with semaphore:
        while not os.path.exists(result_filename):
            print(f"[{width} | {init_E}] Result not found. Running script...")
            try:
                subprocess.run(["python", target_script_damaged, width, init_E], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running script for ({width} | {init_E}): {e}")

        print(f"[{width} | {init_E}] Result file found. Done.")

if __name__ == "__main__":
    combinations = list(itertools.product(widths, init_Es))
    semaphore = multiprocessing.Semaphore(max_concurrent_processes)
    # Launch a separate process for each combination
    processes = []
    for width, init_E in combinations:
        p = multiprocessing.Process(target=run_until_done, args=(width, init_E, semaphore))
        p.start()
        processes.append(p)

    # Optional: Wait for all processes to finish
    for p in processes:
        p.join()
