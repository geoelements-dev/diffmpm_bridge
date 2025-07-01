import subprocess
import time
import os
import itertools
import multiprocessing

# Define the combinations
obs_choices = ["full", "sensor"]
cases = ['d', 'dm', 'g', 'gm', 'h', 'v', 'vm', 'gt', 'gtm', 'ht']

# The target script name
target_script = "cs_true.py"
target_script_damaged = "cs_opt_mask.py"

max_concurrent_processes = 3

def run_true(case, semaphore):
    result_filename = f"s_cs_{case}.npy"
    
    with semaphore:
        while not os.path.exists(result_filename):
            print(f"[{case} | {result_filename}] Result not found. Running script...")
            try:
                subprocess.run(["python", target_script, case], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running script for ({case}): {e}")

        print(f"[{case}] Result file found. Done.")


def run_until_done(obs, case, semaphore):
    result_filename = f"r_c_mask_{case}_{obs}_start_{int(1e1)}.json"
    
    with semaphore:
        while not os.path.exists(result_filename):
            print(f"[{obs} | {case}] Result not found. Running script...")
            try:
                subprocess.run(["python", target_script_damaged, obs, case], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running script for ({obs}, {case}): {e}")

        print(f"[{obs} | {case}] Result file found. Done.")

if __name__ == "__main__":
    semaphore = multiprocessing.Semaphore(max_concurrent_processes)
    processes_true = []
    for case in cases:
        p_true = multiprocessing.Process(target=run_true, args=(case, semaphore))
        p_true.start()
        processes_true.append(p_true)

    # Optional: Wait for all processes to finish
    for p_true in processes_true:
        p_true.join()

    combinations = list(itertools.product(obs_choices, cases))
    semaphore = multiprocessing.Semaphore(max_concurrent_processes)
    # Launch a separate process for each combination
    processes = []
    for obs, case in combinations:
        p = multiprocessing.Process(target=run_until_done, args=(obs, case, semaphore))
        p.start()
        processes.append(p)

    # Optional: Wait for all processes to finish
    for p in processes:
        p.join()
