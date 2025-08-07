import subprocess
import time
import os
import itertools
import multiprocessing

# Define the combinations
obs_choices = ["full", "sensor"]
cases = ['d', 'dm', 'g', 'gm', 'h', 'v', 'vm', 'gt', 'gtm', 'ht']
deviation_threshold_list = [0.3, 0.5, 0.8]

# The target script name
target_script = "cs_true.py"
target_script_damaged = "cs_opt.py"

max_concurrent_processes = 10

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


def run_until_done(obs, case, deviation_threshold, semaphore):
    result_filename = f"r_c_{case}_{obs}_{int(deviation_threshold*10)}.json"
    
    with semaphore:
        while not os.path.exists(result_filename):
            print(f"[{obs} | {case} | {deviation_threshold}] Result not found. Running script...")
            try:
                subprocess.run(["python", target_script_damaged, obs, case, str(deviation_threshold)], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running script for ({obs}, {case}, {deviation_threshold}): {e}")

        print(f"[{obs} | {case} | {deviation_threshold}] Result file found. Done.")

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

    combinations = list(itertools.product(obs_choices, cases, deviation_threshold_list))
    semaphore = multiprocessing.Semaphore(max_concurrent_processes)
    # Launch a separate process for each combination
    processes = []
    for obs, case, deviation in combinations:
        p = multiprocessing.Process(target=run_until_done, args=(obs, case, deviation, semaphore))
        p.start()
        processes.append(p)

    # Optional: Wait for all processes to finish
    for p in processes:
        p.join()

