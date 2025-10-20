import subprocess
import time
import os
import itertools
import multiprocessing

# Define the combinations
obs_choices = ["full", "sensor"]
cases = ['d', 'dm', 'g', 'gm', 'h', 'v', 'vm', 'gt', 'gtm', 'ht']
n_searchs = ["60"]

# The target script name
target_script = "cs_true.py"
target_script_damaged = "cs_opt_additive.py"

max_concurrent_processes = 5

def run_true(case, semaphore):
    result_filename = f"s_cs_{case}_f.npy"
    
    with semaphore:
        while not os.path.exists(result_filename):
            print(f"[{case} | {result_filename}] Result not found. Running script...")
            try:
                subprocess.run(["python", target_script, case], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running script for ({case}): {e}")

        print(f"[{case}] Result file found. Done.")


def run_until_done(obs, case, n_search, semaphore):
    result_filename = f"r_c_add_{case}_{obs}_{n_search}_p1_2_f.json"
    
    with semaphore:
        while not os.path.exists(result_filename):
            print(f"[{obs} | {case} | {n_search}] Result not found. Running script...")
            try:
                subprocess.run(["python", target_script_damaged, obs, case, n_search], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running script for ({obs}, {case}, {n_search}): {e}")

        print(f"[{obs} | {case} | {n_search}] Result file found. Done.")

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

    

    combinations = list(itertools.product(obs_choices, cases, n_searchs[1:]))
    semaphore = multiprocessing.Semaphore(max_concurrent_processes)
    # Launch a separate process for each combination
    processes = []
    for obs, case, n_search in combinations:
        p = multiprocessing.Process(target=run_until_done, args=(obs, case, n_search, semaphore))
        p.start()
        processes.append(p)

    # Optional: Wait for all processes to finish
    for p in processes:
        p.join()
    
        combinations = list(itertools.product(obs_choices, cases, ["30"]))
    semaphore = multiprocessing.Semaphore(max_concurrent_processes)
    # Launch a separate process for each combination
    processes = []
    for obs, case, n_search in combinations:
        p = multiprocessing.Process(target=run_until_done, args=(obs, case, n_search, semaphore))
        p.start()
        processes.append(p)

    # Optional: Wait for all processes to finish
    for p in processes:
        p.join()

    combinations = list(itertools.product(obs_choices, cases, n_searchs))
    semaphore = multiprocessing.Semaphore(max_concurrent_processes)
    # Launch a separate process for each combination
    processes = []
    for obs, case, n_search in combinations:
        p = multiprocessing.Process(target=run_until_done, args=(obs, case, n_search, semaphore))
        p.start()
        processes.append(p)

    # Optional: Wait for all processes to finish
    for p in processes:
        p.join()