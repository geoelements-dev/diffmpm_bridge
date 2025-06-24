import subprocess
import time
import os
import itertools
import multiprocessing

# Define the combinations
obs_choices = ["full", "sensor"]
cases = ['d', 'dm', 'g', 'gm', 'h', 'v', 'vm', 'gt', 'gtm', 'ht']
n_search_list = [10]

# The target script name
target_script = "opt_additive.py"

# Sleep time between runs (in seconds)
sleep_interval = 15 * 60  # 10 minutes
max_concurrent_processes = 5

def run_until_done(obs, case, n_search, semaphore):
    result_filename = f"result_add_{case}_{obs}_{n_search}_p1.json"
    
    with semaphore:
        while not os.path.exists(result_filename):
            print(f"[{obs} | {case} | {n_search}] Result not found. Running script...")
            try:
                subprocess.run(["python", target_script, obs, case, str(n_search)], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running script for ({obs}, {case}, {n_search}): {e}")
            # print(f"[{obs} | {case} | {n_search}] Sleeping for {sleep_interval // 60} minutes.")
            time.sleep(sleep_interval)

        print(f"[{obs} | {case} | {n_search}] Result file found. Done.")

if __name__ == "__main__":
    combinations = list(itertools.product(obs_choices, cases, n_search_list))
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
