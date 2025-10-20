import subprocess
import time
import os
import itertools
import multiprocessing

# Define the combinations
obss = ["full", "sensor"]
damageds = ["0", "1"]
init_es = ["2000", "4000", "6000"]

# The target script name
target_script = "cs_opt.py"

max_concurrent_processes = 2

def run_until_done(obs, damaged, init_e, semaphore):
    result_filename = f"r_c_{damaged}_{init_e}_{obs}_f.json"
    
    with semaphore:
        while not os.path.exists(result_filename):
            print(f"[{obs} | {damaged} | {init_e}]  Result not found. Running script...")
            try:
                subprocess.run(["python", target_script, obs, damaged, init_e], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running script for [{obs} | {damaged} | {init_e}]: {e}")
            # print(f"[{obs} | {case} | {n_search}] Sleeping for {sleep_interval // 60} minutes.")
            # time.sleep(sleep_interval)

        print(f"[{obs} | {damaged} | {init_e}] Result file found. Done.")

if __name__ == "__main__":
    combinations = list(itertools.product(obss, damageds, init_es))
    semaphore = multiprocessing.Semaphore(max_concurrent_processes)
    # Launch a separate process for each combination
    processes = []
    for obs, damaged, init_e in combinations:
        p = multiprocessing.Process(target=run_until_done, args=(obs, damaged, init_e, semaphore))
        p.start()
        processes.append(p)

    # Optional: Wait for all processes to finish
    for p in processes:
        p.join()
