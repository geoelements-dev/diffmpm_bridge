import subprocess
import time
import os
import itertools
import multiprocessing

# Define the combinations
inits = ["1000", "4000", "8000"]
SNR_dbs = ["-10", "0", "10", "20"]

# The target script name
target_script = "cs_opt.py"

# Sleep time between runs (in seconds)
sleep_interval = 15 * 60  # 10 minutes
max_concurrent_processes = 1

def run_until_done(init, SNR_db, semaphore):
    result_filename = f"r_c_{SNR_db}_{init}.json"
    
    with semaphore:
        while not os.path.exists(result_filename):
            print(f"[{init} | {SNR_db}]  Result not found. Running script...")
            try:
                subprocess.run(["python", target_script, init, SNR_db], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running script for ({init}, {SNR_db}): {e}")
            # print(f"[{obs} | {case} | {n_search}] Sleeping for {sleep_interval // 60} minutes.")
            # time.sleep(sleep_interval)

        print(f"[{init} | {SNR_db}] Result file found. Done.")

if __name__ == "__main__":
    combinations = list(itertools.product(inits, SNR_dbs))
    semaphore = multiprocessing.Semaphore(max_concurrent_processes)
    # Launch a separate process for each combination
    processes = []
    for init, SNR_db in combinations:
        p = multiprocessing.Process(target=run_until_done, args=(init, SNR_db, semaphore))
        p.start()
        processes.append(p)

    # Optional: Wait for all processes to finish
    for p in processes:
        p.join()
