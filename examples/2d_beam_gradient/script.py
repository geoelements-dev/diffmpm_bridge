import subprocess
import time
import os
import itertools
import multiprocessing

# Define the combinations
obss = ["full", "row", "sensor"]
n_blocks_xs = ["36", "18", "9", "12", "6", "4", "3"]
n_blocks_ys = ["1", "2"]

# The target script name
target_script = "cs_opt.py"

# Sleep time between runs (in seconds)
sleep_interval = 15 * 60  # 10 minutes
max_concurrent_processes = 3

def run_until_done(obs, n_blocks_x, n_blocks_y, semaphore):
    result_filename = f"r_c_{n_blocks_x}_{n_blocks_y}_{obs}.json"
    
    with semaphore:
        while not os.path.exists(result_filename):
            print(f"[{obs} | {n_blocks_x} | {n_blocks_y}]  Result not found. Running script...")
            try:
                subprocess.run(["python", target_script, obs, n_blocks_x, n_blocks_y], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running script for [{obs} | {n_blocks_x} | {n_blocks_y}]: {e}")
            # print(f"[{obs} | {case} | {n_search}] Sleeping for {sleep_interval // 60} minutes.")
            # time.sleep(sleep_interval)

        print(f"[{obs} | {n_blocks_x} | {n_blocks_y}] Result file found. Done.")

if __name__ == "__main__":
    combinations = list(itertools.product(obss, n_blocks_xs, n_blocks_ys))
    semaphore = multiprocessing.Semaphore(max_concurrent_processes)
    # Launch a separate process for each combination
    processes = []
    for obs, n_blocks_x, n_blocks_y in combinations:
        p = multiprocessing.Process(target=run_until_done, args=(obs, n_blocks_x, n_blocks_y, semaphore))
        p.start()
        processes.append(p)

    # Optional: Wait for all processes to finish
    for p in processes:
        p.join()
