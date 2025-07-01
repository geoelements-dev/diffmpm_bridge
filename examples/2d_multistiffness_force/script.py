import subprocess
import time

# Settings
script_to_run = "c_opt.py"
sleep_seconds = 10 * 60  # 10 minutes between runs

while True:
    print("Running script...")
    try:
        subprocess.run(["python", script_to_run], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Script exited with error: {e}")
    # print(f"Sleeping for {sleep_seconds // 60} minutes...\n")
    # time.sleep(sleep_seconds)
