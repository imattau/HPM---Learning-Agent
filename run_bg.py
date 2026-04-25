import os
import subprocess
import sys

def run():
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Clean up
    subprocess.run(["pkill", "-f", "hpm_ai_v3"], check=False)
    if os.path.exists("checkpoints/two_phase"):
        import shutil
        shutil.rmtree("checkpoints/two_phase")
        
    print("Starting Two-Phase Experiment...")
    with open("experiment.log", "w") as f:
        subprocess.Popen(
            [sys.executable, "-u", "hpm_ai_v3/task8/run_two_phase_experiment.py"],
            env=env,
            stdout=f,
            stderr=subprocess.STDOUT,
            bufsize=1
        )
    print("Process started in background. Logs redirected to experiment.log")

if __name__ == "__main__":
    run()
