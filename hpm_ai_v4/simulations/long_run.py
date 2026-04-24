
import os
import time
import pickle
from hpm_ai_v4.simulations.wikipedia_sim import run_simulation
from hpm_ai_v4.simulations.data.get_corpus import DEFAULT_OUTPUT_PATH

LOG_FILE = "outputs/long_run.log"
MODEL_FILE = "outputs/wikipedia_agent_long.pkl"

if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)
    
    print(f"Starting Long-Term Simulation (2,000 steps, K=4, Parallel 4 CPUs)...")
    print(f"Logging to {LOG_FILE}")
    
    with open(LOG_FILE, "w") as f:
        f.write("HPM v4 Long Run Initialized\n")
        f.write(f"Timestamp: {time.ctime()}\n")
        f.flush()
        
    start_time = time.time()
    
    # Redirect print to log file for the simulation
    import builtins
    _orig_print = builtins.print
    def log_print(*args, **kwargs):
        with open(LOG_FILE, "a") as f:
            _orig_print(*args, **kwargs, file=f)
    builtins.print = log_print

    try:
        agent, metrics = run_simulation(
            filepath=DEFAULT_OUTPUT_PATH,
            total_chars=2000,
            num_initial_patterns=3,
            log_every=500,
            num_workers=4
        )
        
        duration = time.time() - start_time
        print(f"\nSimulation completed in {duration:.2f}s")
        
        # Save the trained agent
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(agent, f)
        print(f"Model saved to {MODEL_FILE}")
        
    except Exception:
        import traceback
        print(f"Simulation failed:\n{traceback.format_exc()}")
    finally:
        builtins.print = _orig_print
