"""Run the full Cartpole benchmark for v5."""

import time

from hpm_ai_v5.planning.cartpole import CartpoleBenchmark

def run_evaluation():
    print("Starting Cartpole Benchmark (100 episodes, max 1000 steps)...")
    
    benchmark = CartpoleBenchmark()
    
    start_time = time.time()
    all_lengths = []
    total_episodes = 100
    batch_size = 5
    num_batches = total_episodes // batch_size
    
    # Run in smaller batches for more frequent updates
    for batch in range(num_batches):
        result = benchmark.run(
            episodes=batch_size, 
            max_steps=1000, 
            global_episode_start=batch*batch_size, 
            total_episodes=total_episodes
        )
        all_lengths.extend(result.episode_lengths)
        avg = sum(result.episode_lengths) / batch_size
        current_epsilon = benchmark.postprocessor.q_epsilon
        print(f"  Batch {batch+1}/{num_batches}: Avg Length = {avg:.2f} (Epsilon: {current_epsilon:.3f}, Total Avg: {sum(all_lengths)/len(all_lengths):.2f})")
    
    duration = time.time() - start_time
    avg_len = sum(all_lengths) / len(all_lengths)
    passed = avg_len > 500
    
    print("\nBenchmark Results:")
    print(f"  Result: {'success' if passed else 'failure'}")
    print(f"  Average Episode Length: {avg_len:.2f} steps")
    print(f"  Total Duration: {duration:.2f} seconds")

if __name__ == "__main__":
    run_evaluation()
