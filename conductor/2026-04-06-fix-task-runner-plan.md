# Plan to Fix TaskRunner

1. The current `TaskRunner.run_task` uses a dummy numerical simulation. The user expects actual Python code execution for tasks like `return_constant`.
2. Implement a `PythonExecutor` class in `experiment_developmental_cognitive_system.py`.
3. The `Executor` will take the sequence of action IDs from the plan, map them to string tokens using `VOCAB[id]`, construct a Python function string, and use `exec()` to run it.
4. Update `TaskRunner.run_task` to use this `PythonExecutor`.
5. Update the evaluation logic to do strict matching on the result of the executed function versus the `expected_output` (e.g. `1` vs `[1]`).
6. Ensure that `perceive` and learning updates only reinforce the correct structural paths.