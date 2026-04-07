class PythonExecutor:
    def run(self, code_str: str, inputs: list) -> any:
        # Wrap in a test function
        code = f"def test_func(inp):\n    {code_str}\n"
        print(f"Executing:\n{code}")
        try:
            local_ns = {}
            exec(code, {}, local_ns)
            test_func = local_ns["test_func"]
            inp = inputs[0] if inputs else None
            return test_func(inp)
        except Exception as e:
            return str(e)

executor = PythonExecutor()
res = executor.run("return 1", [None])
print(f"Result: {res}, Success: {res == 1 and type(res) == int}")
