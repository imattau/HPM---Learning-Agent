"""Code Recognition Benchmark Execution Script."""

from hpm_ai_v5.planning.code_recognition import CodeRecognitionBenchmark

def get_clone_pairs():
    return [
        (
            "def sum_list(arr):\n    s = 0\n    for x in arr:\n        s += x\n    return s",
            "def total(l):\n    res = 0\n    for i in l:\n        res = res + i\n    return res",
            True
        ),
        (
            "x = 10\ny = 20\nx, y = y, x",
            "temp = a\na = b\nb = temp",
            True
        ),
        (
            "if x > 0:\n    print('pos')",
            "while x > 0:\n    x -= 1",
            False
        )
    ]

def get_idiom_corpus():
    base = [
        "with open('f.txt') as f:\n    data = f.read()",
        "with open('data.csv') as file:\n    content = file.read()",
        "[x*x for x in range(10)]",
        "[i*2 for i in items if i > 0]",
        "try:\n    f = open('x')\nfinally:\n    f.close()"
    ]
    return base * 5 # Repeat 5 times to ensure support

def get_edit_examples():
    train = [
        ("x = 10", "x = 10\nif x is not None:\n    print(x)"),
        ("y = 'a'", "y = 'a'\nif y is not None:\n    print(y)")
    ]
    test = [
        ("z = 5", "z = 5\nif z is not None:\n    print(z)")
    ]
    return train, test

def get_completion_data():
    corpus = [
        "def f():\n    a = 1\n    return a",
        "def g():\n    b = 2\n    return b"
    ]
    # FunctionDef -> arguments -> Name -> Store -> Constant -> Return -> Name -> Load
    # Let's see if it can predict 'Return' after an assignment in a function
    test = [
        ("def f():\n    a = 1", "Return")
    ]
    return corpus, test

def run_benchmark():
    benchmark = CodeRecognitionBenchmark()
    
    # 3.1 Clone Detection
    res_clone = benchmark.run_clone_detection(get_clone_pairs())
    print(f"Clone Detection: {res_clone.status} ({res_clone.score:.2%})")
    
    # 3.2 Idiom Discovery
    res_idiom = benchmark.run_idiom_discovery(get_idiom_corpus(), ["with", "list_comp", "try_finally"])
    print(f"Idiom Discovery: {res_idiom.status} (Archive size: {res_idiom.score})")
    
    # 3.3 Edit Transfer
    train_edits, test_edits = get_edit_examples()
    res_edit = benchmark.run_edit_transfer(train_edits, test_edits)
    print(f"Edit Transfer: {res_edit.status} ({res_edit.score:.2%})")
    
    # 3.4 Code Completion
    train_comp, test_comp = get_completion_data()
    res_comp = benchmark.run_completion(train_comp, test_comp)
    print(f"Code Completion: {res_comp.status} ({res_comp.score:.2%})")
    
    print("\nBenchmark Complete.")

if __name__ == "__main__":
    run_benchmark()
