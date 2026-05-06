"""Run the Cross-Language Transfer (CLT) Benchmark."""

from __future__ import annotations

from hpm_ai_v5.planning.clt import CrossLanguageTransferBenchmark


def get_resource_data():
    train_py = [
        "with open('f.txt') as f: data = f.read()",
        "with open('f.txt', 'r') as f: print(f.read())"
    ]
    test_java = [
        "try (BufferedReader br = new BufferedReader(new FileReader(file))) { String line = br.readLine(); }",
        "try (Scanner sc = new Scanner(new File('f.txt'))) { while(sc.hasNext()) { sc.next(); } }"
    ]
    return train_py, test_java


def get_null_check_data():
    train_py = [
        "if x is None: raise ValueError('bad')",
        "if val is None: raise Exception()"
    ]
    test_java = [
        "if (x == null) throw new IllegalArgumentException('bad');",
        "if (node == null) { throw new NullPointerException(); }"
    ]
    return train_py, test_java


def get_map_data():
    train_py = [
        "res = [x*2 for x in items]",
        "y = [a.lower() for a in list]"
    ]
    test_java = [
        "List<Integer> res = items.stream().map(x -> x * 2).collect(Collectors.toList());",
        "var y = list.stream().map(String::toLowerCase).toList();"
    ]
    return train_py, test_java


def run_clt_benchmark():
    benchmark = CrossLanguageTransferBenchmark()
    
    results = []
    
    # Task 1: Resource Management
    py, java = get_resource_data()
    results.append(benchmark.run_transfer_task("Resource Management", py, java))
    
    # Task 2: Null Check + Exception
    py, java = get_null_check_data()
    results.append(benchmark.run_transfer_task("Null Check + Exception", py, java))
    
    # Task 3: Map Operation (Structural)
    py, java = get_map_data()
    results.append(benchmark.run_transfer_task("Map Operation", py, java))
    
    print("\nCLT Benchmark Results:")
    for res in results:
        print(f"{res.task}: {res.status} ({res.score:.2%})")


if __name__ == "__main__":
    run_clt_benchmark()
