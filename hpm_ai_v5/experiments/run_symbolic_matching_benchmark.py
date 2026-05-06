"""Run the Symbolic Pattern Matching Benchmark."""

from __future__ import annotations

from hpm_ai_v5.planning.symbolic_matching import SymbolicPatternMatchingBenchmark


def get_benchmark_data():
    tool_corpus = [
        {"name": "get_weather", "params": ["city", "units"]},
        {"name": "send_email", "params": ["recipient", "subject", "body"]},
        {"name": "calculate", "params": ["expression"]},
    ]
    
    training_data = [
        {
            "query": "What's the temperature in Paris?", 
            "tool": "get_weather", 
            "placeholders": {"paris": "PARAM_CITY"}
        },
        {
            "query": "Give me the weather forecast for London in Fahrenheit.", 
            "tool": "get_weather", 
            "placeholders": {"london": "PARAM_CITY", "fahrenheit": "PARAM_UNITS"}
        },
        {
            "query": "Send an email to Alice about the meeting.", 
            "tool": "send_email", 
            "placeholders": {"alice": "PARAM_RECIPIENT", "meeting": "PARAM_SUBJECT"}
        },
        {
            "query": "Compute 5 plus 10.", 
            "tool": "calculate", 
            "placeholders": {"5 plus 10": "PARAM_EXPRESSION"}
        },
    ]
    
    test_data = [
        {
            "query": "Show me the current conditions in Tokyo using metric units.", 
            "tool": "get_weather", 
            "placeholders": {"tokyo": "PARAM_CITY", "metric": "PARAM_UNITS"}
        },
        {
            "query": "Mail Bob the project update.", 
            "tool": "send_email", 
            "placeholders": {"bob": "PARAM_RECIPIENT", "project update": "PARAM_BODY"}
        },
        {
            "query": "Add 2 and 2.", 
            "tool": "calculate", 
            "placeholders": {"2 and 2": "PARAM_EXPRESSION"}
        },
    ]
    
    distractor_data = [
        "How are you today?",
        "Tell me a joke.",
        "What time is it?",
    ]
    
    return tool_corpus, training_data, test_data, distractor_data


def run_symbolic_matching_benchmark():
    corpus, train, test, distractors = get_benchmark_data()
    
    benchmark = SymbolicPatternMatchingBenchmark(tool_corpus=corpus)
    
    # 1. Train
    benchmark.train(train)
    
    # 2. Test
    result = benchmark.run_test(test, distractors)
    
    print("\nSymbolic Pattern Matching Benchmark Results:")
    print(f"Tool Accuracy: {result.tool_accuracy:.2%}")
    print(f"Parameter F1: {result.parameter_f1:.2%}")
    print(f"Distractor Rejection: {result.distractor_rejection:.2%}")
    print(f"Status: {result.status}")


if __name__ == "__main__":
    run_symbolic_matching_benchmark()
