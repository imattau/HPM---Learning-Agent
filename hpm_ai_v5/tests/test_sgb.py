from hpm_ai_v5.planning.sgb import SymbolicGeneralisationBenchmark


def test_symbolic_generalisation_benchmark() -> None:
    result = SymbolicGeneralisationBenchmark().run()
    assert result.result == "success", (
        f"SGB failed: {result.reason}. "
        f"cross_scale_symbolic={result.cross_scale_symbolic_accuracy:.2f} "
        f"numeric={result.cross_scale_numeric_accuracy:.2f}"
    )
