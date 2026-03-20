from hpm.dynamics.density import PatternDensity


class _ZeroStruct:
    """Stub pattern with zero connectivity and compress — all structural components zero."""
    def connectivity(self): return 0.0
    def compress(self): return 0.0


def test_density_low_when_saturation_and_field_zero():
    """With all three components zero (zero struct, high loss, zero field_freq), D == 0.0."""
    pd = PatternDensity(alpha_conn=0.33, alpha_sat=0.33, alpha_amp=0.34)
    p = _ZeroStruct()
    D = pd.compute(p, loss=1e9, capacity=0.0, field_freq=0.0)
    assert D == 0.0


def test_density_one_when_all_components_maxed():
    """Max connectivity+compress, zero loss, full capacity, full field_freq -> D near 1."""
    pd = PatternDensity(alpha_conn=0.33, alpha_sat=0.33, alpha_amp=0.34)
    class HighDensityPattern:
        def connectivity(self): return 1.0
        def compress(self): return 1.0
    p = HighDensityPattern()
    D = pd.compute(p, loss=0.0, capacity=1.0, field_freq=1.0)
    assert abs(D - 1.0) < 1e-6


def test_structural_component_uses_connectivity_and_compress():
    pd = PatternDensity(alpha_conn=1.0, alpha_sat=0.0, alpha_amp=0.0)
    class FakePattern:
        def connectivity(self): return 0.6
        def compress(self): return 0.4
    p = FakePattern()
    D = pd.compute(p, loss=0.0, capacity=0.0, field_freq=0.0)
    expected_structural = (0.6 + 0.4) / 2  # = 0.5
    assert abs(D - expected_structural) < 1e-9


def test_saturation_high_for_low_loss_high_capacity():
    pd = PatternDensity(alpha_conn=0.0, alpha_sat=1.0, alpha_amp=0.0)
    p = _ZeroStruct()
    D = pd.compute(p, loss=0.01, capacity=0.9, field_freq=0.0)
    assert D > 0.8


def test_saturation_low_for_high_loss():
    pd = PatternDensity(alpha_conn=0.0, alpha_sat=1.0, alpha_amp=0.0)
    p = _ZeroStruct()
    D = pd.compute(p, loss=100.0, capacity=1.0, field_freq=0.0)
    assert D < 0.05


def test_field_freq_zero_contribution():
    pd = PatternDensity(alpha_conn=0.0, alpha_sat=0.0, alpha_amp=1.0)
    p = _ZeroStruct()
    D_zero = pd.compute(p, loss=0.0, capacity=0.0, field_freq=0.0)
    D_half = pd.compute(p, loss=0.0, capacity=0.0, field_freq=0.5)
    assert abs(D_zero) < 1e-9
    assert abs(D_half - 0.5) < 1e-9


def test_negative_loss_clamped_to_zero():
    """Defensive: negative loss should be treated as zero."""
    pd = PatternDensity(alpha_conn=0.0, alpha_sat=1.0, alpha_amp=0.0)
    p = _ZeroStruct()
    D_neg = pd.compute(p, loss=-5.0, capacity=1.0, field_freq=0.0)
    D_zero = pd.compute(p, loss=0.0, capacity=1.0, field_freq=0.0)
    assert abs(D_neg - D_zero) < 1e-9


def test_output_clamped_to_unit_interval():
    """Alphas > 1 should not produce D > 1 due to final clamp."""
    pd = PatternDensity(alpha_conn=5.0, alpha_sat=5.0, alpha_amp=5.0)
    class MaxPattern:
        def connectivity(self): return 1.0
        def compress(self): return 1.0
    p = MaxPattern()
    D = pd.compute(p, loss=0.0, capacity=1.0, field_freq=1.0)
    assert 0.0 <= D <= 1.0
