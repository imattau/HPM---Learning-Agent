# hpm_ai_v3/tests/test_innate_substrate_extended.py
import pytest
from hpm_ai_v3.tools.innate_substrate import InnateCognitiveSubstrate

S = InnateCognitiveSubstrate()


# ── Group D ────────────────────────────────────────────────────────────────

class TestBuildMapping:
    def test_basic(self):
        assert S.build_mapping(["a", "b"], [1, 2]) == {"a": 1, "b": 2}

    def test_truncates_to_shorter(self):
        assert S.build_mapping(["a", "b", "c"], [1, 2]) == {"a": 1, "b": 2}

    def test_empty(self):
        assert S.build_mapping([], []) == {}

    def test_non_list_inputs(self):
        assert S.build_mapping("x", 9) == {"x": 9}


class TestInvertMapping:
    def test_basic(self):
        assert S.invert_mapping({"a": 1, "b": 2}) == {1: "a", 2: "b"}

    def test_empty(self):
        assert S.invert_mapping({}) == {}

    def test_duplicate_values_last_wins(self):
        result = S.invert_mapping({"a": 1, "b": 1})
        assert result == {1: "b"}

    def test_unhashable_value_skipped(self):
        result = S.invert_mapping({"a": [1, 2], "b": 3})
        assert result == {3: "b"}


class TestGetNested:
    def test_dict_access(self):
        assert S.get_nested({"a": {"b": 42}}, "a.b") == 42

    def test_list_index(self):
        assert S.get_nested({"a": [10, 20, 30]}, "a.1") == 20

    def test_missing_key_returns_none(self):
        assert S.get_nested({"a": 1}, "b") is None

    def test_empty_path_returns_obj(self):
        obj = {"x": 1}
        assert S.get_nested(obj, "") == obj

    def test_deep_missing(self):
        assert S.get_nested({"a": {"b": 1}}, "a.c.d") is None


class TestFlatten:
    def test_basic(self):
        assert S.flatten([1, [2, [3, 4]], 5]) == [1, 2, 3, 4, 5]

    def test_scalar(self):
        assert S.flatten(7) == [7]

    def test_empty(self):
        assert S.flatten([]) == []

    def test_dict_not_recursed(self):
        assert S.flatten([{"a": 1}, 2]) == [{"a": 1}, 2]

    def test_deeply_nested(self):
        assert S.flatten([[[1]], [[2, 3]]]) == [1, 2, 3]


class TestGroupBy:
    def test_basic(self):
        result = S.group_by([1, 2, 3, 4], lambda x: x % 2)
        assert result == {1: [1, 3], 0: [2, 4]}

    def test_empty(self):
        assert S.group_by([], lambda x: x) == {}

    def test_key_fn_raises_item_skipped(self):
        result = S.group_by([1, "bad", 3], lambda x: 1 / x)
        assert 1.0 in result or "bad" not in str(result)

    def test_strings(self):
        result = S.group_by(["apple", "avocado", "banana"], lambda x: x[0])
        assert result == {"a": ["apple", "avocado"], "b": ["banana"]}

# ── Group E ────────────────────────────────────────────────────────────────

class TestDetectTrend:
    def test_increasing(self):
        assert S.detect_trend([1, 2, 3, 4]) == "increasing"

    def test_decreasing(self):
        assert S.detect_trend([4, 3, 2, 1]) == "decreasing"

    def test_stable(self):
        assert S.detect_trend([5, 5, 5]) == "stable"

    def test_volatile(self):
        assert S.detect_trend([1, 3, 2, 5]) == "volatile"

    def test_short_series(self):
        assert S.detect_trend([42]) == "stable"


class TestDiffSequence:
    def test_basic(self):
        assert S.diff_sequence([1, 3, 6, 10]) == [2, 3, 4]

    def test_empty(self):
        assert S.diff_sequence([]) == []

    def test_single(self):
        assert S.diff_sequence([5]) == []

    def test_negatives(self):
        assert S.diff_sequence([5, 3, 1]) == [-2, -2]


class TestFindRepeating:
    def test_basic_period_2(self):
        assert S.find_repeating([1, 2, 1, 2, 1, 2]) == [1, 2]

    def test_period_3(self):
        assert S.find_repeating([1, 2, 3, 1, 2, 3]) == [1, 2, 3]

    def test_no_repeat(self):
        assert S.find_repeating([1, 2, 3, 4]) is None

    def test_empty(self):
        assert S.find_repeating([]) is None

    def test_single(self):
        assert S.find_repeating([7]) is None


class TestSlidingWindow:
    def test_basic(self):
        assert S.sliding_window([1, 2, 3, 4], 2) == [[1, 2], [2, 3], [3, 4]]

    def test_window_equals_length(self):
        assert S.sliding_window([1, 2, 3], 3) == [[1, 2, 3]]

    def test_window_too_large(self):
        assert S.sliding_window([1, 2], 5) == []

    def test_n_zero(self):
        assert S.sliding_window([1, 2, 3], 0) == []

    def test_empty_sequence(self):
        assert S.sliding_window([], 2) == []

# ── Group F ────────────────────────────────────────────────────────────────

class TestNormalize:
    def test_basic(self):
        result = S.normalize([1, 1, 2])
        assert abs(sum(result) - 1.0) < 1e-9
        assert abs(result[2] - 0.5) < 1e-9

    def test_all_zeros(self):
        result = S.normalize([0, 0, 0])
        assert result == [1/3, 1/3, 1/3]

    def test_single(self):
        assert S.normalize([5]) == [1.0]

    def test_empty(self):
        assert S.normalize([]) == []


class TestEntropy:
    def test_uniform_2(self):
        result = S.entropy([0.5, 0.5])
        assert abs(result - 1.0) < 1e-9

    def test_certain(self):
        assert S.entropy([1.0, 0.0]) == 0.0

    def test_empty(self):
        assert S.entropy([]) == 0.0

    def test_uniform_4(self):
        result = S.entropy([0.25, 0.25, 0.25, 0.25])
        assert abs(result - 2.0) < 1e-9


class TestArgmax:
    def test_basic(self):
        assert S.argmax([1, 3, 2]) == 1

    def test_first_tie(self):
        assert S.argmax([3, 3, 1]) == 0

    def test_single(self):
        assert S.argmax([7]) == 0

    def test_empty(self):
        assert S.argmax([]) == -1


class TestClamp:
    def test_below_lo(self):
        assert S.clamp(-5, 0, 10) == 0.0

    def test_above_hi(self):
        assert S.clamp(15, 0, 10) == 10.0

    def test_in_range(self):
        assert S.clamp(5, 0, 10) == 5.0

    def test_lo_greater_than_hi_swaps(self):
        assert S.clamp(5, 10, 0) == 5.0

    def test_non_numeric(self):
        assert S.clamp("bad", 0, 1) == 0.0

# ── Group G ────────────────────────────────────────────────────────────────

class TestDecomposeText:
    def test_basic(self):
        result = S.decompose_text("find the maximum value")
        assert result["verb"] == "find"

    def test_empty(self):
        result = S.decompose_text("")
        assert result == {"verb": "", "object": "", "modifier": ""}

    def test_known_verb(self):
        result = S.decompose_text("compute the sum of all numbers")
        assert result["verb"] == "compute"

    def test_returns_dict_keys(self):
        result = S.decompose_text("sort items by value")
        assert set(result.keys()) == {"verb", "object", "modifier"}

    def test_modifier_extracted(self):
        result = S.decompose_text("filter items by category")
        assert result["modifier"] != "" or result["object"] != ""


class TestEstimateProgress:
    def test_half(self):
        assert S.estimate_progress(5, 10) == 0.5

    def test_complete(self):
        assert S.estimate_progress(10, 10) == 1.0

    def test_over_target_clamped(self):
        assert S.estimate_progress(15, 10) == 1.0

    def test_zero_target_zero_current(self):
        assert S.estimate_progress(0, 0) == 1.0

    def test_zero_target_nonzero_current(self):
        assert S.estimate_progress(5, 0) == 0.0

    def test_non_numeric(self):
        assert S.estimate_progress("bad", 10) == 0.0


class TestCheckConstraint:
    def test_gt(self):
        assert S.check_constraint(5, "> 3") is True
        assert S.check_constraint(2, "> 3") is False

    def test_gte(self):
        assert S.check_constraint(3, ">= 3") is True

    def test_lt(self):
        assert S.check_constraint(1, "< 3") is True

    def test_lte(self):
        assert S.check_constraint(3, "<= 3") is True

    def test_eq(self):
        assert S.check_constraint(4, "== 4") is True

    def test_neq(self):
        assert S.check_constraint(4, "!= 5") is True

    def test_in_list(self):
        assert S.check_constraint(2, "in [1, 2, 3]") is True
        assert S.check_constraint(9, "in [1, 2, 3]") is False

    def test_type_check(self):
        assert S.check_constraint(42, "type:int") is True
        assert S.check_constraint("hello", "type:str") is True
        assert S.check_constraint([1, 2], "type:list") is True

    def test_len_check(self):
        assert S.check_constraint([1, 2, 3], "len == 3") is True
        assert S.check_constraint([1, 2], "len > 3") is False

    def test_unknown_constraint_permissive(self):
        assert S.check_constraint(42, "nonsense") is True
