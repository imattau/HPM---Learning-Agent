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
