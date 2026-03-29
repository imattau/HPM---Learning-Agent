"""Tests for the Query base class."""

import numpy as np
import pytest
from hfn.query import Query


class FixedQuery(Query):
    """Subclass that returns fixed strings."""
    def fetch(self, gap_mu, context=None):
        return ["token_a", "token_b", "sig: def token_a ..."]


class EmptyQuery(Query):
    """Subclass that explicitly returns empty list."""
    def fetch(self, gap_mu, context=None):
        return []


class BaseQuery(Query):
    """Direct instantiation of Query base class."""
    pass


def test_base_query_returns_empty_list():
    q = Query()
    mu = np.zeros(10)
    result = q.fetch(mu)
    assert result == []


def test_base_query_with_context_returns_empty_list():
    q = Query()
    mu = np.ones(5)
    result = q.fetch(mu, context={"some": "context"})
    assert result == []


def test_fixed_query_returns_strings():
    q = FixedQuery()
    mu = np.zeros(10)
    result = q.fetch(mu)
    assert len(result) == 3
    assert "token_a" in result
    assert "token_b" in result
    assert "sig: def token_a ..." in result


def test_fixed_query_returns_list_of_str():
    q = FixedQuery()
    mu = np.zeros(10)
    result = q.fetch(mu)
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, str)


def test_empty_subclass_returns_empty():
    q = EmptyQuery()
    mu = np.zeros(10)
    result = q.fetch(mu)
    assert result == []


def test_query_gap_mu_passed_through():
    """Subclass can use gap_mu to determine results."""
    class TokenQuery(Query):
        def fetch(self, gap_mu, context=None):
            idx = int(np.argmax(gap_mu))
            return [f"token_{idx}"]

    q = TokenQuery()
    mu = np.zeros(8)
    mu[3] = 1.0
    result = q.fetch(mu)
    assert result == ["token_3"]
