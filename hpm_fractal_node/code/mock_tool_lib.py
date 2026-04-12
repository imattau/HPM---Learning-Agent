"""
Mock Tool Library for SP55 Experiment 45.
Contains functions with distinct behavioral signatures for discovery.
"""

def find_pairs(data):
    """Returns all unique combinations of 2 elements from the data."""
    if not isinstance(data, (list, tuple)) or len(data) < 2:
        return []
    import itertools
    return list(itertools.combinations(data, 2))

def uniquify(data):
    """Returns a list of unique elements from the input data, preserving order-ish."""
    if not isinstance(data, (list, tuple)):
        return data
    seen = set()
    return [x for x in data if not (x in seen or seen.add(x))]

def invert_nested(data):
    """Reverses the order of elements in a nested list structure."""
    if not isinstance(data, list):
        return data
    return [x[::-1] if isinstance(x, (list, tuple, str)) else x for x in data[::-1]]

def scale_by_mean(data):
    """Scales all numeric elements in a list by their mean."""
    if not isinstance(data, list) or not data:
        return data
    numeric = [x for x in data if isinstance(x, (int, float))]
    if not numeric:
        return data
    m = sum(numeric) / len(numeric)
    return [x * m if isinstance(x, (int, float)) else x for x in data]
