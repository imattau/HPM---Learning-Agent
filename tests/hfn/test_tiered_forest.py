import psutil
import pytest

def test_psutil_importable():
    mem = psutil.virtual_memory()
    assert mem.available > 0
