"""Project-wide configuration constants."""
import os

# Single shared pattern library — all builders and crawlers read/write here by default.
DEFAULT_LIBRARY_PATH = os.environ.get(
    "HPM_LIBRARY_PATH",
    os.path.join(os.path.expanduser("~"), ".hpm", "library.pkl"),
)
