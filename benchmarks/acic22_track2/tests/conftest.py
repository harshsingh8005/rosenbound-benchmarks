"""Put the benchmark directory on the path so tests can import run/src."""

import sys
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parents[1]
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))
