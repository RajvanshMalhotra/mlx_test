"""Shared helpers for the MLX feasibility tests: where results go, and peak-memory readings."""
from __future__ import annotations

import json
import platform
import socket
import sys
import time
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def host_tag() -> str:
    return socket.gethostname().split(".")[0].lower()


def save(name: str, data: dict) -> Path:
    """Write results/<host>/<name>.json, stamped with machine details."""
    out = RESULTS / host_tag()
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.json"
    stamped = {
        "host": host_tag(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **data,
    }
    path.write_text(json.dumps(stamped, indent=2))
    print(f"saved {path}")
    return path


def _memory_fn(mx, name: str):
    # Newer MLX exposes these at the top level; older versions only under mx.metal.
    return getattr(mx, name, None) or getattr(getattr(mx, "metal", None), name, None)


def peak_memory_gb(mx) -> float | None:
    fn = _memory_fn(mx, "get_peak_memory")
    return round(fn() / 1e9, 3) if fn else None


def reset_peak_memory(mx) -> None:
    fn = _memory_fn(mx, "reset_peak_memory")
    if fn:
        fn()
