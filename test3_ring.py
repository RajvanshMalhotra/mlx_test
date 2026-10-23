"""Test 3: can machines (Mac + NVIDIA) do an MLX all-reduce together over the network?

On EVERY machine, with the same hosts.json (make it with make_hostfile.py):
  macOS / Linux / WSL:   MLX_RANK=0 MLX_HOSTFILE=hosts.json python test3_ring.py
  Windows PowerShell:    $env:MLX_RANK=1; $env:MLX_HOSTFILE="hosts.json"; python test3_ring.py
Each machine uses its own rank (0, 1, 2) matching its line in hosts.json. Start them within ~30 s of each other.
"""
from __future__ import annotations

import os
import time
import traceback

import numpy as np

from common import save

SIZES_MB = (1, 10, 50)
REPEATS = 3


def all_sum(mx, x):
    try:
        return mx.distributed.all_sum(x)
    except Exception:
        # Some backends only run collectives on the CPU stream.
        return mx.distributed.all_sum(x, stream=mx.cpu)


def main() -> int:
    result: dict = {"rank_env": os.environ.get("MLX_RANK"), "hostfile_env": os.environ.get("MLX_HOSTFILE")}
    try:
        import mlx.core as mx

        if result["rank_env"] is None or result["hostfile_env"] is None:
            raise RuntimeError("set MLX_RANK and MLX_HOSTFILE first (see the top of this file)")
        result["device"] = str(mx.default_device())
        world = mx.distributed.init(backend="ring")
        rank, size = world.rank(), world.size()
        result.update(rank=rank, world_size=size)
        print(f"rank {rank} of {size} on {result['device']}")

        # 1. Exact integers: every rank sends rank+1, everyone must get 1+2+...+size.
        got = all_sum(mx, mx.ones(10) * (rank + 1))
        mx.eval(got)
        result["integer_sum_ok"] = bool(np.allclose(np.array(got), size * (size + 1) / 2))

        # 2. Random floats from per-rank seeds: everyone can recompute the exact expected sum.
        mine = np.random.RandomState(100 + rank).normal(size=4096).astype(np.float32)
        expected = sum(np.random.RandomState(100 + r).normal(size=4096).astype(np.float32) for r in range(size))
        got = all_sum(mx, mx.array(mine))
        mx.eval(got)
        result["float_sum_max_abs_err"] = float(np.max(np.abs(np.array(got) - expected)))

        # 3. Speed at gradient-like sizes.
        speeds = {}
        for mb in SIZES_MB:
            x = mx.ones(mb * 1024 * 1024 // 4)
            mx.eval(all_sum(mx, x))  # warm-up
            start = time.perf_counter()
            for _ in range(REPEATS):
                mx.eval(all_sum(mx, x))
            seconds = (time.perf_counter() - start) / REPEATS
            speeds[f"{mb}MB"] = {"seconds": round(seconds, 4), "MB_per_s": round(mb / seconds, 1)}
            print(f"all_sum {mb:>3} MB: {seconds:.3f} s  ({mb / seconds:.1f} MB/s)")
        result.update(ok=True, all_sum=speeds)
        code = 0
    except Exception as e:
        result.update(ok=False, error=repr(e), traceback=traceback.format_exc())
        print(traceback.format_exc())
        code = 1
    save(f"test3_ring_rank{result.get('rank', result['rank_env'])}", result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
