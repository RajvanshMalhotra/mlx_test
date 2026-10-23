"""Test 0: what is this machine, and can MLX use its GPU?

Run: python env_check.py
"""
from __future__ import annotations

import platform
import shutil
import subprocess

from common import save


def main() -> None:
    info: dict = {"system": platform.system(), "machine": platform.machine()}

    try:
        import mlx.core as mx

        info["mlx_version"] = mx.__version__
        info["default_device"] = str(mx.default_device())
        a = mx.random.normal((1024, 1024))
        mx.eval(a @ a)
        info["gpu_matmul_ok"] = True
    except Exception as e:  # report, don't crash: a failure here IS the result
        info["mlx_error"] = repr(e)

    try:
        import mlx_lm

        info["mlx_lm_version"] = getattr(mlx_lm, "__version__", "unknown")
    except Exception as e:
        info["mlx_lm_error"] = repr(e)

    if shutil.which("nvidia-smi"):
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True,
        )
        info["nvidia_smi"] = out.stdout.strip() or out.stderr.strip()

    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        pass

    save("test0_env", info)
    for key, value in info.items():
        print(f"{key:16} {value}")


if __name__ == "__main__":
    main()
