"""Test 5: what is this machine, can PyTorch use its NVIDIA GPU, and is gloo (network) available?

Run: python torch_env_check.py
"""
from __future__ import annotations

import platform

from common import save


def main() -> None:
    info: dict = {"system": platform.system(), "machine": platform.machine()}
    try:
        import torch
        import torch.distributed as dist

        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["distributed_available"] = dist.is_available()
        info["gloo_available"] = dist.is_available() and dist.is_gloo_available()
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            info["gpu_name"] = p.name
            info["dedicated_vram_gb"] = round(p.total_memory / 1e9, 2)   # dedicated only, not "shared"
            info["compute_capability"] = f"{p.major}.{p.minor}"
            info["bf16_supported"] = torch.cuda.is_bf16_supported()
            a = torch.randn(2048, 2048, device="cuda")
            torch.cuda.synchronize()
            info["gpu_matmul_ok"] = bool(torch.isfinite(a @ a).all())
    except Exception as e:  # report, don't crash: a failure here IS the result
        info["torch_error"] = repr(e)

    for mod in ("transformers", "peft", "bitsandbytes"):
        try:
            info[f"{mod}_version"] = __import__(mod).__version__
        except Exception as e:
            info[f"{mod}_error"] = repr(e)
    try:
        import psutil

        info["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        pass

    save("test5_torch_env", info)
    for key, value in info.items():
        print(f"{key:22} {value}")


if __name__ == "__main__":
    main()
