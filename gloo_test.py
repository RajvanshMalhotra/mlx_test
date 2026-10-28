"""Test 7: can the laptops average arrays together with PyTorch's gloo backend over Wi-Fi?
This is exactly what one data-parallel training step sends (20 MB of LoRA gradients).

Start every laptop within ~60 s, each with its own rank (the Mac is rank 0):
  Mac:      python gloo_test.py --rank 0 --world 3 --master <Mac's IP>
  Windows:  python gloo_test.py --rank 1 --world 3 --master <Mac's IP>
            python gloo_test.py --rank 2 --world 3 --master <Mac's IP>
If it hangs at "joining", see the README (firewall, GLOO_SOCKET_IFNAME).
"""
from __future__ import annotations

import argparse
import datetime
import os
import platform
import time
import traceback

from common import save


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rank", type=int, required=True)
    p.add_argument("--world", type=int, required=True)
    p.add_argument("--master", required=True, help="IP of the rank-0 laptop")
    p.add_argument("--port", type=int, default=29500)
    p.add_argument("--mb", type=float, default=20.0, help="payload size; 20 MB = our LoRA gradients")
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--iface", help="network interface for gloo, e.g. en0 / Wi-Fi")
    a = p.parse_args()
    if a.iface:
        os.environ["GLOO_SOCKET_IFNAME"] = a.iface
    result: dict = {"rank": a.rank, "world": a.world, "master": a.master, "payload_mb": a.mb}
    try:
        import torch
        import torch.distributed as dist

        print(f"joining {a.master}:{a.port} as rank {a.rank}/{a.world} ...", flush=True)
        # Windows PyTorch builds have no libuv, so the store must not ask for it.
        store = dist.TCPStore(a.master, a.port, a.world, is_master=a.rank == 0,
                              timeout=datetime.timedelta(seconds=120),
                              use_libuv=platform.system() != "Windows")
        dist.init_process_group("gloo", store=store, rank=a.rank, world_size=a.world,
                                timeout=datetime.timedelta(seconds=120))
        x = torch.full((int(a.mb * 2**20 / 4),), float(a.rank + 1))
        dist.all_reduce(x)  # warm-up + correctness
        expected = a.world * (a.world + 1) / 2
        result["sum_correct"] = bool(torch.allclose(x[:10], torch.full((10,), expected)))
        times = []
        for i in range(a.iters):
            dist.barrier()
            t = time.perf_counter()
            dist.all_reduce(x)
            times.append(time.perf_counter() - t)
            print(f"all_reduce {i + 1}/{a.iters}: {times[-1]:.3f} s", flush=True)
        med = sorted(times)[len(times) // 2]
        # Ring all-reduce moves 2(n-1)/n of the payload per laptop.
        result.update(ok=True, allreduce_s_median=round(med, 3), allreduce_s_max=round(max(times), 3),
                      bus_MB_per_s=round(2 * (a.world - 1) / a.world * a.mb / med, 2))
        dist.destroy_process_group()
        code = 0
    except Exception as e:
        result.update(ok=False, error=repr(e), traceback=traceback.format_exc())
        print(traceback.format_exc())
        code = 1
    print(result)
    save(f"test7_gloo_rank{a.rank}", result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
