"""Write hosts.json for Test 3. List IPs in rank order: rank 0 first.

Run:  python make_hostfile.py 192.168.1.10 192.168.1.20 [192.168.1.30] [--port 5000]
Find a machine's IP:  macOS `ipconfig getifaddr en0`  ·  Windows `ipconfig`  ·  Linux/WSL `hostname -I`
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("ips", nargs="+")
    p.add_argument("--port", type=int, default=5000)
    p.add_argument("--out", default="hosts.json")
    a = p.parse_args()
    hosts = [[f"{ip}:{a.port}"] for ip in a.ips]
    Path(a.out).write_text(json.dumps(hosts, indent=2))
    print(f"wrote {a.out}:")
    for rank, entry in enumerate(hosts):
        print(f"  rank {rank}: {entry[0]}")


if __name__ == "__main__":
    main()
