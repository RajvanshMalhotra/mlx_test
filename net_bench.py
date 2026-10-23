"""Plain network speed test between two machines (no MLX needed). Use it if Test 3 fails,
to tell a network problem (firewall, blocked Wi-Fi) apart from an MLX problem.

On machine A:  python net_bench.py server --port 5001
On machine B:  python net_bench.py client <A's IP> --port 5001
"""
from __future__ import annotations

import argparse
import socket
import time

from common import save

PAYLOAD_MB = 100
CHUNK = 1 << 20
PINGS = 50


def recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        part = conn.recv(n - len(buf))
        if not part:
            raise ConnectionError("connection closed early")
        buf.extend(part)
    return bytes(buf)


def server(port: int) -> None:
    with socket.create_server(("0.0.0.0", port)) as srv:
        print(f"waiting on port {port} ...")
        conn, addr = srv.accept()
        with conn:
            print(f"connected: {addr[0]}")
            for _ in range(PINGS):
                conn.sendall(recv_exact(conn, 1))
            remaining = PAYLOAD_MB * CHUNK
            while remaining:
                remaining -= len(conn.recv(min(CHUNK, remaining)))
            conn.sendall(b"k")
    print("done")


def client(host: str, port: int) -> None:
    with socket.create_connection((host, port), timeout=30) as conn:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        rtts = []
        for _ in range(PINGS):
            start = time.perf_counter()
            conn.sendall(b"p")
            recv_exact(conn, 1)
            rtts.append((time.perf_counter() - start) * 1000)
        chunk = b"\0" * CHUNK
        start = time.perf_counter()
        for _ in range(PAYLOAD_MB):
            conn.sendall(chunk)
        recv_exact(conn, 1)
        seconds = time.perf_counter() - start
    rtts.sort()
    result = {
        "peer": host,
        "rtt_ms_median": round(rtts[len(rtts) // 2], 2),
        "rtt_ms_p95": round(rtts[int(len(rtts) * 0.95) - 1], 2),
        "MB_per_s": round(PAYLOAD_MB / seconds, 1),
        "Mbit_per_s": round(PAYLOAD_MB * 8 / seconds, 1),
    }
    print(result)
    save("net_bench", result)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["server", "client"])
    p.add_argument("host", nargs="?")
    p.add_argument("--port", type=int, default=5001)
    a = p.parse_args()
    if a.mode == "server":
        server(a.port)
    else:
        if not a.host:
            p.error("client needs the server's IP")
        client(a.host, a.port)


if __name__ == "__main__":
    main()
