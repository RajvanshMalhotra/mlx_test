"""Compare Test 4 results from two machines.

Run:  python compare.py <host_a> <host_b>      (folder names under results/)
"""
from __future__ import annotations

import json
import sys

import numpy as np

from common import RESULTS


def load(host: str):
    data = json.loads((RESULTS / host / "test4_drift.json").read_text())
    if not data.get("ok"):
        sys.exit(f"{host}: test 4 failed there: {data.get('error')}")
    grads = np.load(RESULTS / host / "test4_grads.npz")
    return data, grads


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a_name, b_name = sys.argv[1], sys.argv[2]
    a, ga = load(a_name)
    b, gb = load(b_name)
    if a["model"] != b["model"]:
        sys.exit(f"different models: {a['model']} vs {b['model']}")

    print(f"{a_name} ({a['device']})  vs  {b_name} ({b['device']})")
    rel = abs(a["loss"] - b["loss"]) / abs(a["loss"])
    print(f"loss               {a['loss']:.6f} vs {b['loss']:.6f}   relative diff {rel:.2e}")

    norm_rel = [
        abs(a["grad_norms"][k] - b["grad_norms"][k]) / max(abs(a["grad_norms"][k]), 1e-12)
        for k in a["grad_norms"]
    ]
    print(f"grad norms         worst relative diff {max(norm_rel):.2e}  median {np.median(norm_rel):.2e}")

    for k in ga.files:
        x, y = ga[k].ravel(), gb[k].ravel()
        l2 = np.linalg.norm(x - y) / max(np.linalg.norm(x), 1e-12)
        cos = float(x @ y / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12))
        print(f"{k.replace('__', '.')}\n  max abs diff {np.max(np.abs(x - y)):.2e}  relative L2 {l2:.2e}  cosine {cos:.6f}")

    print("\nRule of thumb: relative diffs below ~1e-2 and cosine above ~0.999 are normal rounding noise;"
          " much larger means the two backends are not computing the same thing.")


if __name__ == "__main__":
    main()
