"""Test 4: same model, same adapter weights, same batch -- do Metal (Mac) and CUDA (NVIDIA) agree?

Every machine computes one loss and its LoRA gradients from identical inputs and saves them.
Then compare two machines with:  python compare.py <host_a> <host_b>

Run:  python test4_drift.py
"""
from __future__ import annotations

import argparse
import traceback

import numpy as np

from common import RESULTS, host_tag, save
from model_setup import DEFAULT_MODEL, fixed_batch, load_with_lora, loss_fn

SAVED_PARAMS = 2  # full gradients saved for the first and last adapter matrices; norms for all


def deterministic_adapters(model, seed: int = 1234):
    """Overwrite every LoRA matrix with numpy-generated values, identical on every machine.

    lora_b is made non-zero (normally it starts at 0) so that both matrices get gradients.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    rng = np.random.RandomState(seed)
    new = []
    for name, value in sorted(tree_flatten(model.trainable_parameters())):
        if name.endswith("lora_a"):
            bound = 1 / np.sqrt(value.shape[0])
            arr = rng.uniform(-bound, bound, size=value.shape)
        else:
            arr = rng.normal(0.0, 0.001, size=value.shape)
        new.append((name, mx.array(arr.astype(np.float32)).astype(value.dtype)))
    model.update(tree_unflatten(new))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--seq", type=int, default=256)
    a = p.parse_args()
    result: dict = {"model": a.model, "seq": a.seq}
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_flatten

        result["device"] = str(mx.default_device())
        model = load_with_lora(a.model)
        deterministic_adapters(model)
        x, y = fixed_batch(1, a.seq)
        loss, grads = nn.value_and_grad(model, loss_fn)(model, x, y)
        mx.eval(loss, grads)

        flat = sorted(tree_flatten(grads))
        as_np = {name: np.array(g.astype(mx.float32)) for name, g in flat}
        result.update(
            ok=True,
            loss=float(loss),
            grad_norms={name: float(np.linalg.norm(v)) for name, v in as_np.items()},
            total_grad_norm=float(np.sqrt(sum(float(np.sum(v * v)) for v in as_np.values()))),
        )
        keep = [flat[0][0], flat[-1][0]][:SAVED_PARAMS]
        out = RESULTS / host_tag()
        out.mkdir(parents=True, exist_ok=True)
        np.savez(out / "test4_grads.npz", **{k.replace(".", "__"): as_np[k] for k in keep})
        print(f"loss {result['loss']:.6f}  total grad norm {result['total_grad_norm']:.6f}")
        code = 0
    except Exception as e:
        result.update(ok=False, error=repr(e), traceback=traceback.format_exc())
        print(traceback.format_exc())
        code = 1
    save("test4_drift", result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
