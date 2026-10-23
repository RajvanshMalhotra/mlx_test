"""Test 1: can this machine fine-tune a 4-bit model with LoRA in MLX? How fast, how much memory?

Run:  python test1_train.py                      (3B model, 10 steps)
      python test1_train.py --model mlx-community/Qwen2.5-1.5B-Instruct-4bit   (if 3B does not fit)
"""
from __future__ import annotations

import argparse
import time
import traceback

import numpy as np

from common import peak_memory_gb, reset_peak_memory, save
from model_setup import DEFAULT_MODEL, LORA_CONFIG, LORA_LAYERS, fixed_batch, load_with_lora, loss_fn


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    a = p.parse_args()
    result: dict = {"model": a.model, "batch": a.batch, "seq": a.seq, "steps": a.steps,
                    "lora": LORA_CONFIG, "lora_layers": LORA_LAYERS}
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten

        result["device"] = str(mx.default_device())
        t0 = time.perf_counter()
        model = load_with_lora(a.model)
        result["load_s"] = round(time.perf_counter() - t0, 1)
        result["trainable_params"] = int(sum(v.size for _, v in tree_flatten(model.trainable_parameters())))

        x, y = fixed_batch(a.batch, a.seq)
        optimizer = optim.Adam(learning_rate=a.lr)
        loss_and_grad = nn.value_and_grad(model, loss_fn)

        reset_peak_memory(mx)
        times, losses = [], []
        for i in range(a.steps):
            start = time.perf_counter()
            loss, grads = loss_and_grad(model, x, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            times.append(time.perf_counter() - start)
            losses.append(float(loss))
            print(f"step {i + 1}/{a.steps}  loss {losses[-1]:.4f}  {times[-1]:.2f} s")

        steady = times[1:] or times  # step 1 includes warm-up
        result.update(
            ok=True,
            step_s_median=round(float(np.median(steady)), 3),
            tokens_per_s=round(a.batch * a.seq / float(np.median(steady)), 1),
            peak_memory_gb=peak_memory_gb(mx),
            loss_first=round(losses[0], 4),
            loss_last=round(losses[-1], 4),
            loss_went_down=losses[-1] < losses[0],
        )
        code = 0
    except Exception as e:  # a failure (out of memory, missing kernel) is a result too
        result.update(ok=False, error=repr(e), traceback=traceback.format_exc())
        print(traceback.format_exc())
        code = 1
    save("test1_train", result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
