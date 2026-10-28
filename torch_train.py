"""Test 6: QLoRA fine-tuning speed and memory with PyTorch (Hugging Face Transformers + PEFT +
bitsandbytes 4-bit) on an NVIDIA GPU. Same settings as the Mac's test1_train.py, so the two
speeds can be compared directly.

Run:  python torch_train.py                                  (1.5B model, the size measured on the Mac)
      python torch_train.py --model Qwen/Qwen2.5-3B-Instruct  (3B, if it fits)
      python torch_train.py --smoke                           (tiny model on CPU, checks the script only)
"""
from __future__ import annotations

import argparse
import time
import traceback

import numpy as np

from common import save
from model_setup import LORA_CONFIG, LORA_LAYERS, TOKEN_LIMIT

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--smoke", action="store_true", help="0.5B model, CPU, no 4-bit, 3 steps")
    a = p.parse_args()
    if a.smoke:
        a.model, a.steps, a.seq = "Qwen/Qwen2.5-0.5B-Instruct", 3, 64
    result: dict = {"model": a.model, "batch": a.batch, "seq": a.seq, "steps": a.steps,
                    "lora": LORA_CONFIG, "lora_layers": LORA_LAYERS, "framework": "pytorch+peft",
                    "smoke": a.smoke}
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        cuda = torch.cuda.is_available() and not a.smoke
        device = "cuda" if cuda else "cpu"
        result["device"] = torch.cuda.get_device_name(0) if cuda else "cpu"
        t0 = time.perf_counter()
        if cuda:
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                       bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)
            model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=quant,
                                                         device_map={"": 0}, torch_dtype=dtype)
            result["base_format"] = "nf4 (bitsandbytes)"
        else:
            model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32)
            result["base_format"] = "float32 (smoke test)"
        result["load_s"] = round(time.perf_counter() - t0, 1)

        n_layers = model.config.num_hidden_layers
        # Same LoRA as the Mac: rank 8, strength 20 (PEFT: alpha / r = 20), last 16 blocks, all linear layers.
        cfg = LoraConfig(r=LORA_CONFIG["rank"], lora_alpha=LORA_CONFIG["scale"] * LORA_CONFIG["rank"],
                         lora_dropout=LORA_CONFIG["dropout"], target_modules=TARGETS,
                         layers_to_transform=list(range(max(0, n_layers - LORA_LAYERS), n_layers)),
                         task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
        params = [p for p in model.parameters() if p.requires_grad]
        result["trainable_params"] = int(sum(p.numel() for p in params))
        result["lora_grad_mb"] = round(result["trainable_params"] * 4 / 2**20, 2)

        ids = np.random.RandomState(0).randint(0, TOKEN_LIMIT, size=(a.batch, a.seq + 1))
        tokens = torch.tensor(ids, dtype=torch.long, device=device)
        x, y = tokens[:, :-1], tokens[:, 1:]
        opt = torch.optim.Adam(params, lr=a.lr)
        if cuda:
            torch.cuda.reset_peak_memory_stats()

        times, losses = [], []
        for i in range(a.steps):
            start = time.perf_counter()
            logits = model(input_ids=x).logits.float()
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
            losses.append(float(loss))
            print(f"step {i + 1}/{a.steps}  loss {losses[-1]:.4f}  {times[-1]:.2f} s")

        steady = times[1:] or times  # step 1 includes warm-up
        result.update(
            ok=True,
            step_s_median=round(float(np.median(steady)), 3),
            tokens_per_s=round(a.batch * a.seq / float(np.median(steady)), 1),
            peak_memory_gb=round(torch.cuda.max_memory_allocated() / 1e9, 3) if cuda else None,
            loss_first=round(losses[0], 4),
            loss_last=round(losses[-1], 4),
            loss_went_down=losses[-1] < losses[0],
        )
        code = 0
    except Exception as e:  # out of memory, missing kernel, ... is a result too
        result.update(ok=False, error=repr(e), traceback=traceback.format_exc())
        print(traceback.format_exc())
        code = 1
    save("test6_torch_train" + ("_smoke" if a.smoke else ""), result)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
