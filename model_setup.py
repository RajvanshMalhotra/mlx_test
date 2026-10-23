"""Load a 4-bit model with LoRA adapters, and build a fixed batch every machine can reproduce."""
from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "mlx-community/Qwen2.5-3B-Instruct-4bit"
SMOKE_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"  # tiny, for checking the scripts work

# Written out explicitly: mlx-lm's defaults (scale 20, last 16 blocks, all linear layers)
# differ from PEFT's, so every machine must use exactly these values.
LORA_CONFIG = {"rank": 8, "scale": 20.0, "dropout": 0.0}
LORA_LAYERS = 16
TOKEN_LIMIT = 32000  # below every modern vocabulary size, so the same ids are valid everywhere


def load_with_lora(model_id: str, lora_layers: int = LORA_LAYERS):
    from mlx_lm import load
    from mlx_lm.tuner.utils import linear_to_lora_layers

    model, _ = load(model_id)
    model.freeze()
    linear_to_lora_layers(model, lora_layers, LORA_CONFIG)
    return model


def fixed_batch(batch: int, seq: int, seed: int = 0):
    """The same token ids on every machine (numpy's RandomState is stable across platforms)."""
    import mlx.core as mx

    ids = np.random.RandomState(seed).randint(0, TOKEN_LIMIT, size=(batch, seq + 1)).astype(np.int32)
    tokens = mx.array(ids)
    return tokens[:, :-1], tokens[:, 1:]


def loss_fn(model, x, y):
    import mlx.core as mx
    import mlx.nn as nn

    logits = model(x).astype(mx.float32)
    return nn.losses.cross_entropy(logits, y).mean()
