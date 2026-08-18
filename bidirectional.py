"""Bidirectional attention for ColQwen3.5 retrieval.

Released `colpali-engine` (through 0.3.17) builds ColQwen3.5 with the causal
Qwen3.5 masks it inherits from the generative backbone. EVIE was trained and
evaluated with the full-attention layers encoder-ized, so the checkpoint must be
switched before it reproduces the reported scores.

Qwen3.5 interleaves GatedDeltaNet (`linear_attention`) and `full_attention`
layers. Only the full-attention layers are flipped here; the recurrent layers
are order-dependent by construction and are left untouched.
"""

from typing import Any

_ATTENTION_CLASSES = ("Qwen3_5Attention", "Qwen3Attention")


def enable_bidirectional_attention(model: Any) -> None:
    """Encoder-ize the full-attention layers of a ColQwen3.5 model, in place."""
    config = getattr(model, "config", None)
    for cfg in (config, getattr(config, "text_config", None)):
        # `create_causal_mask` falls back to `create_bidirectional_mask` on this flag.
        if cfg is not None:
            cfg.is_causal = False

    for module in model.modules():
        if module.__class__.__name__ in _ATTENTION_CLASSES and hasattr(module, "is_causal"):
            module.is_causal = False
