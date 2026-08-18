#!/usr/bin/env python3
"""Score document page images against a text query with EVIE-Preview-4.5B."""

import argparse

import torch
from PIL import Image

from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor

from bidirectional import enable_bidirectional_attention


def load(model_id: str, device: str = "cuda"):
    model = ColQwen3_5.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="flash_attention_2",
    ).eval()
    enable_bidirectional_attention(model)
    return model, ColQwen3_5Processor.from_pretrained(model_id)


@torch.inference_mode()
def score(model, processor, images, queries):
    image_embeddings = model(**processor.process_images(images).to(model.device))
    model.rope_deltas = None
    query_embeddings = model(**processor.process_queries(queries).to(model.device))
    return processor.score(query_embeddings, image_embeddings)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="EVIE-Preview-4.5B")
    ap.add_argument("--query", required=True, action="append")
    ap.add_argument("--image", required=True, action="append")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    model, processor = load(args.model, args.device)
    scores = score(model, processor, [Image.open(p) for p in args.image], args.query)
    for query, row in zip(args.query, scores):
        ranked = sorted(zip(args.image, row.tolist()), key=lambda x: -x[1])
        print(query)
        for path, value in ranked:
            print("  %8.3f  %s" % (value, path))


if __name__ == "__main__":
    main()
