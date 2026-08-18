---
language:
- en
- fr
- de
- it
- es
- pt
- zh
- ja
- multilingual
license: apache-2.0
library_name: colpali-engine
pipeline_tag: visual-document-retrieval
tags:
- visual-document-retrieval
- vision-language
- colpali
- colbert
- late-interaction
- multi-vector
- qwen3_5
- vidore
- document-retrieval
- multimodal
- state-of-the-art
- sentence-transformers
base_model:
- Qwen/Qwen3.5-4B
datasets:
- vidore/vidore_benchmark
- vidore/vidore_benchmark_v2
inference: false
---

<div align="center">

# 🏆 EVIE-Preview-4.5B

### **Rank #1 on ViDoRe V3 · Rank #1 on ViDoRe V1+V2**

**The most accurate visual document retriever, with native 128-dimensional token vectors.**

<p align="center">
  <a href="#-vidore-v3--rank-1"><img src="https://img.shields.io/badge/🥇_ViDoRe_V3-65.36_·_Rank_%231-FFD700?style=for-the-badge&labelColor=1a1a2e" alt="ViDoRe V3 Rank 1"></a>
  <a href="#-vidore-v1--v2--rank-1"><img src="https://img.shields.io/badge/🥇_ViDoRe_V1+V2-85.77_·_Rank_%231-FFD700?style=for-the-badge&labelColor=1a1a2e" alt="ViDoRe V1+V2 Rank 1"></a>
</p>

<p align="center">
  <a href="#-index-cost"><img src="https://img.shields.io/badge/Token_Dim-128D_native-39d4bd?style=flat-square&logo=vectorworks&logoColor=white" alt="128D"></a>
  <a href="#-index-cost"><img src="https://img.shields.io/badge/Index-420.5_GiB_%2F_1M_pages-45c8f5?style=flat-square&logo=databricks&logoColor=white" alt="Index Cost"></a>
  <a href="#-model-footprint"><img src="https://img.shields.io/badge/Params-4.54B-b285f7?style=flat-square&logo=pytorch&logoColor=white" alt="4.54B"></a>
  <a href="#-multilingual"><img src="https://img.shields.io/badge/Languages-7_query_langs-f4c45e?style=flat-square&logo=googletranslate&logoColor=white" alt="Multilingual"></a>
</p>

<p align="center">
  <a href="LICENSE.txt"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg?style=flat-square" alt="License"></a>
  <a href="https://huggingface.co/Qwen/Qwen3.5-4B"><img src="https://img.shields.io/badge/Base-Qwen3.5--4B-purple.svg?style=flat-square" alt="Base Model"></a>
  <a href="https://github.com/illuin-tech/colpali"><img src="https://img.shields.io/badge/Framework-ColPali_Engine-orange.svg?style=flat-square" alt="Framework"></a>
  <a href="https://huggingface.co/tencent/EVIE-Preview-4.5B"><img src="https://img.shields.io/badge/🤗_Hugging_Face-Model-FFD21E.svg?style=flat-square" alt="Hugging Face"></a>
  <a href="https://github.com/Tencent/EVIE-Preview-4.5B"><img src="https://img.shields.io/badge/GitHub-Source-181717.svg?style=flat-square&logo=github" alt="GitHub"></a>
</p>

[🏆 Results](#-vidore-v3--rank-1) • [💾 Index Cost](#-index-cost) • [⚡ Quick Start](#-quick-start) • [🔬 Reproducing](#-reproducing) • [🧠 Architecture](#-architecture) • [📚 Citation](#-citation)

</div>

---

## 🥇 ViDoRe V3 — Rank #1

8 public domains × 6 query languages, nDCG@10.

| # | Model | Params | Token Dim | **V3 public** |
| :---: | :--- | ---: | ---: | ---: |
| 🥇 **1** | **EVIE-Preview-4.5B** | 4.54B | 128D | **65.36** |
| 🥈 2 | webAI-ColVec1.1-8b | 8.40B | 640D | 65.32 |
| 🥉 3 | webAI-ColVec1.1-4b | 4.54B | 640D | 63.90 |
| 4 | nemotron-colembed-vl-8b-v2 | 8B | — | 63.54 |
| 5 | tomoro-colqwen3-embed-8b | 8B | — | 61.60 |
| 6 | nemotron-colembed-vl-4b-v2 | 4B | — | 61.42 |
| 7 | tomoro-colqwen3-embed-4b | 4B | — | 60.16 |
| 8 | llama-nemotron-colembed-vl-3b-v2 | 3B | — | 59.70 |
| 9 | colnomic-embed-multimodal-7b | 7B | — | 57.64 |
| 10 | jina-embeddings-v4 | ~3.8B | — | 57.54 |

### Two deployment tiers, one checkpoint

| Visual tokens / page | **V3 public** | Vectors / page | Raw index / 1M pages (BF16) |
| :--- | ---: | ---: | ---: |
| 768 (training budget) | 64.56 | **751.62** | **179.2 GiB** |
| 1,792 (extrapolated) | **65.36** | 1,763.58 | 420.5 GiB |

**EVIE was trained at 768 visual tokens per page.** The 1,792 tier is pure test-time extrapolation — the same weights, never trained or fine-tuned at that budget, and never re-exported. That the model does not merely hold up but *gains* 0.80 nDCG@10 at more than twice its training budget, improving in 7 of the 8 domains, is a direct read on how well its page representation generalises beyond the resolution it was fit to.

Pick whichever tier fits your compute budget.

The lighter tier holds a million pages in under 180 GiB.

### Per-domain breakdown

| Model | **Avg** | CompSci | Energy | Finance EN | Finance FR | HR | Industrial | Pharma | Physics |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 🥇 **EVIE-Preview-4.5B** | **65.36** | 80.65 | 71.36 | 70.50 | 54.44 | 67.34 | 58.76 | 69.20 | 50.62 |
| webAI-ColVec1.1-8b | 65.32 | 80.08 | 70.12 | 71.90 | 54.87 | 68.55 | 57.65 | 67.88 | 51.50 |
| webAI-ColVec1.1-4b | 63.90 | 80.34 | 69.50 | 69.18 | 53.13 | 66.90 | 56.36 | 67.25 | 51.24 |
| nemotron-colembed-vl-8b-v2 | 63.54 | 79.30 | 69.82 | 67.29 | 51.54 | 66.32 | 56.03 | 67.19 | 50.84 |
| tomoro-colqwen3-embed-8b | 61.60 | 75.35 | 68.41 | 65.08 | 49.10 | 63.98 | 54.41 | 66.36 | 50.13 |
| nemotron-colembed-vl-4b-v2 | 61.42 | 78.56 | 67.48 | 65.02 | 49.01 | 62.39 | 53.91 | 66.10 | 48.86 |
| llama-nemotron-colembed-vl-3b-v2 | 59.70 | 77.09 | 64.88 | 64.23 | 44.41 | 62.28 | 51.71 | 66.04 | 46.93 |
| colnomic-embed-multimodal-7b | 57.64 | 76.20 | 63.58 | 56.57 | 45.46 | 58.67 | 50.13 | 62.26 | 48.25 |
| jina-embeddings-v4 | 57.54 | 71.81 | 63.50 | 59.30 | 46.10 | 59.53 | 50.38 | 63.09 | 46.63 |

*EVIE rows measured with [`reproduce.sh`](reproduce.sh). Comparison rows are the vendors' published ViDoRe V3 public scores.*

---

## 🥇 ViDoRe V1 + V2 — Rank #1

14 tasks, nDCG@5. **First place on the classic boards too.**

| # | Model | **Avg** | ArxivQA | DocVQA | InfoVQA | ShiftProj | SynAI | SynEnergy | SynGov | SynHealth | Tabfquad | Tatdqa | BioMed | ESGHL | ESG | Econ |
| :---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 🥇 **1** | **EVIE-Preview-4.5B** | **85.77** | 90.73 | 64.53 | 93.26 | 93.85 | 99.63 | 98.26 | 98.89 | 98.89 | 97.32 | 81.93 | 70.17 | 79.84 | 64.95 | 68.53 |
| 🥈 2 | Ops-Colqwen3-4B | 84.90 | 91.80 | 66.50 | 94.00 | 90.80 | 99.60 | 97.30 | 98.00 | 99.60 | 93.60 | 82.40 | 65.50 | 78.60 | 66.00 | 64.50 |
| 🥉 3 | nemotron-colembed-vl-8b-v2 | 84.80 | 93.10 | 68.10 | 94.60 | 93.30 | 100.0 | 97.90 | 98.90 | 99.60 | 97.70 | 83.40 | 66.20 | 73.20 | 60.60 | 60.80 |
| 4 | nemotron-colembed-vl-4b-v2 | 83.90 | 92.00 | 67.40 | 93.30 | 92.30 | 99.30 | 96.20 | 98.00 | 98.50 | 98.10 | 81.20 | 64.30 | 71.40 | 61.50 | 60.80 |
| 5 | colqwen3.5-4.5B-v3 | 83.70 | 91.90 | 66.60 | 93.60 | 90.20 | 100.0 | 97.10 | 97.30 | 98.90 | 95.90 | 84.00 | 65.30 | 73.80 | 58.00 | 59.90 |
| 6 | llama-nemotron-colembed-vl-3b-v2 | 83.60 | 90.40 | 67.20 | 94.70 | 92.00 | 100.0 | 98.00 | 98.00 | 98.90 | 97.30 | 81.00 | 63.20 | 73.10 | 58.60 | 58.60 |
| 7 | tomoro-colqwen3-embed-8b | 83.50 | 91.20 | 66.40 | 94.50 | 87.90 | 99.30 | 96.70 | 97.60 | 99.10 | 94.20 | 80.90 | 65.50 | 76.00 | 60.70 | 59.50 |
| 8 | EvoQwen2.5-VL-Retriever-7B-v1 | 83.40 | 91.50 | 65.10 | 94.10 | 88.80 | 99.60 | 96.60 | 96.30 | 98.90 | 93.60 | 82.30 | 65.20 | 77.00 | 59.70 | 59.10 |
| 9 | tomoro-colqwen3-embed-4b | 83.20 | 90.60 | 66.30 | 94.30 | 87.40 | 99.30 | 96.90 | 97.20 | 99.60 | 94.30 | 79.90 | 65.40 | 74.60 | 62.40 | 56.30 |
| 10 | SauerkrautLM-ColQwen3-8b-v0.1 | 82.90 | 93.80 | 64.70 | 94.50 | 90.40 | 98.60 | 96.50 | 96.80 | 99.30 | 92.20 | 84.00 | 63.30 | 70.80 | 57.90 | 58.00 |

*Tasks 1–10: ViDoRe V1. Tasks 11–14: ViDoRe V2. Board aggregates: **V1 91.73** · **V2 70.87**.*

---

## 💾 Index Cost

Index size is what decides whether multi-vector retrieval actually ships. EVIE emits native **128D** token vectors, so the index stays compact at both page budgets.

| Raw BF16 index | 768 tokens/page | 1,792 tokens/page |
| :--- | ---: | ---: |
| 1M pages | **179.2 GiB** | 420.5 GiB |
| 10M pages | **1.8 TB** | 4.1 TB |

```text
1,763.58 vectors/page × 128 dim × 2 bytes × 1,000,000 pages ÷ 2^30 = 420.5 GiB
```

Scoring stays cheap for the same reason: MaxSim is a late-interaction dot product over the token vectors, so a narrower vector cuts the scoring work exactly as it cuts storage.

---

## 🧠 Architecture

```text
  Text Query ───────► ColQwen3_5 (BiDir Attn) ─────► Query Token Embeddings (128D)
                                                                 │
                                                       Late Interaction (MaxSim) ──► Relevance Score
                                                                 │
Document Image ─────► ColQwen3_5 (Dynamic Vision) ──► Doc Token Embeddings (128D)
```

1. **Vision-Language Backbone** — `Qwen3.5-4B` with interleaved GatedDeltaNet linear attention and full attention.
2. **Compact Projection** — contextual token states projected directly into native 128-dimensional representations.
3. **Late-Interaction Retrieval** — token-level MaxSim between query tokens and document visual tokens.

### 🌍 Multilingual

Queries in **English, French, German, Italian, Spanish, Portuguese and Chinese**, retrieving over charts, tables, scientific reports, financial filings and scanned forms — including Japanese-language pages.

### 📦 Model Footprint

| | Value |
| :--- | ---: |
| Parameters | 4.54B |
| Checkpoint (BF16) | 8.5 GB |
| Token embedding | 128D |
| Max visual tokens | 768 / 1,792 |

---

## ⚡ Quick Start

ColPali Engine is the reference path — every number on this card comes from it. A [Sentence Transformers](#sentence-transformers) path is also available for late-interaction pipelines already built on that API.

### Installation

```bash
pip install -r requirements.txt
```

### Python Inference

```python
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor

from bidirectional import enable_bidirectional_attention

model_id = "tencent/EVIE-Preview-4.5B"

# 1. Load model and enable bidirectional attention
model = ColQwen3_5.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    attn_implementation="flash_attention_2",
).eval()
enable_bidirectional_attention(model)

# 2. Load processor
processor = ColQwen3_5Processor.from_pretrained(model_id)

# 3. Prepare inputs — four example document pages
pages = [
    hf_hub_download("sentence-transformers/example-documents", f"doc{i}.jpg", repo_type="dataset")
    for i in range(1, 5)
]
images = [Image.open(p).convert("RGB") for p in pages]
queries = [
    "What is the variable represented on the y-axis of the graph?",
    "Total outlay is maximum in which year?",
]

image_batch = processor.process_images(images).to(model.device)
query_batch = processor.process_queries(queries).to(model.device)

# 4. Generate multi-vector embeddings and score
with torch.inference_mode():
    image_embeddings = model(**image_batch)
    model.rope_deltas = None  # required before query forward
    query_embeddings = model(**query_batch)

scores = processor.score(query_embeddings, image_embeddings)
print(scores)
# tensor([[17.3750, 10.9375,  7.8750,  7.3438],
#         [ 6.5938, 13.3750,  6.2188,  6.0938]])
print("Best page per query:", scores.argmax(dim=1))
# Best page per query: tensor([0, 1])
```

> ⚠️ Apply `enable_bidirectional_attention(model)` once after loading, and reset `model.rope_deltas = None` before every query forward pass. Both are required to reach the scores above. [`bidirectional.py`](bidirectional.py) ships with this repository because released `colpali-engine` builds ColQwen3.5 with causal masks; run the snippet from the repository root so the import resolves.

### CLI

```bash
python infer.py --query "Quarterly revenue report" --image page_1.png --image page_2.png
```

### Sentence Transformers

EVIE also loads as a [Sentence Transformers](https://www.sbert.net/) `MultiVectorEncoder`, exposing the familiar `encode_query` / `encode_document` / `similarity` API with MaxSim scoring built in. Bidirectional attention is baked into the shipped configuration, so no extra call is needed.

`MultiVectorEncoder` requires Sentence Transformers 6.0.0, which is not on PyPI yet — install from source until it is released:

```bash
pip install "sentence-transformers[image] @ git+https://github.com/huggingface/sentence-transformers.git"
```

```python
from sentence_transformers import MultiVectorEncoder

model = MultiVectorEncoder("tencent/EVIE-Preview-4.5B")

queries = [
    "What is the variable represented on the y-axis of the graph?",
    "Total outlay is maximum in which year?",
]
documents = [
    "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc1.jpg",
    "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc2.jpg",
    "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc3.jpg",
    "https://huggingface.co/datasets/sentence-transformers/example-documents/resolve/main/doc4.jpg",
]

query_embeddings = model.encode_query(queries)
document_embeddings = model.encode_document(documents)
print(query_embeddings[0].shape, document_embeddings[0].shape)
# torch.Size([23, 128]) torch.Size([755, 128])

scores = model.similarity(query_embeddings, document_embeddings)
print(scores)
# tensor([[17.3457, 10.8008,  7.8613,  7.3174],
#         [ 6.5547, 13.3828,  6.2207,  6.0771]])
print("Best page per query:", scores.argmax(dim=1))
# Best page per query: tensor([0, 1])
```

Both paths above run on the same four example pages, so they are directly comparable. The two sets of scores agree closely; the small differences come from the attention backend and dtype, and the ranking is identical.

Documents may be file paths, URLs or `PIL.Image` objects. Text passed to `encode_document` is rendered as a query, since this model has no separate text-document format.

The default page budget is the 768-token tier. To score the 1,792-token tier, raise the pixel budget through `processor_kwargs`:

```python
model = MultiVectorEncoder(
    "tencent/EVIE-Preview-4.5B",
    model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "cuda:0"},
    processor_kwargs={"size": {"longest_edge": 1792 * 32 * 32, "shortest_edge": 65536}},
)
```

---

## 🔬 Reproducing

Every number on this card is reproducible with the shipped script across all visible GPUs:

```bash
bash reproduce.sh
```

On the first run, [`download_data.py`](download_data.py) fetches the 22 public ViDoRe datasets (~55 GB) from Hugging Face. To reuse an existing directory:

```bash
bash reproduce.sh /path/to/vidore
```

### Target aggregates

```text
ViDoRe V1        nDCG@5    91.73   (10 tasks)
ViDoRe V2        nDCG@5    70.87   (4 tasks)
ViDoRe V1+V2     nDCG@5    85.77   (14 tasks)
ViDoRe V3 public nDCG@10   64.56   (8 domains x 6 languages, 768 visual tokens)
ViDoRe V3 public nDCG@10   65.36   (8 domains x 6 languages, 1792 visual tokens)
```

To score the 1,792-token tier directly:

```bash
python -m torch.distributed.run --nproc_per_node=$(nvidia-smi -L | wc -l) reproduce.py \
  --boards v3 --max-visual-tokens 1792 --data-root /path/to/vidore
```

---

## 🎓 Training Details

EVIE was trained on approximately **0.8 million high-quality image-query pairs** spanning multilingual documents, technical reports, complex financial tables, infographics and document visual QA.

### Hard Negative Mining & Evidence Judging

Every mined negative is re-judged by a large multimodal judge before it reaches the loss:

- 🟢 Candidates that actually answer the query are **promoted to positives**.
- 🟡 Partially relevant or ambiguous candidates are **masked out of the loss**.
- 🔴 Only strictly irrelevant pages survive as **true hard negatives**.

Multi-positive rows are group-aware weighted by `1/positive_count` so that positives from the same query never penalise each other in-batch. Rows with empty queries, corrupted images or degraded text are dropped.

---

## 🙏 Acknowledgements

- Built on the [ColPali Engine](https://github.com/illuin-tech/colpali) by Illuin Technology.
- Powered by the [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) vision-language backbone.
- Evaluated on the [ViDoRe Benchmark](https://huggingface.co/vidore) family.

---

## 📚 Citation

```bibtex
@misc{tencent2026evie,
  title        = {EVIE-Preview-4.5B: Rank-1 Multilingual Visual Document Retrieval with 128-Dimensional Multi-Vector Embeddings},
  author       = {{Tencent}},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/tencent/EVIE-Preview-4.5B}}
}
```

<div align="center">

**🏆 Rank #1 on ViDoRe V3 · 🏆 Rank #1 on ViDoRe V1+V2 · 💾 native 128D token vectors**

</div>
