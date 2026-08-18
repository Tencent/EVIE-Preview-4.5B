#!/usr/bin/env python3
"""Reproduce the ViDoRe V1 / V2 / V3 numbers reported in the model card.

Reads the datasets as published on the Hub, one directory per repository under
--data-root, as produced by download_data.py.

V1 follows the official QA protocol: every page of a dataset is a candidate and
queries are deduplicated. V2 and V3 use the released qrels, including graded
relevance. V3 averages the six query languages within each domain; a domain has
one corpus, so it is encoded once and reused across its languages.
"""

import argparse
import glob
import io
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

from colpali_engine.models import ColQwen3_5, ColQwen3_5Processor
from colpali_engine.utils.maxsim import maxsim_inbatch

from bidirectional import enable_bidirectional_attention

V1 = [
    "arxivqa_test_subsampled",
    "docvqa_test_subsampled",
    "infovqa_test_subsampled",
    "shiftproject_test",
    "syntheticDocQA_artificial_intelligence_test",
    "syntheticDocQA_energy_test",
    "syntheticDocQA_government_reports_test",
    "syntheticDocQA_healthcare_industry_test",
    "tabfquad_test_subsampled",
    "tatdqa_test",
]
V2 = ["biomedical_lectures_v2", "economics_reports_v2", "esg_reports_v2",
      "esg_reports_human_labeled_v2"]
V3 = ["computer_science", "energy", "finance_en", "finance_fr", "hr", "industrial",
      "pharmaceuticals", "physics"]
V3_LANGS = ["english", "french", "german", "italian", "portuguese", "spanish"]
META = ("image_filename", "query", "corpus-id", "corpus_id", "id")
CORPUS_ID = ("corpus-id", "corpus_id", "id")
QUERY_ID = ("query-id", "query_id", "id")


def parquets(directory: Path) -> list[str]:
    found = sorted(glob.glob(str(directory / "*.parquet")))
    test = [f for f in found if Path(f).name.startswith("test-")]
    return test or found


def dataset_dir(root: Path, board: str, name: str) -> Path:
    """Datasets live at <root>/<repo name>; a <root>/<board>/<short name> tree also works."""
    short = name.removesuffix("_v2").removeprefix("vidore_v3_")
    for candidate in (root / name, root / board / short, root / board / name):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("%s: download it into %s" % (name, root))


def text_rows(paths):
    for path in paths:
        handle = pq.ParquetFile(path)
        columns = [c for c in handle.schema_arrow.names if c != "image"]
        for group in range(handle.num_row_groups):
            yield from handle.read_row_group(group, columns=columns).to_pylist()


def to_pil(value):
    if isinstance(value, dict):
        value = value["bytes"]
    return Image.open(io.BytesIO(value)).convert("RGB")


def pad_cat(chunks):
    width = max(c.shape[1] for c in chunks)
    return torch.cat([F.pad(c, (0, 0, 0, width - c.shape[1])) for c in chunks], 0)


@torch.inference_mode()
def embed_images(model, processor, paths, batch: int):
    out, metadata, pending = [], [], []

    def flush():
        if pending:
            out.append(model(**processor.process_images(pending).to(model.device)))
            pending.clear()

    for path in paths:
        handle = pq.ParquetFile(path)
        names = handle.schema_arrow.names
        columns = ["image"] + [c for c in META if c in names]
        for group in range(handle.num_row_groups):
            table = handle.read_row_group(group, columns=columns)
            images = table.column("image")
            others = {c: table.column(c) for c in columns[1:]}
            for j in range(table.num_rows):
                metadata.append({c: others[c][j].as_py() for c in others})
                pending.append(to_pil(images[j].as_py()))
                if len(pending) == batch:
                    flush()
    flush()
    return pad_cat(out), metadata


@torch.inference_mode()
def embed_queries(model, processor, texts, batch: int):
    out = []
    for start in range(0, len(texts), batch):
        chunk = [t if t.strip() else " " for t in texts[start : start + batch]]
        model.rope_deltas = None
        out.append(model(**processor.process_queries(chunk).to(model.device)))
    return pad_cat(out)


def maxsim(queries, docs, chunk_q=64, chunk_d=256):
    scores = torch.zeros(queries.shape[0], docs.shape[0], device=queries.device)
    for qi in range(0, queries.shape[0], chunk_q):
        for di in range(0, docs.shape[0], chunk_d):
            scores[qi : qi + chunk_q, di : di + chunk_d] = maxsim_inbatch(
                queries[qi : qi + chunk_q].contiguous(), docs[di : di + chunk_d].contiguous()
            )
    return scores


def ndcg(order, gains: dict, k: int) -> float:
    dcg = sum(
        (2 ** gains[doc] - 1) / math.log2(rank + 2)
        for rank, doc in enumerate(order[:k])
        if gains.get(doc, 0.0) > 0
    )
    ideal = sorted((2 ** g - 1 for g in gains.values() if g > 0), reverse=True)[:k]
    best = sum(g / math.log2(rank + 2) for rank, g in enumerate(ideal))
    return dcg / best if best else 0.0


def ident(record, keys) -> str:
    for key in keys:
        if record.get(key) is not None:
            return str(record[key])
    raise KeyError(keys)


def text_of(record) -> str:
    value = record.get("text", record.get("query"))
    value = "" if value is None else str(value).strip()
    return "" if value.lower() == "none" else value


def eval_qa(model, processor, directory: Path, batch: int, k: int) -> float:
    """Official ViDoRe QA protocol: full-page corpus, deduplicated queries."""
    docs, metadata = embed_images(model, processor, parquets(directory / "data"), batch)
    names = [str(m.get("image_filename") or i) for i, m in enumerate(metadata)]

    page_of, column_page = {}, []
    for name in names:
        column_page.append(page_of.setdefault(name, len(page_of)))
    gold_of = {text_of(m): names[i] for i, m in enumerate(metadata) if text_of(m)}

    queries = list(dict.fromkeys(text_of(m) for m in metadata if text_of(m)))
    scores = maxsim(embed_queries(model, processor, queries, batch), docs)

    groups = torch.tensor(column_page, device=scores.device).expand(scores.shape[0], -1)
    merged = torch.full((scores.shape[0], len(page_of)), float("-inf"), device=scores.device)
    merged.scatter_reduce_(1, groups, scores, reduce="amax")
    ranking = merged.argsort(dim=1, descending=True).tolist()

    return sum(
        ndcg(ranking[i], {page_of[gold_of[q]]: 1.0}, k) for i, q in enumerate(queries)
    ) / len(queries)


def eval_beir(model, processor, corpus: Path, queries: Path, qrels: Path, batch: int, k: int,
              cache: dict, language: str = "") -> float:
    if str(corpus) not in cache:
        cache.clear()
        docs, metadata = embed_images(model, processor, parquets(corpus), batch)
        index = {ident(m, CORPUS_ID): i for i, m in enumerate(metadata)}
        cache[str(corpus)] = (docs, index)
    docs, index = cache[str(corpus)]

    gains = defaultdict(dict)
    for record in text_rows(parquets(qrels)):
        doc = index.get(ident(record, CORPUS_ID))
        relevance = float(record.get("score") or 0.0)
        if doc is not None and relevance > 0:
            gains[ident(record, QUERY_ID)][doc] = relevance

    texts, ids = [], []
    for record in text_rows(parquets(queries)):
        if language and record.get("language") != language:
            continue
        key = ident(record, QUERY_ID)
        if text_of(record) and gains.get(key):
            texts.append(text_of(record))
            ids.append(key)

    scores = maxsim(embed_queries(model, processor, texts, batch), docs)
    ranking = scores.argsort(dim=1, descending=True).tolist()
    return sum(ndcg(ranking[i], gains[key], k) for i, key in enumerate(ids)) / len(ids)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve().parent
    ap.add_argument("--model", default=str(here))
    ap.add_argument("--data-root", default=str(here / "vidore"))
    ap.add_argument("--max-visual-tokens", type=int, default=768)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--attn", default="flash_attention_2")
    ap.add_argument("--boards", default="v1,v2,v3", help="subset of v1,v2,v3 to run")
    ap.add_argument("--output", default="")
    args = ap.parse_args()

    boards = {b.strip().lower() for b in args.boards.split(",") if b.strip()}
    unknown = boards - {"v1", "v2", "v3"}
    if unknown:
        ap.error("unknown board(s): %s" % sorted(unknown))

    rank, world = 0, 1
    if os.environ.get("WORLD_SIZE"):
        dist.init_process_group("nccl")
        rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    def mine(items):
        """Shard whole tasks across ranks; V3 shards by domain so each corpus is encoded once."""
        return [x for i, x in enumerate(items) if i % world == rank]

    root = Path(args.data_root)
    processor = ColQwen3_5Processor.from_pretrained(
        args.model, max_num_visual_tokens=args.max_visual_tokens
    )
    model = ColQwen3_5.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation=args.attn
    )
    enable_bidirectional_attention(model)
    model = model.to("cuda").eval()

    result, cache = {}, {}
    for task in mine(V1) if "v1" in boards else []:
        result["V1/" + task] = eval_qa(model, processor, dataset_dir(root, "eval", task), args.batch, 5)
        print("V1  %-46s nDCG@5  %6.2f" % (task, 100 * result["V1/" + task]), flush=True)

    for task in mine(V2) if "v2" in boards else []:
        base = dataset_dir(root, "eval_v2", task)
        result["V2/" + task] = eval_beir(
            model, processor, base / "corpus", base / "queries", base / "qrels", args.batch, 5, cache
        )
        print("V2  %-46s nDCG@5  %6.2f" % (task, 100 * result["V2/" + task]), flush=True)

    for domain in mine(V3) if "v3" in boards else []:
        base = dataset_dir(root, "eval_v3", "vidore_v3_" + domain)
        legacy = (base / (V3_LANGS[0] + "-corpus")).is_dir()
        for language in V3_LANGS:
            corpus = base / (V3_LANGS[0] + "-corpus") if legacy else base / "corpus"
            queries = base / (language + "-queries") if legacy else base / "queries"
            qrels = base / (language + "-qrels") if legacy else base / "qrels"
            result["V3/%s/%s" % (domain, language)] = eval_beir(
                model, processor, corpus, queries, qrels, args.batch, 10, cache,
                language="" if legacy else language,
            )
        scores = [result["V3/%s/%s" % (domain, l)] for l in V3_LANGS]
        print("V3  %-46s nDCG@10 %6.2f" % (domain, 100 * sum(scores) / len(scores)), flush=True)

    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, result)
        result = {k: v for part in gathered for k, v in part.items()}

    def mean(prefix):
        picked = [v for k, v in result.items() if k.startswith(prefix)]
        return sum(picked) / len(picked) if picked else None

    v1, v2, v3 = mean("V1/"), mean("V2/"), mean("V3/")
    combined = (10 * v1 + 4 * v2) / 14 if None not in (v1, v2) else None
    if rank != 0:
        dist.destroy_process_group()
        return 0

    print()
    for line, value in (
        ("ViDoRe V1        nDCG@5   %6.2f  (10 tasks)", v1),
        ("ViDoRe V2        nDCG@5   %6.2f  (4 tasks)", v2),
        ("ViDoRe V1+V2     nDCG@5   %6.2f  (14 tasks)", combined),
        ("ViDoRe V3 public nDCG@10  %6.2f  (8 domains x 6 languages)", v3),
    ):
        if value is not None:
            print(line % (100 * value))

    if args.output:
        Path(args.output).write_text(json.dumps(
            {"model": args.model, "max_visual_tokens": args.max_visual_tokens,
             "per_task": result,
             "average": {"v1_ndcg@5": v1, "v2_ndcg@5": v2, "v1v2_ndcg@5": combined,
                         "v3_public_ndcg@10": v3}},
            ensure_ascii=False, indent=2) + "\n")
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
