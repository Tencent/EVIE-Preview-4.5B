#!/usr/bin/env python3
"""Download the public ViDoRe V1, V2 and V3 datasets used by reproduce.py."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

from reproduce import V1, V2, V3

REPOS = ["vidore/" + n for n in V1 + V2] + ["vidore/vidore_v3_" + d for d in V3]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "vidore"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    for index, repo in enumerate(REPOS, 1):
        target = out / repo.split("/")[1]
        print("[%2d/%d] %s" % (index, len(REPOS), repo), flush=True)
        snapshot_download(
            repo,
            repo_type="dataset",
            local_dir=str(target),
            allow_patterns=["*.parquet"],
            max_workers=args.workers,
        )
    print("\n%d datasets under %s" % (len(REPOS), out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
