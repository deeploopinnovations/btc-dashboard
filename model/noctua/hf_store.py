"""
noctua/hf_store.py
=====================================================================
Store the large research inputs on Hugging Face, and restore them from there,
so a lost container never again means a lost corpus (DATA_LOSS_2026-09-12).

WHERE: the public dataset repo Deeploopinnovations/noctua-btcusd-corpus
(created 2026-09-29). The 1-minute corpus is derived from
ff137/bitstamp-btcusd-minute-data, MIT-licensed; it is redistributed under the
same licence with attribution (the README says so).

WHAT IS UPLOADED, AND THE REFUSAL THAT GUARDS IT

    btcusd_1min.parquet                                     the research corpus
    btcusd_1min_AFTER_CORPUS_END_do_not_use_in_research.parquet  (if present)
    corpus_manifest.json, README.md

Before anything is uploaded the local corpus is hashed with
regenerate.content_sha256 (column data, not file bytes) and compared with the
COMMITTED manifest. A mismatch refuses: the store must hold the corpus the
committed results were measured on, not whatever happens to be on disk.
Restoring verifies the same hash after download.

Uploading needs a write token in the environment variable HF_TOKEN (the
owner adds it in the environment settings; it is never pasted into chat).

    python -m model.noctua.hf_store --check            # hash only, no network
    python -m model.noctua.hf_store --upload           # needs HF_TOKEN
    python -m model.noctua.hf_store --restore          # public, no token needed
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from noctua.regenerate import MANIFEST, RESEARCH, UNSEEN, content_sha256  # noqa: E402

REPO = "Deeploopinnovations/noctua-btcusd-corpus"
ARTIFACTS = Path("model/artifacts")
MANIFEST_PATH = Path("model/research") / MANIFEST

README = """---
license: mit
tags: [bitcoin, btcusd, volatility, time-series, finance]
pretty_name: NOCTUA BTC/USD 1-minute research corpus
---
# NOCTUA BTC/USD 1-minute research corpus (pinned)

The exact 1-minute BTC/USD corpus the NOCTUA research in
[deeploopinnovations/btc-dashboard](https://github.com/deeploopinnovations/btc-dashboard)
was measured on, stored so a lost container can restore it and verify it.

* `btcusd_1min.parquet` -- {rows:,} minutes ending {end}; content SHA-256 of the
  column data (`model/noctua/regenerate.py`, `content_sha256`): `{sha}`
* `btcusd_1min_AFTER_CORPUS_END_do_not_use_in_research.parquet` -- source minutes after
  the pinned end, kept separate: part of them lie inside a frozen forward holdout.
* `corpus_manifest.json` -- the pin and its fidelity statistics.

Uploaded by `model/noctua/hf_store.py`, which refuses unless the local corpus matches the
committed hash. Restore with `python -m model.noctua.hf_store --restore`.

**Source and licence.** Derived from
[ff137/bitstamp-btcusd-minute-data](https://github.com/ff137/bitstamp-btcusd-minute-data)
(MIT licence), Bitstamp BTC/USD 1-minute OHLCV; redistributed under the same MIT licence
with attribution. Educational research only; not financial advice.
"""


def verify(path: Path, man: dict) -> str:
    d = pd.read_parquet(path)
    digest = content_sha256(d)
    if digest != man["content_sha256"] or len(d) != man["corpus_rows"]:
        raise SystemExit(
            f"REFUSING: {path} hashes to {digest} ({len(d):,} rows); the committed "
            f"manifest says {man['content_sha256']} ({man['corpus_rows']:,}). This is "
            f"not the corpus the committed results were measured on.")
    return digest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="store/restore the corpus on HF")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--upload", action="store_true")
    g.add_argument("--restore", action="store_true")
    a = ap.parse_args(argv)
    man = json.loads(MANIFEST_PATH.read_text())

    if a.restore:
        from huggingface_hub import hf_hub_download
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        for f in (RESEARCH, UNSEEN):
            p = hf_hub_download(REPO, f, repo_type="dataset", local_dir=ARTIFACTS)
            print(f"restored {p}")
        print(f"verified sha256 {verify(ARTIFACTS / RESEARCH, man)}")
        return 0

    digest = verify(ARTIFACTS / RESEARCH, man)
    print(f"local corpus matches the committed manifest: {digest}")
    if a.check:
        return 0
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("REFUSING: HF_TOKEN is not set. The owner adds a write token "
                         "as an environment variable in the environment settings.")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    readme = README.format(rows=man["corpus_rows"], end=man["corpus_end_utc"], sha=digest)
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                    repo_id=REPO, repo_type="dataset", commit_message="README")
    api.upload_file(path_or_fileobj=str(MANIFEST_PATH), path_in_repo=MANIFEST,
                    repo_id=REPO, repo_type="dataset", commit_message="corpus manifest")
    for f in (RESEARCH, UNSEEN):
        p = ARTIFACTS / f
        if p.exists():
            api.upload_file(path_or_fileobj=str(p), path_in_repo=f, repo_id=REPO,
                            repo_type="dataset",
                            commit_message=f"{f} (hash-verified against the committed manifest)")
            print(f"uploaded {f} ({p.stat().st_size / 1e6:.1f} MB)")
    print(f"done: https://huggingface.co/datasets/{REPO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
