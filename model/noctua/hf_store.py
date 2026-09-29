"""
noctua/hf_store.py
=====================================================================
Store the large research inputs on Hugging Face, and restore them from there,
so a lost container never again means a lost corpus (DATA_LOSS_2026-09-12).

WHERE: a public dataset repo named noctua-btcusd-corpus under the account that
OWNS the token in HF_TOKEN (the owner uses a separate Hugging Face account for
this, deliberately not the one connected to this workspace). The repo id is
taken from the token at upload time (or HF_REPO, if set) and written, with the
upload's commit, to research/hf_store.json -- committed, so a restore knows
where to look without any token. The 1-minute corpus is derived from
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

Uploading needs TWO things from the environment settings (never pasted into
chat): a write token in the environment variable HF_TOKEN, and huggingface.co
allowed by the environment's network access (on 2026-09-29 the egress proxy
answered 403 to huggingface.co, so neither upload nor restore can reach the Hub
from this container until it is allowed).

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

REPO_NAME = "noctua-btcusd-corpus"
POINTER = Path("model/research/hf_store.json")
ARTIFACTS = Path("model/artifacts")
MANIFEST_PATH = Path("model/research") / MANIFEST

README = """---
license: mit
tags: [bitcoin, btcusd, volatility, time-series, finance]
pretty_name: NOCTUA BTC/USD 1-minute research corpus
---
# NOCTUA BTC/USD 1-minute research corpus (pinned)

The exact 1-minute BTC/USD corpus the NOCTUA volatility research was measured
on, stored so a lost container can restore it and verify it.

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


def reachable() -> None:
    """Fail with the actual remedy when the Hub is blocked, not a stack trace."""
    import urllib.error
    import urllib.request
    try:
        urllib.request.urlopen("https://huggingface.co/", timeout=20)
    except urllib.error.HTTPError:
        return                                  # an HTTP answer means the Hub is reachable
    except (urllib.error.URLError, OSError) as e:
        raise SystemExit(
            f"REFUSING: huggingface.co is not reachable from here ({e}). If the "
            f"environment's network policy denies it (a 403 from the proxy), allow "
            f"huggingface.co in the environment's network settings; do not retry.")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="store/restore the corpus on HF")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--upload", action="store_true")
    g.add_argument("--restore", action="store_true")
    a = ap.parse_args(argv)
    man = json.loads(MANIFEST_PATH.read_text())

    if a.restore:
        repo, rev, files = os.environ.get("HF_REPO"), None, (RESEARCH,)
        if repo is None:
            if not POINTER.exists():
                raise SystemExit(f"REFUSING: no {POINTER} (nothing has been uploaded yet) "
                                 f"and no HF_REPO set -- nowhere to restore from.")
            ptr = json.loads(POINTER.read_text())
            repo, rev, files = ptr["repo"], ptr.get("revision"), tuple(ptr["files"])
        reachable()
        from huggingface_hub import hf_hub_download
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        for f in files:
            p = hf_hub_download(repo, f, repo_type="dataset", revision=rev,
                                local_dir=ARTIFACTS)
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
    reachable()
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    repo = os.environ.get("HF_REPO") or f"{api.whoami()['name']}/{REPO_NAME}"
    api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
    readme = README.format(rows=man["corpus_rows"], end=man["corpus_end_utc"], sha=digest)
    api.upload_file(path_or_fileobj=readme.encode(), path_in_repo="README.md",
                    repo_id=repo, repo_type="dataset", commit_message="README")
    api.upload_file(path_or_fileobj=str(MANIFEST_PATH), path_in_repo=MANIFEST,
                    repo_id=repo, repo_type="dataset", commit_message="corpus manifest")
    last = None
    for f in (RESEARCH, UNSEEN):
        p = ARTIFACTS / f
        if p.exists():
            last = api.upload_file(path_or_fileobj=str(p), path_in_repo=f, repo_id=repo,
                                   repo_type="dataset",
                                   commit_message=f"{f} (hash-verified against the committed manifest)")
            print(f"uploaded {f} ({p.stat().st_size / 1e6:.1f} MB)")
    rev = getattr(last, "oid", None)
    POINTER.write_text(json.dumps({
        "repo": repo, "repo_type": "dataset", "revision": rev,
        "content_sha256": digest, "corpus_rows": man["corpus_rows"],
        "files": [f for f in (RESEARCH, UNSEEN) if (ARTIFACTS / f).exists()]}, indent=2) + "\n")
    print(f"done: https://huggingface.co/datasets/{repo}  (revision {rev})")
    print(f"wrote {POINTER} -- commit it so a restore needs no token")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
