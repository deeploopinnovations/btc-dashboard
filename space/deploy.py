"""
space/deploy.py -- build the Space from this repo and push it to Hugging Face.

    HF_TOKEN=... python space/deploy.py                  # static Space (free)
    HF_TOKEN=... python space/deploy.py --mode gradio    # CPU Space (needs HF PRO)

Both create the Space PRIVATE. The token is read from the environment and never
written to disk.

static  The free option. The page (space/static/index.html) reads the forward
        log that .github/workflows/noctua-forward.yml commits each day, so the
        model runs on GitHub Actions and the Space only displays it. Since
        2026, Hugging Face hosts Gradio and Docker Spaces on cpu-basic only
        for PRO accounts; static Spaces stay free.
gradio  The model runs inside the Space (space/app.py, 30-minute loop). Also
        creates a PRIVATE dataset repo for the log and stores HF_TOKEN as a
        Space secret and LOG_REPO as a Space variable. Uploads only what
        inference needs: the NOCTUA package, the pinned forecaster
        (serve/noctua_v2.npz, never the refreshed one), the v1 policy weights,
        and the committed hourly/DVOL/funding history.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SDK_VERSION = "6.30.0"

FILES = [
    "space/app.py", "space/forward.py",
    "model/serve/runtime.py", "model/serve/fetch.py", "model/serve/history.py",
    "model/serve/noctua_v2.npz",
    "model/policy/__init__.py", "model/policy/backtest.py", "model/policy/dataset.py",
    "model/policy/paper.py", "model/policy/runtime.py",
    "model/policy/weights/noctua_trader_v1.npz",
    "model/eval/harvest_newdata.py",
    "data/assets/btc_history.parquet", "data/noctua_history.parquet",
    "data/newdata/dvol_btc.parquet", "data/newdata/funding_btc.parquet",
]
DIRS = ["model/noctua"]

REQUIREMENTS = "numpy>=1.26\nscipy>=1.11\npandas>=2.0\npyarrow>=14.0\nhuggingface_hub>=0.30\n"


STATIC_README = """---
title: NOCTUA forward holdout
emoji: 🦉
colorFrom: indigo
colorTo: gray
sdk: static
pinned: false
---

# NOCTUA-Trader forward holdout

Shows the frozen NOCTUA-Trader v1 policy's paper decisions from 2026-10-10 on.
The model runs daily on GitHub Actions in
github.com/deeploopinnovations/btc-dashboard
(`.github/workflows/noctua-forward.yml`) and commits
`data/forward/forward_log.json`; this page reads that file. Paper trading
only. No orders, no exchange keys.
"""


def readme(log_repo: str) -> str:
    return f"""---
title: NOCTUA forward holdout
emoji: 🦉
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: {SDK_VERSION}
python_version: "3.12"
app_file: space/app.py
pinned: false
---

# NOCTUA-Trader forward holdout

Runs the frozen NOCTUA-Trader v1 policy (open NumPy weights) on top of the
pinned NOCTUA volatility forecaster, on the free CPU tier. It decides a BTC
position at 17:00 UTC each day from 2026-10-10 on, holds it 24 hours, and
logs every decision and its result to the private dataset `{log_repo}`.

Paper trading only. No orders, no exchange keys. Built from
github.com/deeploopinnovations/btc-dashboard (`space/`); redeploy with
`python space/deploy.py`.
"""


def stage(dst: Path, log_repo: str) -> None:
    for f in FILES:
        (dst / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, dst / f)
    for d in DIRS:
        shutil.copytree(ROOT / d, dst / d, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (dst / "requirements.txt").write_text(REQUIREMENTS)
    (dst / "README.md").write_text(readme(log_repo))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", default="noctua-forward")
    ap.add_argument("--log", default="noctua-forward-log")
    ap.add_argument("--mode", choices=["static", "gradio"], default="static")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    token = os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("HF_TOKEN is not set")
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    user = api.whoami()["name"]
    space_id, log_id = f"{user}/{a.space}", f"{user}/{a.log}"

    if a.mode == "static":
        with tempfile.TemporaryDirectory() as tmp:
            shutil.copy2(ROOT / "space/static/index.html", Path(tmp) / "index.html")
            (Path(tmp) / "README.md").write_text(STATIC_README)
            if a.dry_run:
                return 0
            api.create_repo(space_id, repo_type="space", space_sdk="static", private=True, exist_ok=True)
            api.upload_folder(repo_id=space_id, repo_type="space", folder_path=tmp,
                              commit_message="Deploy NOCTUA forward holdout page from btc-dashboard")
        print(f"https://huggingface.co/spaces/{space_id}")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        stage(Path(tmp), log_id)
        size = sum(p.stat().st_size for p in Path(tmp).rglob("*") if p.is_file())
        print(f"staged {size / 1e6:.1f} MB for {space_id}")
        if a.dry_run:
            return 0
        api.create_repo(log_id, repo_type="dataset", private=True, exist_ok=True)
        api.create_repo(space_id, repo_type="space", space_sdk="gradio", private=True, exist_ok=True)
        api.add_space_secret(space_id, "HF_TOKEN", token)
        api.add_space_variable(space_id, "LOG_REPO", log_id)
        api.upload_folder(repo_id=space_id, repo_type="space", folder_path=tmp,
                          commit_message="Deploy NOCTUA forward holdout from btc-dashboard")
    print(f"https://huggingface.co/spaces/{space_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
