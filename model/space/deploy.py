"""
space/deploy.py
=====================================================================
Create (or update) the paper-agent Space and its log dataset on the account
that owns HF_TOKEN -- never any other: it refuses unless that account is the
one recorded in research/hf_store.json (the owner's separate HF account).

  <user>/noctua-paper-agent   Space, gradio, free cpu-basic, private
  <user>/noctua-rl-paper      dataset, private: state.json + log.jsonl

The Space gets HF_TOKEN as a SECRET (so it can write its own dataset) and
GIT_REPO / GIT_REF / RL_DATASET as variables. With --warm-start, and only if
the dataset has no state yet, the agent starts from what MV learned over the
registered replay (eval/rl_replay.py's cached inputs), flat, with no pending
decisions; otherwise it starts from nothing.

    python -m model.space.deploy --ref claude/btc-volatility-model-1xepb9 --warm-start
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

HERE = Path(__file__).resolve().parent
POINTER = HERE.parents[0] / "research" / "hf_store.json"
SPACE_NAME, DATASET_NAME = "noctua-paper-agent", "noctua-rl-paper"


def warm_state() -> dict:
    """MV after the registered replay. The replay's inputs are committed
    (model/artifacts/rl_inputs.parquet, ~200 KB); if they are missing they are
    rebuilt from the corpus, and without either this stops with the remedy
    instead of a FileNotFoundError (Codex review, PR #14)."""
    import numpy as np
    from eval.rl_replay import ART, CACHE, build_inputs, xmat
    from rl.agent import make_agent
    if not CACHE.exists():
        if not (ART / "btcusd_1h.parquet").exists():
            raise SystemExit(f"--warm-start needs {CACHE} (committed) or the corpus to rebuild it: "
                             "python -m model.noctua.hf_store --restore, then python -m model.eval.rl_replay")
        build_inputs()
    df = __import__("pandas").read_parquet(CACHE)
    df = df[np.isfinite(df.R) & np.isfinite(df.sigma)].reset_index(drop=True)
    X, S, R = xmat(df), df.sigma.to_numpy(), df.R.to_numpy()
    ag, w = make_agent("MV"), 0.0
    for i in range(len(R)):
        a = ag.act(X[i], S[i], w)
        ag.update(X[i], S[i], w, a, R[i])
        w = a
    return {"agent": ag.state(), "w": 0.0, "pending": [], "last_slot": None, "last_settled": None,
            "base_w": {k: 0.0 for k in ("FLAT", "HALF", "HOLD", "VT")},
            "warm_start": {"source": "P5-rl-paper replay", "steps": int(len(R)),
                           "through": int(df.E.iloc[-1])}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="deploy the NOCTUA paper-agent Space")
    ap.add_argument("--ref", default="main", help="git branch the Space runs")
    ap.add_argument("--warm-start", action="store_true")
    ap.add_argument("--public", action="store_true", help="make the Space public (default private)")
    a = ap.parse_args(argv)
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set")
    api = HfApi(token=token)
    user = api.whoami()["name"]
    owner = json.loads(POINTER.read_text())["repo"].split("/")[0]
    if user != owner:
        raise SystemExit(f"REFUSING: HF_TOKEN belongs to {user!r}, the project's account is {owner!r}")
    space, dataset = f"{user}/{SPACE_NAME}", f"{user}/{DATASET_NAME}"

    api.create_repo(dataset, repo_type="dataset", private=True, exist_ok=True)
    files = {f.rfilename for f in api.dataset_info(dataset).siblings or []}
    if a.warm_start and "state.json" not in files:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "state.json"
            st = warm_state()
            p.write_text(json.dumps(st))
            api.upload_file(path_or_fileobj=str(p), path_in_repo="state.json", repo_id=dataset,
                            repo_type="dataset", commit_message="warm start from the P5-rl-paper replay")
        print(f"[deploy] warm state uploaded: MV after {st['warm_start']['steps']} replay steps")
    else:
        print(f"[deploy] dataset {dataset}: existing files {sorted(files)} (state kept)")

    api.create_repo(space, repo_type="space", space_sdk="gradio", private=not a.public, exist_ok=True)
    api.add_space_secret(space, "HF_TOKEN", token)
    for k, v in (("GIT_REPO", "https://github.com/deeploopinnovations/btc-dashboard"),
                 ("GIT_REF", a.ref), ("RL_DATASET", dataset)):
        api.add_space_variable(space, k, v)
    api.upload_folder(folder_path=str(HERE / "hf"), repo_id=space, repo_type="space",
                      commit_message=f"bootstrap: run btc-dashboard@{a.ref}")
    print(f"[deploy] https://huggingface.co/spaces/{space}  (GIT_REF={a.ref}, dataset {dataset})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
