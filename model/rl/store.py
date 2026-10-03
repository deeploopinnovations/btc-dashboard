"""
rl/store.py
=====================================================================
Where the agent keeps what it has learned and everything it did.

  state.json   the agent, the position, the decisions awaiting settlement
  log.jsonl    one record per settled step (append-only): inputs, action, R,
               utility, what FLAT/HALF/HOLD/VT earned on the same R, the
               utility of every action, regret -- the data to study losses with

LocalStore keeps them in a directory. HFStore mirrors that directory to a
Hugging Face dataset repo, because a Space's own disk is wiped on every
restart: on start it restores both files, after every change it uploads them.
An upload failure is reported, never raised -- the agent keeps trading on
paper and retries on the next change.
"""
from __future__ import annotations

import json
from pathlib import Path


class LocalStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.log_path = self.root / "log.jsonl"

    def load_state(self):
        return json.loads(self.state_path.read_text()) if self.state_path.exists() else None

    def save_state(self, st: dict) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(st))
        tmp.replace(self.state_path)

    def append_log(self, rec: dict) -> None:
        with self.log_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")

    def read_log(self) -> list:
        if not self.log_path.exists():
            return []
        return [json.loads(ln) for ln in self.log_path.read_text().splitlines() if ln.strip()]

    def flush(self) -> str:
        return "local"


class HFStore(LocalStore):
    """LocalStore mirrored to a dataset repo, e.g. <user>/noctua-rl-paper."""

    def __init__(self, root: Path, repo_id: str, token: str | None = None, api=None):
        super().__init__(root)
        from huggingface_hub import HfApi
        self.repo_id = repo_id
        self.api = api or HfApi(token=token)
        self.last_error = None

    def restore(self) -> str:
        from huggingface_hub import hf_hub_download
        got = []
        for name in ("state.json", "log.jsonl"):
            try:
                p = hf_hub_download(self.repo_id, name, repo_type="dataset",
                                    token=self.api.token, local_dir=self.root)
                got.append(Path(p).name)
            except Exception as e:                           # first run: nothing there yet
                self.last_error = f"restore {name}: {type(e).__name__}"
        return f"restored {got}" if got else "nothing to restore"

    def flush(self) -> str:
        try:
            self.api.create_repo(self.repo_id, repo_type="dataset", private=True, exist_ok=True)
            for p in (self.state_path, self.log_path):
                if p.exists():
                    self.api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name,
                                         repo_id=self.repo_id, repo_type="dataset",
                                         commit_message=f"paper agent: {p.name}")
            self.last_error = None
            return "uploaded"
        except Exception as e:                               # noqa: BLE001 -- keep trading on paper
            self.last_error = f"{type(e).__name__}: {e}"
            return "upload failed: " + self.last_error
