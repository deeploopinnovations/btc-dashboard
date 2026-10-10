"""
Hugging Face Space bootstrap for the NOCTUA paper-trading agent.

Clones github.com/deeploopinnovations/btc-dashboard at GIT_REF (sparse: only
the serving code, the agent and the history bundle), then runs
model/space/run.py. A restart therefore always runs the latest code on GIT_REF.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = os.environ.get("GIT_REPO", "https://github.com/deeploopinnovations/btc-dashboard")
REF = os.environ.get("GIT_REF", "main")
DEST = Path(os.environ.get("CLONE_DIR", "/tmp/btc-dashboard"))
PATHS = ["/model/serve/", "/model/noctua/", "/model/rl/", "/model/space/",
         "/data/noctua_history.parquet"]


def git(*args):
    subprocess.run(["git", *args], check=True)


if DEST.exists():
    shutil.rmtree(DEST)
git("clone", "--depth", "1", "--branch", REF, "--filter=blob:none", "--no-checkout", REPO, str(DEST))
git("-C", str(DEST), "sparse-checkout", "set", "--no-cone", *PATHS)
git("-C", str(DEST), "checkout", REF)
print(f"[space] cloned {REPO}@{REF}: "
      f"{subprocess.run(['git', '-C', str(DEST), 'log', '-1', '--format=%h %s'], capture_output=True, text=True).stdout.strip()}",
      flush=True)

sys.path.insert(0, str(DEST / "model"))
from space import run  # noqa: E402

run.main()
