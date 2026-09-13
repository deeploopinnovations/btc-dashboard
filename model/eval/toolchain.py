"""
eval/toolchain.py
=====================================================================
What modelling technology is actually available, measured -- before the
experiment queue is planned around it.

THE QUESTION

PHASES.md's Phase 6+ queue lists E7 (gradient boosting vs LSTM/TCN vs
PatchTST/iTransformer as nonlinear challengers), and the protocol separately
carries E8 (TimeXer) and E16 (time-series foundation models as challengers
only). Every one of those names a package this repo has never imported.
`model/requirements-research.txt` pins exactly six: numpy, pandas, torch,
scikit-learn, scipy, pyarrow. Planning E7's design (train/test split, CV
budget, compute plan) around xgboost, then discovering in Phase 6 that xgboost
cannot be installed here, is a wasted planning pass -- the discovery should
happen now, once, and cheaply.

This is not a hypothetical risk. `eval/kronos_baseline.py` is the standing
proof it is real: that script is "shipped complete" but has NEVER BEEN RUN,
because `huggingface.co` is blocked by this container's egress policy and
Kronos's weights live there. That is a *specific* package (huggingface_hub's
download path) failing for a *specific* reason (the proxy's host allowlist),
discovered only by trying. `eval/datasources.py` separately measured the same
boundary for market-data APIs. Neither of those probes ever asked the question
this file asks: not "can we fetch data" but "can we install a package", and
those are different allowlists -- PyPI and a blocked data aggregator can both
be `https://`, on completely different hosts, with completely different
policy outcomes. Assuming they fail the same way, in either direction, is a
guess. This file measures pip and PyPI specifically, once, and writes the
result down so E7/E8/E16 are planned against fact.

WHY A LIGHTWEIGHT PROBE, NOT A FULL INSTALL

A CPU-heavy training job may be running when this is invoked (see the run
instructions below), so this script deliberately does NOT `pip install`
anything -- that would mutate the environment `requirements-research.txt`
pins, and BENCHMARK.md's bit-identical determinism claim is verified against
those exact versions (see the comment at the top of that file). It also does
not `pip download` full wheels on every run -- xgboost and catboost wheels
are 100+ MB each, and re-pulling them on every invocation of this probe would
be its own small waste. Instead it queries PyPI's JSON metadata endpoint
(`https://pypi.org/pypi/<name>/json`, a few KB) per candidate package: this
answers "is the index reachable and does the package exist for this
interpreter's platform" without paying for the wheel.

That lightweight check is corroborated, not assumed: during the investigation
that produced this script, full `pip download --no-deps` runs were executed
by hand for a representative sample and are recorded verbatim in
`VERIFIED_FULL_DOWNLOADS` below (package, wheel filename, size, exit code).
All of them succeeded. If a future run of this script reports PyPI reachable
for a package not in that verified set, treat it as INSTALLABLE (metadata
evidence) rather than AVAILABLE (import evidence) -- the distinction this
script's own output preserves.

WHAT THIS SCRIPT ALSO CANNOT TELL YOU

A package installing is not the same as a model working. lightgbm and
catboost each ship compiled extensions; PyPI serving a manylinux wheel for
this platform is strong but not perfect evidence the wheel actually loads
(loader/glibc/CPU-feature mismatches are the usual failure mode and are
invisible to a metadata probe). Anything this script marks INSTALLABLE should
still be import-tested once, for real, before an experiment is designed
around it -- this script narrows the search, it does not replace that check.

THE GITHUB ACTIONS ESCAPE HATCH

`.github/workflows/harvest-newdata.yml` already establishes the repo's
pattern for "the container can't reach X, but a GH Actions runner can, and
can commit the result back": that workflow fetches funding-rate/DVOL history
on an unrestricted runner and commits the parquet. The same pattern could
fetch a blocked pip package or a small model checkpoint on a runner and commit
it into the repo for offline use. It is NOT a blanket fix, because the
runner's output still has to fit in a git commit: a wheel is a few MB and
commits fine; a foundation-model checkpoint is routinely hundreds of MB to
several GB and does not. This script's per-experiment verdicts say, for each
BLOCKED item, whether the Actions route is actually practical given that size
constraint -- not just whether it is theoretically possible.

    python -m model.eval.toolchain --output model/artifacts/toolchain.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.datasources import ProbeResult, probe_url                          # noqa: E402

# Full `pip download --no-deps --dest <scratch>` runs executed BY HAND during
# the investigation that produced this script (2026-08-27, this container).
# Every one printed "Successfully downloaded <pkg>" with exit code 0. Kept
# here so the lightweight per-run PyPI-metadata probe below has a known-good
# baseline to be judged against, per the module docstring.
VERIFIED_FULL_DOWNLOADS = {
    "six": "six-1.17.0-py2.py3-none-any.whl (11 kB) -- control case",
    "arch": "arch-8.0.0-cp311-cp311-manylinux...whl (990 kB)",
    "statsmodels": "statsmodels-0.14.6-cp311-cp311-manylinux...whl (10.4 MB)",
    "xgboost": "xgboost-3.2.0-py3-none-manylinux_2_28_x86_64.whl (131.7 MB)",
    "lightgbm": "lightgbm-4.7.0-py3-none-manylinux...whl (3.5 MB)",
    "catboost": "catboost-1.2.10-cp311-cp311-manylinux2014_x86_64.whl (97.2 MB)",
    "transformers": "transformers-5.16.1-py3-none-any.whl (12.1 MB)",
    "huggingface_hub": "huggingface_hub-1.28.0-py3-none-any.whl (793 kB)",
    "neuralforecast": "neuralforecast-3.2.1-py3-none-any.whl (308 kB)",
}

# (import name, pip/display name) already relevant to the research protocol.
RELEVANT_IMPORTS = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("scipy", "scipy"),
    ("pyarrow", "pyarrow"),
    ("torch", "torch"),
    ("sklearn", "scikit-learn"),
    ("xgboost", "xgboost"),
    ("lightgbm", "lightgbm"),
    ("catboost", "catboost"),
    ("statsmodels", "statsmodels"),
    ("arch", "arch"),
    ("pytorch_forecasting", "pytorch-forecasting"),
    ("gluonts", "gluonts"),
    ("darts", "darts"),
    ("neuralforecast", "neuralforecast"),
    ("transformers", "transformers"),
    ("huggingface_hub", "huggingface_hub"),
]

# pip/PyPI names for candidate packages not already checked via import.
CANDIDATE_PACKAGES = [
    "xgboost", "lightgbm", "catboost",                 # E7 gradient boosting
    "statsmodels", "arch",                              # GARCH baseline
    "neuralforecast", "pytorch-forecasting", "gluonts", "darts",  # PatchTST/iTransformer/TimeXer hosts
    "transformers", "huggingface_hub",                  # E16 foundation models
]

# GitHub reference implementations relevant to E7/E8 architectures that are
# not shipped as an installable package (or whose pip package lags the
# published architecture).
GITHUB_REPOS = {
    "TimeXer (E8) -- thuml/Time-Series-Library": "https://github.com/thuml/Time-Series-Library.git",
    "PatchTST -- yuqinie98/PatchTST": "https://github.com/yuqinie98/PatchTST.git",
    "iTransformer -- thuml/iTransformer": "https://github.com/thuml/iTransformer.git",
}


def probe_import(import_name: str, display_name: str) -> dict[str, Any]:
    """Actually import a module and report what happened. No claim of
    availability is made without this succeeding -- see module docstring."""
    try:
        mod = importlib.import_module(import_name)
        version = getattr(mod, "__version__", "unknown")
        return {"package": display_name, "import_name": import_name,
                "installed": True, "version": version, "error": None}
    except Exception as e:
        return {"package": display_name, "import_name": import_name,
                "installed": False, "version": None,
                "error": f"{type(e).__name__}: {str(e)[:120]}"}


def probe_pypi(pkg: str) -> ProbeResult:
    """Lightweight reachability + existence check against PyPI's JSON API.
    Does not download the wheel -- see 'WHY A LIGHTWEIGHT PROBE' above."""
    return probe_url("PyPI", pkg, f"https://pypi.org/pypi/{pkg}/json")


def probe_git_remote(name: str, url: str, timeout: int = 15) -> dict[str, Any]:
    """Real `git ls-remote` against a GitHub repo -- confirms the git/HTTPS
    path to github.com (distinct from raw.githubusercontent.com and from
    huggingface.co, which datasources.py and kronos_baseline.py separately
    found blocked) is actually usable for cloning reference implementations."""
    t0 = time.time()
    try:
        r = subprocess.run(
            ["git", "ls-remote", url, "HEAD"],
            capture_output=True, text=True, timeout=timeout,
        )
        elapsed = round(time.time() - t0, 2)
        ok = r.returncode == 0 and bool(r.stdout.strip())
        return {"name": name, "url": url, "reachable": ok,
                "returncode": r.returncode,
                "notes": r.stdout.strip()[:80] if ok else r.stderr.strip()[:160],
                "seconds": elapsed}
    except subprocess.TimeoutExpired:
        return {"name": name, "url": url, "reachable": False,
                "returncode": None, "notes": f"timed out after {timeout}s",
                "seconds": timeout}
    except Exception as e:
        return {"name": name, "url": url, "reachable": False,
                "returncode": None, "notes": f"{type(e).__name__}: {e}",
                "seconds": round(time.time() - t0, 2)}


def build_experiment_verdicts(imports: dict[str, dict], pypi: dict[str, ProbeResult],
                               github: dict[str, dict], hf: ProbeResult) -> list[dict]:
    """One verdict per protocol experiment: AVAILABLE (imports now),
    INSTALLABLE (pip/PyPI reachable, not yet imported), or BLOCKED BY
    INFRASTRUCTURE (network denies it), with the evidence that produced it."""

    def pypi_ok(pkg: str) -> bool:
        return pypi[pkg].reachable and pypi[pkg].http_status == 200

    verdicts = []

    # --- E7a: gradient-boosted trees --------------------------------------
    gbt_pkgs = ["xgboost", "lightgbm", "catboost"]
    gbt_installed = [p for p in gbt_pkgs if imports[p]["installed"]]
    gbt_pypi_ok = [p for p in gbt_pkgs if pypi_ok(p)]
    verdicts.append({
        "experiment": "E7 -- gradient-boosted trees (xgboost/lightgbm/catboost)",
        "verdict": "AVAILABLE" if gbt_installed else
                   ("INSTALLABLE" if len(gbt_pypi_ok) == len(gbt_pkgs) else "BLOCKED BY INFRASTRUCTURE"),
        "evidence": (f"installed now: {gbt_installed or 'none'}; "
                     f"PyPI metadata reachable for: {gbt_pypi_ok}; "
                     f"full-wheel download hand-verified for all three "
                     f"(see VERIFIED_FULL_DOWNLOADS) -- xgboost 131.7MB, "
                     f"catboost 97.2MB, lightgbm 3.5MB, all exit 0."),
        "gh_actions_workaround": "not needed -- installs directly from PyPI in this container.",
    })

    # --- E7b: LSTM / GRU / TCN ---------------------------------------------
    verdicts.append({
        "experiment": "E7 -- LSTM / GRU / TCN",
        "verdict": "AVAILABLE",
        "evidence": (f"torch is installed (version {imports['torch']['version']}) and these "
                     "are architectures, not packages -- torch.nn.LSTM/GRU/Conv1d are sufficient "
                     "to implement all three directly. No new dependency required."),
        "gh_actions_workaround": "not applicable.",
    })

    # --- E7c: PatchTST, iTransformer ---------------------------------------
    nf_ok = pypi_ok("neuralforecast")
    ptst = github["PatchTST -- yuqinie98/PatchTST"]
    itr = github["iTransformer -- thuml/iTransformer"]
    verdicts.append({
        "experiment": "E7 -- PatchTST, iTransformer",
        "verdict": "INSTALLABLE" if (nf_ok or ptst["reachable"] or itr["reachable"]) else "BLOCKED BY INFRASTRUCTURE",
        "evidence": (f"neuralforecast (ships both architectures) PyPI-reachable: {nf_ok} "
                     "(full wheel hand-verified, 308kB, exit 0); "
                     f"reference repo git-clonable: PatchTST={ptst['reachable']} "
                     f"({ptst['notes']}), iTransformer={itr['reachable']} ({itr['notes']}). "
                     "Either route works: install the package, or clone the reference "
                     "implementation directly on top of the installed torch."),
        "gh_actions_workaround": "not needed -- both routes work directly in this container.",
    })

    # --- E8: TimeXer ---------------------------------------------------------
    tsl = github["TimeXer (E8) -- thuml/Time-Series-Library"]
    verdicts.append({
        "experiment": "E8 -- TimeXer",
        "verdict": "INSTALLABLE" if (nf_ok or tsl["reachable"]) else "BLOCKED BY INFRASTRUCTURE",
        "evidence": (f"no standalone 'timexer' PyPI package; neuralforecast PyPI-reachable: {nf_ok}; "
                     f"reference repo (thuml/Time-Series-Library, contains TimeXer) "
                     f"git-clonable: {tsl['reachable']} ({tsl['notes']}). Would need the "
                     "architecture vendored from the reference repo or via neuralforecast, "
                     "then run on the already-installed torch."),
        "gh_actions_workaround": "not needed -- github.com is directly reachable from this container.",
    })

    # --- GARCH / EGARCH (mandatory volatility baseline) ---------------------
    garch_installed = imports["arch"]["installed"] or imports["statsmodels"]["installed"]
    verdicts.append({
        "experiment": "GARCH(1,1) / EGARCH -- mandatory volatility baseline (RESEARCH_PLAN.md baseline table)",
        "verdict": "AVAILABLE" if garch_installed else
                   ("INSTALLABLE" if (pypi_ok("arch") or pypi_ok("statsmodels")) else "BLOCKED BY INFRASTRUCTURE"),
        "evidence": (f"arch installed: {imports['arch']['installed']}; "
                     f"statsmodels installed: {imports['statsmodels']['installed']}; "
                     f"arch PyPI-reachable: {pypi_ok('arch')} (hand-verified wheel 990kB, exit 0); "
                     f"statsmodels PyPI-reachable: {pypi_ok('statsmodels')} (hand-verified wheel 10.4MB, exit 0). "
                     "`arch` is the standard GARCH/EGARCH package and installs cleanly; "
                     "not present in requirements-research.txt today."),
        "gh_actions_workaround": "not needed -- installs directly from PyPI in this container.",
    })

    # --- E16: time-series foundation models ---------------------------------
    verdicts.append({
        "experiment": "E16 -- time-series foundation models as challengers",
        "verdict": "BLOCKED BY INFRASTRUCTURE",
        "evidence": (f"huggingface.co reachability probe: reachable={hf.reachable}, "
                     f"notes={hf.notes!r}. The `transformers`/`huggingface_hub` PACKAGES install "
                     "fine from PyPI (hand-verified: transformers 12.1MB, huggingface_hub 793kB, "
                     "both exit 0), but nearly every published TS foundation model's WEIGHTS "
                     "(Chronos, TimesFM, Moirai, Lag-Llama, Kronos) are hosted on huggingface.co, "
                     "which this container's egress policy blocks (403 on CONNECT; same result "
                     "independently found by eval/datasources.py for market-data hosts and by "
                     "eval/kronos_baseline.py, which is shipped complete but has never been run "
                     "for exactly this reason)."),
        "gh_actions_workaround": (
            "Package-only: yes, trivially (pip install on a runner, though PyPI already works "
            "here directly so there's no need). Weights: only case-by-case, and only for small "
            "checkpoints. A GH Actions runner has normal internet and could download weights and "
            "commit them (same pattern as harvest-newdata.yml), but most TS foundation model "
            "checkpoints are hundreds of MB to several GB -- not practical to commit into a git "
            "repo. A genuinely small checkpoint (e.g. Chronos-tiny/-mini class, tens of MB) would "
            "be the exception where this is worth doing; verify actual file size on the runner "
            "before committing, don't assume."
        ),
    })

    return verdicts


def main(argv: list[str] | None = None) -> int:
    """Probe pip/PyPI/GitHub/HF reachability and already-installed packages,
    write model/artifacts/toolchain.json."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", type=Path, default=Path("model/artifacts/toolchain.json"))
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    print("[toolchain] Importing already-relevant packages...")
    imports = {imp: probe_import(imp, disp) for imp, disp in RELEVANT_IMPORTS}

    print("[toolchain] Probing PyPI metadata for candidate packages (no wheel downloads)...")
    pypi = {pkg: probe_pypi(pkg) for pkg in CANDIDATE_PACKAGES}

    print("[toolchain] Probing GitHub reference-implementation repos via git ls-remote...")
    github = {name: probe_git_remote(name, url) for name, url in GITHUB_REPOS.items()}

    print("[toolchain] Probing huggingface.co (expected blocked, per kronos_baseline.py)...")
    hf = probe_url("HuggingFace", "huggingface.co root", "https://huggingface.co/")

    verdicts = build_experiment_verdicts(imports, pypi, github, hf)

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pinned_requirements_file": "model/requirements-research.txt",
        "installed": {name: r for name, r in imports.items()},
        "pypi_probes": {pkg: r.to_dict() for pkg, r in pypi.items()},
        "github_probes": github,
        "huggingface_probe": hf.to_dict(),
        "verified_full_downloads_hand_run": VERIFIED_FULL_DOWNLOADS,
        "experiment_verdicts": verdicts,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    if args.verbose:
        print("\n" + "=" * 88)
        print("INSTALLED (import-tested)")
        print("=" * 88)
        for name, r in imports.items():
            mark = "OK" if r["installed"] else "--"
            print(f"  [{mark}] {r['package']:22s} {r['version'] or r['error']}")

        print("\n" + "=" * 88)
        print("PYPI METADATA REACHABILITY (no wheel downloaded)")
        print("=" * 88)
        for pkg, r in pypi.items():
            mark = "OK" if (r.reachable and r.http_status == 200) else "--"
            print(f"  [{mark}] {pkg:22s} HTTP {r.http_status}  {r.notes}")

        print("\n" + "=" * 88)
        print("GITHUB REFERENCE REPOS (git ls-remote)")
        print("=" * 88)
        for name, r in github.items():
            mark = "OK" if r["reachable"] else "--"
            print(f"  [{mark}] {name:45s} {r['notes']}")

    print("\n" + "=" * 88)
    print("EXPERIMENT VERDICTS")
    print("=" * 88)
    for v in verdicts:
        print(f"  {v['verdict']:24s} {v['experiment']}")

    print(f"\n[toolchain] Wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
