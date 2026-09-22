"""The dispersion hook must be a bit-identical no-op by default.

`infer.predict` grew a `disp_lambda` that scales the atom spread about the
predicted median. The shipped artifact must be unable to move because of it, so
this asserts EQUALITY OF BYTES on every returned array at the default, not
merely that other tests still pass -- the failure mode being guarded against is
a silent numerical drift in the production forecast introduced by a research
knob nobody set.

It also asserts the hook does what it claims when it IS set: the median is
untouched (so no level change smuggles in) while the atom spread and the
barrier inputs move. A knob that could not be shown to do anything would be
the other kind of defect.

    python model/tests/test_dispersion_hook.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch                                                    # noqa: E402
from noctua import infer as I                                   # noqa: E402
from noctua.model import Noctua                                 # noqa: E402

FAIL = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def make_batch(seed: int = 0, n: int = 64):
    """A small deterministic model and batch. The hook is a pure function of
    the atoms, so a trained artifact is unnecessary and would make this test
    depend on one."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_feat, n_base, n_shape = 8, 4, 5
    m = Noctua(n_feat, n_base, n_shape, 16)
    m.eval()
    d = {"Xa": rng.normal(size=(n, n_feat)).astype(np.float32),
         "Xb": rng.normal(size=(n, n_base)).astype(np.float32),
         "Xs": rng.normal(size=(n, n_shape)).astype(np.float32),
         "H": np.full(n, 19.0)}
    return m, d


def main() -> int:
    print("dispersion hook")
    m, d = make_batch()

    base = I.predict(m, d)
    same = I.predict(m, d, disp_lambda=1.0)

    keys = [k for k, v in base.items() if isinstance(v, np.ndarray)]
    check("returns the arrays the committee reads",
          {"qa", "sigma_atoms", "sigma_med", "sigma_mean"} <= set(keys),
          f"{len(keys)} arrays")

    bad = [k for k in keys
           if not np.array_equal(base[k], same[k], equal_nan=True)]
    check("disp_lambda=1.0 is BIT-IDENTICAL to the default on every array",
          not bad, "differs: " + ", ".join(bad) if bad else "all equal")

    # And the knob must actually do something, or the no-op test above is
    # vacuous -- a hook wired to nothing would pass it perfectly.
    narrow = I.predict(m, d, disp_lambda=0.5)
    check("sigma_med is UNCHANGED by the hook (no level smuggled in)",
          np.allclose(base["sigma_med"], narrow["sigma_med"], rtol=0, atol=0))
    spread_b = np.std(np.log(base["sigma_atoms"]), axis=1)
    spread_n = np.std(np.log(narrow["sigma_atoms"]), axis=1)
    check("lambda<1 narrows the atom spread",
          float(np.mean(spread_n / spread_b)) < 0.6,
          f"ratio {float(np.mean(spread_n / spread_b)):.3f}")
    check("sigma_mean falls toward sigma_med when narrowed",
          bool(np.all(narrow["sigma_mean"] <= base["sigma_mean"] + 1e-12))
          and float(np.mean(narrow["sigma_mean"] / base["sigma_mean"])) < 1.0)
    check("the barrier inputs move (the curves can see this)",
          not np.array_equal(base["sigma_atoms"], narrow["sigma_atoms"]))

    # Widening must go the other way; a one-sided check would pass on a hook
    # that clamped.
    wide = I.predict(m, d, disp_lambda=1.5)
    check("lambda>1 widens the spread",
          float(np.mean(np.std(np.log(wide["sigma_atoms"]), axis=1)
                        / spread_b)) > 1.4)

    print(f"\n{'PASS' if not FAIL else 'FAIL: ' + ', '.join(FAIL)}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
