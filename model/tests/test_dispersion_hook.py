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

from noctua import infer as I                                   # noqa: E402

# TWO TIERS, because the serving CI job installs the torch-free serving
# runtime and a gate that imports torch cannot run there. The first version of
# this file imported torch at module scope and failed CI with
# ModuleNotFoundError on two commits.
#
# Tier 1 (ALWAYS) exercises `infer.scale_atoms`, pure NumPy, which is where the
# property that matters lives: lam == 1.0 must be an exact identity.
# Tier 2 (only where torch exists) exercises the whole forward pass for
# bit-identity on every returned array.
#
# The absence of torch is REPORTED, not silently skipped -- a suite that
# quietly runs fewer checks and still prints PASS is the kind of guard this
# project keeps having to repair.
try:
    import torch                                               # noqa: E402
    from noctua.model import Noctua                            # noqa: E402
    HAVE_TORCH = True
except ModuleNotFoundError:
    HAVE_TORCH = False

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


def tier1() -> None:
    """Pure-NumPy checks on the helper. These run everywhere."""
    rng = np.random.default_rng(1)
    atoms = np.sort(rng.normal(-4.0, 0.4, size=(50, 32)), axis=1)
    centre = atoms[:, 16][:, None]

    same = I.scale_atoms(atoms, centre, 1.0)
    check("lam=1.0 returns the SAME OBJECT (identity, not recomputation)",
          same is atoms)
    check("lam=1.0 is bit-identical", np.array_equal(same, atoms))

    narrow = I.scale_atoms(atoms, centre, 0.5)
    wide = I.scale_atoms(atoms, centre, 1.5)
    sd = lambda a: float(np.mean(np.std(a, axis=1)))            # noqa: E731
    check("lam<1 narrows the spread", sd(narrow) < 0.55 * sd(atoms),
          f"{sd(narrow)/sd(atoms):.3f}")
    check("lam>1 widens the spread", sd(wide) > 1.45 * sd(atoms),
          f"{sd(wide)/sd(atoms):.3f}")
    check("the centre is preserved exactly at every lam",
          np.array_equal(narrow[:, 16][:, None], centre)
          and np.array_equal(wide[:, 16][:, None], centre))
    # A scaling that collapsed to the centre would pass "narrows" trivially.
    check("lam=0 collapses to the centre exactly",
          np.array_equal(I.scale_atoms(atoms, centre, 0.0),
                         np.broadcast_to(centre, atoms.shape)))


class _StubNoctua:
    """A NumpyNoctua with its three learned stages replaced by fixed maps.

    Subclassing and overriding the stages, rather than building a synthetic
    weights archive, keeps the test about the ONE thing it is for: that
    `predict` applies the hook about the blended median and threads it. Every
    line of `predict` between the stages -- the blend, the atom interpolation,
    the hook, sigma_atoms, sigma_med, sigma_mean -- is the real one.
    """

    def __init__(self, n: int = 16, k: int = 9, blend_w: float = 0.25):
        from serve.runtime import NumpyNoctua
        self.cls = NumpyNoctua
        self.levels = np.linspace(0.05, 0.95, k)
        self.median_idx = int(np.argmin(np.abs(self.levels - 0.5)))
        self.blend_w = blend_w
        rng = np.random.default_rng(3)
        base = rng.normal(-4.0, 0.2, size=(n, 1))
        # a monotone quantile curve per episode, which is what stage A emits
        self._qa = base + np.linspace(-0.8, 0.8, k)[None, :]
        self._har = rng.normal(-4.0, 0.2, size=n)
        self.w = {}

    # the three learned stages, stubbed
    def stage_a(self, Xa, Xb):
        return self._qa

    def har_logvol(self, d):
        return self._har

    def has_mx(self):
        return False

    def stage_b(self, Xs, log_sigma):
        z = np.repeat(log_sigma, 5, axis=1)
        return z, np.abs(z), np.abs(z) + 1.0, None

    def predict(self, d, **kw):
        return self.cls.predict(self, d, **kw)


def tier1b() -> None:
    """The SERVING path, pure NumPy, so it runs in the torch-free CI job too.

    `infer.predict` had this hook and `serve.runtime` did not, which meant a
    lambda could clear the barrier battery and still have nowhere to go. These
    checks are what stop that from silently reverting.
    """
    m = _StubNoctua()
    n = m._qa.shape[0]
    d = {"Xa": np.zeros((n, 1)), "Xb": np.zeros((n, 1)), "Xs": np.zeros((n, 2)),
         "H": np.full(n, 19.0)}

    base = m.predict(d, n_atoms=16)
    same = m.predict(d, n_atoms=16, disp_lambda=1.0)
    keys = [k for k, v in base.items() if isinstance(v, np.ndarray)]
    bad = [k for k in keys if not np.array_equal(base[k], same[k], equal_nan=True)]
    check("serving: disp_lambda=1.0 is bit-identical on every array",
          not bad, "differs: " + ", ".join(bad) if bad else "all equal")

    narrow = m.predict(d, n_atoms=16, disp_lambda=0.5)
    check("serving: sigma_med is untouched, to the bit",
          np.array_equal(base["sigma_med"], narrow["sigma_med"]))
    sb = np.std(np.log(base["sigma_atoms"]), axis=1)
    sn = np.std(np.log(narrow["sigma_atoms"]), axis=1)
    check("serving: lambda<1 halves the atom spread",
          abs(float(np.mean(sn / sb)) - 0.5) < 1e-9,
          f"ratio {float(np.mean(sn / sb)):.6f}")
    check("serving: sigma_mean falls toward sigma_med",
          bool(np.all(narrow["sigma_mean"] <= base["sigma_mean"] + 1e-12))
          and float(np.mean(narrow["sigma_mean"] / base["sigma_mean"])) < 1.0)
    # The centre must be the BLENDED median, not stage A's. With blend_w < 1
    # those differ, and centring on the wrong one would move sigma_med.
    check("serving: the hook centres on the median it actually serves",
          np.array_equal(m.predict(d, n_atoms=16, disp_lambda=0.1)["sigma_med"],
                         base["sigma_med"]))

    # THE ENSEMBLE MUST THREAD IT. A default here rather than a pass-through
    # would leave the knob a no-op on the only class serving instantiates
    # while every check above still passed. Asserted on the parse tree because
    # exercising it needs a seed-scoped weights archive, and an assertion that
    # needs an artifact is one that stops running when the artifact moves.
    import ast as _ast
    import inspect as _inspect
    from serve import runtime as _rt
    tree = _ast.parse(_inspect.getsource(_rt))
    fn = next(x for x in _ast.walk(tree)
              if isinstance(x, _ast.ClassDef) and x.name == "NoctuaV2")
    calls = [c for c in _ast.walk(fn) if isinstance(c, _ast.Call)
             and getattr(c.func, "attr", "") == "predict"]
    check("serving: NoctuaV2 forwards disp_lambda to every seed",
          bool(calls) and all(any(k.arg == "disp_lambda" for k in c.keywords)
                              for c in calls),
          f"{len(calls)} inner predict call(s)")


def main() -> int:
    print("dispersion hook -- tier 1 (pure NumPy, runs everywhere)")
    tier1()
    print("\ndispersion hook -- tier 1b (serving path, pure NumPy)")
    tier1b()
    if not HAVE_TORCH:
        print("\n  torch is NOT installed here, so the full forward-pass "
              "bit-identity\n  checks did NOT run. That is expected in the "
              "serving CI job, which\n  installs the torch-free runtime; "
              "precommit runs both tiers.")
        print(f"\n{'PASS (tier 1 only)' if not FAIL else 'FAIL: ' + ', '.join(FAIL)}")
        return 1 if FAIL else 0

    print("\ndispersion hook -- tier 2 (full forward pass)")
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
