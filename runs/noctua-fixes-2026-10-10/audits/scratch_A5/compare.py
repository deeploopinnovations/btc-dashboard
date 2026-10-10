import sys, numpy as np
a = np.load(sys.argv[1]); b = np.load(sys.argv[2])
assert set(a.files) == set(b.files)
bad = []; maxd = {}
for k in a.files:
    x, y = a[k], b[k]
    eq = x.shape == y.shape and np.array_equal(x, y)
    if not eq:
        bad.append(k)
        maxd[k] = float(np.max(np.abs(x - y))) if x.shape == y.shape else "shape"
print(f"{len(a.files)} arrays compared; identical: {len(a.files)-len(bad)}; differing: {len(bad)}")
by = {}
for k in bad:
    name = k.split("__", 1)[1]
    by.setdefault(name, []).append(maxd[k])
for n, v in sorted(by.items()):
    print(f"  {n:18s} n={len(v):3d} max|diff|={max(v):.6g}")
if "--show" in sys.argv:
    fac = [k for k in a.files if k.endswith("__factor")]
    print("factors head:", [round(float(a[k][0]),4) for k in fac[:8]])
    print("factors cur :", [round(float(b[k][0]),4) for k in fac[:8]])
