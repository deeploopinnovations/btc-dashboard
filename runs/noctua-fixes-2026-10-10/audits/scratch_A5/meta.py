import json, sys
import numpy as np
z = np.load("model/serve/noctua_v2.npz", allow_pickle=False)
meta = json.loads(bytes(z["meta_json"]).decode())
print("keys:", sorted(meta.keys()))
for k in ("feat_cols", "base_cols", "shape_cols"):
    print(k, len(meta[k]), meta[k])
print("weekend_column" in meta, meta.get("weekend_column"))
import pandas as pd
f = pd.read_parquet("model/artifacts/features.parquet")
print("features.parquet cols with cal_:", [c for c in f.columns if c.startswith("cal_")])
print("features.parquet n rows", len(f))
