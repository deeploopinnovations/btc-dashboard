"""Dump train.prepare() inputs for fold 0 train mask. usage: prep_dump.py <tree> <mode:head|cur> <out.npz>"""
import sys
tree, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, tree)
import numpy as np, pandas as pd
from noctua import splits as S
from noctua.train import load_all, prepare
import noctua.train as T
assert T.__file__.startswith(tree)
ep, X = load_all(__import__("pathlib").Path("/home/user/btc-dashboard/model/artifacts"))
folds = S.walk_forward_folds(ep)
m = np.asarray(folds[0]["train"], bool)
if mode == "head":
    X = X.drop(columns=["cal_weekend_frac_ss"])            # HEAD never had the new column
else:
    X = X.copy(); X["cal_weekend_frac_ss"] = X["cal_weekend_frac"].to_numpy()   # M0s: legacy values under the new name
tr, stds = prepare(ep, X, m)
np.savez(out, Xa=tr["Xa"], Xb=tr["Xb"], Xs=tr["Xs"], y=tr["y"], log_sigma=tr["log_sigma"],
         all=np.array(tr["cols"]["all"]), base=np.array(tr["cols"]["base"]), shape=np.array(tr["cols"]["shape"]))
print(mode, "Xa", tr["Xa"].shape, "Xb", tr["Xb"].shape, "Xs", tr["Xs"].shape, "n", int(m.sum()))
