# A5 adversarial audit: NOCTUA code fixes (2026-10-10)

Branch `claude/noctua-disproof-audit-aprkpa`. No tracked file edited, no commit, no branch switch, no hearthbot calls.
Scratch: `runs/noctua-fixes-2026-10-10/audits/scratch_A5/` (`head/` is `git archive HEAD model`, used for before/after).
Data: `data/noctua_history.parquet` (9,600 h, 0 gaps), `model/artifacts/features.parquet` (510,485 rows, has `cal_weekend_frac_ss`).

## Claim 1: weekend column

**1a. `cal_weekend_frac_ss` is the exact Sat+Sun UTC fraction. HOLDS.**
`python -I scratch_A5/calendar_check.py`: 490 episodes, H in {1,2,3,6,12,19,23,24,25,47,48,97,167,168}, including Sun 23:00 with H=1, Sat 00:00 with H=168, Fri 23:00, and non-hour-aligned anchors. Compared with `pandas.dayofweek` over `anchor + 3600*arange(H)`:
max |ss - ref| = 0.000e+00 (exact `==`). Legacy column vs Fri+Sat: 0.000e+00. H=168 gives 2/7 to 1e-12. Values in [0,1]; `ss*H` integer.

**1b. No lookahead. HOLDS.**
Same script: corrupting every hourly field after the largest probe row leaves `cal_weekend_frac_ss` bitwise unchanged. Built with `build_features` on the frame, so the column depends only on `anchor_ts` and `H` (`noctua/features.py:296-311`). `audit_lookahead` also reports leak_free=True, but on only 21 probed episodes, so that check is weak.

**1c. Served outputs bit-identical. HOLDS (with adaptive held constant).**
`python -I scratch_A5/side.py <tree> <out> --no-adaptive` for HEAD (`scratch_A5/head/model`) and the working tree, then `compare.py`. Covered 41 anchors (40 random rows in [9000, end) plus the last row, H=19). Compared `qa`, `sigma_*`, `q_r/q_up/q_dn`, `sigma_atoms`, and the legacy feature column plus `Xa`.
Result: 410/410 arrays identical. The shipped `noctua_v2.npz` is byte-identical to HEAD (`cmp`).
Adaptive enabled (`side.py` without the flag): 246/410 identical. The 164 differing arrays are only `factor`, `sigma_med`, `sigma_mean`, `sigma_atoms`, and only on the 41 anchors where the factor changed (see 2c). The raw network outputs are identical everywhere.

**1d. A newly trained model cannot read the legacy column. HOLDS for the shipped training path.**
`scratch_A5/prep_dump.py` builds `train.prepare()` inputs for fold 0 (102,087 train rows) from HEAD and from the working tree. Xa, Xb, Xs, y and log_sigma are bitwise equal, with the same 39 Xa columns and the same position for the replaced column. The legacy name is absent from every block (`noctua/train.py:74` drops `NON_MODEL_COLS`, which includes `LEGACY_COLS`).
Caveats, not breaking: an explicit `shape_cols` argument naming the legacy column is not blocked. `test_features.py:160-162` exempts the legacy name whenever `_ss` is absent, which weakens the guard (it is only a problem if a new artifact omits `_ss`, which current code never does).

**1e. Other code that hardcodes the column. BROKEN (silent mismatch) in research code, one latent in tooling.**
- BROKEN: `model/eval/lag.py:583-584` (`har_fast_cal`) names `cal_weekend_frac` (legacy) explicitly. Its comparator `log_har_cal` (`lag.py:579`, via `BASE_COLS`) now reads `cal_weekend_frac_ss`. The arms no longer share "the same two calendar terms" as the comment at `lag.py:581-582` says.
- BROKEN (pre-existing, not touched by the fix): `model/eval/hour_anchor_cond.py:78,120` labels "weekend" as `cal_weekend_frac > 0`, i.e. Thu/Fri/Sat anchors.
- WEAKENED: `model/eval/vol_matrix.py:185-193` is a copy of the +4 formula, and `_verify_per_anchor` (`vol_matrix.py:230`) checks only the legacy column. `cal_weekend_frac_ss` is added to `H_DEPENDENT` (`vol_matrix.py:181`) but its values are never checked.
- LATENT: `model/noctua/export.py:46-49` writes `base_cols`/`shape_cols` from the current `spec` constants, not from the checkpoint. It has no callers in the repo, but re-exporting a legacy-trained checkpoint would stamp `_ss` names onto legacy weights.
- WEAKENED: `baselines.py:114` and `committee.py:406` change `log_har_cal` and `GATE_FEATURES` for every eval that calls them, so published baseline numbers will not reproduce from the same data without this change.

## Claim 2: adaptive.py same-hour selector

**2a. Strictly causal (row + H <= anchor). HOLDS.**
`python -I scratch_A5/selector_bruteforce.py`: 4,186 (anchor, H) pairs, H in {1..168}, anchors from the 30-day floor to the end. Failures: causal 0, same-hour 0, window bound 0, floor 0, duplicates 0, count > 60 0. Rows range 1 to 60.

**2b. All rows share the anchor's hour of day. BROKEN when the grid has a gap; HOLDS on the committed bundle.**
The bundle has 0 gaps. But `serve/history.py:165` sets `contiguous = bad == 0 or largest_gap <= max_gap_hours` (default 2). A single missing hour (diff = 2) therefore passes `check_continuity` and `get_hours` (`history.py:209`).
`scratch_A5/gap_hod.py`: drop one row from the bundle. `check_continuity` returns `contiguous: True`. Settled rows counted by index are then one hour off in 52 of 60 cases, because row-24 is 25 hours before the anchor for every row before the gap.
`hours_from_bars` drops incomplete hours (`history.py:129-137`), so a missing hour is a realistic live input. `test_adaptive.py` has no hour-of-day assertion, so nothing catches this.

**2c. Window bounds and sparse history. WEAKENED.**
- (i) Cliff. `scratch_A5/cliff.py`: anchor row 9239 has 19 complete-feature episodes, so factor 1.0 (gate off). Row 9263, 24 h later, has 20 and factor 0.7907. Served sigma and every barrier level move by 21% in one day.
- (ii) Production sample is truncated. The 60-day window needs 365 days of feature warm-up (`reg_rv_vs_year`, 8,760 h). The 400-day bundle leaves 34 usable episodes at the live anchor (`scratch_A5/full_vs_bundle.py`, `same_anchor.py`). The full history at the same anchor (2026-10-05 17:00 UTC) gives 60 episodes. Margin over the gate is 14. Factors agree at this anchor (0.7915 vs 0.7956), but the effective window is about 36 days, not 60.
- (iii) The gate counts complete-feature rows, not settled rows (`adaptive.py:130-132`). The docstring says "settled".
- (iv) Frequency. Over 120 daily 17:00 anchors, HEAD applied the correction on 30, the working tree on 15 (`factor_series.py`). The applied values changed from about 0.97-0.99 to 0.79.
- Small-history refusal (anchor 24*31 returns factor 1.0) is correct and tested.
- Production effect at the live anchor: HEAD 0.9993, working tree 0.7915. Bootstrap 90% CI of the median ratio [0.758, 0.894] on 34 episodes; 76.5% of realized vols below sigma (`last_anchor_detail.py`). Direction matches the audit's stated 17:00 bias, but the size rests on 34 noisy points.

**2d. Off-by-one. HOLDS.**
The most recent eligible same-hour row is included. Brute force found 0 misses. For H <= 24 it is anchor-24 (closes at or before anchor). For H=30 at anchor 5000 it is row 4952 (offset 48), correct because offset 24 would close at 5006 > 5000. The window reaches exactly 60 days (offset 1440).

**test_adaptive.py.** `python -I model/tests/test_adaptive.py`: all checks pass, factor 0.7915 from 34 episodes. It does not test hour alignment (gap in 2b) or the cliff in 2c.

**Evidence quality (not a code bug, but it governs the claimed benefit).** `runs/noctua-fixes-2026-10-10/adaptive_same_hour.py` re-implements the selector instead of importing `serve/adaptive.py`. It runs on the full 2012+ history, not the 400-day bundle production uses, so it cannot see 2c(i)-(ii). Its per-hour table (`evals/adaptive_same_hour.json`) shows same-hour QLIKE worse at 0-13 UTC (hour 10: -0.0207, CI [-0.027, -0.014]) and better at 15-20 UTC (hour 17: +0.0652). The pooled +0.0065 is net of that. The docstring admits the era is not out of sample for design.

## Claim 3: weekend_fix.py arms

**M0s data equals the shipped-era inputs. HOLDS.** `weekend_fix.py:125-126` refuses unless the legacy column matches Fri+Sat. The legacy values are copied to the new name (`:133`). Verified bitwise in `prep_dump.py` (see 1d).

**M0s equals the shipped configuration. BROKEN (pre-existing, not introduced by this patch).**
The shipped artifact trains Stage B on a causal reference, `sigma_ref = clip(exp(har_1d)*sqrt(H))` (`noctua/train_v2.py:118-125`; artifact meta `stage_b_sigma_ref: causal_har_1d_clipped`). `weekend_fix.py` calls `run_fold` at lines 173, 174, 188 and 213 without `sigma_ref_fn`, so `prepare()` falls back to realized RV. `eval/benchmark.py:725-748` documents that exact fallback as "a configuration that does not ship". So M0s, Ws, Ps and Fs all use the non-shipped target. The Ws "artifact increment" claim is measured against a Stage B that does not ship.

**Fs gets the true column everywhere. HOLDS (by reading).** `weekend_fix.py:141-144` sets both names to `wf_true` for `XF`. Fs runs use `XF` (`:174`, `:213`). The legacy name is excluded from Xa by `train.py:74`.

**Self-test.** `python -m model.eval.weekend_fix --selftest`: 5/5 pass.

## Other tests run

- `python -I model/tests/test_weekend_column.py`: 6/6 pass.
- `python -m model.tests.test_features`: all checks pass.
- `python -I model/tests/test_adaptive.py`: all checks pass.

## Verdicts

| Claim | Verdict | Where |
|---|---|---|
| 1a exact Sat+Sun, any H/anchor | HOLDS | features.py:309-311 |
| 1b no lookahead | HOLDS | features.py:296-311 |
| 1c v2 outputs bit-identical | HOLDS | scratch side.py/compare.py |
| 1d new training cannot read legacy | HOLDS (shape_cols override unguarded) | train.py:74 |
| 1e other code hardcodes column | BROKEN in research code; WEAKENED in vol_matrix; LATENT in export | lag.py:583-584; hour_anchor_cond.py:78; export.py:46-49 |
| 2a strictly causal | HOLDS | adaptive.py:96-101 |
| 2b same hour of day | BROKEN under a 1-h gap; HOLDS on gap-free bundle | history.py:165; adaptive.py:100-101 |
| 2c window, MIN_EPISODES | WEAKENED (cliff, 34-episode production sample) | adaptive.py:130-132 |
| 2d most recent day included | HOLDS | adaptive.py:101 |
| 3 M0s = shipped config | BROKEN (pre-existing) | weekend_fix.py:173,174,188,213; train_v2.py:118-125 |
| 3 Fs true everywhere | HOLDS | weekend_fix.py:141-144 |
