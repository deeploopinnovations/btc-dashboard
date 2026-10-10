"""
rl/loop.py
=====================================================================
The paper trader the Space runs: call `tick(now, hours)` as often as you like
(the Space does it every minute); it is idempotent.

Each tick, in this order:
  1. SETTLE every pending decision whose window has closed: R from the feed,
     the utility of the action taken, of every action (regret), and of the
     baselines FLAT/HALF/HOLD/VT on their own position paths; append the record
     to the log; the agent learns from it.
  2. DECIDE the current slot, once, if the hour before it is in the feed and the
     slot began less than `grace_s` ago. Slots that passed without a decision
     (a feed outage, a sleeping Space) HOLD the position -- its P&L still counts
     when they settle -- and are logged as missed. A fresh agent starts at the
     next slot it can make on time; it does not backfill.
  3. PERSIST state (and mirror it, for an HFStore) if anything changed.

Learning happens only from settled windows, and each window once.

ROBUSTNESS (audit G, 2026-10-08 -- each line below was a reproduced failure):
  - settled windows are a SET, not a settled-up-to watermark: a window whose
    bars arrive after a later window settled still settles;
  - each baseline's position is fixed at DECISION time, so its cost term is
    right whatever order windows settle in;
  - a record already in the log is never appended again, and state is saved
    right after settling, so a crash in the next decision cannot duplicate a
    record or a lesson;
  - a window whose bars never arrive is logged VOID after env.VOID_AFTER;
  - a clock more than env.MAX_FEED_LAG beyond the newest bar is refused.
"""
from __future__ import annotations

import numpy as np

from rl import env as E
from rl.agent import make_agent


class PaperTrader:
    def __init__(self, store, agent_kind: str = "MV", forecaster=None, model=None,
                 seed: int = 0, grace_s: int = 50 * 60):
        self.store, self.grace = store, grace_s
        self._forecaster, self._model = forecaster, model
        st = store.load_state()
        if st:
            self.agent = make_agent(st["agent"]["kind"], seed)
            self.agent.load_state(st["agent"])
            self.w, self.pending = float(st["w"]), list(st["pending"])
            self.last_slot = st["last_slot"]
            self.settled = set(st.get("settled", []))
            self.base_w = dict(st["base_w"])
        else:
            self.agent = make_agent(agent_kind, seed)
            self.w, self.pending, self.last_slot = 0.0, [], None
            self.settled = set()
            self.base_w = {k: 0.0 for k in ("FLAT", "HALF", "HOLD", "VT")}
        self.logged = {r["E"] for r in store.read_log() if r.get("type") in ("settled", "void")}
        self.last_flush = None

    @property
    def forecaster(self):
        if self._forecaster is None:
            self._forecaster = E.noctua_forecaster(self._model)
        return self._forecaster

    # ------------------------------------------------------------------
    def _baselines(self, d: dict) -> dict:
        """Fix each baseline's position for this slot, and the one it came from."""
        targets = (E.baseline_weights(d["sigma"]) if d["sigma"] is not None
                   else dict(self.base_w))
        d["base_prev"] = {k: float(v) for k, v in self.base_w.items()}
        d["base_w"] = {k: float(targets[k]) for k in self.base_w}
        self.base_w = dict(d["base_w"])
        return d

    def _hold(self, slot: int, now: int) -> dict:
        return self._baselines({"E": slot, "decided_at": now, "w_prev": self.w, "a": self.w,
                                "x": None, "sigma": None, "missed": True})

    def _record(self, rec: dict) -> None:
        if rec["E"] not in self.logged:                   # a crash may have logged it already
            self.store.append_log(rec)
            self.logged.add(rec["E"])
        self.settled.add(rec["E"])

    def _void(self, d: dict, now: int) -> dict:
        rec = {"type": "void", "E": d["E"], "agent": self.agent.kind, "a": d["a"],
               "w_prev": d["w_prev"], "missed": d["missed"], "voided_at": now,
               "reason": "the window's entry or exit bar never arrived; P&L unknown, nothing learned"}
        self._record(rec)
        return rec

    def _settle(self, d: dict, R: float, now: int) -> dict:
        a, wp = d["a"], d["w_prev"]
        # every utility in the record uses the AGENT's own cost and risk
        # aversion, restored with its state, so a change to env defaults cannot
        # switch the logged series' convention mid-run (Codex review, PR #14)
        c, g = self.agent.cost, self.agent.gamma
        u = E.utility(a, R, wp, c, g)
        u_all = [E.utility(b, R, wp, c, g) for b in E.ACTIONS]
        if "base_w" in d:
            u_base = {k: E.utility(d["base_w"][k], R, d["base_prev"][k], c, g) for k in d["base_w"]}
        else:                                             # decided before 2026-10-08
            targets = (E.baseline_weights(d["sigma"]) if d["sigma"] is not None
                       else dict(self.base_w))
            u_base = {k: E.utility(targets[k], R, self.base_w[k], c, g) for k in self.base_w}
        rec = {"type": "settled", "E": d["E"], "agent": self.agent.kind, "a": a, "w_prev": wp,
               "R": R, "u": u, "u_all": u_all, "regret": max(u_all) - u, "u_baseline": u_base,
               "cost": c, "gamma": g,
               "x": d["x"], "sigma": d["sigma"], "p_up": d.get("p_up"),
               "p_vol_amplify": d.get("p_vol_amplify"), "missed": d["missed"], "settled_at": now}
        if d["x"] is not None:
            self.agent.update(np.array(d["x"]), d["sigma"], wp, a, R)
        self._record(rec)
        return rec

    def _save(self) -> None:
        self.store.save_state({"agent": self.agent.state(), "w": self.w,
                               "pending": self.pending, "last_slot": self.last_slot,
                               "settled": sorted(self.settled)[-5000:], "base_w": self.base_w})

    def tick(self, now: int, hours) -> list:
        if len(hours) == 0:
            raise E.ClockAheadOfFeed("no bars at all")
        lag = int(now) - (int(hours["hour_ts"].max()) + E.HOUR)
        if lag > E.MAX_FEED_LAG:
            raise E.ClockAheadOfFeed(f"now is {lag / 3600:.1f} h past the newest bar; refusing to tick")
        events, changed = [], False
        keep = []
        for d in self.pending:
            if d["E"] in self.settled:
                changed = True                            # already done; just drop it
                continue
            R = E.realized(hours, d["E"])
            if R is not None:
                events.append(self._settle(d, R, now))
                changed = True
            elif now > d["E"] + E.H * E.HOUR + E.VOID_AFTER:
                events.append(self._void(d, now))
                changed = True
            else:
                keep.append(d)
        self.pending = keep
        if changed:
            self._save()                                  # before deciding: a crash there loses nothing

        slot = E.slot_floor(int(now))
        if self.last_slot is not None:
            nxt = self.last_slot + E.H * E.HOUR
            while nxt < slot:                                  # slept through these: hold
                d = self._hold(nxt, now)
                self.pending.append(d)
                events.append({"type": "missed", **d})
                self.last_slot, nxt, changed = nxt, nxt + E.H * E.HOUR, True
        if self.last_slot is None or slot > self.last_slot:
            on_time = now - slot <= self.grace
            try:
                inp = E.inputs_at(None, hours, slot, forecaster=self.forecaster) if on_time else None
            except E.NotReady:
                inp = None
            if inp is not None:
                a = self.agent.act(inp["x"], inp["sigma"], self.w)
                d = {"E": slot, "decided_at": now, "w_prev": self.w, "a": a,
                     "x": inp["x"].tolist(), "sigma": inp["sigma"], "p_up": inp["p_up"],
                     "p_vol_amplify": inp["p_vol_amplify"], "trailing_rv": inp["trailing_rv"],
                     "spot": inp["spot"], "r24": inp["r24"], "missed": False}
                self._baselines(d)
                events.append({"type": "decision", **d})
            elif not on_time and self.last_slot is not None:
                d = self._hold(slot, now)
                events.append({"type": "missed", **d})
            else:
                d = None                                       # wait: inside the grace window, or a fresh start
            if d is not None:
                self.pending.append(d)
                self.w, self.last_slot, changed = d["a"], slot, True

        if changed:
            self._save()
            self.last_flush = self.store.flush()
        return events
