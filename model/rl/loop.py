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
            self.last_slot, self.last_settled = st["last_slot"], st["last_settled"]
            self.base_w = dict(st["base_w"])
        else:
            self.agent = make_agent(agent_kind, seed)
            self.w, self.pending, self.last_slot, self.last_settled = 0.0, [], None, None
            self.base_w = {k: 0.0 for k in ("FLAT", "HALF", "HOLD", "VT")}
        self.last_flush = None

    @property
    def forecaster(self):
        if self._forecaster is None:
            self._forecaster = E.noctua_forecaster(self._model)
        return self._forecaster

    # ------------------------------------------------------------------
    def _hold(self, slot: int, now: int) -> dict:
        return {"E": slot, "decided_at": now, "w_prev": self.w, "a": self.w, "x": None,
                "sigma": None, "missed": True}

    def _settle(self, d: dict, R: float, now: int) -> dict:
        a, wp = d["a"], d["w_prev"]
        u = E.utility(a, R, wp)
        u_all = [E.utility(b, R, wp) for b in E.ACTIONS]
        targets = (E.baseline_weights(d["sigma"]) if d["sigma"] is not None
                   else dict(self.base_w))
        u_base = {k: E.utility(targets[k], R, self.base_w[k]) for k in self.base_w}
        self.base_w = {k: float(targets[k]) for k in self.base_w}
        rec = {"type": "settled", "E": d["E"], "agent": self.agent.kind, "a": a, "w_prev": wp,
               "R": R, "u": u, "u_all": u_all, "regret": max(u_all) - u, "u_baseline": u_base,
               "x": d["x"], "sigma": d["sigma"], "p_up": d.get("p_up"),
               "p_vol_amplify": d.get("p_vol_amplify"), "missed": d["missed"], "settled_at": now}
        if d["x"] is not None:
            self.agent.update(np.array(d["x"]), d["sigma"], wp, a, R)
        self.last_settled = d["E"]
        self.store.append_log(rec)
        return rec

    def tick(self, now: int, hours) -> list:
        events, changed = [], False
        keep = []
        for d in self.pending:
            R = None if (self.last_settled is not None and d["E"] <= self.last_settled) \
                else E.realized(hours, d["E"])
            if R is None:
                keep.append(d)
                continue
            events.append(self._settle(d, R, now))
            changed = True
        self.pending = keep

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
            self.store.save_state({"agent": self.agent.state(), "w": self.w,
                                   "pending": self.pending, "last_slot": self.last_slot,
                                   "last_settled": self.last_settled, "base_w": self.base_w})
            self.last_flush = self.store.flush()
        return events
