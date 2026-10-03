"""
rl/agent.py
=====================================================================
The two learners P5-rl-paper compares. Both see the same state (rl/env.py
features from NOCTUA's served forecast, plus the position held) and choose an
exposure in env.ACTIONS every six hours.

MV -- learns from every action, its own and the ones it did not take.
  The agent's trades do not move the BTC price, so once a window closes the
  utility EVERY action would have earned is known exactly. MV uses that: it
  learns the window's expected return mu(x) (forgotten ridge regression) and its
  second moment kappa * s^2 (s = NOCTUA's sigma, kappa a forgotten calibration
  ratio), then plans one step with the cost of moving from the current position
  known in advance:  argmax_a  a mu - (gamma/2) a^2 kappa s^2 - c|a - w_prev|.
  A loss teaches it twice: through mu (it was wrong about the direction) and
  through kappa (it was wrong about the size of the move).
  It acts only on the part of mu its own standard error supports:
  mu_eff = sign(mu) * max(|mu| - z_conf * se(mu), 0), z_conf = 1. Without that
  (MV0, the registered form) it traded its estimation noise: fed pure noise it
  was flat on only 23% of steps and lost 8 bps of utility per step
  (tests/test_rl.py; P5-rl-paper amendment, made before any replay number).
  And it plans past the next step. One-step planning never left a position it
  had drifted into: staying cost one step of risk, leaving cost the fee once, so
  it held forever. MV values each action as this step's utility, minus the fee,
  plus the discounted value of the position it leaves -- a Bellman equation
  over the five positions, delta = 0.98 (about twelve days), with the
  continuation valued under the AVERAGE edge (x = 0) and typical risk, solved
  by value iteration. That is the reinforcement-learning core: a learned
  reward model and exact planning over the position MDP.

TS -- classic RL, learns only from what it did.
  A contextual bandit: per-action Bayesian linear regression of the realised
  utility on [1, x, w_prev], Thompson sampling to explore. It never sees what
  the other actions would have earned, and it pays for exploration.

Both forget at rho per step (an effective memory of ~500 steps, four months),
so an old regime fades instead of anchoring the policy. State is plain JSON.
"""
from __future__ import annotations

import numpy as np

from rl import env as E

RHO = 0.998


class MVAgent:
    kind = "MV"

    def __init__(self, rho: float = RHO, prior_n: float = 50.0, gamma: float = E.GAMMA,
                 cost: float = E.COST, seed: int = 0, z_conf: float = 1.0,
                 delta: float = 0.98):
        self.rho, self.prior_n, self.gamma, self.cost = rho, prior_n, gamma, cost
        self.z_conf, self.delta = z_conf, delta
        self.A = np.zeros((6, 6))
        self.b = np.zeros(6)
        self.k_num = 0.0
        self.k_den = 0.0
        self.n_eff = 0.0
        self.n_updates = 0

    @staticmethod
    def _z(x):
        return np.r_[1.0, np.asarray(x, np.float64)]

    def mu(self, x) -> float:
        theta = np.linalg.solve(self.A + self.prior_n * np.eye(6), self.b)
        return float(self._z(x) @ theta)

    def mu_se(self, x) -> float:
        z = self._z(x)
        r2 = (self.k_num + self.prior_n * E.S_REF ** 2) / (self.n_eff + self.prior_n)
        return float(np.sqrt(z @ np.linalg.solve(self.A + self.prior_n * np.eye(6), z) * r2))

    def mu_eff(self, x) -> float:
        m = self.mu(x)
        if self.z_conf <= 0:
            return m
        return float(np.sign(m) * max(abs(m) - self.z_conf * self.mu_se(x), 0.0))

    def kappa(self) -> float:
        w = self.prior_n * E.S_REF ** 2
        return (self.k_num + w) / (self.k_den + w)

    def _f(self, m, q) -> np.ndarray:
        a = np.array(E.ACTIONS)
        return a * m - 0.5 * self.gamma * a * a * q

    def continuation(self) -> np.ndarray:
        """V(w): value of holding w under the average edge, by value iteration."""
        if self.delta <= 0:
            return np.zeros(len(E.ACTIONS))
        s2 = (self.k_den + self.prior_n * E.S_REF ** 2) / (self.n_eff + self.prior_n)
        f = self._f(self.mu_eff(np.zeros(5)), self.kappa() * s2)
        a = np.array(E.ACTIONS)
        C = self.cost * np.abs(a[None, :] - a[:, None])          # C[w, a]
        V = np.zeros(len(a))
        for _ in range(2000):
            Vn = np.max(f[None, :] - C + self.delta * V[None, :], axis=1)
            if np.max(np.abs(Vn - V)) < 1e-12:
                break
            V = Vn
        return Vn

    def values(self, x, s, w_prev) -> np.ndarray:
        f = self._f(self.mu_eff(x), self.kappa() * s * s)
        a = np.array(E.ACTIONS)
        return f - self.cost * np.abs(a - w_prev) + self.delta * self.continuation()

    def act(self, x, s, w_prev, rng=None) -> float:
        v = self.values(x, s, w_prev)
        best = np.flatnonzero(v >= v.max() - 1e-15)
        # ties: keep the position, else the smaller exposure
        return float(min((E.ACTIONS[i] for i in best), key=lambda a: (a != w_prev, abs(a))))

    def update(self, x, s, w_prev, a, R) -> None:
        z = self._z(x)
        self.A = self.rho * self.A + np.outer(z, z)
        self.b = self.rho * self.b + z * R
        self.k_num = self.rho * self.k_num + R * R
        self.k_den = self.rho * self.k_den + s * s
        self.n_eff = self.rho * self.n_eff + 1.0
        self.n_updates += 1

    def state(self) -> dict:
        return {"kind": self.kind, "z_conf": self.z_conf, "delta": self.delta, "A": self.A.tolist(),
                "b": self.b.tolist(), "k_num": self.k_num, "k_den": self.k_den,
                "n_eff": self.n_eff, "n_updates": self.n_updates}

    def load_state(self, st: dict) -> None:
        self.A, self.b = np.array(st["A"]), np.array(st["b"])
        self.k_num, self.k_den = float(st["k_num"]), float(st["k_den"])
        self.n_eff, self.z_conf = float(st["n_eff"]), float(st["z_conf"])
        self.delta = float(st["delta"])
        self.n_updates = int(st["n_updates"])


class TSAgent:
    kind = "TS"

    def __init__(self, rho: float = RHO, lam: float = 1.0, noise: float = 0.01,
                 gamma: float = E.GAMMA, cost: float = E.COST, seed: int = 0):
        self.rho, self.lam, self.noise, self.gamma, self.cost = rho, lam, noise, gamma, cost
        k = len(E.ACTIONS)
        self.A = np.zeros((k, 7, 7))
        self.b = np.zeros((k, 7))
        self.rng = np.random.default_rng(seed)
        self.n_updates = 0

    @staticmethod
    def _z(x, w_prev):
        return np.r_[1.0, np.asarray(x, np.float64), w_prev]

    def act(self, x, s, w_prev, rng=None) -> float:
        rng = rng if rng is not None else self.rng
        z = self._z(x, w_prev)
        vals = []
        for i in range(len(E.ACTIONS)):
            P = self.A[i] + self.lam * np.eye(7)
            cov = np.linalg.inv(P)
            mean = cov @ self.b[i]
            theta = rng.multivariate_normal(mean, self.noise ** 2 * (cov + cov.T) / 2)
            vals.append(z @ theta)
        return float(E.ACTIONS[int(np.argmax(vals))])

    def update(self, x, s, w_prev, a, R) -> None:
        z = self._z(x, w_prev)
        i = E.ACTIONS.index(a)
        self.A *= self.rho
        self.b *= self.rho
        self.A[i] += np.outer(z, z)
        self.b[i] += z * E.utility(a, R, w_prev, self.cost, self.gamma)
        self.n_updates += 1

    def state(self) -> dict:
        return {"kind": self.kind, "A": self.A.tolist(), "b": self.b.tolist(),
                "rng": self.rng.bit_generator.state, "n_updates": self.n_updates}

    def load_state(self, st: dict) -> None:
        self.A, self.b = np.array(st["A"]), np.array(st["b"])
        self.rng.bit_generator.state = st["rng"]
        self.n_updates = int(st["n_updates"])


class MV0Agent(MVAgent):
    """MV as registered: the point estimate, one step ahead (z_conf = 0, delta = 0)."""
    kind = "MV0"

    def __init__(self, seed: int = 0, **kw):
        super().__init__(seed=seed, z_conf=0.0, delta=0.0, **kw)


def make_agent(kind: str, seed: int = 0):
    return {"MV": MVAgent, "MV0": MV0Agent, "TS": TSAgent}[kind](seed=seed)
