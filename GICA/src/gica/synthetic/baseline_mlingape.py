# m_LinGapE.py
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.optimize import linprog


@dataclass
class _LPResult:
    w: np.ndarray
    p: np.ndarray
    success: bool
    message: str


class m_LinGapE:
    """
    m-LinGapE (Table 1 in the GIFA paper), using paired indices:
        B_{i,j}(t) = (mu_hat_i - mu_hat_j) + C_{t,delta} * ||x_i - x_j||_{Sigma_hat}
    where Sigma_hat = sigma^2 V_t^{-1}.

    Table 1:
      - J(t) = top-m by mu_hat
      - b_t = argmax_{j in J} max_{i notin J} B_{i,j}
      - c_t = argmax_{i notin J} B_{i,b_t}
      - stop if B_{c_t,b_t} <= epsilon
      - selection: largest_variance | greedy | optimized
    """

    def __init__(
        self,
        paths: Dict[int, List[int]],
        feature_matrix: np.ndarray,
        m: int,
        d: int,
        lambda_reg: float = 1.0,
        epsilon: float = 0.05,
        delta: float = 0.1,
        R: float = 0.1,      # sigma
        S_0: float = 2.0,    # ||theta*|| bound
        selection_rule: str = "largest_variance",
        init_each_arm_once: bool = False,
        seed: int = 0,
        lp_cache: bool = True,
    ):
        self.paths = paths
        self.features = feature_matrix
        self.m = int(m)
        self.d = int(d)

        self.lambda_reg = float(lambda_reg)
        self.epsilon = float(epsilon)
        self.delta = float(delta)

        self.sigma = float(R)
        self.S = float(S_0)

        self.selection_rule = str(selection_rule)
        if self.selection_rule not in {"largest_variance", "greedy", "optimized"}:
            raise ValueError("selection_rule must be 'largest_variance', 'greedy', or 'optimized'")

        self.rng = np.random.default_rng(seed)

        self.g_pi: Dict[int, np.ndarray] = {}
        for pid, step_idx in self.paths.items():
            self.g_pi[int(pid)] = np.mean(self.features[step_idx], axis=0).astype(float)

        self.all_ids: List[int] = sorted(self.g_pi.keys())
        self.K = len(self.all_ids)
        if not (1 <= self.m < self.K):
            raise ValueError("Need 1 <= m < number of arms")

        self.X = np.stack([self.g_pi[pid] for pid in self.all_ids], axis=0)  # (K,d)
        self.id_to_idx = {pid: i for i, pid in enumerate(self.all_ids)}
        self.idx_to_id = {i: pid for pid, i in self.id_to_idx.items()}

        self.L = float(np.max(np.linalg.norm(self.X, axis=1)))

        self.V_inv = (1.0 / self.lambda_reg) * np.eye(self.d)
        self.b_vec = np.zeros(self.d, dtype=float)
        self.theta_hat = np.zeros(self.d, dtype=float)
        self.counts = np.zeros(self.K, dtype=int)

        self.t = 0
        self.total_comparisons = 0
        self.best_G_history: List[float] = []
        self.min_lcb_history: List[float] = []
        self.round_time_history: List[float] = []

        self._init_queue = list(range(self.K)) if init_each_arm_once else []

        self._lp_cache_enabled = bool(lp_cache)
        self._lp_cache: Dict[Tuple[int, int], _LPResult] = {}

    def _C(self) -> float:
        term1 = 2.0 * math.log(1.0 / self.delta)
        term2 = self.d * math.log(1.0 + ((self.t + 1.0) * (self.L ** 2)) / ((self.lambda_reg ** 2) * self.d))
        return math.sqrt(term1 + term2) + (math.sqrt(self.lambda_reg) / self.sigma) * self.S

    def _ls_update(self, arm_idx: int, reward: float) -> None:
        x = self.X[arm_idx]
        v_inv_x = self.V_inv @ x
        denom = 1.0 + float(x.T @ v_inv_x)
        self.V_inv = self.V_inv - np.outer(v_inv_x, v_inv_x) / denom

        self.b_vec = self.b_vec + float(reward) * x
        self.theta_hat = self.V_inv @ self.b_vec
        self.counts[arm_idx] += 1

    def _compute_B_matrix(self):
        C = self._C()
        mu = self.X @ self.theta_hat

        M = self.X @ self.V_inv
        G = M @ self.X.T
        diag = np.clip(np.diag(G), 0.0, None)

        diff_var = diag[:, None] + diag[None, :] - 2.0 * G
        diff_var = np.maximum(diff_var, 1e-12)

        W = self.sigma * np.sqrt(diff_var)
        B = (mu[:, None] - mu[None, :]) + C * W
        return B, mu, C, diag

    def _select_largest_variance(self, b_idx: int, c_idx: int, diag: np.ndarray) -> int:
        return int(b_idx if diag[b_idx] >= diag[c_idx] else c_idx)

    def _select_greedy(self, b_idx: int, c_idx: int) -> int:
        diff = self.X[b_idx] - self.X[c_idx]
        v_inv_diff = self.V_inv @ diff

        xVx = np.sum(self.X * (self.X @ self.V_inv), axis=1)
        diffVx = self.X @ v_inv_diff

        denom = 1.0 + np.maximum(xVx, 0.0)
        score = (diffVx ** 2) / np.maximum(denom, 1e-12)
        return int(np.argmax(score))

    def _solve_l1_representation(self, diff: np.ndarray) -> _LPResult:
        K = self.K
        A_eq = np.concatenate([self.X.T, -self.X.T], axis=1)
        b_eq = diff.astype(float)
        c = np.ones(2 * K, dtype=float)
        bounds = [(0.0, None)] * (2 * K)

        res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
        if not res.success:
            w = np.zeros(K, dtype=float)
            p = np.ones(K, dtype=float) / K
            return _LPResult(w=w, p=p, success=False, message=str(res.message))

        u = res.x[:K]
        v = res.x[K:]
        w = u - v
        absw = np.abs(w)
        s = float(np.sum(absw))
        p = (absw / s) if s > 1e-12 else (np.ones(K, dtype=float) / K)
        return _LPResult(w=w, p=p, success=True, message="ok")

    def _select_optimized(self, b_idx: int, c_idx: int) -> int:
        key = (b_idx, c_idx)
        if self._lp_cache_enabled and key in self._lp_cache:
            lp = self._lp_cache[key]
        else:
            diff = self.X[b_idx] - self.X[c_idx]
            lp = self._solve_l1_representation(diff)
            if self._lp_cache_enabled:
                self._lp_cache[key] = lp

        if not lp.success:
            return self._select_greedy(b_idx, c_idx)

        w = lp.w
        l1 = float(np.sum(np.abs(w)))

        # GIFA Eq. (1): restrict to arms with w_a^*(b,c) > 0
        support = np.where(w > 1e-12)[0]
        if support.size == 0:
            # fall back (paper doesn't specify what to do if no positive entries)
            return self._select_greedy(b_idx, c_idx)

        # scores[a] = N_a(t) * ||w||_1 / |w_a|
        denom = np.maximum(np.abs(w[support]), 1e-12)
        scores = self.counts[support] * l1 / denom

        # optional: random tie-break (tiny noise)
        scores = scores + 1e-12 * self.rng.standard_normal(scores.shape)

        return int(support[int(np.argmax(scores))])


    def select_and_update(self, oracle_callback_path: Callable[[int], float]):
        t0 = time.perf_counter()

        if self._init_queue:
            a_idx = int(self._init_queue.pop(0))
            a_id = self.idx_to_id[a_idx]
            r = float(oracle_callback_path(a_id))
            self._ls_update(a_idx, r)
            self.t += 1

            mu = self.X @ self.theta_hat
            top_idx = np.argsort(mu)[::-1][: self.m]
            top_ids = [self.idx_to_id[int(i)] for i in top_idx]
            self.round_time_history.append(time.perf_counter() - t0)
            return False, top_ids

        B, mu, C, diag = self._compute_B_matrix()

        order = np.argsort(mu)[::-1]
        top_idx = order[: self.m]
        rest_idx = order[self.m :]

        top_ids = [self.idx_to_id[int(i)] for i in top_idx]
        if rest_idx.size == 0:
            self.round_time_history.append(time.perf_counter() - t0)
            return True, top_ids

        self.total_comparisons += int(len(top_idx) * len(rest_idx))

        max_outside_for_j = np.max(B[rest_idx[:, None], top_idx[None, :]], axis=0)
        b_pos = int(np.argmax(max_outside_for_j))
        b_idx = int(top_idx[b_pos])

        col = B[rest_idx, b_idx]
        c_idx = int(rest_idx[int(np.argmax(col))])

        B_cb = float(B[c_idx, b_idx])
        self.best_G_history.append(B_cb)
        self.min_lcb_history.append(-B_cb)

        if B_cb <= self.epsilon:
            self.round_time_history.append(time.perf_counter() - t0)
            return True, top_ids

        if self.selection_rule == "largest_variance":
            a_idx = self._select_largest_variance(b_idx, c_idx, diag)
        elif self.selection_rule == "greedy":
            a_idx = self._select_greedy(b_idx, c_idx)
        else:
            a_idx = self._select_optimized(b_idx, c_idx)

        a_id = self.idx_to_id[a_idx]
        r = float(oracle_callback_path(a_id))
        self._ls_update(a_idx, r)
        self.t += 1

        self.round_time_history.append(time.perf_counter() - t0)
        return False, top_ids
