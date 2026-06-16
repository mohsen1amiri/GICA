# GIFA.py
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np


@dataclass
class _LPResult:
    w: np.ndarray
    p: np.ndarray
    success: bool
    message: str


class LinGIFA:
    """
    LinGIFA (Table 1 in the GIFA paper): a concrete instance of the GIFA framework.

    Uses *paired* gap indices:
        B_{i,j}(t) = (mu_hat_i - mu_hat_j) + C_{t,delta} * ||x_i - x_j||_{Sigma_hat}
    with Sigma_hat = sigma^2 V_t^{-1}.

    LinGIFA specifics (Table 1):
      - compute_Jt  : pick m arms with smallest score_j, where score_j = m_max_{i != j} B_{i,j}(t)
                (m_max = m-th greatest value, per paper notation)
      - compute_bt  : b_t = argmax_{j in J(t)} score_j
      - challenger  : c_t = argmax_{a notin J(t)} B_{a,b_t}(t)
      - stopping    : stop if max_{j in J(t)} score_j <= epsilon
      - selection   : "largest_variance" or "greedy"

    Compatible with your runner (plot_metrics.py):
      - t, m, total_comparisons
      - best_G_history (stopping quantity)
      - min_lcb_history (-best_G_history)
      - round_time_history
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
        selection_rule: str = "largest_variance",  # "largest_variance" | "greedy"
        init_each_arm_once: bool = False,
        seed: int = 0,
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
        if self.selection_rule not in {"largest_variance", "greedy"}:
            raise ValueError("selection_rule must be 'largest_variance' or 'greedy'")

        self.rng = np.random.default_rng(seed)

        # ---- Arm features: x_a = g_pi[path_id] ----
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

        # ---- RLS state ----
        self.V_inv = (1.0 / self.lambda_reg) * np.eye(self.d)
        self.b_vec = np.zeros(self.d, dtype=float)
        self.theta_hat = np.zeros(self.d, dtype=float)
        self.counts = np.zeros(self.K, dtype=int)

        # ---- logs ----
        self.t = 0
        self.total_comparisons = 0
        self.best_G_history: List[float] = []
        self.min_lcb_history: List[float] = []
        self.round_time_history: List[float] = []

        # optional init
        self._init_queue = list(range(self.K)) if init_each_arm_once else []

    def _C(self) -> float:
        """
        GIFA paper (Lemma 3 / Eq.(2)):
            C_{t,δ} = sqrt( 2 ln(1/δ) + d ln( 1 + ((t+1) L^2) / (λ^2 d) ) ) + (sqrt(λ)/σ) S
        """
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

    def _compute_B_matrix(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        B[i,j] = mu_hat[i] - mu_hat[j] + C*sigma*sqrt((x_i-x_j)^T V_inv (x_i-x_j))
        """
        C = self._C()
        mu = self.X @ self.theta_hat  # (K,)

        M = self.X @ self.V_inv
        G = M @ self.X.T
        diag = np.clip(np.diag(G), 0.0, None)

        diff_var = diag[:, None] + diag[None, :] - 2.0 * G
        diff_var = np.maximum(diff_var, 1e-12)

        W = self.sigma * np.sqrt(diff_var)
        B = (mu[:, None] - mu[None, :]) + C * W
        return B, mu, C

    def _select_largest_variance(self, b_idx: int, c_idx: int) -> int:
        xb = self.X[b_idx]
        xc = self.X[c_idx]
        vb = float(xb.T @ self.V_inv @ xb)
        vc = float(xc.T @ self.V_inv @ xc)
        return int(b_idx if vb >= vc else c_idx)

    def _select_greedy(self, b_idx: int, c_idx: int) -> int:
        diff = self.X[b_idx] - self.X[c_idx]
        v_inv_diff = self.V_inv @ diff

        xVx = np.sum(self.X * (self.X @ self.V_inv), axis=1)  # (K,)
        diffVx = self.X @ v_inv_diff                           # (K,)

        denom = 1.0 + np.maximum(xVx, 0.0)
        score = (diffVx ** 2) / np.maximum(denom, 1e-12)
        return int(np.argmax(score))

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

        B, mu, C = self._compute_B_matrix()
        np.fill_diagonal(B, -np.inf)

        # LinGIFA: score_j = m_max_{i != j} B_{i,j}(t)
        # (m_max = m-th greatest value) 
        m = self.m
        kth = self.K - m              # index of the m-th largest in ascending order
        score = np.partition(B, kth, axis=0)[kth, :] # shape (K,)




        # J(t) = m smallest score_j
        J_idx = np.argsort(score)[: self.m]
        J_set = set(int(i) for i in J_idx)

        stop_val = float(np.max(score[J_idx]))
        self.best_G_history.append(stop_val)
        self.min_lcb_history.append(-stop_val)

        top_ids = [self.idx_to_id[int(i)] for i in sorted(J_idx, key=lambda k: mu[k], reverse=True)]

        if stop_val <= self.epsilon:
            self.round_time_history.append(time.perf_counter() - t0)
            return True, top_ids

        b_idx = int(J_idx[int(np.argmax(score[J_idx]))])

        outside = np.array([i for i in range(self.K) if i not in J_set], dtype=int)
        c_idx = int(outside[int(np.argmax(B[outside, b_idx]))])

        if self.selection_rule == "largest_variance":
            a_idx = self._select_largest_variance(b_idx, c_idx)
        else:
            a_idx = self._select_greedy(b_idx, c_idx)

        self.total_comparisons += self.K * (self.K - 1)

        a_id = self.idx_to_id[a_idx]
        r = float(oracle_callback_path(a_id))
        self._ls_update(a_idx, r)
        self.t += 1

        self.round_time_history.append(time.perf_counter() - t0)
        return False, top_ids



