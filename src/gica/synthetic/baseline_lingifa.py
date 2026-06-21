# baseline_lingifa.py
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np


@dataclass
class _LPResult:
    """Lightweight container kept for parity with the m-LinGapE baseline.

    LinGIFA does not solve a linear program, so this type is unused here; it is retained
    so the two baseline modules share the same auxiliary interface.
    """
    w: np.ndarray
    p: np.ndarray
    success: bool
    message: str


class LinGIFA:
    """LinGIFA baseline for fixed-confidence top-m identification in linear bandits.

    Concrete instance of the GIFA (Gap-Index Focused Allocation) framework of
    Réda et al. (2021), Table 1. In the GICA paper it serves as a linear-bandit baseline:
    each candidate reasoning path is an arm with feature ``g(pi)`` (the length-normalized
    mean of its step features), and pulling an arm yields a noisy observation of its
    utility used to update a shared least-squares estimate of theta*.

    It uses the paired upper-gap index
        ``B_{i,j}(t) = (mu_hat_i - mu_hat_j) + C_{t,delta} * ||x_i - x_j||_{Sigma_hat}``,
    with ``Sigma_hat = sigma^2 V_t^{-1}``, and follows Table 1 each round:
      * per-arm score ``score_j = m-th greatest_{i != j} B_{i,j}(t)``;
      * shortlist ``J(t)`` = the ``m`` arms with the smallest ``score_j``;
      * ``b_t = argmax_{j in J(t)} score_j`` (the least-separated shortlisted arm);
      * challenger ``c_t = argmax_{a not in J(t)} B_{a,b_t}(t)``;
      * stop when ``max_{j in J(t)} score_j <= epsilon``;
      * otherwise pull one arm chosen by ``selection_rule`` and update.

    Selection rules:
      * ``"largest_variance"`` : pull whichever of ``b_t, c_t`` has the larger marginal variance.
      * ``"greedy"``           : pull the arm that most reduces variance along ``x_{b_t} - x_{c_t}``.

    Logging attributes consumed by the experiment runner: ``t``, ``m``,
    ``total_comparisons``, ``best_G_history`` (the stopping quantity per round),
    ``min_lcb_history`` (its negation), and ``round_time_history``.
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
        R: float = 0.1,      # sigma (sub-Gaussian noise proxy)
        S_0: float = 2.0,    # bound on ||theta*||
        selection_rule: str = "largest_variance",  # "largest_variance" | "greedy"
        init_each_arm_once: bool = False,
        seed: int = 0,
    ):
        """
        Args:
            paths: Mapping ``arm_id -> list of step indices`` (an arm is a path).
            feature_matrix: Step features of shape ``(num_steps, d)``.
            m: Number of top arms to identify.
            d: Feature dimension.
            lambda_reg: Ridge regularization parameter ``lambda``.
            epsilon: Stopping tolerance on the gap-index score.
            delta: Target error probability (confidence level ``1 - delta``).
            R: Sub-Gaussian noise proxy ``sigma`` of the observations.
            S_0: Upper bound on ``||theta*||``.
            selection_rule: One of ``{"largest_variance", "greedy"}``.
            init_each_arm_once: If True, pull every arm once before adaptive sampling.
            seed: RNG seed (reserved; not used by the deterministic selection rules).
        """
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

        # Arm features: x_a = g(pi_a) = length-normalized mean of the path's step features.
        self.g_pi: Dict[int, np.ndarray] = {}
        for pid, step_idx in self.paths.items():
            self.g_pi[int(pid)] = np.mean(self.features[step_idx], axis=0).astype(float)

        self.all_ids: List[int] = sorted(self.g_pi.keys())
        self.K = len(self.all_ids)
        if not (1 <= self.m < self.K):
            raise ValueError("Need 1 <= m < number of arms")

        # Stacked arm-feature matrix and id<->row-index maps.
        self.X = np.stack([self.g_pi[pid] for pid in self.all_ids], axis=0)  # (K, d)
        self.id_to_idx = {pid: i for i, pid in enumerate(self.all_ids)}
        self.idx_to_id = {i: pid for pid, i in self.id_to_idx.items()}

        self.L = float(np.max(np.linalg.norm(self.X, axis=1)))  # feature-norm bound

        # Shared recursive-least-squares state. V_0 = lambda I => V_0^{-1} = (1/lambda) I.
        self.V_inv = (1.0 / self.lambda_reg) * np.eye(self.d)
        self.b_vec = np.zeros(self.d, dtype=float)    # accumulated sum of reward * x
        self.theta_hat = np.zeros(self.d, dtype=float)
        self.counts = np.zeros(self.K, dtype=int)     # per-arm pull counts

        # Counters and logging traces.
        self.t = 0
        self.total_comparisons = 0
        self.best_G_history: List[float] = []      # stopping quantity max_{j in J} score_j
        self.min_lcb_history: List[float] = []     # its negation (stopping-margin convention)
        self.round_time_history: List[float] = []

        # Optional warm-up queue: pull each arm once before adaptive sampling.
        self._init_queue = list(range(self.K)) if init_each_arm_once else []

    def _C(self) -> float:
        """Confidence multiplier ``C_{t,delta}`` (Réda et al., 2021, Lemma 3 / Eq. (2)).

        ``C_{t,delta} = sqrt( 2 ln(1/delta) + d ln(1 + (t+1) L^2 / (lambda^2 d)) )
                        + (sqrt(lambda) / sigma) * S``.
        """
        term1 = 2.0 * math.log(1.0 / self.delta)
        term2 = self.d * math.log(1.0 + ((self.t + 1.0) * (self.L ** 2)) / ((self.lambda_reg ** 2) * self.d))
        return math.sqrt(term1 + term2) + (math.sqrt(self.lambda_reg) / self.sigma) * self.S

    def _ls_update(self, arm_idx: int, reward: float) -> None:
        """Pull arm ``arm_idx`` with observed ``reward`` and update the linear model.

        Applies the Sherman-Morrison rank-one update to ``V_inv``, accumulates the
        response vector ``b_vec``, recomputes ``theta_hat = V_inv b_vec``, and increments
        the arm's pull count.
        """
        x = self.X[arm_idx]
        v_inv_x = self.V_inv @ x
        denom = 1.0 + float(x.T @ v_inv_x)
        self.V_inv = self.V_inv - np.outer(v_inv_x, v_inv_x) / denom

        self.b_vec = self.b_vec + float(reward) * x
        self.theta_hat = self.V_inv @ self.b_vec
        self.counts[arm_idx] += 1

    def _compute_B_matrix(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Compute the paired upper-gap-index matrix and supporting quantities.

        Returns:
            B (np.ndarray): ``B[i, j] = (mu_i - mu_j) + C * sigma * ||x_i - x_j||_{V^{-1}}``.
            mu (np.ndarray): Plug-in arm utilities ``X @ theta_hat``.
            C (float): Confidence multiplier from :meth:`_C`.
        """
        C = self._C()
        mu = self.X @ self.theta_hat  # (K,)

        # Gram matrix in the V^{-1} geometry: G[i,j] = x_i^T V^{-1} x_j.
        M = self.X @ self.V_inv
        G = M @ self.X.T
        diag = np.clip(np.diag(G), 0.0, None)

        # Pairwise variance ||x_i - x_j||^2_{V^{-1}} = diag_i + diag_j - 2 G_ij.
        diff_var = diag[:, None] + diag[None, :] - 2.0 * G
        diff_var = np.maximum(diff_var, 1e-12)

        W = self.sigma * np.sqrt(diff_var)
        B = (mu[:, None] - mu[None, :]) + C * W
        return B, mu, C

    def _select_largest_variance(self, b_idx: int, c_idx: int) -> int:
        """Pull whichever of the two ambiguous arms has the larger marginal variance."""
        xb = self.X[b_idx]
        xc = self.X[c_idx]
        vb = float(xb.T @ self.V_inv @ xb)
        vc = float(xc.T @ self.V_inv @ xc)
        return int(b_idx if vb >= vc else c_idx)

    def _select_greedy(self, b_idx: int, c_idx: int) -> int:
        """Pull the arm that maximally reduces uncertainty along the ``x_b - x_c`` direction.

        Maximizes ``(x_a^T V^{-1} (x_b - x_c))^2 / (1 + x_a^T V^{-1} x_a)`` over arms ``a``.
        """
        diff = self.X[b_idx] - self.X[c_idx]
        v_inv_diff = self.V_inv @ diff

        xVx = np.sum(self.X * (self.X @ self.V_inv), axis=1)  # (K,)
        diffVx = self.X @ v_inv_diff                           # (K,)

        denom = 1.0 + np.maximum(xVx, 0.0)
        score = (diffVx ** 2) / np.maximum(denom, 1e-12)
        return int(np.argmax(score))

    def select_and_update(self, oracle_callback_path: Callable[[int], float]):
        """Run one LinGIFA round.

        Args:
            oracle_callback_path: ``arm_id -> scalar reward``; one verifier query per call.

        Returns:
            tuple[bool, list[int]]: ``(done, top_ids)`` where ``done`` is the stopping flag
            and ``top_ids`` is the current top-m shortlist (arm ids, best utility first).
        """
        t0 = time.perf_counter()

        # Optional warm-up phase: pull each arm exactly once.
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

        # Paired upper-gap index; exclude self-pairs (i == j) from the per-column reduction.
        B, mu, C = self._compute_B_matrix()
        np.fill_diagonal(B, -np.inf)

        # Per-arm score_j = m-th greatest_{i != j} B_{i,j}(t).
        # With the diagonal at -inf, the (K - m)-th smallest value down each column equals
        # the m-th largest, computed in O(K) per column via np.partition.
        m = self.m
        kth = self.K - m              # index of the m-th largest in ascending order
        score = np.partition(B, kth, axis=0)[kth, :]  # shape (K,)

        # Shortlist J(t) = the m arms with the smallest score_j.
        J_idx = np.argsort(score)[: self.m]
        J_set = set(int(i) for i in J_idx)

        # Stopping quantity = max_{j in J} score_j.
        stop_val = float(np.max(score[J_idx]))
        self.best_G_history.append(stop_val)
        self.min_lcb_history.append(-stop_val)

        # Report the shortlist ordered by plug-in utility (best first).
        top_ids = [self.idx_to_id[int(i)] for i in sorted(J_idx, key=lambda k: mu[k], reverse=True)]

        if stop_val <= self.epsilon:
            self.round_time_history.append(time.perf_counter() - t0)
            return True, top_ids

        # b_t = argmax_{j in J(t)} score_j (the least-separated shortlisted arm).
        b_idx = int(J_idx[int(np.argmax(score[J_idx]))])

        # Challenger c_t = argmax_{a not in J(t)} B_{a,b_t}(t).
        outside = np.array([i for i in range(self.K) if i not in J_set], dtype=int)
        c_idx = int(outside[int(np.argmax(B[outside, b_idx]))])

        # Choose which arm to pull according to the configured selection rule.
        if self.selection_rule == "largest_variance":
            a_idx = self._select_largest_variance(b_idx, c_idx)
        else:
            a_idx = self._select_greedy(b_idx, c_idx)

        # Each round inspects all ordered arm pairs to build the gap-index scores.
        self.total_comparisons += self.K * (self.K - 1)

        # Query the chosen arm and update the shared linear model.
        a_id = self.idx_to_id[a_idx]
        r = float(oracle_callback_path(a_id))
        self._ls_update(a_idx, r)
        self.t += 1

        self.round_time_history.append(time.perf_counter() - t0)
        return False, top_ids
