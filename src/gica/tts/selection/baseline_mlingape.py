# baseline_mlingape.py  (test-time-scaling adaptation)
import math
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
from scipy.optimize import linprog


@dataclass
class _LPResult:
    """Result of the L1 representation LP used by the 'optimized' selection rule.

    Attributes:
        w: Signed coefficients expressing the gap direction (x_b - x_c) as a linear
            combination of the arm features.
        p: Normalized |w| (a distribution over arms); used only as a fallback.
        success: Whether the LP solved successfully.
        message: Solver status message.
    """
    w: np.ndarray
    p: np.ndarray
    success: bool
    message: str


class m_LinGapE:
    """m-LinGapE baseline for the test-time-scaling pipeline.

    Top-m form of LinGapE (Xu et al., 2018), as tabulated in Table 1 of the LinGIFA paper
    (Réda et al., 2021). In the GICA paper it is one of the linear-bandit baselines: each
    candidate reasoning path is an arm with feature ``g(pi)`` (the length-normalized mean
    of its step features), and pulling an arm yields a noisy verifier observation used to
    update a shared least-squares estimate of theta*. (This TTS variant is identical to the
    synthetic implementation; it is invoked by the TTS drivers through the PRM oracle.)

    The algorithm maintains the paired upper-gap index
        ``B_{i,j}(t) = (mu_hat_i - mu_hat_j) + C_{t,delta} * ||x_i - x_j||_{Sigma_hat}``,
    with ``Sigma_hat = sigma^2 V_t^{-1}``, and follows Table 1 each round:
      * ``J(t)``  = the top-m arms by ``mu_hat``;
      * ``b_t``   = ``argmax_{j in J} max_{i not in J} B_{i,j}`` (most contested shortlisted arm);
      * ``c_t``   = ``argmax_{i not in J} B_{i,b_t}`` (its hardest challenger);
      * stop when ``B_{c_t,b_t} <= epsilon``;
      * otherwise pull one arm chosen by ``selection_rule`` and update.

    Selection rules:
      * ``"largest_variance"`` : pull whichever of ``b_t, c_t`` has the larger marginal variance.
      * ``"greedy"``           : pull the arm that most reduces variance along ``x_{b_t} - x_{c_t}``.
      * ``"optimized"``        : pull the LinGapE optimal-allocation arm from the L1 representation
                                 of ``x_{b_t} - x_{c_t}`` (requires the LP solver).
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
        selection_rule: str = "largest_variance",
        init_each_arm_once: bool = False,
        seed: int = 0,
        lp_cache: bool = True,
    ):
        """
        Args:
            paths: Mapping ``arm_id -> list of step indices`` (an arm is a path).
            feature_matrix: Step features of shape ``(num_steps, d)``.
            m: Number of top arms to identify.
            d: Feature dimension.
            lambda_reg: Ridge regularization parameter ``lambda``.
            epsilon: Stopping tolerance on the paired gap index.
            delta: Target error probability (confidence level ``1 - delta``).
            R: Sub-Gaussian noise proxy ``sigma`` of the observations.
            S_0: Upper bound on ``||theta*||``.
            selection_rule: One of ``{"largest_variance", "greedy", "optimized"}``.
            init_each_arm_once: If True, pull every arm once before adaptive sampling.
            seed: RNG seed (used only for tie-breaking in the 'optimized' rule).
            lp_cache: Cache LP solutions per ``(b, c)`` pair in the 'optimized' rule.
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
        if self.selection_rule not in {"largest_variance", "greedy", "optimized"}:
            raise ValueError("selection_rule must be 'largest_variance', 'greedy', or 'optimized'")

        self.rng = np.random.default_rng(seed)

        # Composite (length-normalized) arm features g(pi) = mean of the path's steps.
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

        # Shared linear model state. V_0 = lambda I => V_0^{-1} = (1/lambda) I.
        self.V_inv = (1.0 / self.lambda_reg) * np.eye(self.d)
        self.b_vec = np.zeros(self.d, dtype=float)    # accumulated sum of reward * x
        self.theta_hat = np.zeros(self.d, dtype=float)
        self.counts = np.zeros(self.K, dtype=int)     # per-arm pull counts N_a(t)

        # Counters and logging traces.
        self.t = 0
        self.total_comparisons = 0
        self.best_G_history: List[float] = []      # B_{c_t,b_t} per round
        self.min_lcb_history: List[float] = []     # -B_{c_t,b_t} per round (stopping quantity)
        self.round_time_history: List[float] = []

        # Optional warm-up queue: pull each arm once before adaptive sampling.
        self._init_queue = list(range(self.K)) if init_each_arm_once else []

        # LP-solution cache for the 'optimized' selection rule.
        self._lp_cache_enabled = bool(lp_cache)
        self._lp_cache: Dict[Tuple[int, int], _LPResult] = {}

    def _C(self) -> float:
        """Confidence multiplier ``C_{t,delta}`` of LinGapE / m-LinGapE.

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

    def _compute_B_matrix(self):
        """Compute the full paired upper-gap-index matrix and supporting quantities.

        Returns:
            B (np.ndarray): ``B[i, j] = (mu_i - mu_j) + C * sigma * ||x_i - x_j||_{V^{-1}}``.
            mu (np.ndarray): Plug-in arm utilities ``X @ theta_hat``.
            C (float): Confidence multiplier from :meth:`_C`.
            diag (np.ndarray): Marginal variances ``x_a^T V^{-1} x_a`` (clipped at 0).
        """
        C = self._C()
        mu = self.X @ self.theta_hat

        # Gram matrix in the V^{-1} geometry: G[i,j] = x_i^T V^{-1} x_j.
        M = self.X @ self.V_inv
        G = M @ self.X.T
        diag = np.clip(np.diag(G), 0.0, None)

        # Pairwise variance ||x_i - x_j||^2_{V^{-1}} = diag_i + diag_j - 2 G_ij.
        diff_var = diag[:, None] + diag[None, :] - 2.0 * G
        diff_var = np.maximum(diff_var, 1e-12)

        W = self.sigma * np.sqrt(diff_var)
        B = (mu[:, None] - mu[None, :]) + C * W
        return B, mu, C, diag

    def _select_largest_variance(self, b_idx: int, c_idx: int, diag: np.ndarray) -> int:
        """Pull whichever of the two ambiguous arms has the larger marginal variance."""
        return int(b_idx if diag[b_idx] >= diag[c_idx] else c_idx)

    def _select_greedy(self, b_idx: int, c_idx: int) -> int:
        """Pull the arm that maximally reduces uncertainty along the ``x_b - x_c`` direction.

        Maximizes ``(x_a^T V^{-1} (x_b - x_c))^2 / (1 + x_a^T V^{-1} x_a)`` over arms ``a``.
        """
        diff = self.X[b_idx] - self.X[c_idx]
        v_inv_diff = self.V_inv @ diff

        xVx = np.sum(self.X * (self.X @ self.V_inv), axis=1)
        diffVx = self.X @ v_inv_diff

        denom = 1.0 + np.maximum(xVx, 0.0)
        score = (diffVx ** 2) / np.maximum(denom, 1e-12)
        return int(np.argmax(score))

    def _solve_l1_representation(self, diff: np.ndarray) -> _LPResult:
        """Express the gap direction ``diff = x_b - x_c`` as a min-L1 combination of arms.

        Solves ``min sum|w|`` s.t. ``X^T w = diff`` via the standard nonnegative split
        ``w = u - v`` (a linear program). Returns the signed weights ``w`` and their
        normalization ``p``; on solver failure returns ``success=False`` with a uniform
        fallback.
        """
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
        """LinGapE optimal-allocation rule for the ambiguous pair ``(b_idx, c_idx)``.

        Uses the L1 representation of ``x_b - x_c`` and, restricting to its positive
        support, pulls ``argmax_a N_a(t) * ||w||_1 / |w_a|``. Falls back to the greedy rule
        if the LP fails or has empty support.
        """
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

        # Restrict to arms with strictly positive weight in the L1 representation.
        support = np.where(w > 1e-12)[0]
        if support.size == 0:
            return self._select_greedy(b_idx, c_idx)

        # Optimal-allocation score: N_a(t) * ||w||_1 / |w_a|.
        denom = np.maximum(np.abs(w[support]), 1e-12)
        scores = self.counts[support] * l1 / denom

        # Tiny random perturbation to break exact ties deterministically per seed.
        scores = scores + 1e-12 * self.rng.standard_normal(scores.shape)

        return int(support[int(np.argmax(scores))])

    def select_and_update(self, oracle_callback_path: Callable[[int], float]):
        """Run one m-LinGapE round.

        Args:
            oracle_callback_path: ``arm_id -> scalar reward``; one verifier query per call.

        Returns:
            tuple[bool, list[int]]: ``(done, top_ids)`` where ``done`` is the stopping flag
            and ``top_ids`` is the current top-m shortlist (arm ids).
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

        # Paired upper-gap index over all arm pairs.
        B, mu, C, diag = self._compute_B_matrix()

        # Empirical top-m shortlist J(t) and its challenger set.
        order = np.argsort(mu)[::-1]
        top_idx = order[: self.m]
        rest_idx = order[self.m :]

        top_ids = [self.idx_to_id[int(i)] for i in top_idx]
        if rest_idx.size == 0:
            self.round_time_history.append(time.perf_counter() - t0)
            return True, top_ids

        self.total_comparisons += int(len(top_idx) * len(rest_idx))

        # b_t = argmax_{j in J} max_{i not in J} B_{i,j}.
        max_outside_for_j = np.max(B[rest_idx[:, None], top_idx[None, :]], axis=0)
        b_pos = int(np.argmax(max_outside_for_j))
        b_idx = int(top_idx[b_pos])

        # c_t = argmax_{i not in J} B_{i,b_t}.
        col = B[rest_idx, b_idx]
        c_idx = int(rest_idx[int(np.argmax(col))])

        # Stopping quantity B_{c_t,b_t}.
        B_cb = float(B[c_idx, b_idx])
        self.best_G_history.append(B_cb)
        self.min_lcb_history.append(-B_cb)
        print("B_cb",B_cb)
        if B_cb <= self.epsilon:
            self.round_time_history.append(time.perf_counter() - t0)
            return True, top_ids

        # Choose which arm to pull according to the configured selection rule.
        if self.selection_rule == "largest_variance":
            a_idx = self._select_largest_variance(b_idx, c_idx, diag)
        elif self.selection_rule == "greedy":
            a_idx = self._select_greedy(b_idx, c_idx)
        else:
            a_idx = self._select_optimized(b_idx, c_idx)

        # Query the chosen arm and update the shared linear model.
        a_id = self.idx_to_id[a_idx]
        r = float(oracle_callback_path(a_id))
        self._ls_update(a_idx, r)
        self.t += 1

        self.round_time_history.append(time.perf_counter() - t0)
        return False, top_ids
