# baseline_case.py  (test-time-scaling adaptation)
import time
import math
import numpy as np


class CASE:
    """CASE baseline for the test-time-scaling pipeline.

    Implementation of Algorithm 1 (CASE) of Purohit et al. (2025), adapted to the
    compositional setting of the GICA paper: each candidate reasoning path is an arm whose
    feature ``x_a = g(pi_a)`` is the length-normalized mean of its step features, and
    pulling an arm yields a noisy verifier observation used to update a shared
    least-squares estimate of theta*.

    CASE maintains a top set ``U`` (current best-m estimate, size ``m``) and a sampled
    *challenger shortlist* ``N`` (size ``m'``). Each round it (i) optionally swaps the worst
    arm in ``U`` with the best arm in ``N``, (ii) resamples ``N`` from the complement, (iii)
    identifies the most ambiguous boundary pair ``(b in U, s in N)`` via the upper-gap
    index, and (iv) pulls the arm that most reduces the pairwise uncertainty along
    ``x_b - x_s``. Restricting attention to the challenger shortlist (rather than all
    challengers) is what makes CASE's per-round cost smaller than full gap-index methods.

    Gap index:
        ``B_t(i, j) = rho_hat_t(i) - rho_hat_t(j) + W_t(i, j)``,
        ``W_t(i, j) = C_{t,delta} * (||x_i||_{Sigma_hat} + ||x_j||_{Sigma_hat})``,
        ``Sigma_hat = sigma^2 * V_t^{-1}``.

    Selection rule (Réda et al., 2021):
        ``a_{t+1} = argmin_{a in U union N} ||x_b - x_s||_{(V_t + x_a x_a^T)^{-1}}``.

    Public API:
        ``select_and_update(oracle_callback_path) -> (done: bool, top_ids: list[int])``.

    Logging attributes consumed by the experiment runner: ``t``, ``m``,
    ``total_comparisons``, ``best_G_history`` (``B_t(s_t, b_t)`` per round),
    ``min_lcb_history`` (``(rho_hat(b_t) - rho_hat(s_t)) - W_t(b_t, s_t)``), and
    ``round_time_history``.

    Note:
        This TTS variant keeps the arms in dict form keyed by path id and recomputes the
        ambiguous pair with explicit loops; it also uses a confidence scalar based on the
        running pull count ``N = t`` (see :meth:`_C`). It is therefore a distinct, slower
        implementation from the vectorized synthetic ``CASE`` and is intentionally kept
        as-is for the TTS experiments.
    """

    def __init__(
        self,
        paths,
        feature_matrix,
        m,
        d,
        lambda_reg=1.0,
        epsilon=0.05,
        delta=0.1,
        R=0.1,          # sigma
        S_0=2.0,        # S
        challenger_size=5,   # m' : size of the challenger shortlist N_t
        challenger_batch=5,  # number of arms sampled per round (kept equal to m')
        seed=0,
    ):
        """
        Args:
            paths: Mapping ``arm_id -> list of step indices`` (an arm is a path).
            feature_matrix: Step features of shape ``(num_steps, d)``.
            m: Number of top arms to identify.
            d: Feature dimension.
            lambda_reg: Ridge regularization parameter ``lambda``.
            epsilon: Stopping tolerance on the upper-gap index.
            delta: Target error probability (confidence level ``1 - delta``).
            R: Sub-Gaussian noise proxy ``sigma`` of the observations.
            S_0: Upper bound on ``||theta*||``.
            challenger_size: The shortlist size ``m'`` (size of ``N_t``).
            challenger_batch: Number of arms sampled from the complement each round;
                enforced equal to ``m'`` for faithful Algorithm-1 behavior.
            seed: RNG seed (controls shortlist sampling and the initial theta).
        """
        self.paths = paths
        self.features = feature_matrix
        self.m = int(m)
        self.d = int(d)

        # Algorithm parameters / constants.
        self.lambda_reg = float(lambda_reg)
        self.epsilon = float(epsilon)
        self.delta = float(delta)

        self.sigma = float(R)   # sigma
        self.S = float(S_0)     # S

        # Challenger shortlist size m'.
        self.m_prime = int(challenger_size)

        # For faithful Algorithm-1 behavior the per-round sample size equals m'
        # (kept as a separate argument for backward compatibility).
        self.sample_m_prime = int(challenger_batch)

        self.rng = np.random.default_rng(seed)

        if self.m <= 0 or self.m_prime <= 0:
            raise ValueError("m and challenger_size (m') must be >= 1")
        if self.m + self.m_prime > len(paths):
            raise ValueError("Need at least m + m' distinct arms")
        if self.sample_m_prime != self.m_prime:
            # Enforce the paper's behavior (sample size equals m').
            self.sample_m_prime = self.m_prime

        # Arm features x_a = g(pi_a) = length-normalized mean of the path's step features.
        self.g_pi = {}
        for pid, step_indices in self.paths.items():
            self.g_pi[int(pid)] = np.mean(self.features[step_indices], axis=0).astype(float)

        self.all_ids = sorted(self.g_pi.keys())

        # Feature-norm bound L = max_a ||x_a||_2.
        self.L = float(max(np.linalg.norm(self.g_pi[pid]) for pid in self.all_ids))

        # Regularized design-matrix inverse: V_0^{-1} = (lambda I)^{-1}.
        self.V_inv = (1.0 / self.lambda_reg) * np.eye(self.d)

        # Accumulated response vector b = sum_l r_l x_{a_l}.
        self.b_vec = np.zeros(self.d)

        # Initial parameter drawn from N(0, 1) (used only for the first scoring round).
        self.theta_hat = self.rng.normal(0.0, 1.0, size=self.d)

        # Per-arm pull counts N_a (keyed by arm id).
        self.counts = {pid: 0 for pid in self.all_ids}

        # Initialize the top set U_0 and the challenger shortlist N_0 at random.
        perm = self.rng.permutation(self.all_ids)
        self.U = set(int(x) for x in perm[: self.m])
        self.N = set(int(x) for x in perm[self.m : self.m + self.m_prime])

        # Current ambiguous boundary pair (b in U, s in N).
        self.b = None
        self.s = None

        # Counters and logging traces.
        self.t = 0  # number of pulls so far
        self.total_comparisons = 0
        self.round_time_history = []
        self.best_G_history = []
        self.min_lcb_history = []

        # Establish the initial ambiguous pair (b_1, s_1) for the stopping criterion.
        self._recompute_ambiguous_arms()

    # ---------------- Confidence scalar and gap index ----------------

    def _C(self):
        """Confidence scalar ``C_{t,delta}`` for this TTS variant.

        ``C_{t,delta} = sqrt( 2 ln(1/delta) + N ln(1 + (t+1) L^2 / (lambda^2 N)) )
                        + (sqrt(lambda) / sigma) * S``,
        where ``N = t`` is the number of pulls so far (the second term is dropped while
        ``N == 0``).
        """
        N = self.t
        base = 2.0 * math.log(1.0 / self.delta)
        if N <= 0:
            second = 0.0
        else:
            second = N * math.log(
                1.0 + ((self.t + 1.0) * (self.L ** 2)) / ((self.lambda_reg ** 2) * N)
            )
        return math.sqrt(base + second) + (math.sqrt(self.lambda_reg) / self.sigma) * self.S

    def _rho_hat_all(self):
        """Plug-in utilities ``rho_hat(a) = x_a^T theta_hat`` for every arm."""
        return {pid: float(self.g_pi[pid] @ self.theta_hat) for pid in self.all_ids}

    def _sigma_norm(self, pid):
        """Confidence half-width ``||x_a||_{Sigma_hat} = sigma * sqrt(x_a^T V_inv x_a)``."""
        x = self.g_pi[pid]
        v = float(x.T @ self.V_inv @ x)
        v = max(v, 0.0)
        return self.sigma * math.sqrt(v)

    def _B(self, i, j, mu, norm, C):
        """Upper-gap index ``B_t(i, j) = rho_hat(i) - rho_hat(j) + C*(||x_i|| + ||x_j||)``."""
        return ((mu[i] - mu[j]) + C * (norm[i] + norm[j]))

    # ---------------- Ambiguous-pair selection (Algorithm 1, lines 17-18) ----------------

    def _recompute_ambiguous_arms(self):
        """Set the ambiguous boundary pair ``self.b`` in ``U`` and ``self.s`` in ``N``.

        Computes (lines 17-18)::

            b = argmax_{b in U} max_{a in N} B(a, b)
            s = argmax_{s in N} B(s, b)

        directly via nested loops over the current ``U`` and ``N``.
        """
        mu = self._rho_hat_all()
        C = self._C()
        involved = self.U.union(self.N)
        norm = {pid: self._sigma_norm(pid) for pid in involved}

        # b = argmax_{b in U} max_{a in N} B(a, b).
        best_b = None
        best_b_val = -float("inf")
        for b in self.U:
            inner = -float("inf")
            for a in self.N:
                inner = max(inner, self._B(a, b, mu, norm, C))
            if inner > best_b_val:
                best_b_val = inner
                best_b = b

        # s = argmax_{s in N} B(s, b).
        best_s = None
        best_s_val = -float("inf")
        for s in self.N:
            val = self._B(s, best_b, mu, norm, C)
            if val > best_s_val:
                best_s_val = val
                best_s = s

        self.b = int(best_b)
        self.s = int(best_s)

    # ---------------- Pull-selection rule (Algorithm 1, line 20) ----------------

    def _selection_rule(self, b, s):
        """Pull the arm that most contracts the variance along ``x_b - x_s``.

        Implements ``a* = argmin_{a in U union N} ||x_b - x_s||^2_{(V + x_a x_a^T)^{-1}}``.
        Via Sherman-Morrison on ``V_inv``,

            score(a) = diff^T V_inv diff - (diff^T V_inv x_a)^2 / (1 + x_a^T V_inv x_a),

        with ``diff = x_b - x_s``; the constant first term ``base`` is shared across
        candidates, so minimizing ``score`` maximizes the subtracted reduction.
        """
        diff = self.g_pi[b] - self.g_pi[s]
        base = float(diff.T @ self.V_inv @ diff)

        best_a = None
        best_score = float("inf")
        for a in self.U.union(self.N):
            x = self.g_pi[a]
            v_inv_x = self.V_inv @ x
            denom = 1.0 + float(x.T @ v_inv_x)
            num = float(diff.T @ v_inv_x)
            score = base - (num * num) / denom
            if score < best_score:
                best_score = score
                best_a = a

        return int(best_a)

    # ---------------- Least-squares update (Algorithm 1, lines 22-23) ----------------

    def _ls_update(self, arm_id, reward):
        """Pull arm ``arm_id`` with observed ``reward`` and update the linear model.

        Applies ``V <- V + x x^T`` (rank-one Sherman-Morrison update of ``V_inv``),
        accumulates ``b_vec <- b_vec + reward * x``, recomputes ``theta_hat = V_inv b_vec``,
        and increments the arm's pull count.
        """
        x = self.g_pi[arm_id]
        v_inv_x = self.V_inv @ x
        denom = 1.0 + float(x.T @ v_inv_x)
        self.V_inv = self.V_inv - np.outer(v_inv_x, v_inv_x) / denom

        self.b_vec = self.b_vec + float(reward) * x
        self.theta_hat = self.V_inv @ self.b_vec

        self.counts[arm_id] += 1

    # ---------------- Main iteration ----------------

    def select_and_update(self, oracle_callback_path):
        """Run one CASE iteration (one arm pull + model update).

        Args:
            oracle_callback_path: ``arm_id -> scalar reward``; one verifier query per call.

        Returns:
            tuple[bool, list[int]]: ``(done, top_ids)`` where ``done`` is the stopping flag
            and ``top_ids`` is the current top set ``U`` ordered by plug-in utility.
        """
        t0 = time.perf_counter()

        mu = self._rho_hat_all()
        C = self._C()
        involved = self.U.union(self.N)
        norm = {pid: self._sigma_norm(pid) for pid in involved}

        # Stopping quantity uses the current ambiguous pair (s_t, b_t).
        B_sb = self._B(self.s, self.b, mu, norm, C)

        # Logging traces.
        self.best_G_history.append(float(B_sb))
        W_bs = C * (norm[self.b] + norm[self.s])
        gap_bs = mu[self.b] - mu[self.s]
        self.min_lcb_history.append(float(gap_bs - W_bs))

        # Current output estimate: U ordered by plug-in utility (best first).
        top_ids = sorted(self.U, key=lambda a: mu[a], reverse=True)

        # Stop when B_t(s_t, b_t) <= epsilon.
        print("B_sb", B_sb)
        if B_sb <= self.epsilon:
            self.round_time_history.append(time.perf_counter() - t0)
            return True, top_ids

        # Lines 9-13: swap the worst arm in U with the best arm in N if the latter is better.
        n_t = min(self.U, key=lambda a: mu[a])  # worst in U
        c_t = max(self.N, key=lambda a: mu[a])  # best in N
        if mu[c_t] >= mu[n_t]:
            self.U.remove(n_t)
            self.U.add(c_t)
            self.N.remove(c_t)
            self.N.add(n_t)

        # Line 14: sample M_t uniformly from the complement (U union N)^c.
        excluded = self.U.union(self.N)
        complement = [a for a in self.all_ids if a not in excluded]
        if len(complement) > 0:
            ssz = min(self.sample_m_prime, len(complement))
            M_t = list(self.rng.choice(complement, size=ssz, replace=False))
        else:
            M_t = []

        # Line 15: refresh the challenger shortlist N_t <- top-m' of (M_t union N) by utility.
        candidates = list(set(M_t).union(self.N))
        candidates_sorted = sorted(candidates, key=lambda a: mu[a], reverse=True)
        self.N = set(candidates_sorted[: self.m_prime])

        # Lines 17-18: recompute the ambiguous pair (b_{t+1}, s_{t+1}).
        self.total_comparisons += len(self.U) * len(self.N) + len(self.N)
        self._recompute_ambiguous_arms()

        # Line 20: choose the arm to pull (depends on the current b, s).
        a_next = self._selection_rule(self.b, self.s)

        # Line 21: query its reward.
        r_next = float(oracle_callback_path(a_next))

        # Lines 22-23: update the shared linear model.
        self._ls_update(a_next, r_next)

        self.t += 1
        self.round_time_history.append(time.perf_counter() - t0)
        return False, top_ids
