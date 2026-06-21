# baseline_case.py
import time
import math
import numpy as np


class CASE:
    """CASE baseline for fixed-confidence top-m identification in linear bandits.

    Implementation of Algorithm 1 (CASE) of Purohit et al. (2025), adapted to the
    compositional setting of the GICA paper: each candidate reasoning path is an arm
    whose feature ``x_a = g(pi_a)`` is the length-normalized mean of its step features,
    and pulling an arm yields a noisy observation of its utility used to update a shared
    least-squares estimate of theta*.

    CASE maintains a top set ``U`` (current best-m estimate) and a sampled *challenger
    shortlist* ``N`` of size ``m'``. Each round it (i) optionally swaps the worst arm in
    ``U`` with the best arm in ``N``, (ii) resamples ``N`` from the complement, (iii)
    identifies the most ambiguous boundary pair ``(b in U, s in N)`` via the upper-gap
    index, and (iv) pulls the single arm that most reduces the pairwise uncertainty along
    ``x_b - x_s``. Restricting attention to the challenger shortlist (rather than all
    challengers) is what makes CASE's per-round cost smaller than full gap-index methods.

    The upper-gap index used throughout is
        ``B(a, b) = (mu_hat_a - mu_hat_b) + C_{t,delta} * (||x_a||_{Sigma_hat} + ||x_b||_{Sigma_hat})``,
    with ``Sigma_hat = sigma^2 V_t^{-1}``.

    Logging attributes consumed by the experiment runner: ``t``, ``m``,
    ``total_comparisons``, ``best_G_history``, ``min_lcb_history``, ``round_time_history``,
    and the model state ``V_inv`` / ``theta_hat``.
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
        R=0.1,          # sigma (sub-Gaussian noise proxy)
        S_0=2.0,        # bound on ||theta*||
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

        # For faithful Algorithm-1 behavior the per-round sample size equals m'.
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
        self.K = len(self.all_ids)

        # Stacked arm-feature matrix and id<->row-index maps (matrix form for speed).
        self.id_to_idx = {pid: i for i, pid in enumerate(self.all_ids)}
        self.idx_to_id = {i: pid for pid, i in self.id_to_idx.items()}
        self.X = np.stack([self.g_pi[pid] for pid in self.all_ids], axis=0)  # (K, d)

        # Feature-norm bound L = max_a ||x_a||_2.
        self.L = float(np.max(np.linalg.norm(self.X, axis=1)))

        # Regularized design-matrix inverse: V_0^{-1} = (lambda I)^{-1}.
        self.V_inv = (1.0 / self.lambda_reg) * np.eye(self.d)

        # Accumulated response vector b = sum_l r_l x_{a_l}.
        self.b_vec = np.zeros(self.d)

        # Initial parameter drawn from N(0, 1) (used only for the first scoring round).
        self.theta_hat = self.rng.normal(0.0, 1.0, size=self.d)

        # Per-arm pull counts N_a.
        self.counts = np.zeros(self.K, dtype=int)

        # Initialize the top set U_0 and the challenger shortlist N_0 at random.
        perm = self.rng.permutation(self.all_ids)
        self.U = set(int(x) for x in perm[: self.m])
        self.N = set(int(x) for x in perm[self.m: self.m + self.m_prime])

        # Current ambiguous boundary pair (b in U, s in N).
        self.b = None
        self.s = None

        # Counters and logging traces.
        self.t = 0  # number of pulls so far
        self.total_comparisons = 0
        self.round_time_history = []
        self.best_G_history = []
        self.min_lcb_history = []

        # Establish the initial ambiguous pair.
        mu_vec = self._mu_vec()
        C = self._C()
        norm_vec = self._norm_vec()
        self._recompute_ambiguous_arms(mu_vec=mu_vec, C=C, norm_vec=norm_vec)

    # ---------------- Confidence scalar ----------------

    def _C(self) -> float:
        """Confidence multiplier ``C_{t,delta}`` (shared with the LinGIFA / m-LinGapE baselines).

        ``C_{t,delta} = sqrt( 2 ln(1/delta) + d ln(1 + (t+1) L^2 / (lambda^2 d)) )
                        + (sqrt(lambda) / sigma) * S``.
        """
        term1 = 2.0 * math.log(1.0 / self.delta)
        term2 = self.d * math.log(
            1.0 + ((self.t + 1.0) * (self.L ** 2)) / ((self.lambda_reg ** 2) * self.d)
        )
        return math.sqrt(term1 + term2) + (math.sqrt(self.lambda_reg) / self.sigma) * self.S

    # ---------------- Fast mu / norm helpers ----------------

    def _mu_vec(self) -> np.ndarray:
        """Plug-in utilities ``mu[k] = x_k^T theta_hat`` for all arms."""
        return self.X @ self.theta_hat  # (K,)

    def _norm_vec(self) -> np.ndarray:
        """Per-arm confidence half-width ``||x_k||_{Sigma_hat} = sigma * sqrt(x_k^T V_inv x_k)``."""
        # diag = diag(X V_inv X^T) = sum_i x_i * (V_inv x_i).
        XV = self.X @ self.V_inv                  # (K, d)
        diag = np.einsum("ij,ij->i", self.X, XV)  # (K,)
        diag = np.clip(diag, 0.0, None)
        return self.sigma * np.sqrt(diag)

    # ---------------- Ambiguous-pair selection (Algorithm 1, lines 17-18) ----------------

    def _recompute_ambiguous_arms(self, *, mu_vec: np.ndarray, C: float, norm_vec: np.ndarray) -> None:
        """Set the ambiguous boundary pair ``self.b`` in ``U`` and ``self.s`` in ``N``.

        The paper defines (lines 17-18)::

            b = argmax_{b in U} max_{a in N} B(a, b)
            s = argmax_{s in N} B(s, b)

        With ``B(a, b) = mu[a] - mu[b] + C * (norm[a] + norm[b])`` the inner maximum over
        ``a`` is ``max_{a in N}(mu[a] + C*norm[a]) - (mu[b] - C*norm[b])``. The first term
        is independent of ``b``, so this reduces exactly (no approximation) to::

            b = argmin_{b in U} (mu[b] - C*norm[b])
            s = argmax_{s in N} (mu[s] + C*norm[s])

        which is computed in O(|U| + |N|) instead of O(|U| * |N|).
        """
        if not self.U or not self.N:
            raise RuntimeError("CASE: U or N is empty; cannot recompute ambiguous arms.")

        U_idx = np.fromiter((self.id_to_idx[a] for a in self.U), dtype=int)
        N_idx = np.fromiter((self.id_to_idx[a] for a in self.N), dtype=int)

        # b = argmin_{b in U} (mu[b] - C*norm[b]).
        scores_b = mu_vec[U_idx] - float(C) * norm_vec[U_idx]
        b_idx = int(U_idx[int(np.argmin(scores_b))])

        # s = argmax_{s in N} (mu[s] + C*norm[s]).
        scores_s = mu_vec[N_idx] + float(C) * norm_vec[N_idx]
        s_idx = int(N_idx[int(np.argmax(scores_s))])

        self.b = int(self.idx_to_id[b_idx])
        self.s = int(self.idx_to_id[s_idx])

    # ---------------- Pull-selection rule (Algorithm 1, line 20) ----------------

    def _selection_rule(self, b_id: int, s_id: int) -> int:
        """Pull the arm that most contracts the variance along ``x_b - x_s``.

        Implements ``a* = argmin_{a in U union N} ||x_b - x_s||^2_{(V + x_a x_a^T)^{-1}}``,
        evaluated via Sherman-Morrison as::

            score(a) = diff^T V_inv diff - (diff^T V_inv x_a)^2 / (1 + x_a^T V_inv x_a),

        where ``diff = x_b - x_s``. The constant first term ``base`` is shared across
        candidates; minimizing ``score`` maximizes the subtracted reduction.
        """
        b_idx = self.id_to_idx[int(b_id)]
        s_idx = self.id_to_idx[int(s_id)]
        diff = self.X[b_idx] - self.X[s_idx]
        base = float(diff.T @ self.V_inv @ diff)

        cand_ids = list(self.U.union(self.N))

        best_a = None
        best_score = float("inf")
        for a_id in cand_ids:
            a_idx = self.id_to_idx[int(a_id)]
            x = self.X[a_idx]
            v_inv_x = self.V_inv @ x
            denom = 1.0 + float(x.T @ v_inv_x)
            num = float(diff.T @ v_inv_x)
            score = base - (num * num) / denom
            if score < best_score:
                best_score = score
                best_a = int(a_id)

        return int(best_a)

    # ---------------- Least-squares update (Algorithm 1, lines 22-23) ----------------

    def _ls_update(self, arm_id: int, reward: float) -> None:
        """Pull arm ``arm_id`` with observed ``reward`` and update the linear model.

        Applies ``V <- V + x x^T`` (rank-one Sherman-Morrison update of ``V_inv``),
        accumulates ``b_vec <- b_vec + reward * x``, recomputes ``theta_hat = V_inv b_vec``,
        and increments the arm's pull count.
        """
        arm_id = int(arm_id)
        a_idx = self.id_to_idx[arm_id]
        x = self.X[a_idx]

        v_inv_x = self.V_inv @ x
        denom = 1.0 + float(x.T @ v_inv_x)
        self.V_inv = self.V_inv - np.outer(v_inv_x, v_inv_x) / denom

        self.b_vec = self.b_vec + float(reward) * x
        self.theta_hat = self.V_inv @ self.b_vec

        self.counts[a_idx] += 1

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

        # Compute the round's utilities, confidence multiplier and half-widths once.
        mu_vec = self._mu_vec()
        C = self._C()
        norm_vec = self._norm_vec()

        b_idx = self.id_to_idx[int(self.b)]
        s_idx = self.id_to_idx[int(self.s)]

        # Stopping quantity B(s, b) = mu[s] - mu[b] + C*(norm[s] + norm[b]).
        B_sb = float((mu_vec[s_idx] - mu_vec[b_idx]) + C * (norm_vec[s_idx] + norm_vec[b_idx]))

        self.best_G_history.append(B_sb)

        # LCB-style trace: (mu[b] - mu[s]) - C*(norm[b] + norm[s]).
        gap_bs = float(mu_vec[b_idx] - mu_vec[s_idx])
        W_bs = float(C * (norm_vec[b_idx] + norm_vec[s_idx]))
        self.min_lcb_history.append(gap_bs - W_bs)

        # Current output estimate: U ordered by plug-in utility (best first).
        top_ids = sorted(self.U, key=lambda a: float(mu_vec[self.id_to_idx[int(a)]]), reverse=True)

        if B_sb <= self.epsilon:
            self.round_time_history.append(time.perf_counter() - t0)
            return True, top_ids

        # Lines 9-13: swap the worst arm in U with the best arm in N if the latter is better.
        n_t = min(self.U, key=lambda a: float(mu_vec[self.id_to_idx[int(a)]]))  # worst in U
        c_t = max(self.N, key=lambda a: float(mu_vec[self.id_to_idx[int(a)]]))  # best in N
        if float(mu_vec[self.id_to_idx[int(c_t)]]) >= float(mu_vec[self.id_to_idx[int(n_t)]]):
            self.U.remove(int(n_t))
            self.U.add(int(c_t))
            self.N.remove(int(c_t))
            self.N.add(int(n_t))

        # Line 14: sample M_t uniformly from the complement (U union N)^c.
        excluded = self.U.union(self.N)
        complement = [a for a in self.all_ids if a not in excluded]
        if complement:
            ssz = min(self.sample_m_prime, len(complement))
            M_t = list(self.rng.choice(complement, size=ssz, replace=False))
        else:
            M_t = []

        # Line 15: refresh the challenger shortlist N_t <- top-m' of (M_t union N) by utility.
        candidates = list(set(M_t).union(self.N))
        candidates_sorted = sorted(candidates, key=lambda a: float(mu_vec[self.id_to_idx[int(a)]]), reverse=True)
        self.N = set(int(x) for x in candidates_sorted[: self.m_prime])

        # Lines 17-18: recompute the ambiguous pair (reusing this round's mu / C / norm).
        self.total_comparisons += len(self.U) * len(self.N) + len(self.N)
        self._recompute_ambiguous_arms(mu_vec=mu_vec, C=C, norm_vec=norm_vec)

        # Line 20: choose the arm to pull.
        a_next = self._selection_rule(self.b, self.s)

        # Line 21: query its reward.
        r_next = float(oracle_callback_path(a_next))

        # Lines 22-23: update the shared linear model.
        self._ls_update(a_next, r_next)

        self.t += 1
        self.round_time_history.append(time.perf_counter() - t0)
        return False, top_ids
