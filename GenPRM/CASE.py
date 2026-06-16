# CASE.py
import time
import math
import numpy as np


class CASE:
    """
    Faithful implementation of Algorithm 1 (CASE), adapted to "paths" (arms).

    This drop-in version fixes the main runtime issue in your previous CASE.py:
      - It avoids recomputing (mu, C, norms) twice per iteration.
      - It also removes the O(m*m') nested loop in _recompute_ambiguous_arms by using an
        algebraically equivalent O(m+m') computation (exact, not an approximation).

    Arms: path IDs (ints). Feature for arm a is x_a = g_pi[a] (mean feature over steps).

    Logged fields (kept compatible with plot_metrics.py expectations):
      - t
      - m
      - total_comparisons
      - best_G_history
      - min_lcb_history
      - round_time_history
      - V_inv, theta_hat
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
        R=0.1,          # paper's sigma
        S_0=2.0,        # paper's S
        challenger_size=5,   # paper's m' (size of N_t)
        challenger_batch=5,  # paper samples m' arms; kept for compatibility
        seed=0,
    ):
        self.paths = paths
        self.features = feature_matrix
        self.m = int(m)
        self.d = int(d)

        # Paper parameters / constants
        self.lambda_reg = float(lambda_reg)
        self.epsilon = float(epsilon)
        self.delta = float(delta)

        self.sigma = float(R)   # σ
        self.S = float(S_0)     # S

        # CASE m' (paper uses m')
        self.m_prime = int(challenger_size)

        # For exact Algorithm 1 behavior, sampling size equals m'
        self.sample_m_prime = int(challenger_batch)

        self.rng = np.random.default_rng(seed)

        if self.m <= 0 or self.m_prime <= 0:
            raise ValueError("m and challenger_size (m') must be >= 1")
        if self.m + self.m_prime > len(paths):
            raise ValueError("Need at least m + m' distinct arms")
        if self.sample_m_prime != self.m_prime:
            # Enforce paper behavior (sample size equals m')
            self.sample_m_prime = self.m_prime

        # ---- precompute arm feature vectors x_a = g_pi[a] ----
        self.g_pi = {}
        for pid, step_indices in self.paths.items():
            self.g_pi[int(pid)] = np.mean(self.features[step_indices], axis=0).astype(float)

        self.all_ids = sorted(self.g_pi.keys())
        self.K = len(self.all_ids)

        # Matrix form for speed: X[k] = g_pi[all_ids[k]]
        self.id_to_idx = {pid: i for i, pid in enumerate(self.all_ids)}
        self.idx_to_id = {i: pid for pid, i in self.id_to_idx.items()}
        self.X = np.stack([self.g_pi[pid] for pid in self.all_ids], axis=0)  # (K, d)

        # Paper constant L: upper bound on ||x_a||_2
        self.L = float(np.max(np.linalg.norm(self.X, axis=1)))

        # ---- Regularized design matrix inverse: V_0^{-1} = (lambda I)^{-1} ----
        self.V_inv = (1.0 / self.lambda_reg) * np.eye(self.d)

        # b_vec = Σ r_l x_{a_l}
        self.b_vec = np.zeros(self.d)

        # Paper initializes alpha_1 ~ N(0,1) (used for initial scoring)
        self.theta_hat = self.rng.normal(0.0, 1.0, size=self.d)

        # counts N_a (keep both dict-like access via ids and fast array)
        self.counts = np.zeros(self.K, dtype=int)

        # ---- Initialize U_0 and N_0 randomly ----
        perm = self.rng.permutation(self.all_ids)
        self.U = set(int(x) for x in perm[: self.m])
        self.N = set(int(x) for x in perm[self.m: self.m + self.m_prime])

        # current ambiguous pair (b_t in U, s_t in N)
        self.b = None
        self.s = None

        # ---- counters / logs ----
        self.t = 0  # number of pulls so far
        self.total_comparisons = 0
        self.round_time_history = []
        self.best_G_history = []
        self.min_lcb_history = []

        # initial ambiguous pair
        mu_vec = self._mu_vec()
        C = self._C()
        norm_vec = self._norm_vec()
        self._recompute_ambiguous_arms(mu_vec=mu_vec, C=C, norm_vec=norm_vec)

    # ---------------- Confidence scalar ----------------

    def _C(self) -> float:
        """
        Bound consistent with your GIFA/m-LinGapE implementations:
          C_{t,δ} = sqrt( 2 ln(1/δ) + d ln( 1 + ((t+1)L^2)/(λ^2 d) ) ) + (sqrt(λ)/σ) S
        """
        term1 = 2.0 * math.log(1.0 / self.delta)
        term2 = self.d * math.log(
            1.0 + ((self.t + 1.0) * (self.L ** 2)) / ((self.lambda_reg ** 2) * self.d)
        )
        return math.sqrt(term1 + term2) + (math.sqrt(self.lambda_reg) / self.sigma) * self.S

    # def _C(self) -> float:
    #     """
    #     Heuristic threshold matching GIFA paper Section 5.
    #     """
    #     t = max(self.t, 1)
    #     return math.sqrt(2.0 * math.log((math.log(t) + 1.0) / self.delta))

    # ---------------- Fast mu/norm helpers ----------------

    def _mu_vec(self) -> np.ndarray:
        """mu[k] = x_k^T theta_hat for k=0..K-1"""
        return self.X @ self.theta_hat  # (K,)

    def _norm_vec(self) -> np.ndarray:
        """
        norm[k] = ||x_k||_{Sigma_hat} where Sigma_hat = sigma^2 V_inv
               = sigma * sqrt(x_k^T V_inv x_k)
        """
        # diag = diag(X V_inv X^T) = sum_i x_i * (V_inv x_i)
        XV = self.X @ self.V_inv                 # (K,d)
        diag = np.einsum("ij,ij->i", self.X, XV) # (K,)
        diag = np.clip(diag, 0.0, None)
        return self.sigma * np.sqrt(diag)

    # ---------------- Algorithm 1 lines 17-18 ----------------

    def _recompute_ambiguous_arms(self, *, mu_vec: np.ndarray, C: float, norm_vec: np.ndarray) -> None:
        """
        Sets self.b in U and self.s in N.

        Original paper lines 17-18:
          b = argmax_{b in U} max_{a in N} B(a,b)
          s = argmax_{s in N} B(s,b)

        With B(a,b) = mu[a] - mu[b] + C*(norm[a] + norm[b]), we can rewrite exactly:
          max_{a in N} B(a,b) = max_{a in N}(mu[a] + C*norm[a]) - (mu[b] - C*norm[b])
        The first term doesn't depend on b, so:
          b = argmin_{b in U} (mu[b] - C*norm[b])
          s = argmax_{s in N} (mu[s] + C*norm[s])   (given b fixed, constant shift)
        """
        if not self.U or not self.N:
            raise RuntimeError("CASE: U or N is empty; cannot recompute ambiguous arms.")

        U_idx = np.fromiter((self.id_to_idx[a] for a in self.U), dtype=int)
        N_idx = np.fromiter((self.id_to_idx[a] for a in self.N), dtype=int)

        # b = argmin_{b in U} (mu[b] - C*norm[b])
        scores_b = mu_vec[U_idx] - float(C) * norm_vec[U_idx]
        b_idx = int(U_idx[int(np.argmin(scores_b))])

        # s = argmax_{s in N} (mu[s] + C*norm[s])
        scores_s = mu_vec[N_idx] + float(C) * norm_vec[N_idx]
        s_idx = int(N_idx[int(np.argmax(scores_s))])

        self.b = int(self.idx_to_id[b_idx])
        self.s = int(self.idx_to_id[s_idx])

    # ---------------- Paper selection rule (line 20) ----------------

    def _selection_rule(self, b_id: int, s_id: int) -> int:
        """
        a* = argmin_{a in U ∪ N} ||x_b - x_s||_{(V + x_a x_a^T)^{-1}}^2

        Score(a) = diff^T (V + x_a x_a^T)^{-1} diff
                 = base - (diff^T V_inv x_a)^2 / (1 + x_a^T V_inv x_a)
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

    # ---------------- Least squares update (Algorithm 1 lines 22-23) ----------------

    def _ls_update(self, arm_id: int, reward: float) -> None:
        """
        Update:
          V <- V + x x^T
          b <- b + r x
          theta_hat <- V^{-1} b
        using Sherman–Morrison for V_inv.
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
        """
        One CASE iteration = one arm pull + model update.
        Returns (done, top_ids) where top_ids is U_t sorted by current mu_hat.
        """
        t0 = time.perf_counter()

        # Compute once per iteration (no duplicate recompute)
        mu_vec = self._mu_vec()
        C = self._C()
        norm_vec = self._norm_vec()

        b_idx = self.id_to_idx[int(self.b)]
        s_idx = self.id_to_idx[int(self.s)]

        # Stopping quantity B(s,b) = mu[s]-mu[b] + C*(norm[s]+norm[b])
        B_sb = float((mu_vec[s_idx] - mu_vec[b_idx]) + C * (norm_vec[s_idx] + norm_vec[b_idx]))

        self.best_G_history.append(B_sb)

        # LCB-like trace: (mu[b]-mu[s]) - C*(norm[b]+norm[s])
        gap_bs = float(mu_vec[b_idx] - mu_vec[s_idx])
        W_bs = float(C * (norm_vec[b_idx] + norm_vec[s_idx]))
        self.min_lcb_history.append(gap_bs - W_bs)

        # Current output estimate: U sorted by mu
        top_ids = sorted(self.U, key=lambda a: float(mu_vec[self.id_to_idx[int(a)]]), reverse=True)

        if B_sb <= self.epsilon:
            self.round_time_history.append(time.perf_counter() - t0)
            return True, top_ids

        # ---- Lines 9-13: swap worst in U with best in N if better ----
        n_t = min(self.U, key=lambda a: float(mu_vec[self.id_to_idx[int(a)]]))  # worst in U
        c_t = max(self.N, key=lambda a: float(mu_vec[self.id_to_idx[int(a)]]))  # best in N
        if float(mu_vec[self.id_to_idx[int(c_t)]]) >= float(mu_vec[self.id_to_idx[int(n_t)]]):
            self.U.remove(int(n_t))
            self.U.add(int(c_t))
            self.N.remove(int(c_t))
            self.N.add(int(n_t))

        # ---- Line 14: sample M_t uniformly from complement (U ∪ N)^c ----
        excluded = self.U.union(self.N)
        complement = [a for a in self.all_ids if a not in excluded]
        if complement:
            ssz = min(self.sample_m_prime, len(complement))
            M_t = list(self.rng.choice(complement, size=ssz, replace=False))
        else:
            M_t = []

        # ---- Line 15: N_t <- top_{m'}(M_t ∪ N; rho_hat) ----
        candidates = list(set(M_t).union(self.N))
        candidates_sorted = sorted(candidates, key=lambda a: float(mu_vec[self.id_to_idx[int(a)]]), reverse=True)
        self.N = set(int(x) for x in candidates_sorted[: self.m_prime])

        # ---- Lines 17-18: recompute b_{t+1}, s_{t+1} (NO recompute of mu/C/norm) ----
        self.total_comparisons += len(self.U) * len(self.N) + len(self.N)
        self._recompute_ambiguous_arms(mu_vec=mu_vec, C=C, norm_vec=norm_vec)

        # ---- Line 20: selection rule ----
        a_next = self._selection_rule(self.b, self.s)

        # ---- Line 21: pull reward ----
        r_next = float(oracle_callback_path(a_next))

        # ---- Lines 22-23: update ----
        self._ls_update(a_next, r_next)

        self.t += 1
        self.round_time_history.append(time.perf_counter() - t0)
        return False, top_ids