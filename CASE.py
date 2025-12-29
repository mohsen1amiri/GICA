import time
import math
import numpy as np


class CASE:
    """
    Faithful implementation of Algorithm 1 (CASE) from the uploaded paper,
    adapted to "paths" (arms) with feature vectors x_a in R^d.

    Arms: path IDs (ints). Feature for arm a is x_a = g_pi[a].

    Maintains:
      - U_t: estimated top-m arms (size m)
      - N_t: estimated next-best m' arms (size m_prime)

    Gap index (paper):
        B_t(i,j) = rho_hat_t(i) - rho_hat_t(j) + W_t(i,j)
        W_t(i,j) = C_{t,delta} (||x_i||_{Sigma_hat_t^lambda} + ||x_j||_{Sigma_hat_t^lambda})
        Sigma_hat_t^lambda = sigma^2 * V_t^{-1}

    Selection rule (paper / Réda et al., 2021):
        a_{t+1} = argmin_{a in U_t ∪ N_t} ||x_b - x_s||_{(V_t + x_a x_a^T)^{-1}}

    Public API:
        select_and_update(oracle_callback_path) -> (done: bool, top_ids: list[int])

    Logged fields (kept compatible with your plotting expectations):
      - t
      - m
      - total_comparisons
      - best_G_history     : B_t(s_t, b_t) each iteration (stopping quantity)
      - min_lcb_history    : (rho_hat(b_t)-rho_hat(s_t)) - W_t(b_t,s_t)
      - round_time_history
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

        self.sigma = float(R)   # paper uses σ
        self.S = float(S_0)     # paper uses S

        # CASE m' (paper uses m')
        self.m_prime = int(challenger_size)

        # For exact Algorithm 1 behavior, sampling size equals m'
        # (kept as separate arg for backward compatibility)
        self.sample_m_prime = int(challenger_batch)

        self.rng = np.random.default_rng(seed)

        if self.m <= 0 or self.m_prime <= 0:
            raise ValueError("m and challenger_size (m') must be >= 1")
        if self.m + self.m_prime > len(paths):
            raise ValueError("Need at least m + m' distinct arms")
        if self.sample_m_prime != self.m_prime:
            # To match the paper exactly, these should be equal.
            # We enforce the paper behavior.
            self.sample_m_prime = self.m_prime

        # ---- precompute arm feature vectors x_a = g_pi[a] ----
        self.g_pi = {}
        for pid, step_indices in self.paths.items():
            self.g_pi[int(pid)] = np.mean(self.features[step_indices], axis=0).astype(float)

        self.all_ids = sorted(self.g_pi.keys())

        # Paper constant L: upper bound on ||x_a||_2
        self.L = float(max(np.linalg.norm(self.g_pi[pid]) for pid in self.all_ids))

        # ---- Regularized design matrix inverse: V_0^{-1} = (lambda I)^{-1} ----
        self.V_inv = (1.0 / self.lambda_reg) * np.eye(self.d)

        # b_vec = Σ r_l x_{a_l}
        self.b_vec = np.zeros(self.d)

        # Paper initializes alpha_1 ~ N(0,1) (used for initial scoring)
        self.theta_hat = self.rng.normal(0.0, 1.0, size=self.d)

        # counts N_a
        self.counts = {pid: 0 for pid in self.all_ids}

        # ---- Initialize U_0 and N_0 randomly (paper gives U_0; N_0 is required by the loop) ----
        perm = self.rng.permutation(self.all_ids)
        self.U = set(int(x) for x in perm[: self.m])
        self.N = set(int(x) for x in perm[self.m : self.m + self.m_prime])

        # current ambiguous pair (b_t in U, s_t in N)
        self.b = None
        self.s = None

        # ---- counters / logs ----
        self.t = 0  # number of pulls so far
        self.total_comparisons = 0
        self.round_time_history = []
        self.best_G_history = []
        self.min_lcb_history = []

        # compute initial (b_1, s_1) for the stopping criterion
        self._recompute_ambiguous_arms()

    # ---------------- Paper confidence scalar and gap index ----------------

    def _C(self):
        """
        Paper's confidence scalar C_{t,delta}:

            C_{t,δ} = sqrt( 2 ln(1/δ) + N ln( 1 + ((t+1) L^2) / (λ^2 N) ) )
                      + (sqrt(λ)/σ) S

        where N is total number of pulls so far.
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
        return {pid: float(self.g_pi[pid] @ self.theta_hat) for pid in self.all_ids}

    def _sigma_norm(self, pid):
        # ||x||_{Sigma_hat} with Sigma_hat = sigma^2 V_inv
        x = self.g_pi[pid]
        v = float(x.T @ self.V_inv @ x)
        v = max(v, 0.0)
        return self.sigma * math.sqrt(v)

    def _B(self, i, j, mu, norm, C):
        # B_t(i,j) = rho_hat(i) - rho_hat(j) + C*(||xi|| + ||xj||)
        return (mu[i] - mu[j]) + C * (norm[i] + norm[j])

    # ---------------- Algorithm 1 lines 17-18 ----------------

    def _recompute_ambiguous_arms(self):
        """
        Implements Algorithm 1 lines 17-18 using current U, N, theta_hat, V_inv.
        Sets self.b (in U) and self.s (in N).
        """
        mu = self._rho_hat_all()
        C = self._C()
        involved = self.U.union(self.N)
        norm = {pid: self._sigma_norm(pid) for pid in involved}

        # b = argmax_{b in U} max_{a in N} B(a,b)
        best_b = None
        best_b_val = -float("inf")
        for b in self.U:
            inner = -float("inf")
            for a in self.N:
                inner = max(inner, self._B(a, b, mu, norm, C))
            if inner > best_b_val:
                best_b_val = inner
                best_b = b

        # s = argmax_{s in N} B(s, b)
        best_s = None
        best_s_val = -float("inf")
        for s in self.N:
            val = self._B(s, best_b, mu, norm, C)
            if val > best_s_val:
                best_s_val = val
                best_s = s

        self.b = int(best_b)
        self.s = int(best_s)

    # ---------------- Paper selection rule (line 20) ----------------

    def _selection_rule(self, b, s):
        """
        Greedy variance-minimization selection rule (paper / Réda et al., 2021):

            a* = argmin_{a in U ∪ N} ||x_b - x_s||_{(V + x_a x_a^T)^{-1}}^2

        Using Sherman–Morrison on V_inv = V^{-1}:

            (V + x x^T)^{-1} = V_inv - (V_inv x x^T V_inv) / (1 + x^T V_inv x)

        Score(a) = diff^T (V + x_a x_a^T)^{-1} diff
                 = base - (diff^T V_inv x_a)^2 / (1 + x_a^T V_inv x_a)
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

    # ---------------- Least squares update (Algorithm 1 lines 22-23) ----------------

    def _ls_update(self, arm_id, reward):
        """
        Update:
          V <- V + x x^T
          b <- b + r x
          theta_hat <- V^{-1} b
        using Sherman–Morrison for V_inv.
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
        """
        One CASE iteration = one arm pull + model update.
        Returns (done, top_ids) where top_ids is U_t sorted by current rho_hat.
        """
        t0 = time.perf_counter()

        mu = self._rho_hat_all()
        C = self._C()
        involved = self.U.union(self.N)
        norm = {pid: self._sigma_norm(pid) for pid in involved}

        # ---- Stopping check uses current (s_t, b_t) ----
        B_sb = self._B(self.s, self.b, mu, norm, C)

        # logs consistent with your previous plotting
        self.best_G_history.append(float(B_sb))
        W_bs = C * (norm[self.b] + norm[self.s])
        gap_bs = mu[self.b] - mu[self.s]
        self.min_lcb_history.append(float(gap_bs - W_bs))

        # current output estimate: U_t sorted by rho_hat
        top_ids = sorted(self.U, key=lambda a: mu[a], reverse=True)

        # Paper stops when B_t(s_t, b_t) <= epsilon
        if B_sb <= self.epsilon:
            self.round_time_history.append(time.perf_counter() - t0)
            return True, top_ids

        # ---- Lines 9-13: swap worst in U with best in N if better ----
        n_t = min(self.U, key=lambda a: mu[a])  # worst in U
        c_t = max(self.N, key=lambda a: mu[a])  # best in N
        if mu[c_t] >= mu[n_t]:
            self.U.remove(n_t)
            self.U.add(c_t)
            self.N.remove(c_t)
            self.N.add(n_t)

        # ---- Line 14: sample M_t uniformly from complement (U ∪ N)^c ----
        excluded = self.U.union(self.N)
        complement = [a for a in self.all_ids if a not in excluded]
        if len(complement) > 0:
            ssz = min(self.sample_m_prime, len(complement))
            M_t = list(self.rng.choice(complement, size=ssz, replace=False))
        else:
            M_t = []

        # ---- Line 15: N_t <- top_{m'}(M_t ∪ N; rho_hat) ----
        candidates = list(set(M_t).union(self.N))
        candidates_sorted = sorted(candidates, key=lambda a: mu[a], reverse=True)
        self.N = set(candidates_sorted[: self.m_prime])

        # ---- Lines 17-18: recompute b_{t+1}, s_{t+1} ----
        self.total_comparisons += len(self.U) * len(self.N) + len(self.N)
        self._recompute_ambiguous_arms()

        # ---- Line 20: selection_rule(U, N) (depends on b,s) ----
        a_next = self._selection_rule(self.b, self.s)

        # ---- Line 21: pull & receive reward ----
        r_next = float(oracle_callback_path(a_next))

        # ---- Lines 22-23: update parameters ----
        self._ls_update(a_next, r_next)

        self.t += 1
        self.round_time_history.append(time.perf_counter() - t0)
        return False, top_ids
