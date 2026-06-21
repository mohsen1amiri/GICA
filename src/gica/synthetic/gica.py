import numpy as np
import time


class GICA:
    """Gap-Index Compositional Arm framework for fixed-confidence top-K identification.

    This is the reference implementation of Algorithm 1 in the paper. Each reasoning
    path ``pi`` is a *compositional arm* whose feature ``g(pi)`` is the length-normalized
    average of its step features, and all paths share a single linear utility parameter
    ``theta*`` (``mu(pi) = g(pi)^T theta*``). Because every step query updates this shared
    parameter, one observation informs every path at once.

    Each round, GICA:
      1. forms the empirical top-K shortlist from the plug-in path utilities;
      2. certifies stopping via the worst-case lower-confidence-bound gap
         ``Gamma_t = min_{top x rest} (gap - W)`` and halts once ``Gamma_t >= -epsilon``;
      3. otherwise locates the most ambiguous boundary pair ``(pi*, pi_dagger)`` by the
         pairwise gap index ``G_t = gap^2 / sigma^2``;
      4. queries, among the steps of that boundary pair, the one maximizing the exact
         one-step variance contraction ``C_t(s)`` (Sherman-Morrison);
      5. applies the rank-one design-matrix update and a recursive-least-squares update
         of ``theta_hat``.

    The self-normalized confidence radius ``beta_t(delta)`` is computed exactly from
    log-determinants for numerical stability.

    Args:
        paths (dict[int, list[int]]): Mapping ``path_id -> list of step indices``.
        feature_matrix (np.ndarray): Step features of shape ``(num_unique_steps, d)``.
        m (int): Number of top paths to identify (the top-set size ``K``).
        d (int): Feature dimension.
        lambda_reg (float): Ridge regularization parameter ``lambda``.
        epsilon (float): Stopping tolerance.
        delta (float): Target error probability (confidence level ``1 - delta``).
        R (float): Sub-Gaussian noise proxy of the step observations.
        S_0 (float): Upper bound on ``||theta*||``.
        step_pool_mode (str): Candidate-step pool per round. ``"paths"`` restricts to the
            steps of the current boundary pair (``pi* union pi_dagger``); ``"all"`` uses
            every step in ``feature_matrix``.
    """

    def __init__(
        self, paths, feature_matrix, m, d,
        lambda_reg=1.0, epsilon=0.01, delta=0.05, R=1.0, S_0=1.0,
        step_pool_mode="paths",
    ):
        self.paths = paths
        self.features = feature_matrix
        self.m = m
        if m <= 0:
            raise ValueError("m must be >= 1")
        if m >= len(paths):
            raise ValueError("m must be < number of paths")

        self.d = d
        self.epsilon = epsilon
        self.delta = delta
        self.lambda_reg = lambda_reg
        self.R = R
        self.S_0 = S_0

        # Shared linear model. Initialized as V_0 = lambda * I, so V_0^{-1} = (1/lambda) I.
        self.V_inv = (1.0 / self.lambda_reg) * np.eye(d)
        self.theta_hat = np.zeros(d)

        # Composite (length-normalized) path features g(pi) = mean of the path's steps.
        self.g_pi = {}
        for pid, step_indices in paths.items():
            path_feats = self.features[step_indices]
            self.g_pi[pid] = np.mean(path_feats, axis=0)

        self.t = 0  # round / verifier-call counter
        self.num_unique_steps = feature_matrix.shape[0]

        # Candidate-step pool. "all" fixes the pool once; "paths" rebuilds it from the
        # boundary pair on every round (so it is left as None here).
        self.step_pool_mode = step_pool_mode
        if self.step_pool_mode == "all":
            self.step_candidates_all = np.arange(self.num_unique_steps, dtype=int)
        elif self.step_pool_mode == "paths":
            self.step_candidates_all = None
        else:
            raise ValueError("step_pool_mode must be 'paths' or 'all'")

        # Diagnostics / logging traces.
        self.total_comparisons = 0       # cumulative number of gap-index comparisons
        self.gap_deficit_history = []    # (W - gap - epsilon) for the chosen pair, per round
        self.round_time_history = []     # wall-clock time per select_and_update call
        self.best_G_history = []         # hardest boundary gap-index G_t, per round
        self.min_lcb_history = []        # stopping quantity Gamma_t = min(gap - W), per round

    def _get_confidence_radius_beta(self):
        """Return the self-normalized confidence radius ``beta_t(delta)``.

        Implements
            beta_t = R * sqrt( 2 * log( det(V_t)^{1/2} det(lambda I)^{-1/2} / delta ) )
                     + sqrt(lambda) * S_0,
        evaluated through log-determinants for numerical stability. Since the design
        matrix is tracked as ``V_inv``, ``log det(V_t) = -log det(V_inv)``.
        """
        # slogdet returns (sign, log|det|); V_inv is positive-definite so sign == 1.
        sign, log_det_V_inv = np.linalg.slogdet(self.V_inv)
        log_det_V_t = -log_det_V_inv

        # det(lambda * I) = lambda^d  =>  log det(lambda I) = d * log(lambda).
        log_det_lambda_I = self.d * np.log(self.lambda_reg)

        # Argument of the logarithm in beta_t.
        log_ratio = 0.5 * log_det_V_t - 0.5 * log_det_lambda_I - np.log(self.delta)

        # Floating-point safety: keep the sqrt argument non-negative.
        if log_ratio < 0:
            log_ratio = 0

        beta_t = self.R * np.sqrt(2 * log_ratio) + np.sqrt(self.lambda_reg) * self.S_0
        return beta_t

    def select_and_update(self, oracle_callback):
        """Run one GICA round.

        Args:
            oracle_callback (callable): ``step_index -> scalar reward``. Each call is one
                verifier query of the selected step.

        Returns:
            tuple[bool, list[int]]: ``(done, top_ids)`` where ``done`` indicates the
            shortlist has been certified ``epsilon``-optimal and ``top_ids`` is the current
            empirical top-K shortlist.
        """
        t_round0 = time.perf_counter()

        # --- 1. Plug-in path utilities and the current confidence radius ---
        mu_hat = {pid: np.dot(g, self.theta_hat) for pid, g in self.g_pi.items()}
        beta_t = self._get_confidence_radius_beta()

        # Empirical top-K shortlist and its challenger set.
        sorted_ids = sorted(mu_hat.keys(), key=lambda k: mu_hat[k], reverse=True)
        P_hat_m = set(sorted_ids[:self.m])

        top_ids = list(sorted_ids[:self.m])
        rest_ids = list(sorted_ids[self.m:])
        n_pairs = len(top_ids) * len(rest_ids)

        # Degenerate case: no challengers left to separate against.
        if len(rest_ids) == 0:
            self.round_time_history.append(time.perf_counter() - t_round0)
            return True, top_ids

        self.total_comparisons += n_pairs

        # --- 2. Stopping rule: Gamma_t = min over (top x rest) of (gap - width) ---
        min_lcb = np.inf
        for pi in top_ids:
            for pj in rest_ids:
                g_diff = self.g_pi[pi] - self.g_pi[pj]
                sig2 = float(g_diff.T @ self.V_inv @ g_diff)
                sig2 = max(sig2, 1e-12)
                W = beta_t * np.sqrt(sig2)
                gap = mu_hat[pi] - mu_hat[pj]
                lcb = gap - W
                if lcb < min_lcb:
                    min_lcb = lcb

        self.min_lcb_history.append(min_lcb)

        # Certified epsilon-optimal: every shortlisted path beats every challenger
        # by more than -epsilon at confidence 1 - delta.
        if min_lcb >= -self.epsilon:
            self.round_time_history.append(time.perf_counter() - t_round0)
            return True, top_ids

        # --- 3. Boundary selection: hardest pair = argmin gap index G_t = gap^2 / sigma^2 ---
        best_G = np.inf
        best_pi = None
        best_pi_dagger = None

        for pi in top_ids:
            for pj in rest_ids:
                g_diff = self.g_pi[pi] - self.g_pi[pj]
                sig2 = float(g_diff.T @ self.V_inv @ g_diff)
                sig2 = max(sig2, 1e-12)
                gap = mu_hat[pi] - mu_hat[pj]
                G = (gap ** 2) / sig2
                if G < best_G:
                    best_G = G
                    best_pi = pi
                    best_pi_dagger = pj

        self.best_G_history.append(best_G)

        # Gap deficit for the chosen boundary pair (-> 0 as the pair becomes separable).
        g_diff = self.g_pi[best_pi] - self.g_pi[best_pi_dagger]
        sig2 = float(g_diff.T @ self.V_inv @ g_diff)
        sig2 = max(sig2, 1e-12)
        W = beta_t * np.sqrt(sig2)
        gap = mu_hat[best_pi] - mu_hat[best_pi_dagger]
        gap_deficit = (W - gap - self.epsilon)
        self.gap_deficit_history.append(gap_deficit)

        # --- 4. Step selection: maximize the exact one-step variance contraction ---
        # C_t(s) = <g_diff, x_s>_{V^{-1}}^2 / (1 + ||x_s||^2_{V^{-1}})  (Sherman-Morrison).
        target_g = self.g_pi[best_pi] - self.g_pi[best_pi_dagger]

        # Admissible step pool U_t: boundary pair's steps ("paths") or all steps ("all").
        if self.step_pool_mode == "paths":
            cand = np.array(
                sorted(set(self.paths[best_pi]).union(self.paths[best_pi_dagger])),
                dtype=int,
            )
        else:
            cand = self.step_candidates_all

        if cand.size == 0:
            raise RuntimeError("Candidate step set is empty (unexpected).")

        scores = []
        for s in cand:
            x_s = self.features[s]
            v_inv_x = self.V_inv @ x_s
            num = float((target_g.T @ v_inv_x) ** 2)
            den = 1.0 + float(x_s.T @ v_inv_x)
            scores.append(num / den)

        step_to_pull = int(cand[int(np.argmax(scores))])

        # --- 5. Query the verifier and update (Sherman-Morrison V_inv + RLS theta_hat) ---
        y_obs = oracle_callback(step_to_pull)
        x_star = self.features[step_to_pull]

        v_inv_x = np.dot(self.V_inv, x_star)
        denom = 1 + np.dot(x_star.T, v_inv_x)
        outer = np.outer(v_inv_x, v_inv_x)

        pred_error = y_obs - np.dot(x_star, self.theta_hat)

        # Rank-one inverse update of the design matrix.
        self.V_inv = self.V_inv - (outer / denom)

        # Recursive-least-squares update of theta_hat using the refreshed V_inv.
        gain_vector = np.dot(self.V_inv, x_star)
        self.theta_hat = self.theta_hat + gain_vector * pred_error

        self.t += 1
        self.round_time_history.append(time.perf_counter() - t_round0)
        return False, sorted_ids[:self.m]
