import numpy as np
import time  

class P_GIHA:
    """
    P-GIHA: Pairwise Gap-Index Framework for Sample-efficient Verification.
    
    Modified for Exact Theoretical Bounds:
      - Uses exact determinant-based confidence radius beta_t.
      - References: Theorem 8.1 .
    """

    def __init__(
    self, paths, feature_matrix, m, d,
    lambda_reg=1.0, epsilon=0.01, delta=0.05, R=1.0, S_0=1.0,
    step_pool_mode="paths"  # "paths" (union of path steps) or "all"
    ):  

        """
        Args:
            paths (dict): Mapping of path_id -> list of step_indices.
            feature_matrix (np.ndarray): Shape (Total_Unique_Steps, d).
            m (int): Number of top paths to identify.
            d (int): Dimension of feature vectors.
            lambda_reg (float): Regularization parameter (lambda).
            epsilon (float): Tolerance for stopping condition.
            delta (float): Confidence level.
            R (float): Sub-Gaussian noise proxy parameter[cite: 311].
            S_0 (float): Bound on the norm of the true parameter theta*[cite: 311].
        """
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


        
        # Initialize Shared Linear Model Parameters
        # V_0 = lambda * I, so V_inv = (1/lambda) * I
        self.V_inv = (1.0 / self.lambda_reg) * np.eye(d)
        self.theta_hat = np.zeros(d)
        
        # Precompute Composite Path Features (g_pi)
        self.g_pi = {}
        for pid, step_indices in paths.items():
            path_feats = self.features[step_indices]
            self.g_pi[pid] = np.mean(path_feats, axis=0)

        self.t = 0  # iteration counter
        self.num_unique_steps = feature_matrix.shape[0]
        # --- Step pool (two options) ---
        # Paper's S is "all steps across all paths" :contentReference[oaicite:4]{index=4}
        # Here we support:
        #   - "paths": only steps that appear in at least one path (union)
        #   - "all": all steps in feature_matrix (your synthetic global pool)
        # --- Step pool (two options) ---
        # "paths": only steps in the TWO boundary paths (pi* ∪ pi†) -> computed each iteration
        # "all": all steps in the feature matrix -> fixed list
        self.step_pool_mode = step_pool_mode

        if self.step_pool_mode == "all":
            self.step_candidates_all = np.arange(self.num_unique_steps, dtype=int)
        elif self.step_pool_mode == "paths":
            self.step_candidates_all = None  # boundary-dependent, built inside select_and_update()
        else:
            raise ValueError("step_pool_mode must be 'paths' or 'all'")

        

        self.total_comparisons = 0
        self.gap_deficit_history = []   # one value per round
        self.round_time_history = []    # optional: time per select_and_update call
        self.best_G_history = []   # store hardest boundary gap-index per round
        self.min_lcb_history = []  # store min_{top x rest} (gap - W) each round (stopping quantity)


    def _get_confidence_radius_beta(self):
        """
        Compute confidence radius beta_t exactly as defined in Theorem 8.1.
        
        Formula[cite: 311]: 
        beta_t(delta) = R * sqrt( 2 * log( (det(V_t)^0.5 * det(lambda*I)^-0.5) / delta ) ) + sqrt(lambda)*S_0
        
        We compute terms using log-determinants for numerical stability:
        log(term) = 0.5 * log(det(V_t)) - 0.5 * log(det(lambda*I)) - log(delta)
        
        Since we maintain V_inv, we use: log(det(V_t)) = -log(det(V_inv))
        """
        # 1. Compute log(det(V_t)) using V_inv
        # slogdet returns (sign, log_absolute_value)
        sign, log_det_V_inv = np.linalg.slogdet(self.V_inv)
        
        # Note: V_inv is positive definite, so sign should always be 1.
        # log(det(V_t)) = -log(det(V_inv))
        log_det_V_t = -log_det_V_inv
        
        # 2. Compute log(det(lambda * I))
        # det(lambda * I) = lambda^d
        # log(det(lambda * I)) = d * log(lambda)
        log_det_lambda_I = self.d * np.log(self.lambda_reg)
        
        # 3. Combine terms inside the logarithm
        # log_ratio corresponds to log( (det(V_t)^0.5 * det(lambda*I)^-0.5) / delta )
        log_ratio = 0.5 * log_det_V_t - 0.5 * log_det_lambda_I - np.log(self.delta)
        
        # Ensure non-negative value inside sqrt (floating point safety)
        if log_ratio < 0:
            log_ratio = 0
            
        # 4. Final Formula
        beta_t = self.R * np.sqrt(2 * log_ratio) + np.sqrt(self.lambda_reg) * self.S_0
        
        return beta_t

    def select_and_update(self, oracle_callback):
        t_round0 = time.perf_counter()

        # ... (rest of the implementation remains the same as previous) ...
        # --- 1. Compute Path Estimates (Linear) ---
        mu_hat = {pid: np.dot(g, self.theta_hat) for pid, g in self.g_pi.items()}
        
        # Now uses the EXACT beta calculation
        beta_t = self._get_confidence_radius_beta()
        
        sorted_ids = sorted(mu_hat.keys(), key=lambda k: mu_hat[k], reverse=True)
        P_hat_m = set(sorted_ids[:self.m])

        top_ids = list(sorted_ids[:self.m])
        rest_ids = list(sorted_ids[self.m:])

        n_pairs = len(top_ids) * len(rest_ids)


        if len(rest_ids) == 0:
            self.round_time_history.append(time.perf_counter() - t_round0)
            return True, top_ids


        self.total_comparisons += n_pairs
        # --- Stopping Condition (paper): min over (top x rest) of (gap - width) >= -epsilon ---
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


        if min_lcb >= -self.epsilon:
            self.round_time_history.append(time.perf_counter() - t_round0)
            return True, top_ids

        

        # --- Choose hardest boundary pair: argmin G_t over (top x rest) ---
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

        # # Optional refinement: among ALL other paths, pick the hardest competitor for best_pi
        # best_G2 = np.inf
        # for pj in sorted_ids:
        #     if pj == best_pi:
        #         continue
        #     g_diff = self.g_pi[best_pi] - self.g_pi[pj]
        #     sig2 = float(g_diff.T @ self.V_inv @ g_diff)
        #     sig2 = max(sig2, 1e-12)
        #     gap = mu_hat[best_pi] - mu_hat[pj]
        #     G = (gap ** 2) / sig2
        #     if G < best_G2:
        #         best_G2 = G
        #         best_pi_dagger = pj
        self.best_G_history.append(best_G)

        # gap-deficit for the chosen boundary pair (should go to 0 at convergence)
        g_diff = self.g_pi[best_pi] - self.g_pi[best_pi_dagger]
        sig2 = float(g_diff.T @ self.V_inv @ g_diff)
        sig2 = max(sig2, 1e-12)
        W = beta_t * np.sqrt(sig2)
        gap = mu_hat[best_pi] - mu_hat[best_pi_dagger]
        gap_deficit = (W - gap - self.epsilon)  # no clipping
        self.gap_deficit_history.append(gap_deficit)



        # ... (Step Selection & Update) ...
        target_g = self.g_pi[best_pi] - self.g_pi[best_pi_dagger]
        
        # Build step candidates based on mode
        if self.step_pool_mode == "paths":
            # Restrict to steps in the two boundary paths (pi* ∪ pi†)
            cand = np.array(
                sorted(set(self.paths[best_pi]).union(self.paths[best_pi_dagger])),
                dtype=int
            )
        else:  # "all"
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



        # Update
        y_obs = oracle_callback(step_to_pull)
        x_star = self.features[step_to_pull]
        
        v_inv_x = np.dot(self.V_inv, x_star)
        denom = 1 + np.dot(x_star.T, v_inv_x)
        outer = np.outer(v_inv_x, v_inv_x)
        
        pred_error = y_obs - np.dot(x_star, self.theta_hat)
        
        self.V_inv = self.V_inv - (outer / denom)
        
        # RLS update with NEW V_inv
        gain_vector = np.dot(self.V_inv, x_star)
        self.theta_hat = self.theta_hat + gain_vector * pred_error

        self.t += 1
        self.round_time_history.append(time.perf_counter() - t_round0)
        return False, sorted_ids[:self.m]