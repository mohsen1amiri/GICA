import numpy as np

class P_GIHA:
    """
    P-GIHA: Pairwise Gap-Index Framework for Sample-efficient Verification.
    
    Modified for Exact Theoretical Bounds:
      - Uses exact determinant-based confidence radius beta_t.
      - References: Theorem 8.1 .
    """

    def __init__(self, paths, feature_matrix, m, d, lambda_reg=1.0, epsilon=0.01, delta=0.05, R=1.0, S_0=1.0):
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

        self.t = 1
        self.num_unique_steps = feature_matrix.shape[0]

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
        # ... (rest of the implementation remains the same as previous) ...
        # --- 1. Compute Path Estimates (Linear) ---
        mu_hat = {pid: np.dot(g, self.theta_hat) for pid, g in self.g_pi.items()}
        
        # Now uses the EXACT beta calculation
        beta_t = self._get_confidence_radius_beta()
        
        sorted_ids = sorted(mu_hat.keys(), key=lambda k: mu_hat[k], reverse=True)
        P_hat_m = set(sorted_ids[:self.m])
        
        # ... (Compute Gaps & Ambiguities) ...
        best_pi = None
        best_pi_dagger = None
        max_GI = -np.inf

        for pi in sorted_ids:
            min_ambiguity = np.inf
            current_competitor = None
            
            for pi_prime in sorted_ids:
                if pi == pi_prime: continue
                
                g_diff = self.g_pi[pi] - self.g_pi[pi_prime]
                width_sq = np.dot(g_diff.T, np.dot(self.V_inv, g_diff))
                # Avoid numerical zero
                if width_sq < 1e-12: width_sq = 1e-12
                
                gap = mu_hat[pi] - mu_hat[pi_prime]
                ambiguity = (gap**2) / width_sq
                
                if ambiguity < min_ambiguity:
                    min_ambiguity = ambiguity
                    current_competitor = pi_prime
            
            if min_ambiguity > max_GI:
                max_GI = min_ambiguity
                best_pi = pi
                best_pi_dagger = current_competitor

        # ... (Stopping Condition) ...
        worst_in_top = sorted_ids[self.m - 1]
        best_in_rest = sorted_ids[self.m]
        
        g_diff_stop = self.g_pi[worst_in_top] - self.g_pi[best_in_rest]
        w_stop = beta_t * np.sqrt(np.dot(g_diff_stop.T, np.dot(self.V_inv, g_diff_stop)))
        delta_stop = mu_hat[worst_in_top] - mu_hat[best_in_rest]
        
        if (delta_stop - w_stop) >= -self.epsilon:
            return True, sorted_ids[:self.m]

        # ... (Step Selection & Update) ...
        target_g = self.g_pi[best_pi] - self.g_pi[best_pi_dagger]
        
        scores = []
        for s in range(self.num_unique_steps):
            x_s = self.features[s]
            v_inv_x = np.dot(self.V_inv, x_s)
            
            num = (np.dot(target_g.T, v_inv_x))**2
            den = 1 + np.dot(x_s.T, v_inv_x)
            scores.append(num / den)
            
        step_to_pull = np.argmax(scores)

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
        return False, sorted_ids[:self.m]
