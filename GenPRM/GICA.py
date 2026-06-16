import numpy as np
import time  

class GICA:
    """
    P-GICA: Pairwise Gap-Index Framework for Sample-efficient Verification.
    
    Modified for Exact Theoretical Bounds and High-Performance Vectorization:
      - Uses exact determinant-based confidence radius beta_t.
      - References: Theorem 8.1.
      - Vectorized pairwise boundary computations & candidate step scoring.
    """

    def __init__(
        self, paths, feature_matrix, K, d,
        lambda_reg=1.0, epsilon=0.01, delta=0.05, R=1.0, S_0=1.0,
        step_pool_mode="paths"  # "paths" (union of path steps) or "all"
    ):  
        self.paths = paths
        self.features = feature_matrix
        self.K = K
        if K <= 0:
            raise ValueError("K must be >= 1")
        if K >= len(paths):
            raise ValueError("K must be < number of paths")

        self.d = d
        self.epsilon = epsilon
        self.delta = delta
        self.lambda_reg = lambda_reg
        self.R = R
        self.S_0 = S_0

        # Initialize Shared Linear Model Parameters
        self.V_inv = (1.0 / self.lambda_reg) * np.eye(d)
        self.theta_hat = np.zeros(d)
        
        # Precompute Composite Path Features (g_pi)
        self.g_pi = {}
        for pid, step_indices in paths.items():
            path_feats = self.features[step_indices]
            self.g_pi[pid] = np.mean(path_feats, axis=0)

        # --- PRECOMPUTED MATRICES FOR VECTORIZATION ---
        self.path_ids = list(self.g_pi.keys())
        self.X_paths = np.stack([self.g_pi[pid] for pid in self.path_ids]) # Shape: (Total Paths, d)

        self.t = 0  # iteration counter
        self.num_unique_steps = feature_matrix.shape[0]

        self.step_pool_mode = step_pool_mode
        if self.step_pool_mode == "all":
            self.step_candidates_all = np.arange(self.num_unique_steps, dtype=int)
        elif self.step_pool_mode == "paths":
            self.step_candidates_all = None  
        else:
            raise ValueError("step_pool_mode must be 'paths' or 'all'")

        self.total_comparisons = 0
        self.gap_deficit_history = []   
        self.round_time_history = []    
        self.best_G_history = []   
        self.min_lcb_history = []  

    def _get_confidence_radius_beta(self):
        """ Compute confidence radius beta_t exactly as defined in Theorem 8.1. """
        sign, log_det_V_inv = np.linalg.slogdet(self.V_inv)
        log_det_V_t = -log_det_V_inv
        log_det_lambda_I = self.d * np.log(self.lambda_reg)
        
        log_ratio = 0.5 * log_det_V_t - 0.5 * log_det_lambda_I - np.log(self.delta)
        if log_ratio < 0:
            log_ratio = 0
            
        beta_t = self.R * np.sqrt(2 * log_ratio) + np.sqrt(self.lambda_reg) * self.S_0
        return beta_t

    def select_and_update(self, oracle_callback):
        t_round0 = time.perf_counter()

        # ---------------------------------------------------------
        # 1. Vectorized Path Estimates
        # ---------------------------------------------------------
        # Instead of a dict comprehension, compute all mu_hat simultaneously
        mu_vec = self.X_paths @ self.theta_hat  # Shape: (Total Paths,)
        
        # Sort indices to find top-m and rest
        sorted_indices = np.argsort(mu_vec)[::-1]
        sorted_ids = [self.path_ids[i] for i in sorted_indices]

        top_indices = sorted_indices[:self.K]
        rest_indices = sorted_indices[self.K:]

        if len(rest_indices) == 0:
            self.round_time_history.append(time.perf_counter() - t_round0)
            return True, sorted_ids[:self.K]

        self.total_comparisons += len(top_indices) * len(rest_indices)
        beta_t = self._get_confidence_radius_beta()

        # ---------------------------------------------------------
        # 2. Vectorized Pairwise Computations (No Python `for` loops)
        # ---------------------------------------------------------
        X_top = self.X_paths[top_indices]   # Shape: (m, d)
        X_rest = self.X_paths[rest_indices] # Shape: (K-m, d)

        # Broadcast diffs: (m, 1, d) - (1, K-m, d) -> (m, K-m, d)
        diffs = X_top[:, None, :] - X_rest[None, :, :]
        
        # Broadcast gaps: (m, 1) - (1, K-m) -> (m, K-m)
        gaps = mu_vec[top_indices][:, None] - mu_vec[rest_indices][None, :]

        # Vectorized variance: sig2 = diffs^T * V_inv * diffs
        # V_inv_diffs shape: (m, K-m, d)
        V_inv_diffs = diffs @ self.V_inv
        # einsum computes the dot product over the last dimension 'd' efficiently
        sig2 = np.einsum('ijk,ijk->ij', diffs, V_inv_diffs)
        sig2 = np.clip(sig2, 1e-12, None) # Shape: (m, K-m)

        # Confidence widths and LCBs
        W = beta_t * np.sqrt(sig2)
        lcbs = gaps - W
        
        # Stopping Condition min(gap - width) >= -epsilon
        min_lcb = float(np.min(lcbs))
        self.min_lcb_history.append(min_lcb)

        if min_lcb >= -self.epsilon:
            self.round_time_history.append(time.perf_counter() - t_round0)
            return True, sorted_ids[:self.K]

        # ---------------------------------------------------------
        # 3. Hardest Boundary Pair
        # ---------------------------------------------------------
        # Compute G for all pairs simultaneously
        G_matrix = (gaps ** 2) / sig2
        
        # Find index of the minimum G in the (m, K-m) grid
        best_idx_tuple = np.unravel_index(np.argmin(G_matrix), G_matrix.shape)
        top_idx, rest_idx = best_idx_tuple
        
        best_G = float(G_matrix[top_idx, rest_idx])
        best_pi = self.path_ids[top_indices[top_idx]]
        best_pi_dagger = self.path_ids[rest_indices[rest_idx]]
        
        self.best_G_history.append(best_G)

        # Log gap deficit for the chosen pair
        best_W = W[top_idx, rest_idx]
        best_gap = gaps[top_idx, rest_idx]
        self.gap_deficit_history.append(float(best_W - best_gap - self.epsilon))

        # ---------------------------------------------------------
        # 4. Vectorized Step Selection
        # ---------------------------------------------------------
        target_g = self.g_pi[best_pi] - self.g_pi[best_pi_dagger]
        
        if self.step_pool_mode == "paths":
            cand = np.array(
                sorted(set(self.paths[best_pi]).union(self.paths[best_pi_dagger])),
                dtype=int
            )
        else:
            cand = self.step_candidates_all

        if cand.size == 0:
            raise RuntimeError("Candidate step set is empty (unexpected).")

        # Extract features for all candidate steps: Shape (Num_Cands, d)
        X_cand = self.features[cand]
        V_inv_X_cand = X_cand @ self.V_inv

        # Compute scores for all steps simultaneously
        # CORRECTED: Project target_g through V_inv
        V_inv_target_g = self.V_inv @ target_g 

        # Now compute the numerator using the projected vector
        nums = (X_cand @ V_inv_target_g) ** 2
        dens = 1.0 + np.einsum('ij,ij->i', X_cand, V_inv_X_cand)
        scores = nums / dens

        step_to_pull = int(cand[np.argmax(scores)])

        # ---------------------------------------------------------
        # 5. RLS Update
        # ---------------------------------------------------------
        y_obs = oracle_callback(step_to_pull)
        x_star = self.features[step_to_pull]
        
        v_inv_x = np.dot(self.V_inv, x_star)
        denom = 1.0 + np.dot(x_star.T, v_inv_x)
        outer = np.outer(v_inv_x, v_inv_x)
        
        pred_error = y_obs - np.dot(x_star, self.theta_hat)
        
        self.V_inv = self.V_inv - (outer / denom)
        
        # RLS update with NEW V_inv
        gain_vector = np.dot(self.V_inv, x_star)
        self.theta_hat = self.theta_hat + gain_vector * pred_error

        self.t += 1
        self.round_time_history.append(time.perf_counter() - t_round0)
        
        return False, sorted_ids[:self.K]