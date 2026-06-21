import numpy as np
import time


class GICA:
    """GICA selector for the test-time-scaling pipeline (Algorithm 1, TTS variant).

    Same gap-index, fixed-confidence top-K identification as the synthetic ``GICA``, with
    one addition specific to the TTS setting (Appendix B.2.2): a *dynamic boundary
    feature*. The last column of the feature matrix is overwritten each round with the
    projection of every step embedding onto the current boundary direction
    (centroid of pi* minus centroid of pi_dagger), letting the shared linear model exploit
    the compositional boundary geometry between the contested paths.

    The step embeddings are supplied externally by assigning ``self.env_step_emb`` after
    construction (the environment builds them); they are used only to recompute that last
    feature column.

    Args:
        paths (dict[int, list[int]]): Mapping ``path_id -> list of step indices``.
        feature_matrix (np.ndarray): Step features ``(num_steps, d)``; the last column is
            reserved for the dynamic boundary feature and is updated in place each round.
        m (int): Number of top paths to identify (the top-set size ``K``).
        d (int): Feature dimension.
        lambda_reg (float): Ridge regularization parameter ``lambda``.
        epsilon (float): Stopping tolerance.
        delta (float): Target error probability (confidence level ``1 - delta``).
        R (float): Sub-Gaussian noise proxy of the verifier observations.
        S_0 (float): Upper bound on ``||theta*||``.
        step_pool_mode (str): Candidate-step pool per round. ``"paths"`` restricts to the
            steps of the current boundary pair; ``"all"`` uses every step.
    """

    def __init__(
        self, paths, feature_matrix, m, d,
        lambda_reg=1.0, epsilon=0.01, delta=0.05,
        R=1.0, S_0=1.0, step_pool_mode="paths"
    ):

        self.paths = paths
        self.features = feature_matrix
        self.m = m
        self.d = d
        self.epsilon = epsilon
        self.delta = delta
        self.lambda_reg = lambda_reg
        self.R = R
        self.S_0 = S_0

        # Shared linear model. V_0 = lambda I => V_0^{-1} = (1/lambda) I.
        self.V_inv = (1.0 / lambda_reg) * np.eye(d)
        self.theta_hat = np.zeros(d)

        # Composite (length-normalized) path features g(pi) = mean of the path's steps.
        self.g_pi = {}
        for pid, step_indices in paths.items():
            path_feats = self.features[step_indices]
            self.g_pi[pid] = np.mean(path_feats, axis=0)

        self.t = 0
        self.step_pool_mode = step_pool_mode

    def _get_confidence_radius_beta(self):
        """Return the self-normalized confidence radius ``beta_t(delta)``.

        Computed from log-determinants for stability; since the design matrix is tracked
        as ``V_inv``, ``log det(V_t) = -log det(V_inv)``.
        """
        sign, log_det_V_inv = np.linalg.slogdet(self.V_inv)
        log_det_V_t = -log_det_V_inv
        log_det_lambda_I = self.d * np.log(self.lambda_reg)

        log_ratio = 0.5 * log_det_V_t - 0.5 * log_det_lambda_I - np.log(self.delta)
        log_ratio = max(log_ratio, 0)

        beta_t = self.R * np.sqrt(2 * log_ratio) + np.sqrt(self.lambda_reg) * self.S_0
        return beta_t

    def select_and_update(self, oracle_callback):
        """Run one GICA round against the PRM oracle.

        Args:
            oracle_callback (callable): ``step_index -> scalar reward`` (one verifier call).

        Returns:
            tuple[bool, list[int]]: ``(done, top_ids)`` with the stopping flag and the
            current empirical top-K shortlist.
        """
        # Plug-in path utilities and current confidence radius.
        mu_hat = {pid: np.dot(g, self.theta_hat) for pid, g in self.g_pi.items()}
        beta_t = self._get_confidence_radius_beta()

        # Empirical top-K shortlist and its challenger set.
        sorted_ids = sorted(mu_hat.keys(), key=lambda k: mu_hat[k], reverse=True)

        top_ids = list(sorted_ids[:self.m])
        rest_ids = list(sorted_ids[self.m:])

        if len(rest_ids) == 0:
            return True, top_ids

        # Stopping quantity: min over (top x rest) of (gap - confidence width).
        min_lcb = np.inf
        print("min_lcb",min_lcb)
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
        print("min_lcb",gap, W, beta_t, np.sqrt(sig2),  min_lcb*0.1, self.epsilon)

        # Certified epsilon-optimal (with the TTS damping factor on the margin).
        if (min_lcb*0.1) >= -self.epsilon:
            return True, top_ids

        # Hardest boundary pair = argmin gap index G_t = gap^2 / sigma^2.
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

        target_g = self.g_pi[best_pi] - self.g_pi[best_pi_dagger]
        step_emb = self.env_step_emb  # step embeddings; attach via self.env_step_emb after construction

        # --- Dynamic boundary feature: project step embeddings onto the boundary direction ---
        pi_star_steps = self.paths[best_pi]
        pi_dagger_steps = self.paths[best_pi_dagger]

        centroid_star = np.mean(step_emb[pi_star_steps], axis=0)
        centroid_star /= np.linalg.norm(centroid_star) + 1e-8

        centroid_dagger = np.mean(step_emb[pi_dagger_steps], axis=0)
        centroid_dagger /= np.linalg.norm(centroid_dagger) + 1e-8

        boundary_dir = centroid_star - centroid_dagger
        boundary_dir /= np.linalg.norm(boundary_dir) + 1e-8

        # Overwrite the reserved last feature column with the boundary projection.
        self.features[:, -1] = step_emb @ boundary_dir

        # Admissible step pool U_t: boundary pair's steps ("paths") or all steps ("all").
        if self.step_pool_mode == "paths":
            cand = np.array(
                sorted(set(self.paths[best_pi]).union(self.paths[best_pi_dagger])),
                dtype=int
            )
        else:
            cand = np.arange(self.features.shape[0])

        # Step selection: maximize the one-step variance contraction C_t(s) (Sherman-Morrison).
        scores = []
        for s in cand:
            x_s = self.features[s]
            v_inv_x = self.V_inv @ x_s
            num = float((target_g.T @ v_inv_x) ** 2)
            den = 1.0 + float(x_s.T @ v_inv_x)
            scores.append(num / den)

        step_to_pull = int(cand[int(np.argmax(scores))])

        # Query the PRM verifier for the selected step.
        y_obs = oracle_callback(step_to_pull)

        # Sherman-Morrison rank-one update of V_inv + recursive-least-squares update of theta_hat.
        x_star = self.features[step_to_pull]
        v_inv_x = self.V_inv @ x_star
        denom = 1 + np.dot(x_star.T, v_inv_x)
        outer = np.outer(v_inv_x, v_inv_x)

        pred_error = y_obs - np.dot(x_star, self.theta_hat)

        self.V_inv = self.V_inv - (outer / denom)
        gain_vector = self.V_inv @ x_star
        self.theta_hat = self.theta_hat + gain_vector * pred_error

        self.t += 1

        return False, sorted_ids[:self.m]
