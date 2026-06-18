import numpy as np

# Synthetic environment for the GICA experiments -- "random instance, control the
# boundary gap, measure rho" design. Drop-in: `from run import ReasoningEnvironment`.
#
# Philosophy (per the paper's model in Section 3):
#   * theta* is a random shared parameter, ||theta*|| <= S_0.
#   * Each path is a random bag of step features x_s in R^d, ||x_s|| <= L.
#   * g(pi) = mean of its step features (length-normalized); mu(pi) = g(pi)^T theta*.
#   * Querying a step returns x_s^T theta* + R-sub-Gaussian (Gaussian) noise.
#   * Top-K identification; the only gap that binds GICA's stopping rule is the
#     rank-K / rank-(K+1) boundary gap Delta_C.
#
# What we CONTROL: Delta_C (the boundary gap), d, theta_norm, L, noise.
# What we DO NOT control: rho_dagger (Assumption 3.2). It is an emergent property
#   of the random geometry, so we MEASURE and report it -- statically over the
#   boundary set C*_K, and (the honest version) along the algorithm's realized
#   trajectory via record_rho()/get_realized_rho().
#
# The ONLY structural intervention is utility calibration: random steps give each
# path a raw quality; we add a constant offset along theta_hat to each path's steps
# so the path's utility hits a chosen target, and we rigidly shift the top-K block
# so the boundary gap is exactly Delta_C. Everything off the theta* axis -- which is
# what determines rho_dagger -- stays fully random and untouched.


class ReasoningEnvironment:
    def __init__(self, num_paths=200, dim=8,
                 noise_std=0.1, path_len_min=20, path_len_max=80,
                 theta_norm=1.0, util_floor=0.30,
                 grid_gap=1e-1, gap_spread=3.0,            # gap_spread accepted, unused
                 rho_target=0.20, rho_spread=2.5,          # accepted, unused (rho is measured)
                 feature_norm=8.0,
                 b_min=0.05, b_max=0.15,                   # accepted, unused
                 seed=None, num_total_steps=None,          # legacy, ignored
                 top_k=None,                               # rank of the binding boundary (pass K/m)
                 step_scale=0.30,                          # std of random step features
                 util_range=1.0):                          # spread of random path qualities
        self.dim = int(dim)
        self.num_paths = int(num_paths)
        self.noise_std = float(noise_std)
        self.path_len_min = int(path_len_min)
        self.path_len_max = int(path_len_max)
        self.theta_norm = float(theta_norm)
        self.util_floor = float(util_floor)
        self.grid_gap = float(grid_gap)
        self.step_scale = float(step_scale)
        self.util_range = float(util_range)
        self.rng = np.random.default_rng(seed)

        M, d = self.num_paths, self.dim
        if d < 2:
            raise ValueError("dim must be >= 2.")
        if top_k is None:
            top_k = max(1, M // 2)
        if not (1 <= top_k < M):
            raise ValueError(f"top_k must be in [1, M-1]; got {top_k}, M={M}.")
        self.top_k = int(top_k)
        self.Delta_C = self.theta_norm * self.grid_gap

        # ---- random shared parameter theta* ----
        th = self.rng.normal(0, 1, size=d); th /= np.linalg.norm(th)
        self.theta_hat = th
        self.true_theta = th * self.theta_norm

        # ---- random target qualities, then pin the boundary gap (rigid top-K shift) ----
        # Random per-path quality levels (genuine spread, not all tied).
        q = self.util_floor + np.sort(self.rng.uniform(0, self.util_range, size=M))[::-1]  # desc
        # Force mu(rank K-1) - mu(rank K) = Delta_C by shifting the top-K block.
        shift = self.Delta_C - (q[self.top_k - 1] - q[self.top_k])
        q[:self.top_k] += shift
        self._u_target = q                          # path id == rank (0 = best); fine for a sim
        self._top_ids = np.arange(0, self.top_k)
        self._chal_ids = np.arange(self.top_k, M)

        # ---- random steps; variable lengths ----
        lengths = self.rng.integers(self.path_len_min, self.path_len_max + 1, size=M)
        feats, self.paths, cursor = [], {}, 0
        for i in range(M):
            T = int(lengths[i])
            X = self.rng.normal(0, self.step_scale, size=(T, d))   # random step features
            mu_raw = float(np.mean(X, axis=0) @ self.true_theta)
            # calibrate ONLY the theta_hat-component so mu(pi_i) == target; rest untouched
            offset = (self._u_target[i] - mu_raw) / self.theta_norm
            X = X + offset * self.theta_hat[None, :]
            feats.append(X)
            self.paths[i] = list(range(cursor, cursor + T))
            cursor += T
        self.feature_matrix = np.vstack(feats)

        # path means
        self._g = np.stack([np.mean(self.feature_matrix[self.paths[i]], axis=0)
                            for i in range(M)])

        # ---- feature-norm budget L (fixed -> c0_tilde fixed across M) ----
        max_norm = float(np.max(np.linalg.norm(self.feature_matrix, axis=1)))
        self.L = float(feature_norm)
        if self.L < max_norm - 1e-9:
            raise ValueError(
                f"feature_norm={self.L:.4f} too small; need >= {max_norm:.4f}. "
                f"Raise feature_norm or lower step_scale / util_range / util_floor.")

        # running realized-rho tracker (for use inside the real GICA loop)
        self._realized_rho = np.inf

    # ----------------------------------------------------------------- paper objects
    def get_ground_truth(self):
        return {pid: float(np.mean(self.feature_matrix[s] @ self.true_theta))
                for pid, s in self.paths.items()}

    def oracle_callback(self, step_idx):
        clean = float(self.feature_matrix[int(step_idx)] @ self.true_theta)
        return clean + self.rng.normal(0, self.noise_std)

    def oracle_callback_path(self, path_id):
        g = np.mean(self.feature_matrix[self.paths[int(path_id)]], axis=0)
        return float(g @ self.true_theta) + self.rng.normal(0, self.noise_std)

    def g(self, pid):
        return self._g[int(pid)]

    # ----------------------------------------------------------------- gaps
    def get_boundary_gap(self):
        """Delta_C: min over (top, challenger) gaps == rank-K vs rank-(K+1). Controlled."""
        mus = np.sort(list(self.get_ground_truth().values()))[::-1]
        return float(mus[self.top_k - 1] - mus[self.top_k])

    def get_min_gap(self):
        """Global min gap (random intra-block; NOT Delta_C -- see get_boundary_gap)."""
        mus = np.sort(list(self.get_ground_truth().values()))
        gp = np.diff(mus); gp = gp[gp > 1e-12]
        return float(gp.min()) if gp.size else 0.0

    def get_c0_tilde(self, lambda_reg):
        return float(lambda_reg / (lambda_reg + self.L ** 2))

    def within_path_score_std(self):
        return float(np.mean([np.std(self.feature_matrix[s] @ self.true_theta)
                              for s in self.paths.values()]))

    # ----------------------------------------------------------------- measure rho_dagger
    @staticmethod
    def _cos2(diff, X, Minv=None):
        if Minv is None:
            dg = diff; xv = X
        else:
            dg = diff @ Minv; xv = X @ Minv
        num = (X @ dg) ** 2
        den = float(diff @ dg) * np.einsum('ij,ij->i', xv, X)
        ok = den > 1e-18
        return num[ok] / den[ok]

    def estimate_rho_dagger(self, lambda_reg=1.0, steps="union", pairs="boundary",
                            n_pairs=3000, per_pair_steps=80, seed=999):
        """STATIC measurement of Assumption 3.2's rho_dagger at the base geometry
        A = lambda I (Euclidean). pairs='boundary' -> over C*_K = {(top, challenger)}
        (the set the proof/algorithm use); 'all' -> over arbitrary pairs (typically ~0).
        steps='union' -> against the pair's own steps U_t (what GICA may query);
        'all' -> against every step (the literal assumption; lower)."""
        rng = np.random.default_rng(seed)
        worst = np.inf
        if pairs == "boundary":
            ti = self._top_ids[rng.integers(0, len(self._top_ids), n_pairs)]
            cj = self._chal_ids[rng.integers(0, len(self._chal_ids), n_pairs)]
            P = zip(ti, cj)
        else:
            ii = rng.integers(0, self.num_paths, n_pairs)
            jj = rng.integers(0, self.num_paths, n_pairs)
            P = ((a, b) for a, b in zip(ii, jj) if a != b)
        all_steps = np.arange(len(self.feature_matrix))
        for p, q in P:
            diff = self._g[int(p)] - self._g[int(q)]
            if np.linalg.norm(diff) < 1e-12:
                continue
            if steps == "union" and pairs == "boundary":
                su = np.array(self.paths[int(p)] + self.paths[int(q)])
            else:
                su = all_steps
            if len(su) > per_pair_steps:
                su = su[rng.integers(0, len(su), per_pair_steps)]
            c = self._cos2(diff, self.feature_matrix[su])
            if c.size:
                worst = min(worst, float(c.min()))
        return worst

    def measure_assumption_3_2(self, lambda_reg=1.0):
        """Convenience: the headline numbers for this random instance."""
        return {
            "rho_boundary_allsteps_euclid": self.estimate_rho_dagger(steps="all"),
            "rho_boundary_Ut_euclid": self.estimate_rho_dagger(steps="union"),
            "rho_allpairs_euclid": self.estimate_rho_dagger(pairs="all", steps="all"),
            "rho_realized_greedy_sim": self.simulate_realized_rho(lambda_reg),
        }

    def simulate_realized_rho(self, lambda_reg=1.0, n_rounds=200, seed=7):
        """Fallback realized rho when GICA is not instrumented. Mimics GICA on the
        binding boundary pair: each round greedily query its best admissible step
        (argmax C_t over U_t) and update V_t. Because querying the same pair piles
        mass into V_t, the per-round best alignment DECLINES; we report the min over
        rounds -- the realized floor that governs the contraction. This is only a
        proxy; the trustworthy number comes from record_rho() on the real run."""
        V = lambda_reg * np.eye(self.dim)
        p = int(self._top_ids[-1]); q = int(self._chal_ids[0])     # binding pair
        diff = self._g[p] - self._g[q]
        su = np.array(self.paths[p] + self.paths[q])
        X = self.feature_matrix[su]
        worst = np.inf
        for _ in range(n_rounds):
            Vi = np.linalg.inv(V)
            dg = X @ (diff @ Vi)
            cscore = dg ** 2 / (1.0 + np.einsum('ij,ij->i', X @ Vi, X))   # C_t(s)
            k = int(np.argmax(cscore))
            c = self._cos2(diff, X[k:k + 1], Vi)
            if c.size:
                worst = min(worst, float(c[0]))
            x = X[k]; V += np.outer(x, x)
        return worst

    # ---------------- Assumption 3.2, measured along the REAL trajectory ----------------
    # rho_t(V_t)   = min over ALL pair-differences Delta g AND ALL steps s of
    #                cos^2_{V_t^{-1}}( g(pi_p,pi_p'), x_s )       (the literal assumption)
    # rho_dagger   = min_t rho_t   over the whole run.
    #
    # Usage inside the real loop (after each GICA.select_and_update, which updates V_t):
    #     env.update_realized_rho(np.linalg.inv(GICA.V))      # or GICA.V_inv
    # then read env.get_realized_rho() once the run finishes.

    def reset_realized_rho(self):
        self._realized_rho = np.inf

    def rho_t(self, V_inv=None, n_pairs=3000, n_steps=3000, seed=None, pairs="all"):
        """Assumption-3.2 quantity at the CURRENT geometry V_t (pass V_inv = inv(V_t);
        omit for the base geometry A = lambda I). Minimum of the squared V_t^{-1}-cosine
        over pair-differences and steps.
          pairs='all'      -> all ordered path pairs (the literal assumption; ~0 in a
                              generic d-dim instance, since some Delta g is ~orthogonal
                              to some step under some V_t).
          pairs='boundary' -> only C*_K = {(top, challenger)} (the set the proof uses).
          n_pairs/n_steps=None -> use ALL pairs / ALL steps (exact; small M only)."""
        rng = np.random.default_rng(seed)
        M = self.num_paths
        g = self._g
        if pairs == "boundary":
            if n_pairs is None:
                ii = np.repeat(self._top_ids, len(self._chal_ids))
                jj = np.tile(self._chal_ids, len(self._top_ids))
            else:
                ii = self._top_ids[rng.integers(0, len(self._top_ids), n_pairs)]
                jj = self._chal_ids[rng.integers(0, len(self._chal_ids), n_pairs)]
        else:
            if n_pairs is None:
                ii, jj = np.triu_indices(M, k=1)
            else:
                ii = rng.integers(0, M, n_pairs); jj = rng.integers(0, M, n_pairs)
                m = ii != jj; ii, jj = ii[m], jj[m]
        diff = g[ii] - g[jj]
        diff = diff[np.linalg.norm(diff, axis=1) > 1e-12]
        if diff.shape[0] == 0:
            return np.inf
        if n_steps is None:
            X = self.feature_matrix
        else:
            X = self.feature_matrix[rng.integers(0, len(self.feature_matrix), n_steps)]
        dG = diff if V_inv is None else diff @ V_inv
        XV = X if V_inv is None else X @ V_inv
        diff_n2 = np.einsum('ij,ij->i', dG, diff)        # Delta g^T V^{-1} Delta g
        x_n2 = np.einsum('ij,ij->i', XV, X)              # x^T V^{-1} x
        worst = np.inf
        CH = 256
        for a in range(0, dG.shape[0], CH):
            cross = dG[a:a + CH] @ X.T                   # Delta g^T V^{-1} x
            den = diff_n2[a:a + CH][:, None] * x_n2[None, :]
            ok = den > 1e-18
            if ok.any():
                cos2 = np.where(ok, cross ** 2 / np.where(ok, den, 1.0), np.inf)
                worst = min(worst, float(cos2.min()))
        return worst

    def update_realized_rho(self, V_inv=None, n_pairs=3000, n_steps=3000, seed=None, pairs="all"):
        """Compute rho_t at the current V_t and fold it into the running rho_dagger=min_t rho_t."""
        r = self.rho_t(V_inv, n_pairs=n_pairs, n_steps=n_steps, seed=seed, pairs=pairs)
        if np.isfinite(r):
            self._realized_rho = min(self._realized_rho, r)
        return r

    def get_realized_rho(self):
        """rho_dagger for this run = min over recorded rounds of rho_t."""
        return float(self._realized_rho)

    def diagnose_geometry(self):
        c = self._g - self._g.mean(axis=0, keepdims=True)
        return {"rank": int(np.linalg.matrix_rank(c, tol=1e-8)),
                "frac_var_along_u": float(np.var(self._g @ self.theta_hat) /
                                          max(np.sum(np.var(self._g, axis=0)), 1e-12))}


def _report(env, lam=1.0):
    geo = env.diagnose_geometry(); rho = env.measure_assumption_3_2(lam)
    print(f"  M={env.num_paths:5d} d={env.dim} Delta_C={env.get_boundary_gap():.4e} "
          f"(min_gap={env.get_min_gap():.4e}) c0_tilde={env.get_c0_tilde(lam):.4e} "
          f"rank(G)={geo['rank']} within_path_std={env.within_path_score_std():.3e} "
          f"max||x||={np.max(np.linalg.norm(env.feature_matrix,axis=1)):.2f}<=L={env.L}")
    print(f"        rho_dagger MEASURED:  C*_K(U_t)={rho['rho_boundary_Ut_euclid']:.4f}  "
          f"C*_K(all steps)={rho['rho_boundary_allsteps_euclid']:.4f}  "
          f"all-pairs={rho['rho_allpairs_euclid']:.4f}  "
          f"realized-greedy={rho['rho_realized_greedy_sim']:.4f}")


if __name__ == "__main__":
    CFG = dict(dim=8, noise_std=0.1, path_len_min=20, path_len_max=80,
               theta_norm=1.0, util_floor=0.30, grid_gap=1e-1,
               step_scale=0.30, util_range=1.0, feature_norm=8.0, top_k=10)
    print("--- random instances: Delta_C controlled, rho_dagger measured ---")
    for M in (200, 500, 1000):
        _report(ReasoningEnvironment(num_paths=M, seed=0, **CFG))
    print("\n--- same M, different seeds: rho_dagger varies (it is emergent, not set) ---")
    for sd in (0, 1, 2, 3):
        e = ReasoningEnvironment(num_paths=500, seed=sd, **CFG)
        print(f"  seed={sd}: Delta_C={e.get_boundary_gap():.4f} (exact)  "
              f"rho_dagger[C*,U_t]={e.estimate_rho_dagger(steps='union'):.4f}  "
              f"realized-greedy={e.simulate_realized_rho():.4f}")
