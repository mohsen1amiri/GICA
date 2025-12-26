import numpy as np
from P_GIHA import P_GIHA

# --- 1. Synthetic Environment Generator (Linear) ---
class ReasoningEnvironment:
    def __init__(self, num_paths=20, num_total_steps=100, dim=10,
                 noise_std=0.1, path_len_min=5, path_len_max=20, seed=None):

        """
        Generates a synthetic reasoning landscape with Linear Scores.
        """
        self.dim = dim
        self.num_paths = num_paths
        self.noise_std = noise_std
        self.path_len_min = path_len_min
        self.path_len_max = path_len_max
        self.rng = np.random.default_rng(seed)

        
        # 1. Generate True Parameter Theta*
        # Using a normalized vector
        self.true_theta = self.rng.normal(0, 1, size=dim)
        self.true_theta /= np.linalg.norm(self.true_theta)
        
        # 2. Generate Feature Matrix
        self.feature_matrix = self.rng.normal(0, 1, size=(num_total_steps, dim))
        self.feature_matrix /= np.linalg.norm(self.feature_matrix, axis=1, keepdims=True)
        
        # 3. Construct Paths with full coverage and random lengths (auto min/max)
        if num_total_steps < num_paths:
            raise ValueError("num_total_steps must be >= num_paths to give each path at least 1 step.")

        rng = self.rng


        # Average steps per path
        base = num_total_steps // num_paths

        # How much variability you want (0.0 -> almost equal lengths, larger -> more variation)
        variation = 0.30
        spread = max(1, int(base * variation))

        MIN_LEN = max(1, base - spread)
        MAX_LEN = base + spread + 1   # +1 ensures enough slack to always fit the remainder

        # Start each path at MIN_LEN
        lengths = np.full(num_paths, MIN_LEN, dtype=int)
        remaining = num_total_steps - num_paths * MIN_LEN

        # Each path can receive up to (MAX_LEN - MIN_LEN) extra steps
        caps = np.full(num_paths, MAX_LEN - MIN_LEN, dtype=int)

        if remaining > caps.sum():
            # In practice should not happen with MAX_LEN = base + spread + 1, but keep it safe.
            raise ValueError("Auto MIN/MAX produced insufficient capacity. Increase MAX_LEN or variation.")

        # Efficient random allocation of the remaining steps under caps:
        # Create a "pool" with each path id repeated 'cap' times, then sample 'remaining' items.
        pool = np.repeat(np.arange(num_paths), caps)
        chosen = rng.choice(pool, size=remaining, replace=False)
        lengths += np.bincount(chosen, minlength=num_paths)

        # Now assign each step exactly once using a random permutation
        perm = rng.permutation(num_total_steps)

        self.paths = {}
        start = 0
        for pid in range(num_paths):
            L = int(lengths[pid])
            self.paths[pid] = perm[start:start + L].tolist()
            start += L

        # Optional sanity checks:
        assert start == num_total_steps
        assert len(set().union(*map(set, self.paths.values()))) == num_total_steps



    def get_ground_truth(self):
        """
        Calculates TRUE score = Average(x^T * theta) for each path.
        No sigmoid. Pure linear aggregation.
        """
        true_scores = {}
        for pid, steps in self.paths.items():
            path_feats = self.feature_matrix[steps]
            
            # Linear Score: x^T * theta
            step_scores = np.dot(path_feats, self.true_theta)
            
            # Path Score = Average of step scores
            true_scores[pid] = np.mean(step_scores)
        return true_scores

    def oracle_callback(self, step_idx):
        """
        Simulates the Oracle with Linear Feedback.
        Returns: x^T * theta + Noise
        """
        feat = self.feature_matrix[step_idx]
        
        # Pure Linear Response
        clean_score = np.dot(feat, self.true_theta)
        
        # Add Gaussian Noise (Standard assumption for Linear Bandits)
        noise = self.rng.normal(0, self.noise_std)

        
        return clean_score + noise


# --- 2. Simulation Runner ---

def run_synthetic_experiment():
    print("--- Setting up Synthetic Environment (Linear Mode) ---")
    
    # Setup Parameters
        # ---------------------------
    # Experiment config
    # ---------------------------
    SEED = 0
    ENV_CFG = dict(
        num_paths=200,
        num_total_steps=50000,
        dim=8,
        noise_std=0.1,
        path_len_min=200,
        path_len_max=300,   # exclusive if you use randint(min, max)
    )

    ALG_CFG = dict(
        m=3,
        lambda_reg=1.0,
        epsilon=0.05,
        delta=0.01,
        R=0.1,        # should match noise_std scale
        S_0=2.0,
        step_pool_mode="paths",   # "paths" (boundary union) or "all"
    )

    max_iter = 3000

    print("SEED:", SEED)
    print("ENV_CFG:", ENV_CFG)
    print("ALG_CFG:", ALG_CFG)
    
    env = ReasoningEnvironment(**ENV_CFG, seed=SEED)

    M_TOP = ALG_CFG["m"]
    DIM = ENV_CFG["dim"]

    # Compute Ground Truth
    true_scores = env.get_ground_truth()
    sorted_truth = sorted(true_scores.items(), key=lambda x: x[1], reverse=True)
    true_top_m = [pid for pid, score in sorted_truth[:M_TOP]]
    
    print(f"Ground Truth Top-{M_TOP} Paths: {true_top_m}")
    print(f"Scores (Linear): Best={sorted_truth[0][1]:.4f}, Worst={sorted_truth[-1][1]:.4f}")
    print("-" * 30)


    p_giha = P_GIHA(
        paths=env.paths,
        feature_matrix=env.feature_matrix,
        m=ALG_CFG["m"],
        d=ENV_CFG["dim"],
        lambda_reg=ALG_CFG["lambda_reg"],
        epsilon=ALG_CFG["epsilon"],
        delta=ALG_CFG["delta"],
        R=ALG_CFG["R"],
        S_0=ALG_CFG["S_0"],
        step_pool_mode=ALG_CFG["step_pool_mode"],
    )


    
    # Loop
    print(f"Starting P-GIHA Optimization...")
    converged = False
    
    for t in range(max_iter):
        # Oracle now returns a float (linear score + noise) instead of 0/1
        converged, estimated_top_m = p_giha.select_and_update(env.oracle_callback)
        
        if t % 2 == 0:
             print(f"Iter {t}: Current Estimate {estimated_top_m}")
         
             
        if converged:
            print("-" * 30)
            print(f"Converged at iteration {t}!")
            print(f"Final Estimated Top-{M_TOP}: {estimated_top_m}")
            print(f"True Ground Truth Top-{M_TOP}: {true_top_m}")
            
            correct_ids = set(estimated_top_m).intersection(set(true_top_m))
            print(f"Accuracy: {len(correct_ids)}/{M_TOP} correct.")
            print(f"Total arm pulls (oracle queries): {p_giha.t}")
            converged = True
            break
            
    if not converged:
        print("Max iterations reached.")

if __name__ == "__main__":
    run_synthetic_experiment()