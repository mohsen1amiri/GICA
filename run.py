import numpy as np
from P_GIHA import P_GIHA

# --- 1. Synthetic Environment Generator (Linear) ---
class ReasoningEnvironment:
    def __init__(self, num_paths=20, num_total_steps=100, dim=10):
        """
        Generates a synthetic reasoning landscape with Linear Scores.
        """
        self.dim = dim
        self.num_paths = num_paths
        
        # 1. Generate True Parameter Theta*
        # Using a normalized vector
        self.true_theta = np.random.normal(0, 1, size=dim)
        self.true_theta /= np.linalg.norm(self.true_theta)
        
        # 2. Generate Feature Matrix
        self.feature_matrix = np.random.normal(0, 1, size=(num_total_steps, dim))
        self.feature_matrix /= np.linalg.norm(self.feature_matrix, axis=1, keepdims=True)
        
        # 3. Construct Paths (Variable Lengths)
        self.paths = {}
        for pid in range(num_paths):
            path_len = np.random.randint(5, 20)
            path_indices = np.random.choice(num_total_steps, size=path_len, replace=False)
            self.paths[pid] = list(path_indices)

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
        noise = np.random.normal(0, 0.1) 
        
        return clean_score + noise


# --- 2. Simulation Runner ---

def run_synthetic_experiment():
    print("--- Setting up Synthetic Environment (Linear Mode) ---")
    
    # Setup Parameters
    N_PATHS = 200
    M_TOP = 3       
    DIM = 8        
    TOTAL_STEPS = 50000 
    
    env = ReasoningEnvironment(num_paths=N_PATHS, num_total_steps=TOTAL_STEPS, dim=DIM)
    
    # Compute Ground Truth
    true_scores = env.get_ground_truth()
    sorted_truth = sorted(true_scores.items(), key=lambda x: x[1], reverse=True)
    true_top_m = [pid for pid, score in sorted_truth[:M_TOP]]
    
    print(f"Ground Truth Top-{M_TOP} Paths: {true_top_m}")
    print(f"Scores (Linear): Best={sorted_truth[0][1]:.4f}, Worst={sorted_truth[-1][1]:.4f}")
    print("-" * 30)

    # Initialize P-GIHA Algorithm
    p_giha = P_GIHA(
        paths=env.paths,
        feature_matrix=env.feature_matrix,
        m=M_TOP,
        d=DIM,
        epsilon=0.05 
    )
    
    # Loop
    print(f"Starting P-GIHA Optimization...")
    max_iter = 300
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
            break
            
    if not converged:
        print("Max iterations reached.")

if __name__ == "__main__":
    run_synthetic_experiment()
