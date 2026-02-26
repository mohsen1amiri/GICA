import numpy as np
import json
from collections import defaultdict
from text_utils import strip_string
from prm import ThinkPRM
from statistics import mean, stdev
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import json
from sentence_transformers import SentenceTransformer
from prm import ThinkPRM
from P_GIHA_llm_setup import P_GIHA
from text_utils import strip_string

def compute_em_from_top_m(question, paths, answers, q_idx, top_m):

    # -------------------------
    # Select winning path
    # -------------------------
    winning_pid = top_m[0]
    print("top m", [paths[pid_win].split("The answer is:")[-1] for pid_win in top_m])
    winning_path = paths[winning_pid]

    # -------------------------
    # Extract predicted answer
    # -------------------------
    winning_path = winning_path.lower().replace("\n", "")

    winning_path = winning_path.split("the answer is:")[-1]
    if "\\boxed" in winning_path:
        winning_path = winning_path.replace("\\boxed{","")
        k = winning_path.rfind("}")
        winning_path = winning_path[:k] #+ "" + winning_path[k+1:]
    pred = strip_string(winning_path.replace("$", ""))

    # -------------------------
    # Ground truth
    # -------------------------
    gt = strip_string(answers[q_idx])

    print("winning_path**", pred, gt.lower())

    em = int(pred.strip().lower() == gt.strip().lower())

    return em
    
# ---------------------------
# Models
# ---------------------------
prm = ThinkPRM(
    model_name_or_path="launch/ThinkPRM-1.5B",
    temperature=0.0,
    max_length=3000,
    n=1
)

embedder = SentenceTransformer("all-MiniLM-L6-v2",device="cuda:1")


# ---------------------------
# PRM Environment
# ---------------------------
class PRMEnvironment:

    def __init__(self, question, paths):

        self.question = question
        self.paths_raw = paths
        self.paths = {}
        self.step_to_path = {}

        self.step_texts = []

        sid = 0
        for pid, path in enumerate(paths):

            steps = path.split(".\n") #path.split("\n") if len(s.strip()) > 0]
            self.paths[pid] = []

            for step in steps:
                self.step_texts.append(step)
                self.paths[pid].append(sid)
                sid += 1
        for pid, ids in self.paths.items():
            for sid in ids:
                self.step_to_path[sid] = pid
        # embeddings
        self.step_emb = embedder.encode(self.step_texts, normalize_embeddings=True)
        self.q_emb = embedder.encode([question], normalize_embeddings=True)[0]

        self.global_centroid = np.mean(self.step_emb, axis=0)
        self.global_centroid /= np.linalg.norm(self.global_centroid)

        self.feature_matrix = self.build_features()


# def build_features(self):

#     feats = []
#     emb = self.step_emb[step_ids]
#     for i, e in enumerate(self.step_emb):

#         pid = self.step_to_path[i]
#         path_ids = self.paths[pid]
#         pos_idx = path_ids.index(i)

#         # -----------------------------
#         # TERM-LEVEL PROJECTIONS
#         # -----------------------------
#         sim_q_vec = self.q_tok_emb @ e
#         sim_path_vec = self.path_tok_emb @ e
#         sim_global_vec = self.global_tok_emb @ e

#         pos_norm = pos_idx / max(1, len(path_ids))

#         feat_vec = np.concatenate([
#             sim_q_vec,
#             sim_path_vec,
#             sim_global_vec,
#             np.array([pos_norm, 1.0])
#         ])

#         feats.append(feat_vec)

#     feats = np.array(feats, dtype=np.float32)

#     # VERY IMPORTANT: column scaling only
#     feats = feats / (np.std(feats, axis=0, keepdims=True) + 1e-6)

#     return feats
    def build_features(self):

        feats = []

        # --- Precompute path centroids ---
        path_centroids = {}
        # path_dispersion = {}

        for pid, step_ids in self.paths.items():

            emb = self.step_emb[step_ids]

            centroid = np.mean(emb, axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-8
            path_centroids[pid] = centroid

        #     dists = np.linalg.norm(emb - centroid, axis=1)
        #     path_dispersion[pid] = float(np.mean(dists))

        # --- Build base features ---
        for i, e in enumerate(self.step_emb):

            pid = self.step_to_path[i]
            path_ids = self.paths[pid]
            pos_idx = path_ids.index(i)
           # print("emb",e.shape,path_centroids[pid].shape,self.global_centroid.shape)
            cos_q = cosine_similarity(e.reshape(1,-1) , self.q_emb.reshape(1,-1))[0][0]
            cos_path = cosine_similarity(e.reshape(1,-1) , path_centroids[pid].reshape(1,-1))[0][0]
            cos_global = cosine_similarity(e.reshape(1,-1) , self.global_centroid.reshape(1,-1))[0][0]
            #print("cos_q",cos_q)
            # prefix_ids = path_ids[:pos_idx + 1]
            # prefix_emb = self.step_emb[prefix_ids]

            # prefix_centroid = np.mean(prefix_emb, axis=0)
            # prefix_centroid /= np.linalg.norm(prefix_centroid) + 1e-8
            # cos_prefix = float(e @ prefix_centroid)

            if pos_idx > 0:
                prev_e = self.step_emb[path_ids[pos_idx - 1]]
                cos_prev = float(e @ prev_e)
            else:
                cos_prev = 0.0

            pos_norm = pos_idx / max(1, len(path_ids))
            length_norm = min(len(self.step_texts[i]) / 200.0, 1.0)
           # dispersion = path_dispersion[pid]
           # print("cos_q",cos_q,cos_path,cos_global)
            # IMPORTANT: last slot reserved for dynamic boundary feature
            feats.append([
                cos_q,
                cos_path,
                cos_global,
                # cos_prefix,
                # cos_prev,
                # dispersion,
                pos_norm,
               # length_norm,
                1.0,   # bias
                0.0    # boundary feature placeholder
            ])
       # print("feats",feats)

        feats = np.array(feats)

        #feats = feats / (np.std(feats, axis=0, keepdims=True) + 1e-8)
      #  print("feats",feats)
        return feats

    def find_path(self, step_id):
        for pid, ids in self.paths.items():
            if step_id in ids:
                return pid

    # -----------------------------------
    # ThinkPRM Oracle
    # -----------------------------------
    def oracle_callback(self, step_idx):

        pid = self.find_path(step_idx)
        pos = self.paths[pid].index(step_idx)

        prefix_ids = self.paths[pid][:pos + 1]
        prefix_steps = [self.step_texts[i] for i in prefix_ids]

        res = prm.predict_correctness_batch(
            questions=[self.question],
            prefix_steps_batch=[prefix_steps]
        )[0]

        return float(res["prefix_score"])


# ---------------------------
# Experiment Runner
# ---------------------------
def run_prm_experiment(question, paths, answers, q_idx):

    env = PRMEnvironment(question, paths)

    p_giha = P_GIHA(
        paths=env.paths,
        feature_matrix=env.feature_matrix,
        m=5,
        d=env.feature_matrix.shape[1],
        lambda_reg=1.0,
        epsilon=0.15,
        delta=0.05,
        R=1.0,
        S_0=1.0,
        step_pool_mode="paths",
    )
    p_giha.env_step_emb = env.step_emb
    p_giha.patience = 0
    previous_J = None
    t=0
    done = False
    while not done and p_giha.patience <10:
        converged, top_m = p_giha.select_and_update(env.oracle_callback)
        if previous_J == top_m:
            p_giha.patience+=1
        else:
            p_giha.patience=0
        previous_J = top_m
        
        print(f"Iter {t} | Top paths: {top_m}")
        t+=1

        if converged:
            print("Converged!")
            done=True
            break
    em = compute_em_from_top_m(
        question,
        paths,
        answers,
        q_idx,
        top_m
    )

    return em


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":

    times = []

    with open("/home/dsv/vevi4591/GIHA/GenPRM/src/data/Deepseek-MathOdyssey-RL-7B.json") as f:
        data = json.load(f)
    exact_match = 0
    total=0
    for idx in range(len(data["answer"])):
        start = time.time()
        print("\n============================")
        print(f"QUESTION {idx}")
        em = run_prm_experiment(
            data["prompt"][idx],#.replace("Please reason step by step, and put your final answer within \\boxed{}",""),
            data["completion"][idx],
            data["answer"],
            idx
        )

        exact_match += em
        total += 1
        end = time.time()
        times.append(end-start)

        print("EM so far:", exact_match / total, mean(times))

    print("Final EM:", (exact_match / total), mean(times),stdev(times))
