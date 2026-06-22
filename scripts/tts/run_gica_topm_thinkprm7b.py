"""
run_gica_topm_thinkprm7b.py
===========================

GICA driver variant using the larger ThinkPRM-7B verifier (Appendix C.2 robustness
ablation / Figure 5). CLI-driven (dataset path and name), it re-ranks the certified top-5
shortlist by mean step-labels before grading.

For each question it builds a :class:`PRMEnvironment`, runs GICA's top-5 identification loop
against the ThinkPRM oracle, selects the winner from the shortlist, and scores it by
Exact-Match. Per-question verifier-call counts and timings are written to a CSV. This
variant also uses a slightly relaxed patience rule (stop when the shortlist has recurred
among the last few rounds).

Run from the repository root, e.g.::

    python scripts/tts/run_gica_topm_thinkprm7b.py \\
        --file_path data/Deepseek-MathOdyssey-RL-7B.json --dataset_name MathOdyssey
"""

import numpy as np
import json
from collections import defaultdict
from gica.tts.answer_extraction import strip_string
from gica.tts.verifier import ThinkPRM
import pandas as pd
import argparse
from statistics import mean, stdev
import time
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import json
from sentence_transformers import SentenceTransformer
from gica.tts.verifier import ThinkPRM
from gica.tts.selection.gica import GICA
from gica.tts.answer_extraction import strip_string


def compute_em_from_top_m(question, paths, answers, q_idx, top_m):
    """Re-rank the shortlist by mean step-labels, extract the winner, and return its EM.

    Each path in ``top_m`` is re-scored by the mean of ThinkPRM's per-step labels; the
    highest-scoring path wins. Its final answer is parsed (handling "the answer is" /
    "the final answer is" and ``\\boxed{...}``), normalized via ``strip_string``, and
    compared to the ground truth.
    """
    # -------------------------
    # Select winning path
    # -------------------------
    scores_prefix = []
    for pid in top_m:
        path =  paths[pid]
        steps = path.split(".\n")

        res = prm.predict_correctness_batch(
            questions=[question],
            prefix_steps_batch=[steps]
        )[0]
        print("final_ranking",steps,res["step_labels"][0])
        scores_prefix.append(mean(res["step_labels"][0]))
    winning_pid = sorted(zip(top_m,scores_prefix), key= lambda x: x[1], reverse=True)[0][0]
    print("top m", [paths[pid_win].split("The answer is:")[-1] for pid_win in top_m])
    winning_path = paths[winning_pid]

    # -------------------------
    # Extract predicted answer
    # -------------------------
    winning_path = winning_path.lower().replace("\n", "")
    if "the answer is" in winning_path:
        winning_path = winning_path.split("the answer is:")[-1]
    else:
        winning_path = winning_path.split("the final answer is")[-1]
    if "\\boxed" in winning_path:
        winning_path = winning_path.replace("\\boxed{","")
        k = winning_path.rfind("}")
        winning_path = winning_path[:k]
    pred = strip_string(winning_path.replace("$", ""))

    # -------------------------
    # Ground truth
    # -------------------------
    gt = strip_string(answers[q_idx])

    print("winning_path**", pred, gt.lower())

    em = int(pred.strip().lower() == gt.strip().lower())

    return em


# ---------------------------
# Models (loaded once at import; require a GPU + vLLM)
# ---------------------------
prm = ThinkPRM(
    model_name_or_path="launch/ThinkPRM-7B",
    temperature=0.0,
    max_length=3000,
    n=1
)

embedder = SentenceTransformer("all-MiniLM-L6-v2",device="cuda")


# ---------------------------
# PRM Environment
# ---------------------------
class PRMEnvironment:
    """Turns a question and its candidate paths into a step-level bandit instance.

    Splits each path into steps (on ``".\\n"``), embeds them, and exposes
    :meth:`oracle_callback` (the ThinkPRM bridge) plus the step feature matrix used by GICA.
    """

    def __init__(self, question, paths):

        self.question = question
        self.paths_raw = paths
        self.paths = {}
        self.step_to_path = {}

        self.step_texts = []

        # Split every path into steps; record step<->path membership.
        sid = 0
        for pid, path in enumerate(paths):

            steps = path.split(".\n")
            self.paths[pid] = []

            for step in steps:
                self.step_texts.append(step)
                self.paths[pid].append(sid)
                sid += 1
        for pid, ids in self.paths.items():
            for sid in ids:
                self.step_to_path[sid] = pid

        # Step / question embeddings and the global step centroid.
        self.step_emb = embedder.encode(self.step_texts, normalize_embeddings=True)
        self.q_emb = embedder.encode([question], normalize_embeddings=True)[0]

        self.global_centroid = np.mean(self.step_emb, axis=0)
        self.global_centroid /= np.linalg.norm(self.global_centroid)

        self.feature_matrix = self.build_features()

    def build_features(self):
        """Build the per-step feature matrix.

        Each step's feature is the 6-vector
        ``[cos(step, question), cos(step, path-centroid), cos(step, global-centroid),
        position-fraction, bias=1, boundary-placeholder=0]``. The final slot is reserved
        for the dynamic boundary feature written by GICA. (Additional candidate features
        were explored but are disabled, and the optional column normalization is left off.)
        """
        feats = []

        # Precompute the per-path centroid in embedding space.
        path_centroids = {}

        for pid, step_ids in self.paths.items():

            emb = self.step_emb[step_ids]

            centroid = np.mean(emb, axis=0)
            centroid /= np.linalg.norm(centroid) + 1e-8
            path_centroids[pid] = centroid

        # Build the base feature for each step.
        for i, e in enumerate(self.step_emb):

            pid = self.step_to_path[i]
            path_ids = self.paths[pid]
            pos_idx = path_ids.index(i)

            cos_q = cosine_similarity(e.reshape(1,-1) , self.q_emb.reshape(1,-1))[0][0]
            cos_path = cosine_similarity(e.reshape(1,-1) , path_centroids[pid].reshape(1,-1))[0][0]
            cos_global = cosine_similarity(e.reshape(1,-1) , self.global_centroid.reshape(1,-1))[0][0]

            if pos_idx > 0:
                prev_e = self.step_emb[path_ids[pos_idx - 1]]
                cos_prev = float(e @ prev_e)
            else:
                cos_prev = 0.0

            pos_norm = pos_idx / max(1, len(path_ids))
            length_norm = min(len(self.step_texts[i]) / 200.0, 1.0)

            # Last slot reserved for the dynamic boundary feature.
            feats.append([
                cos_q,
                cos_path,
                cos_global,
                pos_norm,
                1.0,   # bias
                0.0    # boundary feature placeholder
            ])

        feats = np.array(feats)

        return feats

    def find_path(self, step_id):
        """Return the path id that contains ``step_id``."""
        for pid, ids in self.paths.items():
            if step_id in ids:
                return pid

    # -----------------------------------
    # ThinkPRM Oracle
    # -----------------------------------
    def oracle_callback(self, step_idx):
        """Score the within-path prefix up to ``step_idx`` and return ThinkPRM's prefix score."""
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
    """Run one question end-to-end with GICA (ThinkPRM-7B); return (EM, num_rounds).

    Runs GICA's top-5 identification loop until it converges or a relaxed patience rule
    fires: the patience counter advances when the shortlist repeats the previous round or
    (after a few rounds) matches any of the last three shortlists, and the loop stops once
    patience reaches 6.
    """
    env = PRMEnvironment(question, paths)

    p_giha = GICA(
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
    # Attach step embeddings for GICA's dynamic boundary feature and init the patience state.
    p_giha.env_step_emb = env.step_emb
    p_giha.patience = 0
    previous_J = None
    t=0
    done = False
    previous_topms = []
    while not done and p_giha.patience <6:
        converged, top_m = p_giha.select_and_update(env.oracle_callback)
        if previous_J == top_m or ( t > 2 and (previous_topms[-1] == top_m or previous_topms[-2] == top_m or previous_topms[-3] == top_m )):
            p_giha.patience+=1
        else:
            p_giha.patience=0
        previous_J = top_m
        previous_topms.append(top_m)

        print(f"Iter {t} | Top paths: {top_m}", p_giha.patience)
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

    return em, t


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Linear Top-m identification Bandit Algorithms for TTS verification ')
    parser.add_argument('--file_path', type=str, 
                    help='dataset file path: example: ./data/Deepseek-MathOdyssey-RL-7B.json')
    parser.add_argument('--dataset_name', type=str, 
                    help='dataset file path: example: Mathodyssey, AIME')
    times = []
    args = parser.parse_args()

    with open("{}".format(args.file_path)) as f:
        data = json.load(f)
    exact_match = 0
    total=0
    iterations_data = {"qid":[],"iter":[], "time":[],"EM":[]}
    for idx in range(len(data["answer"])):
        idx1 = idx
        start = time.time()
        print("\n============================")
        print(f"QUESTION {idx1}")
        em, iterat = run_prm_experiment(
            data["prompt"][idx1],
            data["completion"][idx1],
            data["answer"],
            idx1
        )


        exact_match += em
        iterations_data["qid"].append(idx+1)
        iterations_data["iter"].append(iterat)
        total += 1
        end = time.time()
        times.append(end-start)
        iterations_data["time"].append(end-start)
        iterations_data["EM"].append(em)
        print("EM so far:", exact_match / total, mean(times))
        iterations_data_1 = pd.DataFrame(iterations_data)
        iterations_data_1.to_csv("qid_iterations_GIHA_{}_over_top_m.csv".format(args.dataset_name),index=False)

    print("Final EM:", (exact_match / total), mean(times),stdev(times))
