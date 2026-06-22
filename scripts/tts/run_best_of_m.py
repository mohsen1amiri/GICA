"""
run_best_of_m.py
================

Exhaustive Best-of-M verification baseline (the accuracy upper bound) using ThinkPRM-1.5B.

Unlike the bandit selectors, this scores *every step of every candidate path* for every
question: paths are streamed into fixed-size batches and sent to the verifier in single GPU
calls. Once all paths of a question are scored, the path with the highest mean step-label
wins, and its final answer is graded by Exact-Match. Per-question path scores are written
to a JSON file.

Run from the repository root (the dataset path and output filename are hardcoded to AIME;
edit them to switch benchmark)::

    python scripts/tts/run_best_of_m.py
"""

import json
from collections import defaultdict
from gica.tts.answer_extraction import strip_string
from gica.tts.verifier import ThinkPRM
from statistics import mean, stdev
import time

# ======================
# CONFIG
# ======================
BATCH_SIZE = 64
prm = ThinkPRM(
    model_name_or_path="launch/ThinkPRM-1.5B",
    max_length=4096,
    temperature=0.0,
    n=1
)

# ======================
# LOAD DATA
# ======================
with open("data/Deepseek-AIME-RL-7B.json") as f:
    data = json.load(f)

# ======================
# PREP TRACKING STRUCTURES
# ======================
scores_dict = {}                 # (q_idx, path_id) -> mean step-label score
remaining_paths = {}             # q_idx -> number of paths not yet scored
processed_question = set()       # q_idx values already graded
for idx in range(len(data["answer"][:400])):
    remaining_paths[idx] = len(data["completion"][idx])
exact_match = 0
mismatches = 0

# ======================
# STREAM BUFFER
# ======================
buffer_questions = []
buffer_steps = []
buffer_meta = []   # (q_idx, path_id)
responses = {"qid":[],"path_scores":[]}


def flush_buffer():
    """Score the buffered (question, path) pairs in one GPU call and grade completed questions.

    Runs ThinkPRM on the batched prefixes, records each path's mean step-label score, and
    for every question whose paths are now fully scored selects the highest-scoring path,
    extracts and normalizes its answer, and updates the running Exact-Match tally. Clears
    the buffer at the end.
    """
    global exact_match, mismatches
    if not buffer_questions:
        return

    # Single batched verifier call for the whole buffer.
    results = prm.predict_correctness_batch(
        questions=buffer_questions,
        prefix_steps_batch=buffer_steps
    )

    # Store per-path scores; grade a question once all its paths are scored.
    for meta, res in zip(buffer_meta, results):
        q_idx, path_id = meta
        scores_dict[(q_idx, path_id)] = mean(res["step_labels"][0])
        remaining_paths[q_idx] -= 1
        print("ssss",len(res["step_labels"][0]))

        # Question complete: pick the best path and score Exact-Match.
        if remaining_paths[q_idx] == 0 and q_idx not in processed_question:
            processed_question.add(q_idx)
            paths = data["completion"][q_idx]
            path_scores = [
                scores_dict[(q_idx, i)] for i in range(len(paths))
            ]
            responses["qid"].append(q_idx)
            responses["path_scores"].append(path_scores)
            max_idx = max(range(len(path_scores)), key=path_scores.__getitem__)
            winning_path = paths[max_idx].lower().replace("\n", "")
            winning_path = winning_path.split("the answer is:")[-1]
            if "\\boxed" in winning_path:
                winning_path = winning_path.replace("\\boxed{","")
                k = winning_path.rfind("}")
                winning_path = winning_path[:k]
            winning_path = strip_string(winning_path.replace("$", ""))
            ground_truth = strip_string(data["answer"][q_idx])
            print("winning_path**", path_scores,q_idx, winning_path, ground_truth.strip().lower())
            if winning_path.strip().lower() == ground_truth.strip().lower():
                exact_match += 1
            else:
                mismatches += 1
            print("EM so far", exact_match/(exact_match+mismatches))

    # Reset the buffer for the next batch.
    buffer_questions.clear()
    buffer_steps.clear()
    buffer_meta.clear()


# ======================
# STREAMING LOOP
# ======================
times = []
for q_idx in range(len(data["answer"][:400])):
    start = time.time()
    prompt = data["prompt"][q_idx]
    paths = data["completion"][q_idx]
    for path_id, path in enumerate(paths):
        steps = path.split(".\n")
        buffer_questions.append(prompt)
        buffer_steps.append(steps)
        buffer_meta.append((q_idx, path_id))
        if len(buffer_questions) >= BATCH_SIZE:
            flush_buffer()
    end = time.time()
    times.append(end-start)
    with open("final_path_scores_AIME.json","w") as f:
        json.dump(responses,f)

# Flush any paths still buffered after the loop.
flush_buffer()
print("Final EM", exact_match/(exact_match+mismatches), mean(times), stdev(times))
