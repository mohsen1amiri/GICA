import json
from collections import defaultdict
from text_utils import strip_string
from prm import ThinkPRM
from statistics import mean

# ======================
# CONFIG
# ======================
BATCH_SIZE = 64

prm = ThinkPRM(
    model_name_or_path="launch/ThinkPRM-1.5B",
    max_length=2048,
    temperature=0.0,
    n=1
)

# ======================
# LOAD DATA
# ======================
with open("data/Deepseek-MathOdyssey-RL-7B.json") as f:
    data = json.load(f)

# ======================
# PREP TRACKING STRUCTURES
# ======================
scores_dict = {}
remaining_paths = {}
processed_question = set()

for idx in range(len(data["answer"])):
    remaining_paths[idx] = len(data["completion"][idx])

exact_match = 0
mismatches = 0

# ======================
# STREAM BUFFER
# ======================
buffer_questions = []
buffer_steps = []
buffer_meta = []   # (q_idx, path_id)

def flush_buffer():
    global exact_match, mismatches

    if not buffer_questions:
        return

    # SINGLE GPU CALL
    results = prm.predict_correctness_batch(
        questions=buffer_questions,
        prefix_steps_batch=buffer_steps
    )

    # STORE RESULTS
    for meta, res in zip(buffer_meta, results):

        q_idx, path_id = meta

        scores_dict[(q_idx, path_id)] = mean(res["step_labels"][0])
        remaining_paths[q_idx] -= 1
        print("ssss",len(res["step_labels"][0]))

        # CHECK IF QUESTION COMPLETE
        if remaining_paths[q_idx] == 0 and q_idx not in processed_question:

            processed_question.add(q_idx)

            paths = data["completion"][q_idx]

            path_scores = [
                scores_dict[(q_idx, i)] for i in range(len(paths))
            ]

            max_idx = max(range(len(path_scores)), key=path_scores.__getitem__)

            winning_path = paths[max_idx].lower().replace("\n", "")

            winning_path = winning_path.split("the answer is:")[-1]
            if "\\boxed" in winning_path:
                winning_path = winning_path.replace("\\boxed{","").replace("}","")
            winning_path = strip_string(winning_path.replace("$", ""))

            ground_truth = strip_string(data["answer"][q_idx])

            print("winning_path**", path_scores,q_idx, winning_path, ground_truth.strip().lower())

            if winning_path.strip().lower() == ground_truth.strip().lower():
                exact_match += 1
            else:
                mismatches += 1

            print("EM so far", exact_match/(exact_match+mismatches))

    # CLEAR BUFFER
    buffer_questions.clear()
    buffer_steps.clear()
    buffer_meta.clear()


# ======================
# STREAMING LOOP
# ======================
for q_idx in range(len(data["answer"])):

    prompt = data["prompt"][q_idx]
    paths = data["completion"][q_idx]

    for path_id, path in enumerate(paths):

        steps = path.split(".\n")

        buffer_questions.append(prompt)
        buffer_steps.append(steps)
        buffer_meta.append((q_idx, path_id))

        if len(buffer_questions) >= BATCH_SIZE:
            flush_buffer()

# FLUSH REMAINDER
flush_buffer()

print("Final EM", exact_match/(exact_match+mismatches))
