import os
import sys
import json
from statistics import mean
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from text_utils import strip_string

sys.path.append('..')
from prm_evaluation.genprm_inference import GenPRM
from prm_evaluation.genprm_inference import CodeExecutor


# ======================
# Init model (ONLY ONCE)
# ======================
genprm = GenPRM('GenPRM/GenPRM-1.5B', tensor_parallel_size=1)
code_executor = CodeExecutor()

SYSTEM_MSG = {
    "content": "You are a math teacher. Your task is to review and critique the paragraphs in solution step by step.",
    "role": "system"
}


# ======================
# FAST PATH EVALUATION
# ======================
def evaluate_path(prompt, path):

    reward_sum = 0.0
    reward_count = 0

    messages = [SYSTEM_MSG.copy()]

    steps = path.split(".\n")

    messages.append({
        "role": "user",
        "content": "Question: " + prompt + "\n\n" + steps[0]
    })
    messages.append({"role": "assistant", "content": ""})

    for step in steps[1:]:
        messages.append({"role": "user", "content": step})
        messages.append({"role": "assistant", "content": ""})
    #print("messages",len(messages))
    # IMPORTANT:
    # avoid messages[:i] slicing (huge speedup)
    context = []

    for i, msg in enumerate(messages):

        if msg["role"] == "assistant":
            try:
                output, reward = genprm.inference(
                    context,
                    cur_step=i,
                    code_executor=code_executor,
                    logging=False
                )

                msg["content"] = output[0]
                reward_sum += reward
                reward_count += 1

            except Exception:
                pass

        context.append(msg)

    if reward_count == 0:
        return 0.0

    return reward_sum / reward_count


# ======================
# LOAD DATA
# ======================
with open("data/Deepseek-Math-RL-7B.json") as f:
	data = json.load(f)


exact_match = 0
mismatches = 0


# ======================
# MAIN LOOP
# ======================
for idx in range(len(data["answer"])):

    prompt = data["prompt"][idx]
    paths = data["completion"][idx]
    print("paths",len(paths))

    path_scores = [0.0] * len(paths)

    #  PARALLEL EXECUTION ACROSS PATHS
    # Adjust workers based on GPU capacity
    with ThreadPoolExecutor(max_workers=4) as executor:

        futures = {
            executor.submit(evaluate_path, prompt, path): i
            for i, path in enumerate(paths)
        }

        for future in as_completed(futures):
            i = futures[future]
            path_scores[i] = future.result()

    print("index", idx)

    # faster argmax (no numpy needed)
    max_idx = max(range(len(path_scores)), key=path_scores.__getitem__)

    winning_path = paths[max_idx].lower().replace("\n", "")
    winning_path = winning_path.split("the answer is:")[-1]
    winning_path = strip_string(winning_path.replace("$", ""))

    ground_truth = strip_string(data["answer"][idx])

    print("winning_path**", winning_path, ground_truth.strip().lower())

    if "boxed" in winning_path:
        winning_path = winning_path.replace("\\boxed{","").replace("}","")

    if winning_path.strip().lower() == ground_truth.strip().lower():
        exact_match += 1
    else:
        mismatches += 1

    print("EM so far", exact_match / (exact_match + mismatches))

print("Final EM", exact_match / (exact_match + mismatches))
