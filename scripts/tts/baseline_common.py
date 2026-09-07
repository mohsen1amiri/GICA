"""
baseline_common.py
==================

Shared helpers for the ORM drivers (``run_orm_rerank_v2.py`` and
``run_orm_prm_cascade.py``). It exists so those two scripts grade with exactly
the same rules as the reference drivers rather than with a second, subtly
different implementation:

- ``grade_path_best_of_m`` is the winning-path EM rule of ``run_best_of_m.py``;
- ``self_con`` / ``self_con_answer`` are the normalization and tally of
  ``run_majority_vote.py`` (kept verbatim, ``\boxed`` handling included, so the
  numbers stay comparable), and ``weighted_self_con`` is their score-weighted
  variant used for the ORM/PRM-weighted vote;
- ``load_dataset`` reads the ``data/`` JSON schema (parallel ``prompt`` /
  ``completion`` / ``answer`` lists) and truncates it to ``--data_limit``;
- the remaining helpers write the per-question CSV / JSON outputs.

This module is imported by path (the drivers put their own directory on
``sys.path``), so it is not part of the installable ``gica`` package.
"""

import csv
import json
import os

import os, sys
try:
    import gica  # noqa: F401
except ModuleNotFoundError:
    _REPO_SRC = os.path.abspath(os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
    if os.path.isdir(_REPO_SRC) and _REPO_SRC not in sys.path:
        sys.path.insert(0, _REPO_SRC)
from gica.tts.answer_extraction import strip_string


# ----------------------------------------------------------------------
# Data loading (schema: parallel lists `prompt`, `completion`, `answer`)
# ----------------------------------------------------------------------
def load_dataset(file_path, data_limit=None):
    """Load a pre-generated-paths JSON file from `data/` and truncate to `data_limit` questions."""
    with open(file_path) as f:
        data = json.load(f)

    n = len(data["answer"]) if data_limit is None else min(int(data_limit), len(data["answer"]))
    return {
        "prompt": data["prompt"][:n],
        "completion": data["completion"][:n],
        "answer": data["answer"][:n],
        "n_questions": n,
    }


# ======================================================================
# EM rule 1 — winning-PATH grading, verbatim from run_best_of_m.py
# (identical logic in run_gica.py's compute_em_from_top_m)
# ======================================================================
def grade_path_best_of_m(path_text, ground_truth_raw):
    """Return EM (0/1) for a winning path, using the run_best_of_m.py rule verbatim."""
    winning_path = path_text.lower().replace("\n", "")
    if "the answer is" in winning_path:
            winning_path = winning_path.split("the answer is:")[-1]
    else:
            winning_path = winning_path.split("the final answer is")[-1]
   # winning_path = winning_path.split("the answer is:")[-1]
    # if "\\boxed" in winning_path:
    #     winning_path = winning_path.replace("\\boxed{", "")
    #     k = winning_path.rfind("}")
    #     winning_path = winning_path[:k]
    winning_path = strip_string(winning_path.replace("$", ""))
    ground_truth = strip_string(ground_truth_raw)
    print("winning_path**",winning_path, ground_truth)
    if winning_path.strip().lower() == ground_truth.strip().lower():
        return 1
    return 0


# ======================================================================
# ======================================================================
def self_con(tmp_list, verbose=False):
    """Self-consistency vote over a list of candidate solution strings.

    VERBATIM copy of run_majority_vote.py::self_con — same normalization
    (note: intentionally no \\boxed handling), same tally, same ordering.
    Only change: the tally `print(d)` is behind `verbose` so the bootstrap
    sweep doesn't flood stdout; it does not affect the returned value.
    """
    ans_list = []
    for tmp in tmp_list:
        ans = ""
        tmp = tmp.replace("\n", "")
        tmp = tmp.lower().replace("\n", "")
        if "the answer is" in tmp:
            tmp = tmp.split("the answer is:")[-1]
        # else:
        #     tmp = tmp.split("the final answer is")[-1]
        #tmp = tmp.lower().split("the answer is:")[-1]
        # if "\\boxed" in tmp:
        #     tmp = tmp.replace("\\boxed{","")
        #     k = tmp.rfind("}")
        #     tmp = tmp[:k] #+ "" + winning_path[k+1:]
        tmp = tmp.replace("$", "")
        tmp = strip_string(tmp)
        ans_list.append(tmp)
    # Tally normalized answers.
    d = {}
    for i in ans_list:
        if i in d:
            d[i] += 1
        else:
            d[i] = 1
    if verbose:
        print(d)
    n = sorted(d.items(), key=lambda x: x[1], reverse=True)
    return n


def self_con_answer(tmp):
    """Normalize ONE candidate with the exact run_majority_vote.py per-item rule."""
    tmp = tmp.lower().replace("\n", "")
    if "the answer is" in tmp:
        tmp = tmp.split("the answer is:")[-1]
    else:
        tmp = tmp.split("the final answer is")[-1]
    #tmp = tmp.lower().split("the answer is:")[-1]
    # if "\\boxed" in tmp:
    #     tmp = tmp.replace("\\boxed{","")
    #     k = tmp.rfind("}")
    #     tmp = tmp[:k] #+ "" + winning_path[k+1:]
    tmp = tmp.replace("$", "")
    tmp = strip_string(tmp)
    return tmp


def weighted_self_con(tmp_list, scores):
    """Score-weighted variant of self_con: same normalization, weighted tally.

    Returns (answer, total_weight) pairs sorted by descending weight, so
    `weighted_self_con(paths, scores)[0][0]` is the weighted-vote winner —
    same access pattern as `self_con(paths)[0][0]`.
    """
    d = {}
    for tmp, s in zip(tmp_list, scores):
        a = self_con_answer(tmp)
        d[a] = d.get(a, 0.0) + float(s)
    return sorted(d.items(), key=lambda x: x[1], reverse=True)


def grade_answer_string(winning_answer, ground_truth_raw):
    """Grade a voted answer string — the exact final comparison of run_majority_vote.py."""
    ground_truth = strip_string(ground_truth_raw)
    if winning_answer.strip().lower() == ground_truth.strip().lower():
        return 1
    return 0


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def write_rows_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    print(f"[baseline_common] wrote {path} ({len(rows)} rows)")


def save_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)
    print(f"[baseline_common] wrote {path}")


def load_json_if_exists(path):
    if path and os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def count_tokens(tokenizer, texts):
    """Total token count of a list of strings (used to log verification decode tokens)."""
    return sum(len(tokenizer.encode(t, add_special_tokens=False)) for t in texts if t)
