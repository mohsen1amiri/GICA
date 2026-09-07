"""
run_orm_rerank.py
=================

ORM-based reranking baseline: score every COMPLETE candidate path once with an
outcome-level verifier (no step-level verification), then select the answer by

Run from the repository root::

    # generative ORM (no new model; ThinkPRM-1.5B in outcome mode)
    python scripts/tts/run_orm_rerank.py \
        --file_path data/Deepseek-AIME-RL-7B.json --dataset_name AIME

    # discriminative ORM (any seq-classification reward model on HF)
    python scripts/tts/run_orm_rerank.py \
        --file_path data/Deepseek-AIME-RL-7B.json --dataset_name AIME \
        --orm_backend seqcls --orm_model <hf-reward-model-name>
"""

import argparse
import os
import sys
import time
import json
from statistics import mean, stdev

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_common import (grade_answer_string, grade_path_best_of_m,
                             load_dataset, save_json, weighted_self_con,
                             write_rows_csv)


def main():
    ap = argparse.ArgumentParser(description="ORM reranking baseline")
    ap.add_argument("--file_path", type=str, required=True)
    ap.add_argument("--dataset_name", type=str, default="dataset")
    ap.add_argument("--data_limit", type=int, default=10**9)
    ap.add_argument("--orm_backend", type=str, default="thinkprm",
                    choices=["thinkprm", "seqcls", "rlhflow"])
    ap.add_argument("--orm_model", type=str, default=None,
                    help="HF reward-model name (seqcls backend only)")
    ap.add_argument("--prm_model", type=str, default="launch/ThinkPRM-1.5B",
                    help="verifier used by the thinkprm outcome backend")
    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    data = load_dataset(args.file_path, args.data_limit)

    # ------------------------------------------------------------------
    # Build the ORM (heavy imports deferred so --help works anywhere)
    # ------------------------------------------------------------------
    from gica.tts.verifier.orm import build_orm
    if args.orm_backend == "thinkprm":
        from gica.tts.verifier import ThinkPRM
        prm = ThinkPRM(model_name_or_path=args.prm_model,
                       max_length=args.max_length, temperature=0.0, n=1)
        orm = build_orm("thinkprm", prm=prm)
    else:
        orm = build_orm(args.orm_backend, orm_model=args.orm_model,
                        max_length=args.max_length)

    # ------------------------------------------------------------------
    # Score, rerank, grade
    # ------------------------------------------------------------------
    em_argmax = em_wvote = 0
    rows, times = [], []
    all_scores = {"qid": [], "orm_scores": []}
    done = 0
    with open("/mimer/NOBACKUP/groups/naiss2025-5-631/gica_repo/GICA/data/data/math-500-idx.json") as f:
        ids = json.load(f)
    for q_idx in ids["ids"]:
    #for q_idx in range(data["n_questions"]):
        t0 = time.time()
        question = data["prompt"][q_idx]
        paths = data["completion"][q_idx][:100]

        scores, gen_tokens = orm.score_paths(question, paths,
                                             batch_size=args.batch_size)
        dt = time.time() - t0
        times.append(dt)
        all_scores["qid"].append(q_idx)
        all_scores["orm_scores"].append(scores)

        # (a) argmax rerank — winning PATH graded with the run_best_of_m rule
        best = max(range(len(paths)), key=scores.__getitem__)
        hit_a = grade_path_best_of_m(paths[best], data["answer"][q_idx])
        # (b) ORM-weighted vote — self_con normalization + comparison
        print("response*****",weighted_self_con(paths, scores)[0][0], data["answer"][q_idx])
        hit_w = grade_answer_string(weighted_self_con(paths, scores)[0][0],
                                    data["answer"][q_idx])

        em_argmax += hit_a
        em_wvote += hit_w
        rows.append([q_idx, len(paths), gen_tokens, round(dt, 3), hit_a, hit_w])
        done = done + 1
        print(f"[q{q_idx}] calls={len(paths)} gen_toks={gen_tokens} "
              f"time={dt:.1f}s | EM argmax={em_argmax/done:.4f} "
              f"weighted-vote={em_wvote/done:.4f}")

        # checkpoint scores every question (cheap; enables cascade reuse)
        save_json(f"orm_scores_{args.dataset_name}.json", all_scores)

    n = data["n_questions"]
    print(f"\nFinal ORM rerank EM (argmax)        : {em_argmax/n:.4f}")
    print(f"Final ORM rerank EM (weighted vote) : {em_wvote/n:.4f}")
    print(f"ORM calls/question = M = {len(data['completion'][0])}, "
          f"time/question = {mean(times):.1f} +/- "
          f"{(stdev(times) if len(times) > 1 else 0):.1f}s")

    write_rows_csv(f"orm_rerank_{args.orm_backend}_{args.dataset_name}.csv",
                   ["qid", "orm_calls", "gen_tokens", "time", "EM_argmax",
                    "EM_weighted_vote"], rows)


if __name__ == "__main__":
    main()
