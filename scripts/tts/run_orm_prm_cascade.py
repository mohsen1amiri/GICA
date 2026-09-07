"""
run_orm_prm_cascade.py
======================

ORM -> PRM cascaded pipeline a cheap
outcome-level ORM first scores ALL M complete paths, then expensive step-level
ThinkPRM verification is applied ONLY to the top-C shortlist, whose winner is
chosen exactly like `run_best_of_m.py` (argmax of mean step labels; a
PRM-weighted vote over the shortlist is also reported).

    # self-contained (generative ORM = ThinkPRM outcome mode)
    python scripts/tts/run_orm_prm_cascade.py \
        --file_path data/Deepseek-AIME-RL-7B.json --dataset_name AIME \
        --cascade_top_c 5

    # reuse scores from run_orm_rerank.py (skips phase 1 entirely)
    python scripts/tts/run_orm_prm_cascade.py \
        --file_path data/Deepseek-AIME-RL-7B.json --dataset_name AIME \
        --orm_scores orm_scores_AIME.json --cascade_top_c 5

    # discriminative ORM front-end
    python scripts/tts/run_orm_prm_cascade.py \
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
from baseline_common import (count_tokens, grade_answer_string,
                             grade_path_best_of_m, load_dataset,
                             load_json_if_exists, save_json,
                             weighted_self_con, write_rows_csv)


def label_mean(res):
    """Mean step-label score of one verified path (run_best_of_m winner metric)."""
    labels = res["step_labels"][0]
    if not labels:
        return -1.0
    return float(sum(labels)) / len(labels)


def main():
    ap = argparse.ArgumentParser(description="ORM->PRM cascade baseline")
    ap.add_argument("--file_path", type=str, required=True)
    ap.add_argument("--dataset_name", type=str, default="dataset")
    ap.add_argument("--data_limit", type=int, default=10**9)
    ap.add_argument("--cascade_top_c", type=int, default=5,
                    help="shortlist size passed to step-level verification "
                         "(default 5 = GICA's K)")
    ap.add_argument("--orm_scores", type=str, default=None,
                    help="orm_scores_<dataset>.json from run_orm_rerank.py; "
                         "if given, phase 1 is skipped")
    ap.add_argument("--orm_backend", type=str, default="thinkprm",
                    choices=["thinkprm", "seqcls", "rlhflow"])
    ap.add_argument("--orm_model", type=str, default=None)
    ap.add_argument("--prm_model", type=str, default="launch/ThinkPRM-1.5B")
    ap.add_argument("--max_length", type=int, default=4096)
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    data = load_dataset(args.file_path, args.data_limit)
    with open("/mimer/NOBACKUP/groups/naiss2025-5-631/gica_repo/GICA/data/data/math-500-idx.json") as f:
        ids = json.load(f)
    n_q = data["n_questions"]

    # ==================================================================
    # PHASE 1 — outcome-level scores for every path of every question
    # ==================================================================
    #orm_times = [0.0] * len(ids["ids"])
    orm_times = [0.0] * n_q
    cached = load_json_if_exists(args.orm_scores)
    prm = None  # the vLLM engine, created lazily

    if cached is not None:
        score_by_qid = dict(zip(cached["qid"], cached["orm_scores"]))
        #orm_scores = [score_by_qid[q] for q in ids["ids"]]

        orm_scores = [score_by_qid[q] for q in range(n_q)]
        print(f"[phase 1] loaded cached ORM scores from {args.orm_scores}")
    else:
        from gica.tts.verifier.orm import build_orm
        if args.orm_backend == "thinkprm":
            from gica.tts.verifier import ThinkPRM
            prm = ThinkPRM(model_name_or_path=args.prm_model,
                           max_length=args.max_length, temperature=0.0, n=1)
            orm = build_orm("thinkprm", prm=prm)
        else:
            orm = build_orm(args.orm_backend, orm_model=args.orm_model,
                            max_length=args.max_length)

        orm_scores = []
        for q_idx in range(n_q):
            t0 = time.time()
            scores, _ = orm.score_paths(data["prompt"][q_idx],
                                        data["completion"][q_idx],
                                        batch_size=args.batch_size)
            orm_times[q_idx] = time.time() - t0
            orm_scores.append(scores)
            print(f"[phase 1][q{q_idx}] scored {len(scores)} paths "
                  f"in {orm_times[q_idx]:.1f}s")
        save_json(f"orm_scores_{args.dataset_name}.json",
                  {"qid": list(range(n_q)), "orm_scores": orm_scores})
        orm.free()  # no-op for thinkprm backend; frees GPU for seqcls

    # ==================================================================
    # PHASE 2 — step-level PRM verification of the top-C shortlist only
    # ==================================================================
    if prm is None:
        from gica.tts.verifier import ThinkPRM
        prm = ThinkPRM(model_name_or_path=args.prm_model,
                       max_length=args.max_length, temperature=0.0, n=1)

    C = args.cascade_top_c
    em_argmax = em_wvote = 0
    rows, times = [], []
    tracker = 0
    #for q_idx in ids["ids"]:
    for q_idx in range(n_q):
        t0 = time.time()
        question = data["prompt"][q_idx]
        paths = data["completion"][q_idx][:100]
#        scores = orm_scores[tracker]
        scores = orm_scores[q_idx]

        shortlist = sorted(range(len(paths)), key=scores.__getitem__,
                           reverse=True)[:C]

        # one step-level verifier call per shortlisted path, batched together
        # (identical to the run_best_of_m call pattern: all steps in one CoT)
        results = prm.predict_correctness_batch(
            questions=[question] * len(shortlist),
            prefix_steps_batch=[paths[pid].split(".\n") for pid in shortlist],
        )
        prm_scores = [label_mean(res) for res in results]
        gen_tokens = count_tokens(
            prm.tokenizer, [o for res in results for o in res.get("outputs", [])])

        dt = time.time() - t0
        times.append(dt)

        # winner (a): argmax mean step labels, PATH graded with the
        # run_best_of_m rule verbatim
        best_pid = shortlist[max(range(len(shortlist)),
                                 key=prm_scores.__getitem__)]
        #print("*******", paths[best_pid], data["answer"][q_idx])
        hit_a = grade_path_best_of_m(paths[best_pid], data["answer"][q_idx])
        # winner (b): PRM-weighted vote over the shortlist — self_con
        # normalization + comparison
        hit_w = grade_answer_string(
            weighted_self_con([paths[pid] for pid in shortlist],
                              prm_scores)[0][0],
            data["answer"][q_idx])

        em_argmax += hit_a
        em_wvote += hit_w
        tracker+=1
        rows.append([q_idx, len(paths), len(shortlist), gen_tokens,
                     round(orm_times[tracker], 3), round(dt, 3), hit_a, hit_w])
        
        print(f"[phase 2][q{q_idx}] orm_calls={len(paths)} "
              f"prm_calls={len(shortlist)} gen_toks={gen_tokens} "
              f"time={dt:.1f}s | EM argmax={em_argmax/tracker:.4f} "
              f"weighted-vote={em_wvote/tracker:.4f}")

    print(f"\nFinal cascade EM (argmax step-labels): {em_argmax/n_q:.4f}")
    print(f"Final cascade EM (PRM-weighted vote) : {em_wvote/n_q:.4f}")
    print(f"Cost/question: {len(data['completion'][0])} ORM calls "
          f"+ {C} step-level PRM calls "
          f"(exhaustive Best-of-M = {len(data['completion'][0])} PRM calls); "
          f"phase-2 time/question = {mean(times):.1f} +/- "
          f"{(stdev(times) if len(times) > 1 else 0):.1f}s")

    write_rows_csv(
        f"cascade_{args.orm_backend}_top{C}_{args.dataset_name}.csv",
        ["qid", "orm_calls", "prm_calls", "prm_gen_tokens", "orm_time",
         "prm_time", "EM_argmax", "EM_weighted_vote"], rows)


if __name__ == "__main__":
    main()
