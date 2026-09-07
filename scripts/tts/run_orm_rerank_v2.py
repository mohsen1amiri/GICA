"""
run_orm_rerank_v2.py
====================

ORM-based reranking baseline (the "ORM" row of Table 1): score every COMPLETE
candidate path once with an outcome-level verifier (no step-level verification),
then select the answer by (a) argmax of the ORM score and (b) an ORM-weighted
vote over the normalized answers. Both EM variants are reported.

Cost is M outcome-level calls per question, against M step-level calls for
exhaustive Best-of-M.

Outputs: `orm_rerank_<backend>_<dataset>.csv` (per-question calls, generated
tokens, time and both EM variants) and `orm_scores_<dataset>.json`, which
`run_orm_prm_cascade.py --orm_scores ...` can reuse to skip its phase 1.

Run from the repository root::

    # generative ORM (no new model; ThinkPRM-1.5B in outcome mode)
    python scripts/tts/run_orm_rerank_v2.py \
        --file_path data/Deepseek-AIME-RL-7B.json --dataset_name AIME

    # discriminative ORM (any seq-classification reward model on HF)
    python scripts/tts/run_orm_rerank_v2.py \
        --file_path data/Deepseek-AIME-RL-7B.json --dataset_name AIME \
        --orm_backend seqcls --orm_model <hf-reward-model-name>

By default every question in --file_path is scored. Pass --index_file to
restrict the run to a saved list of question indices instead (a JSON file
holding {"ids": [...]}); earlier revisions of this script read such a list from
a fixed cluster path.
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
    ap.add_argument("--file_path", type=str, required=True,
                    help="pre-generated-paths JSON under data/, e.g. "
                         "data/Deepseek-AIME-RL-7B.json")
    ap.add_argument("--dataset_name", type=str, default="dataset",
                    help="short benchmark tag used in the output filenames, "
                         "e.g. Math500, MathOdyssey, AIME")
    ap.add_argument("--data_limit", type=int, default=10**9,
                    help="use only the first N questions (default: all)")
    ap.add_argument("--index_file", type=str, default=None,
                    help="optional JSON {\"ids\": [...]} restricting the run "
                         "to those question indices (default: every question)")
    ap.add_argument("--orm_backend", type=str, default="thinkprm",
                    choices=["thinkprm", "seqcls", "rlhflow"],
                    help="thinkprm reuses --prm_model in outcome mode (no "
                         "extra weights); seqcls/rlhflow load the separate HF "
                         "reward model given by --orm_model")
    ap.add_argument("--orm_model", type=str, default=None,
                    help="HF reward-model name (seqcls / rlhflow backends only)")
    ap.add_argument("--prm_model", type=str, default="launch/ThinkPRM-1.5B",
                    help="verifier used by the thinkprm outcome backend")
    ap.add_argument("--max_length", type=int, default=4096,
                    help="verifier / reward-model context length")
    ap.add_argument("--batch_size", type=int, default=64,
                    help="paths scored per ORM call")
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
    if args.index_file:
        with open(args.index_file) as f:
            q_indices = json.load(f)["ids"]
    else:
        q_indices = range(data["n_questions"])
    for q_idx in q_indices:
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

    print(f"\nFinal ORM rerank EM (argmax)        : {em_argmax/done:.4f}")
    print(f"Final ORM rerank EM (weighted vote) : {em_wvote/done:.4f}")
    print(f"ORM calls/question = M = {len(data['completion'][0])}, "
          f"time/question = {mean(times):.1f} +/- "
          f"{(stdev(times) if len(times) > 1 else 0):.1f}s")

    write_rows_csv(f"orm_rerank_{args.orm_backend}_{args.dataset_name}.csv",
                   ["qid", "orm_calls", "gen_tokens", "time", "EM_argmax",
                    "EM_weighted_vote"], rows)


if __name__ == "__main__":
    main()
