"""
run_generation.py
=================

Generation-side timing for the generation-vs-verification comparison of
Appendix C.2 (Figure 6).

Every other driver in this folder consumes the *pre-generated* candidate paths
shipped in `data/`, so none of them measures how long the generator itself took.
This script closes that gap: it replays the questions of a benchmark file
through the generator LLM under vLLM, samples M paths per question exactly as
the shipped data was produced, and records the per-question wall-clock. Pair its
mean time/query with the mean time/query printed by `run_best_of_m.py` (the
exhaustive verification side) to obtain the two bars of Figure 6.

Prompts are taken VERBATIM from the `prompt` field of `--file_path` rather than
rebuilt from a template, so the generator sees byte-identical inputs to those
that produced the released paths (the two generators use different chat
formats, and the stored prompts already encode them).

Defaults follow the released files, whose `temperature` / `top_p` fields record
M = 100 paths per question at temperature 1.0 and top-p 0.95; the paper reports
generator temperatures in {1.0, 1.1}.

Outputs:
  * `generation_times_<dataset>.csv` with columns
    `qid, n_paths, gen_tokens, time` -- `time` is the per-question generation
    wall-clock that Figure 6 averages.
  * optionally, with `--save_paths`, a JSON in the `data/` schema
    (`prompt` / `completion` / `answer`) holding the regenerated paths.

Run from the repository root, e.g.::

    # DeepSeekMath-RL-7B on MathOdyssey, the Figure 6 generator
    python scripts/tts/run_generation.py \\
        --file_path data/Deepseek-MathOdyssey-RL-7B.json \\
        --dataset_name MathOdyssey --model deepseek-ai/deepseek-math-7b-rl

    # the InternLM2 generator
    python scripts/tts/run_generation.py \\
        --file_path data/InternLM2-Math-MathOdyssey-Plus-7B.json \\
        --dataset_name MathOdyssey --model internlm/internlm2-math-plus-7b

Requires a CUDA GPU and the Track-2 dependencies (`requirements-tts.txt`).
"""

import argparse
import json
import os
import sys
import time
from statistics import mean, stdev

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_common import load_dataset, save_json, write_rows_csv


def main():
    ap = argparse.ArgumentParser(
        description="Generator-side timing for Figure 6 (Appendix C.2)")
    ap.add_argument("--file_path", type=str, required=True,
                    help="benchmark JSON under data/ supplying the questions "
                         "(its `prompt` entries are replayed verbatim), e.g. "
                         "data/Deepseek-MathOdyssey-RL-7B.json")
    ap.add_argument("--dataset_name", type=str, default="dataset",
                    help="short benchmark tag used in the output filenames, "
                         "e.g. Math500, MathOdyssey, AIME")
    ap.add_argument("--model", type=str,
                    default="deepseek-ai/deepseek-math-7b-rl",
                    help="generator on Hugging Face; the paper uses "
                         "deepseek-ai/deepseek-math-7b-rl and "
                         "internlm/internlm2-math-plus-7b")
    ap.add_argument("--data_limit", type=int, default=10**9,
                    help="generate for only the first N questions (default: all)")
    ap.add_argument("--num_paths", type=int, default=100,
                    help="M, paths sampled per question (paper: 100)")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="sampling temperature (paper: 1.0 or 1.1)")
    ap.add_argument("--top_p", type=float, default=0.95,
                    help="nucleus sampling top-p (released data: 0.95)")
    ap.add_argument("--max_tokens", type=int, default=2048,
                    help="max new tokens per path; the released DeepSeek paths "
                         "run to roughly 1k tokens")
    ap.add_argument("--max_model_len", type=int, default=4096,
                    help="generator context length")
    ap.add_argument("--tensor_parallel_size", type=int, default=1,
                    help="number of GPUs for tensor parallelism")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed")
    ap.add_argument("--save_paths", action="store_true",
                    help="also write the regenerated paths as "
                         "generated_paths_<dataset>.json (data/ schema)")
    args = ap.parse_args()

    data = load_dataset(args.file_path, args.data_limit)

    # Heavy imports deferred so --help works without a GPU / vLLM present.
    from vllm import LLM, SamplingParams

    llm = LLM(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        gpu_memory_utilization=0.98,
        max_model_len=args.max_model_len,
        dtype="half",
    )
    sampling_params = SamplingParams(
        n=args.num_paths,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    rows, times = [], []
    completions = []
    for q_idx in range(data["n_questions"]):
        prompt = data["prompt"][q_idx]

        # One timed call per question: vLLM returns all num_paths samples.
        t0 = time.time()
        out = llm.generate([prompt], sampling_params)[0]
        dt = time.time() - t0

        paths = [o.text for o in out.outputs]
        gen_tokens = sum(len(o.token_ids) for o in out.outputs)
        times.append(dt)
        completions.append(paths)
        rows.append([q_idx, len(paths), gen_tokens, round(dt, 3)])

        print(f"[q{q_idx}] paths={len(paths)} gen_toks={gen_tokens} "
              f"time={dt:.1f}s | running mean={mean(times):.1f}s")

    print(f"\nGeneration time/question = {mean(times):.1f} +/- "
          f"{(stdev(times) if len(times) > 1 else 0):.1f}s "
          f"over {len(times)} questions at M = {args.num_paths}")
    print("Pair this mean with run_best_of_m.py's mean time/question "
          "(exhaustive verification) for the two bars of Figure 6.")

    write_rows_csv(f"generation_times_{args.dataset_name}.csv",
                   ["qid", "n_paths", "gen_tokens", "time"], rows)

    if args.save_paths:
        save_json(f"generated_paths_{args.dataset_name}.json",
                  {"prompt": data["prompt"],
                   "completion": completions,
                   "answer": data["answer"]})


if __name__ == "__main__":
    main()
