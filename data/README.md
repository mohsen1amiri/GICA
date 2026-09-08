# Datasets

The test-time-scaling experiments (Track 2) consume **pre-generated candidate reasoning
paths**, not raw benchmarks. Each file contains, per question: the prompt, the list of
`M = 100` candidate chain-of-thought solutions produced by the generator LLM, and the
ground-truth answer.

## Download

**Download the JSON data from:**

> https://osf.io/v7muk/

The download is a single `data.zip` containing a `data/` folder. Unzip it at the
repository root so its contents land in **this `data/` folder**.


## Expected files

The archive contains one candidate-path file per (generator, benchmark) pair:

```
data/
├── Deepseek-Math-RL-7B.json               # MATH-500,    generator = DeepSeekMath-RL-7B
├── Deepseek-MathOdyssey-RL-7B.json        # MathOdyssey, generator = DeepSeekMath-RL-7B
├── Deepseek-AIME-RL-7B.json               # AIME,        generator = DeepSeekMath-RL-7B
├── InternLM2-Math-Plus-7B.json            # MATH-500,    generator = InternLM2-Math-Plus-7B
├── InternLM2-Math-MathOdyssey-Plus-7B.json # MathOdyssey, generator = InternLM2-Math-Plus-7B
├── InternLM2-AIME-7B.json                 # AIME,        generator = InternLM2-Math-Plus-7B
└── math-500-idx.json                      # question-index subset ({"ids": [...]}),
                                           #   for run_orm_rerank_v2.py --index_file
```


## Schema

```jsonc
{
  "prompt":     ["<question 1>", "<question 2>", ...],
  "completion": [ ["<path 1>", "<path 2>", ..., "<path M>"],   // M candidate CoTs for Q1
                  ["<path 1>", ...],                            // for Q2
                  ... ],
  "answer":     ["<ground truth 1>", "<ground truth 2>", ...]
}
```

`prompt`, `completion`, and `answer` are parallel lists indexed by question.
