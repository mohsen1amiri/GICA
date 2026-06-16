# Datasets

The test-time-scaling experiments (Track 2) consume **pre-generated candidate reasoning
paths**, not raw benchmarks. Each file contains, per question: the prompt, the list of
`M = 100` candidate chain-of-thought solutions produced by the generator LLM, and the
ground-truth answer.

## Download

**Download the JSON data from:**

> https://osf.io/v7muk/overview?view_only=aa5acf15fbcd4d0db38d3f53f480dc52

and drop the files into **this `data/` folder** (repository root → `data/`).


## Expected files

The driver scripts reference files named `<Generator>-<Benchmark>-RL-7B.json`, e.g.:

```
data/
├── Deepseek-Math-RL-7B.json          # MATH-500,     generator = DeepSeekMath-RL-7B
├── Deepseek-MathOdyssey-RL-7B.json   # MathOdyssey,  generator = DeepSeekMath-RL-7B
├── Deepseek-AIME-RL-7B.json          # AIME,         generator = DeepSeekMath-RL-7B
├── InternLM2-Math-RL-7B.json         # MATH-500,     generator = InternLM2-Math-Plus-7B
├── InternLM2-MathOdyssey-RL-7B.json  # MathOdyssey,  generator = InternLM2-Math-Plus-7B
└── InternLM2-AIME-RL-7B.json         # AIME,         generator = InternLM2-Math-Plus-7B
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
