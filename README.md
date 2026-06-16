# GICA — The Gap-Index Compositional Arm Framework for Sample-Efficient Test-Time Scaling

> Reference implementation and reproduction package for the paper
> **“GICA: The Gap-Index Compositional Arm Framework for Sample-Efficient Test-Time Scaling.”**

---

## 1. Project Title & Abstract

**Test-time scaling (TTS)** improves the reasoning of large language models (LLMs) by sampling
many candidate chain-of-thought (CoT) solutions and using a **verifier** to select among them.
**Process reward models (PRMs)** that score *every intermediate step* — especially recent
*reasoning-based* PRMs that generate a long verification chain-of-thought before emitting a score —
are the most accurate verifiers, but they are prohibitively expensive: their cost grows with both
the number of candidate paths `M` and the number of steps per path.

**GICA** makes fine-grained, step-level verification practical at scale. It recasts process-level
verification as a **fixed-confidence top-K identification problem over compositional arms**: each
reasoning path is a *parent arm* whose feature is the length-normalized average of its step
features under a *shared* linear utility model. Because every step query updates the shared
parameter, **one verifier call informs every path at once**. GICA adaptively queries only the most
informative steps along the most ambiguous top-vs-challenger boundary, and stops as soon as its
top-K shortlist is statistically certified.

Empirically, GICA **matches the accuracy of exhaustive Best-of-M process verification** while
reducing verifier calls by up to **4.2×** and inference runtime by up to **4.3×** relative to the
strongest bandit baseline, across three math-reasoning benchmarks (MATH-500, MathOdyssey, AIME),
two open-weight generators (DeepSeekMath-RL-7B, InternLM2-Math-Plus-7B), and two reasoning-based
verifiers (ThinkPRM-1.5B, ThinkPRM-7B).

This repository contains **two self-contained experimental tracks**:

- **Track 1 — Synthetic** (`src/gica/synthetic/`): isolates the bandit algorithm’s sample
  efficiency on compositional top-K instances with a known ground-truth parameter. CPU-only,
  deterministic, runs in seconds. Reproduces **Figure 3**.
- **Track 2 — Test-Time Scaling** (`src/gica/tts/` + `scripts/tts/`): the end-to-end TTS pipeline
  with an LLM generator (pre-computed paths), a ThinkPRM verifier, and GICA / baseline selection.
  Requires CUDA GPUs and vLLM. Reproduces **Figures 4–5** and **Tables 1 & 5**.

### A note on naming (important)

The algorithm’s development name was **`P_GIHA`**; throughout the code it has been renamed to its
publication name **`GICA`**. Two further conventions are worth knowing when cross-referencing the
paper:

| In the code                       | In the paper                                 |
|-----------------------------------|----------------------------------------------|
| class `GICA`                      | **GICA** (Algorithm 1)                       |
| constructor argument `m`          | **K** — the size of the top-set to identify  |
| `total_comparisons`               | number of gap-index comparisons              |
| `best_G_history`                  | per-round hardest boundary gap-index `G_t`   |
| `min_lcb_history`                 | stopping quantity `Γ_t = min(Δ̂ − W)`        |
| `XtremeAlg3OnPaths` (IGW)         | an **optional extra** baseline, **not** reported in the paper |

---

## 2. Repository Structure

```text
GICA/
├── README.md                         # this file
├── pyproject.toml                    # installable package definition (provides `import gica`)
├── requirements-synthetic.txt        # Track-1 dependencies (CPU: numpy/scipy/matplotlib)
├── requirements-tts.txt              # Track-2 dependencies (GPU: vLLM/transformers/…)
├── .gitignore
│
├── data/                             # datasets live here (downloaded separately — see §3.3)
│   └── README.md                     # download link + expected filenames + JSON schema
│
├── src/gica/                         # the importable Python package
│   ├── __init__.py
│   │
│   ├── synthetic/                    # ── TRACK 1: synthetic experiments (Figure 3) ──
│   │   ├── environment.py            # ReasoningEnvironment: compositional linear bandit + oracle
│   │   ├── gica.py                   # ★ class GICA — Algorithm 1 (gap-index selection + RLS update)
│   │   ├── baseline_case.py          # CASE baseline (Purohit et al., 2025)
│   │   ├── baseline_lingifa.py       # LinGIFA baseline (Réda et al., 2021)
│   │   ├── baseline_mlingape.py      # m-LinGapE baseline (Xu et al., 2018)
│   │   ├── baseline_igw_extreme.py   # OPTIONAL IGW hierarchical bandit — not in the paper
│   │   └── benchmark.py              # ★ multi-seed benchmark harness + figure generator
│   │
│   └── tts/                          # ── TRACK 2: test-time-scaling library ──
│       ├── verifier/
│       │   ├── __init__.py           # exposes `ThinkPRM`
│       │   └── thinkprm.py           # ★ ThinkPRM-1.5B / 7B wrapper (reasoning-based PRM, vLLM)
│       ├── selection/
│       │   ├── gica.py               # ★ class GICA for TTS (adds the dynamic boundary feature)
│       │   ├── baseline_case.py      # CASE   (TTS adaptation, no compositional feature)
│       │   ├── baseline_lingifa.py   # LinGIFA(TTS adaptation, no compositional feature)
│       │   └── baseline_mlingape.py  # m-LinGapE (TTS adaptation, no compositional feature)
│       ├── utils/
│       │   ├── prompt_template.py    # ThinkPRM verification prompt template
│       │   └── answer_parsing.py     # step-label parsing from the verifier output
│       └── answer_extraction.py      # final-answer extraction + normalization for Exact-Match
│
└── scripts/                          # runnable experiment drivers (entry points)
    ├── synthetic/
    │   └── slurm_benchmark.sbatch    # SLURM launcher for the synthetic benchmark
    └── tts/
        ├── run_gica.py               # ★ GICA TTS driver (ThinkPRM-1.5B; main results)
        ├── run_gica_topm_thinkprm7b.py   # GICA driver, ThinkPRM-7B (CLI args; Table 5 / Fig 5)
        ├── run_gica_topm_steplabels.py   # GICA driver, alternative final-pick by step labels
        ├── run_baselines.py          # CASE / LinGIFA / m-LinGapE TTS driver (CLI args)
        ├── run_best_of_m.py          # ★ exhaustive Best-of-M upper bound (ThinkPRM-1.5B, batched)
        ├── run_top1.py               # Top-1 decoding reference (no verification)
        └── run_majority_vote.py      # majority-vote / self-consistency reference
```

`★` marks the files most central to the paper.

---

## 3. Prerequisites & Installation

The two tracks have **disjoint** dependency sets. Track 1 is CPU-only and tiny; Track 2 needs GPUs
and a heavyweight LLM-serving stack. Install whichever you need (or both).

### 3.0 Common: clone and create the package environment

```bash
git clone <your-repo-url> GICA
cd GICA

# A clean Python 3.10 environment (conda recommended; venv also works)
conda create -n gica python=3.10 -y
conda activate gica

# Install the `gica` package itself (editable, so `import gica` works from anywhere)
pip install -e .
```

### 3.1 Track 1 — Synthetic (CPU)

```bash
pip install -r requirements-synthetic.txt
```

That is all you need to reproduce Figure 3. No GPU, no datasets, no model downloads.

### 3.2 Track 2 — Test-Time Scaling (GPU)

```bash
pip install -r requirements-tts.txt
```

This installs **vLLM 0.7.1**, **transformers 4.48.2**, **sentence-transformers**, etc. (pinned to
the versions used in the paper). You will additionally need access to the following models (they
download automatically on first use via Hugging Face):

- **Verifiers:** `launch/ThinkPRM-1.5B` (main results) and `launch/ThinkPRM-7B` (ablation).
- **Sentence encoder:** `all-MiniLM-L6-v2` (used to embed reasoning steps).

> **Hardware.** Track 2 runs ThinkPRM under vLLM on the GPU. ThinkPRM-1.5B fits on a single
> modern GPU; ThinkPRM-7B benefits from more memory. The drivers set
> `gpu_memory_utilization=0.98` and `tensor_parallel_size=1` by default — adjust inside
> `src/gica/tts/verifier/thinkprm.py` (or the driver call site) for your hardware.

### 3.3 Download the datasets (Track 2 only)

The TTS experiments consume **pre-generated candidate reasoning paths** (not raw benchmarks).

> **Please download the JSON data from**
> **https://drive.google.com/drive/folders/19Pu3OguXDXLguMzY78T9q4YzU2JtNuzT?usp=sharing**
> **and drop the files into the `data/` folder.**

See [`data/README.md`](data/README.md) for the expected filenames and the JSON schema. In short,
each file provides, per question, the `prompt`, a list of `M = 100` candidate `completion` paths,
and the ground-truth `answer`.

---

## 4. Code Architecture & Contents

### 4.1 The big idea (how the data flows)

Both tracks implement the same loop; they differ only in **where the verifier signal comes from**.

```
                 ┌─────────────────── one round of GICA ───────────────────┐
   paths Π  ─▶   │ 1. estimate path utilities μ̂(π) = g(π)·θ̂                │
                 │ 2. form empirical top-K shortlist                        │
                 │ 3. STOP if Γ_t ≥ −ε  (shortlist is certified)            │
                 │ 4. else pick the most ambiguous boundary pair (π⋆, π†)   │
                 │ 5. query the single most informative step on π⋆ ∪ π†  ───┼──▶ verifier → y_t
                 │ 6. Sherman–Morrison update of V⁻¹ and RLS update of θ̂ ◀─┘
                 └──────────────────────────────────────────────────────────┘
```

- **Track 1** replaces the verifier with a synthetic oracle `x_s·θ⋆ + noise`.
- **Track 2** replaces it with **ThinkPRM**, which reads the question + the step’s within-path
  prefix and returns a correctness score.

Because the linear parameter `θ` is **shared across all paths**, a single step observation tightens
the utility estimate of *every* path — this is the compositional information sharing that gives
GICA its `O(1)`-per-round query cost (vs. `O(T_p·M)` for path-arm baselines) and an `M`-independent
sample-complexity bound.

### 4.2 Track 1 — Synthetic (`src/gica/synthetic/`)

- **`environment.py`** — `ReasoningEnvironment` builds a compositional linear bandit instance: a
  unit `true_theta` (`θ⋆`), ℓ2-normalized Gaussian step features, and a full-coverage partition of
  steps into variable-length paths. `get_ground_truth()` returns each path’s true utility
  `μ(π)` (length-normalized average of step utilities); `oracle_callback(step)` returns
  `x_s·θ⋆ + Gaussian noise` (the step-level verifier signal). Run it directly for a one-shot demo.
- **`gica.py`** — `class GICA`: the full Algorithm 1. Per round it computes the exact
  determinant-based confidence radius `β_t(δ)` (in log-space for stability), forms the empirical
  top-K shortlist, evaluates the stopping rule `Γ_t ≥ −ε`, selects the most ambiguous boundary pair
  by the gap-index `G_t = Δ̂² / σ²`, queries the step that maximizes the exact one-step variance
  contraction `C_t(s) = ⟨g(π⋆,π†), x_s⟩²_{V⁻¹} / (1 + ‖x_s‖²_{V⁻¹})`, and applies the
  Sherman–Morrison + recursive-least-squares update. `step_pool_mode="paths"` restricts candidate
  steps to the boundary pair (the paper setting).
- **`baseline_case.py`, `baseline_lingifa.py`, `baseline_mlingape.py`** — faithful path-arm
  reimplementations of CASE, LinGIFA, and m-LinGapE. All expose the same
  `select_and_update(oracle) → (done, top_ids)` interface and the same logging fields as GICA.
- **`baseline_igw_extreme.py`** — a self-contained IGW hierarchical contextual bandit, wired into
  the benchmark for completeness. **Not reported in the paper**; safe to ignore.
- **`benchmark.py`** — the experiment harness that produces Figure 3. Highlights: a `FairOracle`
  that gives each step/path its own deterministic noise stream (so every algorithm sees identical
  feedback); a `step_surrogate` mode that runs the baselines on step-arms and ranks real paths via
  the learned `θ̂` (how the paper adapts them to the compositional setting); and a `run_trials`
  routine that records verifier calls, runtime, gap-index comparisons, and accuracy across seeds.

### 4.3 Track 2 — Test-Time Scaling (`src/gica/tts/` + `scripts/tts/`)

Think of Track 2 in six **roles**:

1. **Data** (`data/*.json`) — question, `M=100` candidate solutions, ground truth.
2. **Environment** (`PRMEnvironment`, defined inside each driver) — splits paths into steps
   (on `".\n"`), embeds them with `all-MiniLM-L6-v2`, and builds a compact **6-dimensional step
   feature** `[cos(step,question), cos(step,path-centroid), cos(step,global-centroid),
   position-fraction, bias=1, boundary-placeholder=0]`. Its `oracle_callback(step)` reconstructs
   the within-path prefix and calls the verifier — **this is what counts as a “verifier call.”**
3. **Verifier** (`tts/verifier/thinkprm.py`) — wraps ThinkPRM under vLLM. Given the question + a
   step prefix, it generates a verification chain-of-thought, then converts the “ Yes”/“ No”
   decision log-probabilities into a confidence in `[0,1]`. The drivers read the `prefix_score`
   (reward `y_t`) and `step_labels` (binary per-step). The prompt is in `tts/utils/prompt_template.py`.
4. **Selection** (`tts/selection/*.py`) — the bandit algorithms. **`gica.py`** is the only one with
   the *dynamic boundary feature*: each round it overwrites the reserved last feature slot with the
   projection of every step embedding onto the current boundary direction
   (centroid(π⋆) − centroid(π†)) — the concrete mechanism for exploiting compositional structure.
   The baselines never touch that slot.
5. **Drivers** (`scripts/tts/*.py`) — glue everything together, loop over questions, and grade.
6. **Aggregation/grading** (`compute_em_from_top_m` inside each driver + `tts/answer_extraction.py`)
   — pick the winning path from the top-K, extract and normalize its answer, compute Exact-Match.

Reference points: **`run_best_of_m.py`** scores *every* step of *every* path (the exhaustive upper
bound), **`run_top1.py`** takes the first sampled path (the floor), and **`run_majority_vote.py`**
does self-consistency over final answers.

---

## 5. Quick Start / Tutorial

The fastest way to confirm everything is wired correctly is the **synthetic track** (no GPU, no
data, ~seconds).

### 5.1 One-line sanity check

```bash
conda activate gica
pip install -e .                      # if not done already
pip install -r requirements-synthetic.txt
```

```bash
python - <<'PY'
from gica.synthetic.environment import ReasoningEnvironment
from gica.synthetic.gica import GICA

env = ReasoningEnvironment(num_paths=40, num_total_steps=800, dim=8, noise_std=0.1, seed=0)
truth = sorted(env.get_ground_truth().items(), key=lambda x: x[1], reverse=True)
true_top5 = {pid for pid, _ in truth[:5]}

g = GICA(env.paths, env.feature_matrix, m=5, d=8,
         lambda_reg=1.0, epsilon=0.1, delta=0.05, R=0.1, S_0=2.0, step_pool_mode="paths")
done = False
while not done and g.t < 5000:
    done, est = g.select_and_update(env.oracle_callback)

print(f"GICA converged in {g.t} verifier calls; "
      f"recovered {len(set(est) & true_top5)}/5 of the true top-5.")
PY
```

Expected output (deterministic given the seed):

```text
GICA converged in 679 verifier calls; recovered 5/5 of the true top-5.
```

### 5.2 Run the full synthetic benchmark (produces Figure 3 plots)

```bash
python -m gica.synthetic.benchmark
```

This runs GICA and all baselines over multiple seeds and writes comparison figures to a
`plots/` directory (created automatically in the current working directory). On a cluster:

```bash
sbatch scripts/synthetic/slurm_benchmark.sbatch
```

### 5.3 A minimal Track-2 run (after installing Track-2 deps + downloading data)

```bash
# from the repository root, with data/Deepseek-MathOdyssey-RL-7B.json present
python scripts/tts/run_gica.py
```

This loads ThinkPRM-1.5B, runs GICA on MathOdyssey, and prints running Exact-Match and timing.

---

## 6. Reproducing the Paper

All commands are run from the **repository root** with the `gica` environment active and (for
Track 2) the datasets present in `data/`.

### 6.1 Figure 3 — synthetic sample efficiency (RQ1)

The paper sweeps **`M ∈ {200, 500, 1000}`**, `d = 8`, `K = 10`, `R = 0.1`, path lengths
uniform in `{20,…,80}`, averaged over seeds `{0,…,9}`, with hyperparameters
`λ=1.0, δ=0.01, ε=0.02, S₀=2.0` (Appendix B.1).

1. Open `src/gica/synthetic/benchmark.py` and set, in the `__main__` block:
   ```python
   ENV_CFG = dict(num_paths=200, num_total_steps=10_000, dim=8,
                  noise_std=0.1, path_len_min=20, path_len_max=80)
   # set every ALG_* dict to: lambda_reg=1.0, delta=0.01, epsilon=0.02, R=0.1, S_0=2.0, m(=K)=10
   N_TRIALS = 10
   ```
2. Run once per scale:
   ```bash
   python -m gica.synthetic.benchmark      # repeat with num_paths = 200, 500, 1000
   ```
3. Read the generated `plots/compare_*` figures: **(a)** gap-index comparisons,
   **(b)** verifier calls, **(c)** runtime — as a function of `M`.

> All algorithms must share `(λ, δ, ε, R, S₀, K)` so that differences reflect the **sampling rule**
> alone. The committed defaults (`ε=0.1, δ=0.05, num_paths=50`) are demo values; override them with
> the Appendix-B.1 values above to match the paper.

### 6.2 Tables 1 & 5 and Figures 4–5 — TTS pipeline (RQ2/RQ3)

Common settings (Appendix B.2): shortlist size `K = 5`, `δ = 0.05`, `λ = 1.0`,
generators `DeepSeekMath-RL-7B` and `InternLM2-Math-Plus-7B`, `M = 100` paths per question.

| Paper artifact                         | Command(s)                                                                                       |
|----------------------------------------|--------------------------------------------------------------------------------------------------|
| **Top-1 decoding** (floor)             | `python scripts/tts/run_top1.py`  *(edit the `data/…` path inside to switch benchmark)*          |
| **Best-of-M** (exhaustive upper bound) | `python scripts/tts/run_best_of_m.py`  *(ThinkPRM-1.5B; edit the `data/…` path + `BATCH_SIZE`)*  |
| **GICA**, ThinkPRM-1.5B (Table 1)      | `python scripts/tts/run_gica.py`  *(edit the `data/…` path inside to switch benchmark)*          |
| **Baselines**, ThinkPRM-7B             | `python scripts/tts/run_baselines.py --baseline_name CASE  --file_path data/Deepseek-MathOdyssey-RL-7B.json --dataset_name MathOdyssey --data_limit 400` |
| **GICA**, ThinkPRM-7B (Table 5/Fig 5)  | `python scripts/tts/run_gica_topm_thinkprm7b.py --file_path data/Deepseek-MathOdyssey-RL-7B.json --dataset_name MathOdyssey` |

For the baseline driver, `--baseline_name ∈ {CASE, GIFA, lingape}`.

**Steps to reproduce a full table row (example: MathOdyssey, DeepSeekMath-RL-7B, ThinkPRM-1.5B):**

```bash
# 1) upper bound and floor
python scripts/tts/run_best_of_m.py        # set the data path to Deepseek-MathOdyssey-RL-7B.json
python scripts/tts/run_top1.py             # set the data path to Deepseek-MathOdyssey-RL-7B.json

# 2) GICA
python scripts/tts/run_gica.py             # already points at Deepseek-MathOdyssey-RL-7B.json

# 3) baselines (one run each)
python scripts/tts/run_gica_topm_thinkprm7b.py \
    --file_path data/Deepseek-MathOdyssey-RL-7B.json --dataset_name MathOdyssey   # (ThinkPRM-7B variant)
```

Each driver prints the running and final **Exact-Match** and the mean/standard-deviation of
per-query time. The CLI drivers (`run_baselines.py`, `run_gica_topm_thinkprm7b.py`) additionally
write a per-question CSV with columns `qid, iter, time, EM`, where **`iter` is the verifier-call
count** (Figures 4–5) and **`time`** is the per-query inference runtime.

- **Figure 4** (ThinkPRM-1.5B): collect the `iter` and `time` columns from the GICA and baseline
  runs across MATH-500 / MathOdyssey / AIME and both generators.
- **Table 5 & Figure 5** (ThinkPRM-7B ablation, Appendix C.2): the `run_gica_topm_thinkprm7b.py`
  and `run_baselines.py` drivers already load `launch/ThinkPRM-7B`; rerun them on each dataset.

To switch the **benchmark** for the hardcoded-path drivers (`run_gica.py`, `run_best_of_m.py`,
`run_top1.py`, `run_majority_vote.py`), edit the single `open("data/…json")` line near the top
of the file. To switch the **verifier**, edit the `model_name_or_path="launch/ThinkPRM-…"` argument
in the driver’s `ThinkPRM(...)` constructor.

### 6.3 Appendix C.1 — sensitivity to the pair–step correlation ρ†

This is a synthetic study: reseed the random geometry in
`src/gica/synthetic/environment.py` / `benchmark.py`, measure the realized ρ† over the boundary
set, and record GICA’s verifier calls and runtime. Expected trend: cost scales as `O(1/ρ†)`, but the
empirical effect is far milder than worst case.

---

## 7. Reproducibility notes & known caveats

- **Determinism.** Track 1 is fully deterministic given the seed. In Track 2, ThinkPRM is run at
  `temperature=0.0` (greedy) with a fixed prompt and step-segmentation rule, so the verifier signal
  is reproducible; using the released `data/*.json` removes generator stochasticity.
- **Hyperparameter drift.** A few committed constants differ slightly from the paper text
  (synthetic demo defaults `ε=0.1, δ=0.05`; the canonical Appendix-B.1 values are `ε=0.02, δ=0.01`;
  the TTS drivers use `ε=0.15` with a small damping factor, while Appendix B.2 reports `ε=0.1`).
  Set them to the Appendix values to match the paper exactly. All methods share `(δ, λ, ε)` so the
  comparison remains fair regardless.
- **Optional extra baseline.** `baseline_igw_extreme.py` (`XtremeAlg3OnPaths`) is included for
  completeness but is **not** part of the paper’s reported results.
- **Early stopping.** The TTS drivers add a small *patience* guard (stop if the top-K is unchanged
  for several rounds) on top of GICA’s statistical stopping rule, so a single hard question never
  runs unboundedly.
- **Feature normalization.** The 6-dimensional TTS features are described in the paper as
  ℓ2-normalized to `L = 1`; the corresponding line is commented out in the driver’s
  `build_features`. Re-enable it to match the paper text precisely.
- **Provenance.** Track 2 builds on the public
  [GenPRM](https://github.com/RyanLiu112/GenPRM) pipeline; the upstream data-synthesis / PRM-training
  scaffolding is **not** required to reproduce the paper’s evaluation and is therefore omitted from
  this unified repository (the verifier prompt template it provided is retained in
  `src/gica/tts/utils/prompt_template.py`).

---

## 8. Citation

If you use this code, please cite the paper:

```bibtex
@article{gica2026,
  title   = {GICA: The Gap-Index Compositional Arm Framework for Sample-Efficient Test-Time Scaling},
  author  = {Anonymous},
  journal = {Transactions on Machine Learning Research (under review)},
  year    = {2026}
}
```

## Acknowledgements

The test-time-scaling pipeline builds on the public
[GenPRM](https://github.com/RyanLiu112/GenPRM) repository. The reasoning-based verifier is
**ThinkPRM** (Khalifa et al., *Process Reward Models That Think*, TMLR 2026). Answer
grading/normalization utilities are adapted from
[Qwen2.5-Math](https://github.com/QwenLM/Qwen2.5-Math). The bandit baselines reimplement
**CASE** (Purohit et al., 2025), **LinGIFA** (Réda et al., 2021), and **m-LinGapE** (Xu et al., 2018).
