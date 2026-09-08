# Appendix Experiments

Companion to [`README.md`](README.md). The README covers the paper's main results — Figure 3,
Table 1 and Figure 4. This file covers everything else: the full synthetic protocol of
Appendix B.1, and the additional figures and tables of Section 4.5 and Appendix C.

All commands run from the repository root with the `gica` environment active and, for the
test-time-scaling experiments, the datasets present in `data/`.

---

## 1. Synthetic protocol (Appendix B.1)

The committed defaults in the `__main__` block of `src/gica/synthetic/benchmark.py` are the
paper's configuration, so `python -m gica.synthetic.benchmark` reproduces the synthetic
protocol as shipped. For reference, those settings are:

| Setting | Value | Where it lives |
|---|---|---|
| Problem scales `M` | 200, 500, 1000 | `ENV_CFG["num_paths"]` — one run per scale |
| Feature dimension `d` | 8 | `ENV_CFG["dim"]` |
| Path lengths | uniform in {20,…,80} | `ENV_CFG["path_len_min"/"path_len_max"]` |
| Step-feature std | 0.30 (per coordinate) | `ENV_CFG["step_scale"]` |
| Feature-norm budget `L` | 2.5 | `ENV_CFG["feature_norm"]` |
| `‖θ⋆‖` | 1.0 (unit sphere) | `ENV_CFG["theta_norm"]` |
| Boundary rank `K` | 10 | `ENV_CFG["top_k"]`, and `m` in every `ALG_*` |
| Noise `R` | 0.1 | `ENV_CFG["noise_std"]` |
| Ridge `λ` | 1.0 | every `ALG_*` |
| Confidence `δ` | 0.01 | every `ALG_*` |
| Tolerance `ε` | 0.02 | every `ALG_*` |
| Norm bound `S₀` | 2.0 | every `ALG_*` |
| Seeds | {0,…,9} | `N_TRIALS = 10` |
| Iteration cap | 100,000 | `MAX_ROUNDS` |
| Baseline arm model | path arms, queried step-by-step | `BASELINE_ARM_MODE = "path"`, `PATH_PULL_MODEL = "avg_steps"` |

The last row is what makes the comparison fair: the path-arm baselines are charged one
verifier call per constituent step, so the per-step verification budget matches GICA's.

> All algorithms share `(λ, δ, ε, R, S₀, K)` so that differences reflect the **sampling rule**
> alone. If you change one, change it in every `ALG_*` dict. Note that `top_k` must stay equal
> to `m`: omitted, it would default to `M // 2` rather than the paper's `K = 10`. The
> environment also still accepts a legacy `num_total_steps` argument, but it is ignored.

---

## 2. Figure 7 — shortlist recovery (Appendix C.3)

Every benchmark run writes `plots/topk_error_K<K>_<timestamp>.dat`, a `round mean std` table
holding GICA's per-round top-`K` identification error in percent, i.e.

```
100 × (1 − |certified top-K ∩ true top-K| / K)
```

averaged over the `N_TRIALS` seeds with one standard deviation. Trials that stop early are
forward-filled with their final value, so every round averages over all seeds.

Figure 7 overlays three such curves at `M = 1000`. Run the benchmark three times with
`ENV_CFG["num_paths"] = 1000` and `m` (and `ENV_CFG["top_k"]`) set to 5, 10 and 20, then plot
the three `.dat` files together.

---

## 3. Table 2 — sensitivity to the pair–step correlation ρ† (Section 4.5)

A synthetic study. ρ† is an emergent property of the random geometry, so it is **measured**
rather than set: reseed the geometry in `src/gica/synthetic/environment.py` / `benchmark.py`,
let the benchmark measure the realized ρ† over the boundary set (`MEASURE_RHO = True`,
`RHO_PAIRS`, `RHO_EVERY` at the top of `benchmark.py`), and record GICA's verifier calls and
runtime for each seeding.

The expected trend is a cost scaling of `O(1/ρ†)`, though the empirical effect is far milder
than the worst case, since ρ† is a uniform infimum set by a few adversarial configurations
while the boundary pairs that actually bottleneck termination are aligned far more favourably.

---

## 4. Table 5 and Figure 5 — verifier scale, ThinkPRM-7B (Appendix C.1)

The verifier-scale ablation repeats the full TTS pipeline with ThinkPRM-7B in place of
ThinkPRM-1.5B, holding the generators, `M = 100`, `K = 5`, the step-segmentation rule, the
feature encoder and all bandit hyperparameters fixed.

Both drivers already load `launch/ThinkPRM-7B`, so no edit is needed — just rerun them on each
dataset:

```bash
# GICA with the 7B verifier
python scripts/tts/run_gica_topm_thinkprm7b.py \
    --file_path data/Deepseek-MathOdyssey-RL-7B.json --dataset_name MathOdyssey

# the bandit baselines with the 7B verifier
python scripts/tts/run_baselines.py --baseline_name CASE \
    --file_path data/Deepseek-MathOdyssey-RL-7B.json \
    --dataset_name MathOdyssey --data_limit 400
```

**Table 5** is the final Exact-Match each driver prints. **Figure 5** is the `iter` and `time`
columns of the per-question CSV they write, exactly as Figure 4 is built from the 1.5B runs.

---

## 5. Figure 6 — generation vs verification runtime (Appendix C.2)

Figure 6 splits end-to-end latency into its two halves for `M = 100` paths per query, with
DeepSeekMath-RL-7B as generator and ThinkPRM-7B as verifier. The percentages in the figure are
each bar over the sum of the two.

**Verification bar** — the per-query time printed by the exhaustive driver:

```bash
python scripts/tts/run_best_of_m.py        # read the final mean(times)
```

Note that `run_best_of_m.py` loads `launch/ThinkPRM-1.5B`; set it to `launch/ThinkPRM-7B` to
match Figure 6.

**Generation bar** — *not currently reproducible from this repository.* Every driver here
consumes the pre-generated candidate paths shipped in `data/`, so none of them times the
generator, and the released JSON files record the sampling configuration (`temperature`,
`top_p`) but no wall-clock. Reproducing this bar requires re-running the generator over the
same questions and timing it; a driver for that is planned but not yet part of this
repository.

---

## 6. Table 6 — sensitivity to K (Appendix C.4)

Task performance as the shortlist size varies over `K ∈ {1, 3, 5, 10}`, with
DeepSeekMath-RL-7B as generator and ThinkPRM-7B as verifier. Run the 7B GICA driver once per
value, changing the `m=` argument of the `GICA(...)` constructor in
`scripts/tts/run_gica_topm_thinkprm7b.py`:

```bash
python scripts/tts/run_gica_topm_thinkprm7b.py \
    --file_path data/Deepseek-MathOdyssey-RL-7B.json --dataset_name MathOdyssey
```

The `K = 5` column reproduces the GICA row of Table 5, which is the consistency check that the
two tables come from the same configuration.

---

## 7. Figure 8 — accuracy–verification-cost Pareto (Appendix C.5)

Figure 8 has no driver of its own: it is a scatter of numbers the runs above already produce,
over all twelve dataset × generator × verifier configurations.

- **y-axis, exact-match accuracy** — the final EM each driver prints; these are the values
  tabulated in **Table 1** (ThinkPRM-1.5B) and **Table 5** (ThinkPRM-7B).
- **x-axis, average verifier calls per query** — the mean of the `iter` column of a driver's
  per-question CSV; these are the values plotted in **Figure 4** (1.5B) and **Figure 5** (7B).
  For the exhaustive Best-of-M reference the count is not adaptive: it is the total number of
  steps over all `M = 100` candidate paths of a question.

A method is *dominated* when another point has both higher accuracy and fewer verifier calls;
the solid curves in the figure connect the non-dominated points per configuration, and the
faded markers are the dominated ones.
