# benchmark.py
import numpy as np
import time
import matplotlib.pyplot as plt

from gica.synthetic.gica import GICA
from gica.synthetic.baseline_lingifa import LinGIFA
from gica.synthetic.baseline_mlingape import m_LinGapE
from gica.synthetic.environment import ReasoningEnvironment
from gica.synthetic.baseline_igw_extreme import XtremeAlg3OnPaths
from gica.synthetic.baseline_case import CASE

import os
from datetime import datetime

# ---- output folder + timestamp helpers ----
PLOTS_DIR = "plots"  # change to e.g. "results/plots" if you want
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(PLOTS_DIR, exist_ok=True)

def outpath(stem: str, ext: str = ".png") -> str:
    return os.path.join(PLOTS_DIR, f"{stem}_{RUN_TS}{ext}")

# ---------------------------
# Evaluation modes
# ---------------------------
# "path": baselines operate on PATH arms (each arm = a CoT/path)
# "step_surrogate": baselines operate on STEP arms (each arm = a step),
#                   then we rank PATHS using the learned surrogate (theta_hat)
BASELINE_ARM_MODE = "step_surrogate"  # "path" or "step_surrogate"

# In step_surrogate mode:
# - "baseline": stop when the baseline stops (often based on step-top-m, not path-top-m)
# - "path_gamma": stop using your paper-style path-level Gamma_t criterion (recommended)
STEP_SURROGATE_STOP_RULE = "baseline"  # "baseline" or "path_gamma"

# How to model a PATH pull (only used in BASELINE_ARM_MODE="path"):
# - "single": one noisy observation of g(pi)^T theta
# - "avg_steps": equivalent to averaging T step observations (noise shrinks by 1/sqrt(T)).
#                Recommended if you charge oracle_cost = len(path).
PATH_PULL_MODEL = "single"  # "single" or "avg_steps"


class FairOracle:
    """
    Deterministic, arm-wise noise oracle (order-independent).
    - Each STEP has its own RNG stream (seeded by trial_seed + step_id)
    - Each PATH has its own RNG stream (seeded by trial_seed + path_id)
    """

    def __init__(self, env, trial_seed: int):
        self.env = env
        self.trial_seed = int(trial_seed)
        self._rng_step = {}  # step_id -> Generator
        self._rng_path = {}  # path_id -> Generator

    @staticmethod
    def _mix64(x: int) -> int:
        """SplitMix64 finalizer (deterministic 64-bit mix)."""
        mask = (1 << 64) - 1
        x &= mask
        x ^= (x >> 30)
        x = (x * 0xBF58476D1CE4E5B9) & mask
        x ^= (x >> 27)
        x = (x * 0x94D049BB133111EB) & mask
        x ^= (x >> 31)
        return x & mask

    def _seed_for(self, kind: int, arm_id: int) -> int:
        # kind: 0=step, 1=path
        x = (self.trial_seed * 0x9E3779B97F4A7C15) ^ (kind * 0xBF58476D1CE4E5B9) ^ (int(arm_id) + 1)
        return self._mix64(x)

    def _get_rng(self, kind: int, arm_id: int):
        store = self._rng_step if kind == 0 else self._rng_path
        arm_id = int(arm_id)
        if arm_id not in store:
            store[arm_id] = np.random.default_rng(self._seed_for(kind, arm_id))
        return store[arm_id]

    # ---- Oracles ----
    def step(self, step_idx: int) -> float:
        step_idx = int(step_idx)
        x = self.env.feature_matrix[step_idx]
        clean = float(x @ self.env.true_theta)
        rng = self._get_rng(0, step_idx)
        noise = float(rng.normal(0.0, self.env.noise_std))
        return clean + noise

    def path(self, path_id: int) -> float:
        path_id = int(path_id)
        steps = self.env.paths[path_id]
        g = np.mean(self.env.feature_matrix[steps], axis=0)
        clean = float(g @ self.env.true_theta)

        rng = self._get_rng(1, path_id)

        if PATH_PULL_MODEL == "single":
            # One noisy observation of g^T theta
            noise = float(rng.normal(0.0, self.env.noise_std))
            return clean + noise

        elif PATH_PULL_MODEL == "avg_steps":
            # Equivalent to averaging T step observations (noise shrinks as 1/sqrt(T))
            T = max(1, len(steps))
            noise = float(rng.normal(0.0, self.env.noise_std / np.sqrt(T)))
            return clean + noise

        else:
            raise ValueError("Unknown PATH_PULL_MODEL")


# ---------------------------
# Helpers for step-surrogate mode
# ---------------------------
def make_save_path(filename: str, out_dir: str, run_ts: str) -> str:
    """
    Turn 'foo.png' into '<out_dir>/foo_<run_ts>.png'
    Works even if filename has no extension.
    """
    base = os.path.basename(filename)
    name, ext = os.path.splitext(base)
    if ext == "":
        ext = ".png"
    return os.path.join(out_dir, f"{name}_{run_ts}{ext}")

def sigma_for_oracle(env, *, oracle_kind: str) -> float:
    """
    Calibrate sigma/R to match the oracle actually used.

    - step oracle: noise std = env.noise_std
    - path oracle:
        * single     : noise std = env.noise_std
        * avg_steps  : noise std = env.noise_std/sqrt(T), worst-case at Tmin
    """
    oracle_kind = str(oracle_kind).lower().strip()
    if oracle_kind == "step":
        return float(env.noise_std)

    if oracle_kind != "path":
        raise ValueError("oracle_kind must be 'step' or 'path'")

    if PATH_PULL_MODEL == "single":
        return float(env.noise_std)

    Tmin = min(len(steps) for steps in env.paths.values())
    return float(env.noise_std / np.sqrt(max(1, Tmin)))


def make_step_as_paths(env):
    """
    Turn steps into singleton 'paths' so existing path-arm baselines can run without rewriting:
      step_id -> [step_id]
    Then g(pi) = mean(feature_matrix[[step_id]]) = x_step.
    """
    return {int(s): [int(s)] for s in range(env.feature_matrix.shape[0])}


class StepSurrogateToPathRanker:
    """
    Wrap a baseline run on STEP-arms (singleton paths) and return TOP-m PATHS by g(pi)^T theta_hat.
    Optionally stop using the PATH-level Gamma criterion (recommended).

    IMPORTANT: We track wrapper post-processing time in self._postprocess_time so the
    experiment runner can subtract it from the baseline runtime.
    """

    def __init__(self, base_algo, env, m_paths, epsilon, delta, R, S_0, lambda_reg, name):
        self.base = base_algo
        self.env = env
        self.m = int(m_paths)
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.R = float(R)
        self.S_0 = float(S_0)
        self.lambda_reg = float(lambda_reg)
        self._name = str(name)

        # Precompute path features g(pi) over REAL paths
        self.g_pi_paths = {
            pid: np.mean(env.feature_matrix[steps], axis=0)
            for pid, steps in env.paths.items()
        }

        # Traces expected by run_trials/plotters
        self.best_G_history = []
        self.min_lcb_history = []
        self.total_comparisons = 0
        self.t = 0  # mirrors base.t

        # Post-processing time accumulator (excluded from baseline runtime)
        self._postprocess_time = 0.0

    @staticmethod
    def _get_theta_hat(base):
        if hasattr(base, "theta_hat"):
            return base.theta_hat
        if hasattr(base, "theta"):
            return base.theta
        if hasattr(base, "theta_est"):
            return base.theta_est
        raise AttributeError("Base algorithm does not expose theta_hat/theta/theta_est for surrogate ranking.")

    @staticmethod
    def _get_V_inv(base):
        if hasattr(base, "V_inv"):
            return base.V_inv
        if hasattr(base, "V"):
            return np.linalg.inv(base.V)
        raise AttributeError("Base algorithm does not expose V_inv or V for confidence widths.")

    def _beta(self, V_inv, d):
        sign, log_det_V_inv = np.linalg.slogdet(V_inv)
        if sign <= 0:
            raise FloatingPointError("V_inv lost PD-ness (sign<=0).")
        log_det_V = -log_det_V_inv
        log_det_lambda_I = d * np.log(self.lambda_reg)
        log_ratio = 0.5 * log_det_V - 0.5 * log_det_lambda_I - np.log(self.delta)
        if log_ratio < 0:
            raise FloatingPointError("Negative log_ratio in beta computation; numerical instability.")
        return self.R * np.sqrt(2 * log_ratio) + np.sqrt(self.lambda_reg) * self.S_0

    def _rank_paths(self, theta_hat):
        scores = {pid: float(g @ theta_hat) for pid, g in self.g_pi_paths.items()}
        top = sorted(scores.keys(), key=lambda pid: scores[pid], reverse=True)[: self.m]
        return top, scores

    def _gamma_paths(self, theta_hat, V_inv):
        top, scores = self._rank_paths(theta_hat)
        top_set = set(top)
        rest = [pid for pid in scores.keys() if pid not in top_set]
        if not rest:
            return np.inf

        d = int(theta_hat.shape[0])
        beta = self._beta(V_inv, d)

        min_lcb = np.inf
        for pi in top:
            for pj in rest:
                gdiff = self.g_pi_paths[pi] - self.g_pi_paths[pj]
                sig2 = float(gdiff.T @ V_inv @ gdiff)
                sig2 = max(sig2, 1e-12)
                W = beta * np.sqrt(sig2)
                gap = scores[pi] - scores[pj]
                lcb = gap - W
                if lcb < min_lcb:
                    min_lcb = lcb
        return min_lcb

    def select_and_update(self, oracle_step):
        # 1) Run ONE iteration of the baseline (core baseline time)
        done_base, _ = self.base.select_and_update(oracle_step)

        # 2) Time ONLY the wrapper post-processing below
        t_post0 = time.perf_counter()

        self.t = getattr(self.base, "t", self.t)
        self.total_comparisons = getattr(self.base, "total_comparisons", self.total_comparisons)

        theta_hat = self._get_theta_hat(self.base)
        V_inv = self._get_V_inv(self.base)

        top_paths, _ = self._rank_paths(theta_hat)

        if hasattr(self.base, "best_G_history") and len(getattr(self.base, "best_G_history")) > len(self.best_G_history):
            self.best_G_history.append(self.base.best_G_history[-1])

        if STEP_SURROGATE_STOP_RULE == "baseline":
            done = bool(done_base)
        elif STEP_SURROGATE_STOP_RULE == "path_gamma":
            gamma = self._gamma_paths(theta_hat, V_inv)
            self.min_lcb_history.append(gamma)
            done = (gamma >= -self.epsilon)
        else:
            raise ValueError("Unknown STEP_SURROGATE_STOP_RULE")

        self._postprocess_time += (time.perf_counter() - t_post0)
        return done, top_paths


# ---------------------------
# Plotting + trial runner utilities
# ---------------------------

def pad_nan(seqs):
    max_len = max(len(s) for s in seqs)
    out = np.full((len(seqs), max_len), np.nan, dtype=float)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return out


def run_trials(
    make_algo_fn,
    n_trials=50,
    max_rounds=2000,
    verbose=True,
    log_every=50,
):
    total_comparisons = []
    runtimes = []
    oracle_costs = []
    G_traces = []
    lcb_traces = []
    acc_traces = []

    for k in range(n_trials):
        algo = make_algo_fn(k)

        oracle_base = algo._oracle
        cost_fn = getattr(algo, "_oracle_cost", lambda _a: 1)

        oracle_cost = 0

        def oracle_wrapped(a):
            nonlocal oracle_cost
            oracle_cost += int(cost_fn(a))
            return oracle_base(a)

        t0 = time.perf_counter()
        done = False
        top = None
        acc_hist = []

        while (not done) and (algo.t < max_rounds):
            done, top = algo.select_and_update(oracle_wrapped)

            correct = len(set(top).intersection(set(algo._true_top_m)))
            acc = correct / algo.m
            acc_hist.append(acc)

            if verbose and (algo.t % log_every == 0):
                print(
                    f"[{algo._name} trial {k+1}/{n_trials}] iter={algo.t} done={done} "
                    f"acc={correct}/{algo.m} ({acc:.2f}) oracle_cost={oracle_cost}"
                )

        rt = time.perf_counter() - t0
        # Subtract wrapper post-processing time (only exists in step_surrogate wrappers)
        rt -= float(getattr(algo, "_postprocess_time", 0.0))
        rt = max(0.0, rt)

        runtimes.append(rt)
        total_comparisons.append(getattr(algo, "total_comparisons", 0))
        oracle_costs.append(oracle_cost)

        G_traces.append(np.array(getattr(algo, "best_G_history", []), dtype=float))
        lcb_traces.append(np.array(getattr(algo, "min_lcb_history", []), dtype=float))
        acc_traces.append(np.array(acc_hist, dtype=float))

        if verbose:
            correct = len(set(top).intersection(set(algo._true_top_m)))
            acc = correct / algo.m
            print(
                f"[{algo._name} trial {k+1}/{n_trials}] finished: iter={algo.t}, done={done}, "
                f"final acc={correct}/{algo.m} ({acc:.2f}), oracle_cost={oracle_cost}, runtime={rt:.2f}s"
            )

    return (
        np.array(total_comparisons),
        np.array(runtimes),
        np.array(oracle_costs),
        G_traces,
        lcb_traces,
        acc_traces,
    )


# ---------------------------
# Plotting helpers (unchanged)
# ---------------------------

def plot_results(total_comparisons, runtimes, G_traces, lcb_traces, label="GICA", epsilon=0.05, save_path="pgihA_metrics.png"):
    avg_comp = float(np.mean(total_comparisons))
    std_comp = float(np.std(total_comparisons, ddof=1)) if len(total_comparisons) > 1 else 0.0

    G = pad_nan(G_traces)
    L = pad_nan(lcb_traces)

    mean_G = np.nanmean(G, axis=0)
    std_G = np.nanstd(G, axis=0, ddof=1)

    mean_L = np.nanmean(L, axis=0)
    std_L = np.nanstd(L, axis=0, ddof=1)

    cnt_G = np.sum(~np.isnan(G), axis=0)
    cnt_L = np.sum(~np.isnan(L), axis=0)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))

    ax = axes[0]
    ax.bar([0], [avg_comp], yerr=[std_comp], capsize=6)
    ax.set_xticks([0])
    ax.set_xticklabels([label], rotation=45, ha="right")
    ax.set_ylabel("Avg no. of comparisons")
    ax.set_title("(a) comparisons across\nsimulations")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax = axes[1]
    ax.boxplot([runtimes], labels=[label], showfliers=True)
    ax.set_ylabel("Avg runtime (seconds)")
    ax.set_title("(b) Average runtime\n(in seconds)")

    ax = axes[2]
    xG = np.arange(len(mean_G))
    ax.plot(xG, mean_G, linewidth=1, color="tab:orange", label="Gap index (mean)")

    maskG = cnt_G >= 2
    ax.fill_between(
        xG[maskG],
        (mean_G - std_G)[maskG],
        (mean_G + std_G)[maskG],
        color="tab:orange",
        alpha=0.2,
        label="±1 std",
    )
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Gap index")
    ax.set_title("(c) Gap index\ncomparison")
    ax.legend(loc="best")

    ax = axes[3]
    xL = np.arange(len(mean_L))
    ax.plot(xL, mean_L, linewidth=1, color="tab:blue", label="min (Δhat - W) mean")

    maskL = cnt_L >= 2
    ax.fill_between(
        xL[maskL],
        (mean_L - std_L)[maskL],
        (mean_L + std_L)[maskL],
        color="tab:blue",
        alpha=0.2,
        label="±1 std",
    )

    ax.axhline(-epsilon, linestyle="--", linewidth=1, color="gray", label="-epsilon")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Δhat - W")
    ax.set_title("(d) stopping quantity")
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {save_path}")
    plt.show()


def plot_compare(resA, resB, nameA="GICA", nameB="CASE", epsilon=0.05, save_path="compare.png"):
    compsA, rtsA, orcA, G_A, L_A, Acc_A = resA
    compsB, rtsB, orcB, G_B, L_B, Acc_B = resB

    GA = pad_nan(G_A)
    GB = pad_nan(G_B)
    LA = pad_nan(L_A)
    LB = pad_nan(L_B)
    AA = pad_nan(Acc_A)
    AB = pad_nan(Acc_B)

    def mean_std_cnt(M):
        mu = np.nanmean(M, axis=0)
        sd = np.nanstd(M, axis=0, ddof=1) if M.shape[0] > 1 else np.zeros_like(mu)
        cnt = np.sum(~np.isnan(M), axis=0)
        return mu, sd, cnt

    muGA, sdGA, cntGA = mean_std_cnt(GA)
    muGB, sdGB, cntGB = mean_std_cnt(GB)
    muLA, sdLA, cntLA = mean_std_cnt(LA)
    muLB, sdLB, cntLB = mean_std_cnt(LB)
    muAA, sdAA, cntAA = mean_std_cnt(AA)
    muAB, sdAB, cntAB = mean_std_cnt(AB)

    fig, axes = plt.subplots(1, 8, figsize=(32, 3.6))

    means = [np.mean(compsA), np.mean(compsB)]
    stds = [
        np.std(compsA, ddof=1) if len(compsA) > 1 else 0.0,
        np.std(compsB, ddof=1) if len(compsB) > 1 else 0.0,
    ]
    axes[0].bar([0, 1], means, yerr=stds, capsize=6)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels([nameA, nameB], rotation=45, ha="right")
    axes[0].set_ylabel("Avg no. of comparisons")
    axes[0].set_title("(a) comparisons across\nsimulations")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    axes[1].boxplot([rtsA, rtsB], labels=[nameA, nameB], showfliers=True)
    axes[1].set_ylabel("Runtime (seconds)")
    axes[1].set_title("(b) Average runtime\n(in seconds)")

    ax = axes[2]
    xA = np.arange(len(muGA))
    xB = np.arange(len(muGB))
    ax.plot(xA, muGA, color="tab:orange", label=f"{nameA}")
    ax.plot(xB, muGB, color="tab:green", label=f"{nameB}")
    ax.fill_between(xA[cntGA >= 2], (muGA - sdGA)[cntGA >= 2], (muGA + sdGA)[cntGA >= 2], color="tab:orange", alpha=0.2)
    ax.fill_between(xB[cntGB >= 2], (muGB - sdGB)[cntGB >= 2], (muGB + sdGB)[cntGB >= 2], color="tab:green", alpha=0.2)
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Gap index")
    ax.set_title("(c) Gap index\ncomparison")
    ax.legend(loc="best")

    ax = axes[3]
    xA = np.arange(len(muLA))
    xB = np.arange(len(muLB))
    ax.plot(xA, muLA, color="tab:blue", label=f"{nameA}")
    ax.plot(xB, muLB, color="tab:red", label=f"{nameB}")
    ax.fill_between(xA[cntLA >= 2], (muLA - sdLA)[cntLA >= 2], (muLA + sdLA)[cntLA >= 2], color="tab:blue", alpha=0.2)
    ax.fill_between(xB[cntLB >= 2], (muLB - sdLB)[cntLB >= 2], (muLB + sdLB)[cntLB >= 2], color="tab:red", alpha=0.2)
    ax.axhline(-epsilon, linestyle="--", color="gray", linewidth=1, label="-epsilon")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Δhat - W")
    ax.set_title("(d) stopping quantity")
    ax.legend(loc="best")

    ax = axes[4]
    xA = np.arange(len(muAA))
    xB = np.arange(len(muAB))
    ax.plot(xA, muAA, color="tab:purple", label=f"{nameA}")
    ax.plot(xB, muAB, color="tab:brown", label=f"{nameB}")
    ax.fill_between(xA[cntAA >= 2], (muAA - sdAA)[cntAA >= 2], (muAA + sdAA)[cntAA >= 2], color="tab:purple", alpha=0.2)
    ax.fill_between(xB[cntAB >= 2], (muAB - sdAB)[cntAB >= 2], (muAB + sdAB)[cntAB >= 2], color="tab:brown", alpha=0.2)
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(e) Accuracy")
    ax.legend(loc="best")

    mean_orcA = float(np.mean(orcA))
    mean_orcB = float(np.mean(orcB))
    std_orcA = float(np.std(orcA, ddof=1)) if len(orcA) > 1 else 0.0
    std_orcB = float(np.std(orcB, ddof=1)) if len(orcB) > 1 else 0.0

    orcA_pos = np.clip(np.array(orcA, dtype=float), 1e-12, None)
    orcB_pos = np.clip(np.array(orcB, dtype=float), 1e-12, None)

    ax = axes[5]
    ax.bar([0, 1], [mean_orcA, mean_orcB], yerr=[std_orcA, std_orcB], capsize=6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([nameA, nameB], rotation=45, ha="right")
    ax.set_ylabel("Avg oracle calls\n(step-equiv.)")
    ax.set_title("(f1) oracle calls\n(linear)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax = axes[6]
    ax.bar(
        [0, 1],
        [np.mean(orcA_pos), np.mean(orcB_pos)],
        yerr=[
            np.std(orcA_pos, ddof=1) if len(orcA_pos) > 1 else 0.0,
            np.std(orcB_pos, ddof=1) if len(orcB_pos) > 1 else 0.0,
        ],
        capsize=6,
    )
    ax.set_yscale("log")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([nameA, nameB], rotation=45, ha="right")
    ax.set_ylabel("Avg oracle calls\n(step-equiv.)")
    ax.set_title("(f2) oracle calls\n(log)")

    ax = axes[7]
    ratio = orcB_pos / orcA_pos
    mean_ratio = float(np.mean(ratio))
    std_ratio = float(np.std(ratio, ddof=1)) if len(ratio) > 1 else 0.0
    ax.bar([0], [mean_ratio], yerr=[std_ratio], capsize=6)
    ax.set_xticks([0])
    ax.set_xticklabels([f"{nameB}/{nameA}"], rotation=45, ha="right")
    ax.set_ylabel("Relative oracle calls")
    ax.set_title("(f3) oracle calls\n(relative)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {save_path}")
    plt.show()


def plot_compare_all(res_dict, epsilon=0.05, save_path="compare_all_algorithms.png"):
    names = list(res_dict.keys())
    nA = len(names)

    def mean_std(x):
        x = np.asarray(x, dtype=float)
        mu = float(np.mean(x))
        sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        return mu, sd

    def mean_std_cnt_from_traces(traces):
        M = pad_nan(traces)
        mu = np.nanmean(M, axis=0)
        sd = np.nanstd(M, axis=0, ddof=1) if M.shape[0] > 1 else np.zeros_like(mu)
        cnt = np.sum(~np.isnan(M), axis=0)
        return mu, sd, cnt

    comps_list, rts_list, orc_list = [], [], []
    G_stats, L_stats, A_stats = {}, {}, {}

    for name in names:
        comps, rts, orc, G_tr, L_tr, Acc_tr = res_dict[name]
        comps_list.append(np.asarray(comps))
        rts_list.append(np.asarray(rts))
        orc_list.append(np.asarray(orc))
        G_stats[name] = mean_std_cnt_from_traces(G_tr)
        L_stats[name] = mean_std_cnt_from_traces(L_tr)
        A_stats[name] = mean_std_cnt_from_traces(Acc_tr)

    fig, axes = plt.subplots(1, 7, figsize=(30, 3.8))

    ax = axes[0]
    means = [mean_std(c)[0] for c in comps_list]
    stds = [mean_std(c)[1] for c in comps_list]
    ax.bar(np.arange(nA), means, yerr=stds, capsize=6)
    ax.set_xticks(np.arange(nA))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Avg no. of comparisons")
    ax.set_title("(a) comparisons")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax = axes[1]
    ax.boxplot(rts_list, labels=names, showfliers=True)
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("(b) runtime")

    ax = axes[2]
    for name in names:
        mu, sd, cnt = G_stats[name]
        x = np.arange(len(mu))
        line, = ax.plot(x, mu, label=name)
        mask = cnt >= 2
        if np.any(mask):
            ax.fill_between(x[mask], (mu - sd)[mask], (mu + sd)[mask], alpha=0.2, color=line.get_color())
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Gap index")
    ax.set_title("(c) gap index")
    ax.legend(loc="best", fontsize=8)

    ax = axes[3]
    for name in names:
        mu, sd, cnt = L_stats[name]
        x = np.arange(len(mu))
        line, = ax.plot(x, mu, label=name)
        mask = cnt >= 2
        if np.any(mask):
            ax.fill_between(x[mask], (mu - sd)[mask], (mu + sd)[mask], alpha=0.2, color=line.get_color())
    ax.axhline(-epsilon, linestyle="--", linewidth=1, color="gray", label="-epsilon")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Δhat - W")
    ax.set_title("(d) stopping qty")
    ax.legend(loc="best", fontsize=8)

    ax = axes[4]
    for name in names:
        mu, sd, cnt = A_stats[name]
        x = np.arange(len(mu))
        line, = ax.plot(x, mu, label=name)
        mask = cnt >= 2
        if np.any(mask):
            ax.fill_between(x[mask], (mu - sd)[mask], (mu + sd)[mask], alpha=0.2, color=line.get_color())
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(e) accuracy")
    ax.legend(loc="best", fontsize=8)

    ax = axes[5]
    means = [mean_std(o)[0] for o in orc_list]
    stds = [mean_std(o)[1] for o in orc_list]
    ax.bar(np.arange(nA), means, yerr=stds, capsize=6)
    ax.set_xticks(np.arange(nA))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Avg oracle calls\n(step-equiv.)")
    ax.set_title("(f1) oracle calls")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax = axes[6]
    orc_pos = [np.clip(np.asarray(o, dtype=float), 1e-12, None) for o in orc_list]
    means = [mean_std(o)[0] for o in orc_pos]
    stds = [mean_std(o)[1] for o in orc_pos]
    ax.bar(np.arange(nA), means, yerr=stds, capsize=6)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(nA))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Avg oracle calls\n(step-equiv.)")
    ax.set_title("(f2) oracle calls (log)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {save_path}")
    plt.show()


# ---------------------------
# Main experiment runner
# ---------------------------
if __name__ == "__main__":
    SEED = 0
    ENV_CFG = dict(
        num_paths=50,
        num_total_steps=1000,
        dim=8,
        noise_std=0.1,
        path_len_min=50,
        path_len_max=300,
    )

    # NOTE: we still keep R in these dicts, but we will OVERRIDE it per-trial
    # using sigma_for_oracle(...) to match the oracle's actual noise.
    ALG_PGIHA = dict(m=10, lambda_reg=1.0, epsilon=0.1, delta=0.05, R=0.1, S_0=2.0, step_pool_mode="paths")
    ALG_CASE  = dict(m=10, lambda_reg=1.0, epsilon=0.1, delta=0.05, R=0.1, S_0=2.0, challenger_size=50, challenger_batch=50, seed=0)
    ALG_MLINGAPE = dict(m=10, lambda_reg=1.0, epsilon=0.1, delta=0.05, R=0.1, S_0=2.0, selection_rule="largest_variance")
    ALG_GIFA  = dict(m=10, lambda_reg=1.0, epsilon=0.1, delta=0.05, R=0.1, S_0=2.0, selection_rule="largest_variance")

    ALG_XTREME = dict(
        m=10,
        k=1,
        r=1,
        lam=1.0,
        gamma_C=1.0,
        branching=2,
        beam_size=ENV_CFG["num_paths"],
        seed=0,
        squash_rewards=True,
        sigmoid_alpha=5.0,
        sigmoid_beta=0.0,
        sigmoid_clip=35.0,
    )

    N_TRIALS = 10
    MAX_ROUNDS = 10_000

    trial_seeds = list(range(N_TRIALS))
    assert len(trial_seeds) == N_TRIALS

    # ---------------------------
    # Factories
    # ---------------------------
    def make_algo_pgiha(k):
        seed_k = trial_seeds[k]
        env = ReasoningEnvironment(**ENV_CFG, seed=seed_k)

        true_scores = env.get_ground_truth()
        sorted_truth = sorted(true_scores.items(), key=lambda x: x[1], reverse=True)
        true_top_m = [pid for pid, _ in sorted_truth[: ALG_PGIHA["m"]]]

        fair = FairOracle(env, seed_k)
        sigma_step = sigma_for_oracle(env, oracle_kind="step")

        algo = GICA(
            paths=env.paths,
            feature_matrix=env.feature_matrix,
            m=ALG_PGIHA["m"],
            d=ENV_CFG["dim"],
            lambda_reg=ALG_PGIHA["lambda_reg"],
            epsilon=ALG_PGIHA["epsilon"],
            delta=ALG_PGIHA["delta"],
            R=sigma_step,
            S_0=ALG_PGIHA["S_0"],
            step_pool_mode=ALG_PGIHA["step_pool_mode"],
        )

        algo._env = env
        algo._oracle = fair.step
        algo._true_top_m = true_top_m
        algo._name = "GICA"
        algo._oracle_cost = lambda _a: 1
        return algo

    def make_algo_case(k):
        seed_k = trial_seeds[k]
        env = ReasoningEnvironment(**ENV_CFG, seed=seed_k)

        true_scores = env.get_ground_truth()
        sorted_truth = sorted(true_scores.items(), key=lambda x: x[1], reverse=True)
        true_top_m = [pid for pid, _ in sorted_truth[: ALG_CASE["m"]]]

        fair = FairOracle(env, seed_k)

        if BASELINE_ARM_MODE == "path":
            sigma_path = sigma_for_oracle(env, oracle_kind="path")

            algo = CASE(
                paths=env.paths,
                feature_matrix=env.feature_matrix,
                m=ALG_CASE["m"],
                d=ENV_CFG["dim"],
                lambda_reg=ALG_CASE["lambda_reg"],
                epsilon=ALG_CASE["epsilon"],
                delta=ALG_CASE["delta"],
                R=sigma_path,
                S_0=ALG_CASE["S_0"],
                challenger_size=ALG_CASE.get("challenger_size", 5),
                challenger_batch=ALG_CASE.get("challenger_batch", 5),
                seed=seed_k,
            )
            algo._env = env
            algo._oracle = fair.path
            algo._true_top_m = true_top_m
            algo._name = "CASE(path-arms)"

            if PATH_PULL_MODEL == "single":
                algo._oracle_cost = lambda _path_id: 1
            else:
                algo._oracle_cost = lambda path_id, paths=env.paths: len(paths[int(path_id)])
            return algo

        elif BASELINE_ARM_MODE == "step_surrogate":
            sigma_step = sigma_for_oracle(env, oracle_kind="step")
            step_paths = make_step_as_paths(env)

            base = CASE(
                paths=step_paths,
                feature_matrix=env.feature_matrix,
                m=ALG_CASE["m"],
                d=ENV_CFG["dim"],
                lambda_reg=ALG_CASE["lambda_reg"],
                epsilon=ALG_CASE["epsilon"],
                delta=ALG_CASE["delta"],
                R=sigma_step,
                S_0=ALG_CASE["S_0"],
                challenger_size=ALG_CASE.get("challenger_size", 5),
                challenger_batch=ALG_CASE.get("challenger_batch", 5),
                seed=seed_k,
            )

            algo = StepSurrogateToPathRanker(
                base_algo=base,
                env=env,
                m_paths=ALG_CASE["m"],
                epsilon=ALG_CASE["epsilon"],
                delta=ALG_CASE["delta"],
                R=sigma_step,
                S_0=ALG_CASE["S_0"],
                lambda_reg=ALG_CASE["lambda_reg"],
                name="CASE(step-arms→rank-paths)",
            )
            algo._env = env
            algo._oracle = fair.step
            algo._true_top_m = true_top_m
            algo._name = "CASE(step-arms→rank-paths)"
            algo._oracle_cost = lambda _a: 1
            return algo

        else:
            raise ValueError("Unknown BASELINE_ARM_MODE")

    def make_algo_mlingape(k):
        seed_k = trial_seeds[k]
        env = ReasoningEnvironment(**ENV_CFG, seed=seed_k)

        true_scores = env.get_ground_truth()
        sorted_truth = sorted(true_scores.items(), key=lambda x: x[1], reverse=True)
        true_top_m = [pid for pid, _ in sorted_truth[: ALG_MLINGAPE["m"]]]

        fair = FairOracle(env, seed_k)

        if BASELINE_ARM_MODE == "path":
            sigma_path = sigma_for_oracle(env, oracle_kind="path")

            algo = m_LinGapE(
                paths=env.paths,
                feature_matrix=env.feature_matrix,
                m=ALG_MLINGAPE["m"],
                d=ENV_CFG["dim"],
                lambda_reg=ALG_MLINGAPE["lambda_reg"],
                epsilon=ALG_MLINGAPE["epsilon"],
                delta=ALG_MLINGAPE["delta"],
                R=sigma_path,
                S_0=ALG_MLINGAPE["S_0"],
                selection_rule=ALG_MLINGAPE["selection_rule"],
                seed=seed_k,
            )
            algo._env = env
            algo._oracle = fair.path
            algo._true_top_m = true_top_m
            algo._name = f"m-LinGapE({ALG_MLINGAPE['selection_rule']}; path-arms)"

            if PATH_PULL_MODEL == "single":
                algo._oracle_cost = lambda _path_id: 1
            else:
                algo._oracle_cost = lambda path_id, paths=env.paths: len(paths[int(path_id)])
            return algo

        elif BASELINE_ARM_MODE == "step_surrogate":
            sigma_step = sigma_for_oracle(env, oracle_kind="step")
            step_paths = make_step_as_paths(env)

            base = m_LinGapE(
                paths=step_paths,
                feature_matrix=env.feature_matrix,
                m=ALG_MLINGAPE["m"],
                d=ENV_CFG["dim"],
                lambda_reg=ALG_MLINGAPE["lambda_reg"],
                epsilon=ALG_MLINGAPE["epsilon"],
                delta=ALG_MLINGAPE["delta"],
                R=sigma_step,
                S_0=ALG_MLINGAPE["S_0"],
                selection_rule=ALG_MLINGAPE["selection_rule"],
                seed=seed_k,
            )

            algo = StepSurrogateToPathRanker(
                base_algo=base,
                env=env,
                m_paths=ALG_MLINGAPE["m"],
                epsilon=ALG_MLINGAPE["epsilon"],
                delta=ALG_MLINGAPE["delta"],
                R=sigma_step,
                S_0=ALG_MLINGAPE["S_0"],
                lambda_reg=ALG_MLINGAPE["lambda_reg"],
                name=f"m-LinGapE({ALG_MLINGAPE['selection_rule']}; step-arms→rank-paths)",
            )
            algo._env = env
            algo._oracle = fair.step
            algo._true_top_m = true_top_m
            algo._name = f"m-LinGapE({ALG_MLINGAPE['selection_rule']}; step-arms→rank-paths)"
            algo._oracle_cost = lambda _a: 1
            return algo

        else:
            raise ValueError("Unknown BASELINE_ARM_MODE")

    def make_algo_gifa(k):
        seed_k = trial_seeds[k]
        env = ReasoningEnvironment(**ENV_CFG, seed=seed_k)

        true_scores = env.get_ground_truth()
        sorted_truth = sorted(true_scores.items(), key=lambda x: x[1], reverse=True)
        true_top_m = [pid for pid, _ in sorted_truth[: ALG_GIFA["m"]]]

        fair = FairOracle(env, seed_k)

        if BASELINE_ARM_MODE == "path":
            sigma_path = sigma_for_oracle(env, oracle_kind="path")

            algo = LinGIFA(
                paths=env.paths,
                feature_matrix=env.feature_matrix,
                m=ALG_GIFA["m"],
                d=ENV_CFG["dim"],
                lambda_reg=ALG_GIFA["lambda_reg"],
                epsilon=ALG_GIFA["epsilon"],
                delta=ALG_GIFA["delta"],
                R=sigma_path,
                S_0=ALG_GIFA["S_0"],
                selection_rule=ALG_GIFA["selection_rule"],
                seed=seed_k,
            )
            algo._env = env
            algo._oracle = fair.path
            algo._true_top_m = true_top_m
            algo._name = f"LinGIFA({ALG_GIFA['selection_rule']}; path-arms)"

            if PATH_PULL_MODEL == "single":
                algo._oracle_cost = lambda _path_id: 1
            else:
                algo._oracle_cost = lambda path_id, paths=env.paths: len(paths[int(path_id)])
            return algo

        elif BASELINE_ARM_MODE == "step_surrogate":
            sigma_step = sigma_for_oracle(env, oracle_kind="step")
            step_paths = make_step_as_paths(env)

            base = LinGIFA(
                paths=step_paths,
                feature_matrix=env.feature_matrix,
                m=ALG_GIFA["m"],
                d=ENV_CFG["dim"],
                lambda_reg=ALG_GIFA["lambda_reg"],
                epsilon=ALG_GIFA["epsilon"],
                delta=ALG_GIFA["delta"],
                R=sigma_step,
                S_0=ALG_GIFA["S_0"],
                selection_rule=ALG_GIFA["selection_rule"],
                seed=seed_k,
            )

            algo = StepSurrogateToPathRanker(
                base_algo=base,
                env=env,
                m_paths=ALG_GIFA["m"],
                epsilon=ALG_GIFA["epsilon"],
                delta=ALG_GIFA["delta"],
                R=sigma_step,
                S_0=ALG_GIFA["S_0"],
                lambda_reg=ALG_GIFA["lambda_reg"],
                name=f"LinGIFA({ALG_GIFA['selection_rule']}; step-arms→rank-paths)",
            )
            algo._env = env
            algo._oracle = fair.step
            algo._true_top_m = true_top_m
            algo._name = f"LinGIFA({ALG_GIFA['selection_rule']}; step-arms→rank-paths)"
            algo._oracle_cost = lambda _a: 1
            return algo

        else:
            raise ValueError("Unknown BASELINE_ARM_MODE")

    def make_algo_xtreme(k):
        seed_k = trial_seeds[k]
        env = ReasoningEnvironment(**ENV_CFG, seed=seed_k)

        true_scores = env.get_ground_truth()
        sorted_truth = sorted(true_scores.items(), key=lambda x: x[1], reverse=True)
        true_top_m = [pid for pid, _ in sorted_truth[: ALG_XTREME["m"]]]

        fair = FairOracle(env, seed_k)
        oracle_xtreme = fair.path

        algo = XtremeAlg3OnPaths(
            num_paths=ENV_CFG["num_paths"],
            m=ALG_XTREME["m"],
            k=ALG_XTREME["k"],
            r=ALG_XTREME["r"],
            beam_size=ALG_XTREME["beam_size"],
            branching=ALG_XTREME["branching"],
            lam=ALG_XTREME["lam"],
            gamma_C=ALG_XTREME["gamma_C"],
            seed=seed_k,
            name="eXtreme(Alg3)",
            squash_rewards=ALG_XTREME["squash_rewards"],
            sigmoid_alpha=ALG_XTREME["sigmoid_alpha"],
            sigmoid_beta=ALG_XTREME["sigmoid_beta"],
            sigmoid_clip=ALG_XTREME.get("sigmoid_clip", 35.0),
        )

        algo._env = env
        algo._oracle = oracle_xtreme
        algo._true_top_m = true_top_m
        algo._name = "eXtreme(Alg3)"

        if PATH_PULL_MODEL == "single":
            algo._oracle_cost = lambda _path_id: 1
        else:
            algo._oracle_cost = lambda path_id, paths=env.paths: len(paths[int(path_id)])
        return algo

    # ---------------------------
    # Run trials (FAIR budget)
    # ---------------------------
    res_pgiha = run_trials(make_algo_fn=make_algo_pgiha, n_trials=N_TRIALS, max_rounds=MAX_ROUNDS, verbose=True, log_every=50)
    res_case = run_trials(make_algo_fn=make_algo_case, n_trials=N_TRIALS, max_rounds=MAX_ROUNDS, verbose=True, log_every=50)
    res_mlingape = run_trials(make_algo_fn=make_algo_mlingape, n_trials=N_TRIALS, max_rounds=MAX_ROUNDS, verbose=True, log_every=50)
    res_gifa = run_trials(make_algo_fn=make_algo_gifa, n_trials=N_TRIALS, max_rounds=MAX_ROUNDS, verbose=True, log_every=50)
    res_xtreme = run_trials(make_algo_fn=make_algo_xtreme, n_trials=N_TRIALS, max_rounds=MAX_ROUNDS, verbose=True, log_every=50)

    all_results = {
        "GICA": res_pgiha,
        "CASE": res_case,
        "m-LinGapE": res_mlingape,
        "LinGIFA": res_gifa,
        "eXtreme(Alg3)": res_xtreme,
    }

    plot_compare(res_pgiha, res_mlingape, nameA="GICA", nameB="m-LinGapE",
                 epsilon=ALG_PGIHA["epsilon"], save_path=outpath("compare_pgiha_mlingape"))

    plot_compare(res_pgiha, res_gifa, nameA="GICA", nameB="LinGIFA",
                 epsilon=ALG_PGIHA["epsilon"], save_path=outpath("compare_pgiha_gifa"))

    plot_compare(res_case, res_mlingape, nameA="CASE", nameB="m-LinGapE",
                 epsilon=ALG_CASE["epsilon"], save_path=outpath("compare_case_mlingape"))

    plot_compare(res_case, res_gifa, nameA="CASE", nameB="LinGIFA",
                 epsilon=ALG_CASE["epsilon"], save_path=outpath("compare_case_gifa"))

    plot_compare(res_case, res_xtreme, nameA="CASE", nameB="eXtreme",
                 epsilon=ALG_CASE["epsilon"], save_path=outpath("compare_case_xtreme"))

    plot_compare(res_pgiha, res_xtreme, nameA="GICA", nameB="eXtreme",
                 epsilon=ALG_PGIHA["epsilon"], save_path=outpath("compare_pgiha_xtreme"))

    plot_compare(res_pgiha, res_case, nameA="GICA", nameB="CASE",
                 epsilon=ALG_PGIHA["epsilon"], save_path=outpath("compare_pgiha_case"))

    plot_compare_all(all_results, epsilon=ALG_PGIHA["epsilon"],
                     save_path=outpath("compare_all_algorithms"))
