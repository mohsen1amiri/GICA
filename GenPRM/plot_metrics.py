# -*- coding: utf-8 -*-
"""
Fully-parameterized experiment runner + plotting (publication-ready) with script-managed logging.

- Parses ALL knobs via argparse (env, algorithms, modes, plotting, trials).
- Supports:
    * --run-seeds <s1> <s2> ...    (runs once per seed you pass; minimal change requested)
    * --runs / --run-seed / --randomize-run-seed (legacy-compatible)
- Logging:
    * Writes a log file per run under --logs-dir (default: logs/)
    * Still prints to console unless --no-console-log is passed
    * Logging time excluded from runtime as before (overhead measured)
- Plots separate figures per algorithm (publication style).
"""
import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from GICA import GICA
from GIFA import LinGIFA
from m_LinGapE import m_LinGapE
from run import ReasoningEnvironment
# from XTreme import XtremeAlg3OnPaths
from CASE import CASE


# ---------------------------
# Global timestamp & output path helper
# ---------------------------
RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")


def outpath(stem: str, seed_tag: str = "", out_dir: str = "plots", ext: str = ".png") -> str:
    """
    Build '<out_dir>/<stem>[_seed<seed_tag>]_YYYYmmdd_HHMMSS.<ext>'
    seed_tag is optional and used to distinguish multi-run outputs.
    """
    os.makedirs(out_dir, exist_ok=True)
    base = stem if seed_tag == "" else f"{stem}_seed{seed_tag}"
    return os.path.join(out_dir, f"{base}_{RUN_TS}{ext}")


# ---------------------------
# Logging helpers
# ---------------------------
def setup_run_logger(run_idx: int, seed_tag: str, logs_dir: str, level: str, console: bool, extra_tags: str = "") -> logging.Logger:
    """
    Create a per-run logger writing to logs_dir/run<idx>_seed<seedtag>_<extra_tags>_<ts>.log
    """
    os.makedirs(logs_dir, exist_ok=True)
    
    # Add the extra_tags to the logger name
    logger_name = f"run{run_idx}_seed{seed_tag or 'none'}{extra_tags}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False 

    # Clear previous handlers if reusing name
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # File handler uses the updated logger_name
    log_file = os.path.join(logs_dir, f"{logger_name}_{RUN_TS}.log")
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # Console handler (optional)
    if console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(getattr(logging, level.upper(), logging.INFO))
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    logger.info(f"[logger] Log file: {log_file}")
    return logger


# ---------------------------
# Deterministic per-arm oracles with per-trial seeds
# ---------------------------
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

    def path(self, path_id: int, path_pull_model: str = "single") -> float:
        path_id = int(path_id)
        steps = self.env.paths[path_id]

        if path_pull_model == "single":
            # Baseline treats the path as a single black-box arm (Cost = 1)
            g = np.mean(self.env.feature_matrix[steps], axis=0)
            clean = float(g @ self.env.true_theta)
            rng = self._get_rng(1, path_id)
            noise = float(rng.normal(0.0, self.env.noise_std))
            return clean + noise
            
        elif path_pull_model == "avg_steps":
            # NO SHORTCUT: Physically query each step one-by-one to measure true runtime.
            # This calls FairOracle.step() for every individual step in the path.
            step_rewards = [self.step(s) for s in steps]
            
            # Return the averaged reward for the baseline to use in its theta update
            return float(np.mean(step_rewards))
            
        else:
            raise ValueError("Unknown PATH_PULL_MODEL")


# ---------------------------
# Helpers for step-surrogate mode
# ---------------------------
def sigma_for_oracle(env, *, oracle_kind: str, path_pull_model: str) -> float:
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

    if path_pull_model == "single":
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

    We track wrapper post-processing time in self._postprocess_time so the
    experiment runner can subtract it from the baseline runtime.
    """

    def __init__(self, base_algo, env, m_paths, epsilon, delta, R, S_0, lambda_reg, name,
                 step_surrogate_stop_rule: str):
        self.base = base_algo
        self.env = env
        self.m = int(m_paths)
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.R = float(R)
        self.S_0 = float(S_0)
        self.lambda_reg = float(lambda_reg)
        self._name = str(name)
        self._stop_rule = step_surrogate_stop_rule

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
        for attr in ("theta_hat", "theta", "theta_est"):
            if hasattr(base, attr):
                return getattr(base, attr)
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

        if self._stop_rule == "baseline":
            done = bool(done_base)
        elif self._stop_rule == "path_gamma":
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
    max_len = max(len(s) for s in seqs) if len(seqs) > 0 else 0
    out = np.full((len(seqs), max_len), np.nan, dtype=float)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return out


def _slugify(name: str) -> str:
    """Safe stem for filenames."""
    s = name.lower()
    for a, b in {" ": "_", "(": "", ")": "", "→": "to", ";": "", "/": "_", ":": ""}.items():
        s = s.replace(a, b)
    return s


def apply_publication_style(dpi: int, use_tex: bool = False, font_size: int = 10):
    """
    Matplotlib rcParams for consistent, publication-quality figures.
    """
    plt.rcParams.update({
        "savefig.dpi": dpi,
        "figure.dpi": dpi,
        "font.size": font_size,
        "axes.titlesize": font_size + 1,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.6,
        "axes.prop_cycle": matplotlib.cycler(color=[
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
            "#bcbd22", "#17becf"
        ]),
        "pdf.fonttype": 42,  # TrueType in PDF
        "ps.fonttype": 42,
    })
    # Serif font; switch to LaTeX if requested
    if use_tex:
        plt.rcParams.update({"text.usetex": True, "font.family": "serif"})
    else:
        plt.rcParams.update({"text.usetex": False, "font.family": "serif"})


def summarize_results(all_results, logger=None):
    """
    Print/log mean ± std across trials for each algorithm, for the metrics returned.
    """
    metrics = [
        (0, "Comparisons",      "{:>14.3e}"),
        (1, "Runtime (s)",      "{:>14.4f}"),
        (2, "Oracle cost",      "{:>14.3e}"),
        (6, "Avg Realized Rho", "{:>14.4e}"),
    ]

    header = f"{'Algorithm':<22}" + "".join(f"{name:>32}" for _, name, _ in metrics)
    sep    = "-" * len(header)
    rows   = [header, sep]

    for name, res in all_results.items():
        cells = []
        for idx, _label, fmt in metrics:
            # Safely grab the array if it exists
            arr = np.asarray(res[idx], dtype=float) if len(res) > idx else np.array([])
            
            # Filter out NaNs
            arr = arr[~np.isnan(arr)] if arr.size > 0 else arr
            
            if arr.size == 0:
                cells.append(f"{'n/a':>32}")
                continue
                
            mean = float(np.mean(arr))
            std  = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            cells.append(f"  {fmt.format(mean)} ± {fmt.format(std)}")
            
        rows.append(f"{name:<22}" + "".join(cells))

    block = "\n".join(rows)
    print("\n=== Summary across trials (mean ± std) ===")
    print(block)
    if logger is not None:
        logger.info("Summary across trials (mean ± std):\n" + block)

# --- PERSISTENCE AND FORMAL PLOTTING ENGINE ---

def save_experimental_results(all_results, out_dir, seed_tag):
    """
    Saves raw trial data and algorithm traces to a compressed NPZ file.
    """
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"raw_data_run_seed{seed_tag}_{RUN_TS}.npz")
    
    data_to_save = {}
    for algo_name, res in all_results.items():
        prefix = _slugify(algo_name)
        data_to_save[f"{prefix}_comps"] = res[0]
        data_to_save[f"{prefix}_rts"] = res[1]
        data_to_save[f"{prefix}_orc"] = res[2]
        data_to_save[f"{prefix}_G_tr"] = np.array(res[3], dtype=object)
        data_to_save[f"{prefix}_L_tr"] = np.array(res[4], dtype=object)
        data_to_save[f"{prefix}_Acc_tr"] = np.array(res[5], dtype=object)

        if len(res) > 6:
            data_to_save[f"{prefix}_Rho"] = res[6]

    np.savez_compressed(save_path, **data_to_save)
    print(f"[Storage] Results saved to: {save_path}")
    return save_path


def plot_single_metric(res_dict, metric_idx, title, ylabel, save_path, is_log=False):
    """
    Generates a formal, single-plot figure for a specific metric (no subplots).
    Metric Index: 0 = Comparisons, 1 = Runtime, 2 = Oracle Calls.
    """
    plt.figure(figsize=(7, 6))
    names = list(res_dict.keys())
    means = [np.mean(res_dict[n][metric_idx]) for n in names]
    stds = [np.std(res_dict[n][metric_idx], ddof=1) if len(res_dict[n][metric_idx]) > 1 else 0 for n in names]

    # Sort for professional presentation
    sorted_idx = np.argsort(means)
    names = [names[i] for i in sorted_idx]
    means = [means[i] for i in sorted_idx]
    stds = [stds[i] for i in sorted_idx]

    colors = plt.cm.get_cmap('tab10')(np.linspace(0, 0.8, len(names)))
    bars = plt.bar(names, means, yerr=stds, capsize=8, color=colors, edgecolor='black', alpha=0.9)
    
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=30, ha='right', fontsize=10)
    
    if is_log:
        plt.yscale('log')
        plt.grid(True, which="both", axis='y', linestyle='--', alpha=0.4)
    else:
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        plt.grid(True, axis='y', linestyle='--', alpha=0.4)

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_gica_convergence_standalone(gica_res, epsilon, save_path):
    """
    Plots the standalone convergence profile of the GICA stopping criterion.
    """
    plt.figure(figsize=(8, 5))
    lcb_traces = gica_res[4] # min_lcb_history trace
    
    # Pad traces with NaN to compute mean across trials of different lengths
    max_len = max(len(t) for t in lcb_traces) if len(lcb_traces) > 0 else 0
    padded = np.full((len(lcb_traces), max_len), np.nan)
    for i, t in enumerate(lcb_traces):
        padded[i, :len(t)] = t
    
    mean_lcb = np.nanmean(padded, axis=0) if len(padded) > 0 else []
    std_lcb = np.nanstd(padded, axis=0, ddof=1) if len(lcb_traces) > 1 else np.zeros_like(mean_lcb)
    
    if len(mean_lcb) > 0:
        rounds = np.arange(len(mean_lcb))
        plt.plot(rounds, mean_lcb, color='#d62728', lw=2.5, label=r'GICA: $\min (\hat{\Delta} - W)$')
        plt.fill_between(rounds, mean_lcb - std_lcb, mean_lcb + std_lcb, color='#d62728', alpha=0.15)
        
    plt.axhline(-epsilon, color='black', linestyle='--', lw=1.5, label=r'Stopping Threshold $-\epsilon$')

    plt.title("GICA Stopping Criterion Convergence", fontsize=14, fontweight='bold')
    plt.xlabel("Iteration (t)", fontsize=12)
    plt.ylabel(r"Verification Margin ($\hat{\Delta} - W$)", fontsize=12)
    plt.legend(loc='best', frameon=True)
    plt.grid(True, linestyle=':', alpha=0.7)
    
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def _get_Vinv(algo):
    """Best-effort fetch of V_t^{-1} from an algo (tries common names; handles the
    step-surrogate wrapper via .base). Returns None if unavailable."""
    for obj in (algo, getattr(algo, "base", None)):
        if obj is None:
            continue
        for name in ("V_inv", "Vinv", "Vt_inv"):
            if hasattr(obj, name):
                return getattr(obj, name)
        for name in ("V", "Vt", "A", "design_matrix"):
            if hasattr(obj, name):
                try:
                    return np.linalg.inv(getattr(obj, name))
                except Exception:
                    return None
    return None


def run_trials(make_algo_fn, target_K, n_trials=100, max_rounds=5000, verbose=True, log_every=50,
               logger: logging.Logger = None,
               measure_rho=True, rho_every=25, rho_npairs=3000, rho_nsteps=3000, rho_pairs="all"):
    total_comparisons, runtimes, oracle_costs = [], [], []
    G_traces, lcb_traces, acc_traces = [], [], []
    rho_traces = []

    # A no-op logger if none is provided
    class _Null:
        def info(self, *_a, **_k): ...
    log = logger or _Null()

    for k in range(n_trials):
        # --- start-of-trial timers ---
        trial_t0 = time.perf_counter()
        log_overhead = 0.0

        algo = make_algo_fn(k)

        if measure_rho and hasattr(algo, "_env") and hasattr(algo._env, "reset_realized_rho"):
            algo._env.reset_realized_rho()

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
        rho_overhead = 0.0   # time spent measuring rho (excluded from runtime)

        while (not done) and (algo.t < max_rounds):
            done, top = algo.select_and_update(oracle_wrapped)

            if measure_rho and (algo.t % rho_every == 0) and hasattr(algo, "_env") \
                    and hasattr(algo._env, "update_realized_rho"):
                _trho0 = time.perf_counter()
                _Vi = _get_Vinv(algo)
                if _Vi is not None:
                    algo._env.update_realized_rho(_Vi, n_pairs=rho_npairs, n_steps=rho_nsteps,
                                                  seed=algo.t, pairs=rho_pairs)
                rho_overhead += time.perf_counter() - _trho0

            # --- THE PERMANENT FIX ---
            # Do NOT rely on algo.m or algo.K. Just pull the truth array we built.
            true_top_set = getattr(algo, "_true_top_K", getattr(algo, "_true_top_m", []))
            
            # The denominator is strictly the length of the truth array. 
            target_size = len(true_top_set)
            if target_size == 0: 
                target_size = 1 # Prevent crash if array is empty
            
            correct = len(set(top).intersection(set(true_top_set)))
            acc = correct / target_size
            acc_hist.append(acc)

            if verbose and (algo.t % log_every == 0):
                # --- build stopping-criterion suffix (works for all algos) ---
                gamma = None
                thr = None

                # Preferred: gamma_t = min_lcb_history[-1] (≡ \hat{Δ} - W)
                if hasattr(algo, "min_lcb_history") and len(algo.min_lcb_history) > 0:
                    gamma = float(algo.min_lcb_history[-1])
                    if hasattr(algo, "epsilon"):
                        thr = -float(algo.epsilon)

                # Fallback: if no gamma but an algorithm logs a B-like quantity
                if gamma is None and hasattr(algo, "best_G_history") and len(algo.best_G_history) > 0 and hasattr(algo, "epsilon"):
                    gamma = -float(algo.best_G_history[-1])
                    thr = -float(algo.epsilon)

                crit_suffix = ""
                if (gamma is not None) and (thr is not None):
                    crit_suffix = f"  gamma={gamma:.4f}  thr={thr:.4f}  stop={gamma >= thr}"

                # --- time the log so we can subtract its overhead from runtime ---
                tlog0 = time.perf_counter()
                msg = (f"[{algo._name} trial {k+1}/{n_trials}] iter={algo.t} done={done} "
                       f"acc={correct}/{target_size} ({acc:.2f}) oracle_cost={oracle_cost}{crit_suffix}")
                print(msg)
                log.info(msg)
                log_overhead += time.perf_counter() - tlog0

        # Calculate exact algorithmic runtime, excluding print/log/rho/wrapper overhead
        rt = time.perf_counter() - t0
        rt -= log_overhead
        rt -= rho_overhead   # EXCLUDE Assumption-3.2 rho measurement from runtime
        # Subtract wrapper post-processing time (only exists in step_surrogate wrappers)
        rt -= float(getattr(algo, "_postprocess_time", 0.0))
        rt = max(0.0, rt)

        runtimes.append(rt)
        total_comparisons.append(getattr(algo, "total_comparisons", 0))
        oracle_costs.append(oracle_cost)

        G_traces.append(np.array(getattr(algo, "best_G_history", []), dtype=float))
        lcb_traces.append(np.array(getattr(algo, "min_lcb_history", []), dtype=float))
        acc_traces.append(np.array(acc_hist, dtype=float))
        rho_dagger = np.nan
        if measure_rho and hasattr(algo, "_env") and hasattr(algo._env, "get_realized_rho"):
            _rd = algo._env.get_realized_rho()
            rho_dagger = _rd if np.isfinite(_rd) else np.nan
        rho_traces.append(rho_dagger)

        if verbose:
            # --- THE PERMANENT FIX ---
            true_top_set = getattr(algo, "_true_top_K", getattr(algo, "_true_top_m", []))
            target_size = len(true_top_set)
            
            if target_size == 0: 
                target_size = 1
            
            correct = len(set(top).intersection(set(true_top_set)))
            acc = correct / target_size

            # compute wall and subtract logging / rho / wrapper overhead
            wall = time.perf_counter() - trial_t0
            rt_pp = max(wall - log_overhead - rho_overhead
                        - float(getattr(algo, "_postprocess_time", 0.0)), 0.0)

            # rebuild criterion suffix (same logic as mid-run)
            gamma = None
            thr = None
            if hasattr(algo, "min_lcb_history") and len(algo.min_lcb_history) > 0:
                gamma = float(algo.min_lcb_history[-1])
                if hasattr(algo, "epsilon"):
                    thr = -float(algo.epsilon)
            elif hasattr(algo, "best_G_history") and len(algo.best_G_history) > 0 and hasattr(algo, "epsilon"):
                gamma = -float(algo.best_G_history[-1])
                thr = -float(algo.epsilon)

            crit_suffix = ""
            if (gamma is not None) and (thr is not None):
                crit_suffix = f",  gamma={gamma:.4f}, thr={thr:.4f}, stop={gamma >= thr}"

            rho_str = f" | rho_dagger={rho_dagger:.4e}" if not np.isnan(rho_dagger) else ""

            tlog0 = time.perf_counter()
            msg = (f"[{algo._name} trial {k+1}/{n_trials}] finished: iter={algo.t}, done={done}, "
                   f"final acc={correct}/{target_size} ({acc:.2f}), oracle_cost={oracle_cost}, "
                   f"runtime={rt_pp:.2f}s{crit_suffix}{rho_str}")
            print(msg)
            log.info(msg)
            log_overhead += time.perf_counter() - tlog0

    return (
        np.array(total_comparisons),
        np.array(runtimes),
        np.array(oracle_costs),
        G_traces,
        lcb_traces,
        acc_traces,
        np.array(rho_traces),
    )


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # ---------------------------
    # CLI: parse ALL parameters
    # ---------------------------
    p = argparse.ArgumentParser(description="Bandit baselines with process-level verification (publication-ready plots).")

    # Output / plotting
    p.add_argument("--plots-dir", type=str, default="plots", help="Output folder for plots.")
    p.add_argument("--file-format", type=str, default="png", choices=["png", "pdf", "svg"], help="Figure file format.")
    p.add_argument("--dpi", type=int, default=300, help="Save/display DPI.")
    p.add_argument("--use-tex", action="store_true", help="Use LaTeX text rendering (requires LaTeX installed).")
    p.add_argument("--font-size", type=int, default=10, help="Base font size for figures.")

    # Logging
    p.add_argument("--logs-dir", type=str, default="logs", help="Directory to save log files.")
    p.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--no-console-log", action="store_true", help="Disable console logging (file logging still on).")

    p.add_argument("--algos", nargs="+", default=["gica", "case", "mlingape", "gifa"],
                   choices=["gica", "case", "mlingape", "gifa"], help="Which algorithms to run.")
    p.add_argument("--measure-rho", dest="measure_rho", action="store_true", default=True)
    p.add_argument("--no-measure-rho", dest="measure_rho", action="store_false")
    p.add_argument("--rho-every", type=int, default=25, help="Compute rho_t every N rounds.")
    p.add_argument("--rho-npairs", type=int, default=3000, help="Pair samples for rho_t (<=0 => ALL).")
    p.add_argument("--rho-nsteps", type=int, default=3000, help="Step samples for rho_t (<=0 => ALL).")
    p.add_argument("--rho-pairs", type=str, default="all", choices=["all", "boundary"])

    # Modes
    p.add_argument("--baseline-arm-mode", type=str, default="path",
                   choices=["path", "step_surrogate"], help="Baselines operate on PATH or STEP arms.")
    p.add_argument("--step-surrogate-stop-rule", type=str, default="baseline",
                   choices=["baseline", "path_gamma"], help="Stopping rule in step_surrogate mode.")
    p.add_argument("--path-pull-model", type=str, default="avg_steps",
                   choices=["single", "avg_steps"], help="How to model a PATH pull when mode=path.")

    # Env config
    p.add_argument("--top-k", type=int, default=10, help="Rank of the binding boundary (K/m).")
    p.add_argument("--step-scale", type=float, default=0.30, help="Std of random step features.")
    p.add_argument("--util-range", type=float, default=1.0, help="Spread of random path qualities.")
    p.add_argument("--num-paths", type=int, default=1_000)  # 200, 500, 1_000
    p.add_argument("--dim", type=int, default=8)
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--path-len-min", type=int, default=20)
    p.add_argument("--path-len-max", type=int, default=80)
    # --- controlled-structure knobs (FIXED across the M-sweep) ---
    p.add_argument("--theta-norm", type=float, default=1.0)   # must be <= every algo's S_0
    p.add_argument("--util-floor", type=float, default=0.01)
    p.add_argument("--grid-gap", type=float, default=1e-6)    # Delta_min = theta_norm * grid_gap
    p.add_argument("--gap-spread", type=float, default=2.0)
    p.add_argument("--feature-norm", type=float, default=2.5) # L: size for the LARGEST M

    p.add_argument("--b-min", type=float, default=0.1,
                help="Lower bound of per-path off-axis bias amplitude b_i.")
    p.add_argument("--b-max", type=float, default=10.0,
                help="Upper bound of per-path off-axis bias amplitude b_i.")

    # GICA
    p.add_argument("--gica-K", type=int, default=10)
    p.add_argument("--gica-lambda", type=float, default=1.0)
    p.add_argument("--gica-epsilon", type=float, default=0.02)
    p.add_argument("--gica-delta", type=float, default=0.01)
    p.add_argument("--gica-R", type=float, default=0.1)
    p.add_argument("--gica-S0", type=float, default=2.0)
    p.add_argument("--gica-step-pool", type=str, default="paths", choices=["paths", "all"])

    # CASE
    p.add_argument("--case-m", type=int, default=10)
    p.add_argument("--case-lambda", type=float, default=1.0)
    p.add_argument("--case-epsilon", type=float, default=0.02)
    p.add_argument("--case-delta", type=float, default=0.01)
    p.add_argument("--case-R", type=float, default=0.1)
    p.add_argument("--case-S0", type=float, default=2.0)
    p.add_argument("--case-challenger-size", type=int, default=10)
    p.add_argument("--case-challenger-batch", type=int, default=10)

    # m-LinGapE
    p.add_argument("--mlingape-m", type=int, default=10)
    p.add_argument("--mlingape-lambda", type=float, default=1.0)
    p.add_argument("--mlingape-epsilon", type=float, default=0.02)
    p.add_argument("--mlingape-delta", type=float, default=0.01)
    p.add_argument("--mlingape-R", type=float, default=0.1)
    p.add_argument("--mlingape-S0", type=float, default=2.0)
    p.add_argument("--mlingape-selection", type=str, default="largest_variance",
                   choices=["largest_variance", "greedy", "optimized"])

    # LinGIFA
    p.add_argument("--gifa-m", type=int, default=10)
    p.add_argument("--gifa-lambda", type=float, default=1.0)
    p.add_argument("--gifa-epsilon", type=float, default=0.02)
    p.add_argument("--gifa-delta", type=float, default=0.01)
    p.add_argument("--gifa-R", type=float, default=0.1)
    p.add_argument("--gifa-S0", type=float, default=2.0)
    p.add_argument("--gifa-selection", type=str, default="largest_variance",
                   choices=["largest_variance", "greedy"])

    # # eXtreme
    # p.add_argument("--xtreme-m", type=int, default=10)
    # p.add_argument("--xtreme-k", type=int, default=1)
    # p.add_argument("--xtreme-r", type=int, default=1)
    # p.add_argument("--xtreme-lam", type=float, default=1.0)
    # p.add_argument("--xtreme-gamma-C", type=float, default=1.0)
    # p.add_argument("--xtreme-branching", type=int, default=2)
    # p.add_argument("--xtreme-beam-size", type=int, default=None)  # default set to num_paths later
    # p.add_argument("--xtreme-squash-rewards", action="store_true")
    # p.add_argument("--xtreme-sigmoid-alpha", type=float, default=5.0)
    # p.add_argument("--xtreme-sigmoid-beta", type=float, default=0.0)
    # p.add_argument("--xtreme-sigmoid-clip", type=float, default=35.0)

    # Runner
    p.add_argument("--n-trials", type=int, default=10)
    p.add_argument("--max-rounds", type=int, default=100_000)
    p.add_argument("--log-every", type=int, default=1, help="Log progress every N rounds.")
    p.add_argument("--verbose", action="store_true", help="Enable logging inside the loops.")

    # Seeding / multi-run
    p.add_argument("--runs", type=int, default=1, help="Number of independent runs.")
    p.add_argument("--run-seed", type=int, default=None, help="Root seed for the first run (int).")
    p.add_argument("--randomize-run-seed", action="store_true",
                   help="If set (and --run-seed omitted), pick a fresh run seed.")
    # Explicit list of run seeds (overrides the three flags above if provided)
    p.add_argument("--run-seeds", type=int, nargs="+", default=None,
                   help="Explicit run seeds, e.g., --run-seeds 101 202 303")

    args = p.parse_args()

    _rho_np = None if args.rho_npairs <= 0 else args.rho_npairs
    _rho_ns = None if args.rho_nsteps <= 0 else args.rho_nsteps

    # Apply plot style
    apply_publication_style(dpi=args.dpi, use_tex=args.use_tex, font_size=args.font_size)

    # Prepare generic ENV and modes
    ENV_CFG = dict(
        num_paths=args.num_paths,
        dim=args.dim,
        noise_std=args.noise_std,
        path_len_min=args.path_len_min,
        path_len_max=args.path_len_max,
        theta_norm=args.theta_norm,
        util_floor=args.util_floor,
        grid_gap=args.grid_gap,
        gap_spread=args.gap_spread,
        feature_norm=args.feature_norm,
        b_min=args.b_min,          
        b_max=args.b_max,          
        top_k=args.top_k,          
        step_scale=args.step_scale, 
        util_range=args.util_range, 
    )
    # --- structure check: confirm the M-sweep is clean ---
    _env_chk = ReasoningEnvironment(**ENV_CFG, seed=0)
    _geom = _env_chk.diagnose_geometry()
    print(f"[STRUCTURE CHECK] M={_env_chk.num_paths}  d={_env_chk.dim}  "
        f"Delta_min={_env_chk.get_min_gap():.6e}  "
        f"rho_dagger~={_env_chk.estimate_rho_dagger():.4f}  "
        f"c0_tilde={_env_chk.get_c0_tilde(args.gica_lambda):.6f}  "
        f"rank(G)={_geom['rank']}  frac_var_along_u={_geom['frac_var_along_u']:.3f}")
    del _env_chk, _geom

    BASELINE_ARM_MODE = args.baseline_arm_mode
    STEP_SURROGATE_STOP_RULE = args.step_surrogate_stop_rule
    PATH_PULL_MODEL = args.path_pull_model

    # Algorithm dicts
    ALG_GICA = dict(K=args.top_k, lambda_reg=args.gica_lambda, epsilon=args.gica_epsilon,
                    delta=args.gica_delta, R=args.gica_R, S_0=args.gica_S0,
                    step_pool_mode=args.gica_step_pool)

    ALG_CASE = dict(m=args.top_k, lambda_reg=args.case_lambda, epsilon=args.case_epsilon,
                    delta=args.case_delta, R=args.case_R, S_0=args.case_S0,
                    challenger_size=args.case_challenger_size, challenger_batch=args.case_challenger_batch)

    ALG_MLINGAPE = dict(m=args.top_k, lambda_reg=args.mlingape_lambda, epsilon=args.mlingape_epsilon,
                         delta=args.mlingape_delta, R=args.mlingape_R, S_0=args.mlingape_S0,
                         selection_rule=args.mlingape_selection)

    ALG_GIFA = dict(m=args.top_k, lambda_reg=args.gifa_lambda, epsilon=args.gifa_epsilon,
                    delta=args.gifa_delta, R=args.gifa_R, S_0=args.gifa_S0,
                    selection_rule=args.gifa_selection)

    # ALG_XTREME = dict(
    #     m=args.xtreme_m, k=args.xtreme_k, r=args.xtreme_r,
    #     lam=args.xtreme_lam, gamma_C=args.xtreme_gamma_C,
    #     branching=args.xtreme_branching,
    #     beam_size=args.xtreme_beam_size or args.num_paths,
    #     squash_rewards=args.xtreme_squash_rewards,
    #     sigmoid_alpha=args.xtreme_sigmoid_alpha,
    #     sigmoid_beta=args.xtreme_sigmoid_beta,
    #     sigmoid_clip=args.xtreme_sigmoid_clip,
    # )

    # ---------------------------
    # Decide which run seeds to use (honor --run-seeds first)
    # ---------------------------
    if args.run_seeds and len(args.run_seeds) > 0:
        run_iter = list(enumerate([int(s) for s in args.run_seeds]))
    else:
        if args.runs < 1:
            raise ValueError("--runs must be >= 1")
        run_iter = []
        for run_idx in range(args.runs):
            if args.run_seed is not None:
                RUN_SEED = int(args.run_seed) + run_idx  # deterministic progression
            elif args.randomize_run_seed:
                RUN_SEED = int(np.random.SeedSequence().entropy)  # fresh entropy per run
            else:
                RUN_SEED = None  # legacy path
            run_iter.append((run_idx, RUN_SEED))

    # ---------------------------
    # Multi-run loop
    # ---------------------------
    for run_idx, RUN_SEED in run_iter:
        # Trial seeds: preserve legacy when no run seed + runs=1 + no explicit run_seeds
        if RUN_SEED is None and args.runs == 1 and not args.run_seeds:
            trial_seeds = list(range(args.n_trials))  # original behavior unchanged
        else:
            rng = np.random.default_rng(RUN_SEED)
            trial_seeds = rng.integers(low=0, high=2**32 - 1,
                                       size=args.n_trials, dtype=np.uint32).astype(int).tolist()

        seed_tag = "" if RUN_SEED is None else str(RUN_SEED)

        # Set up per-run logger
        logger = setup_run_logger(
            run_idx=run_idx,
            seed_tag=seed_tag,
            logs_dir=args.logs_dir,
            level=args.log_level,
            console=(not args.no_console_log),
        )
        logger.info(f"[run:{run_idx}] Starting (RUN_SEED={RUN_SEED}, n_trials={args.n_trials})")

        # Save manifest
        seeds_manifest = {
            "run_ts": RUN_TS,
            "run_idx": run_idx,
            "run_seed": RUN_SEED,
            "n_trials": args.n_trials,
            "trial_seeds": trial_seeds,
            "env": ENV_CFG,
            "modes": {
                "baseline_arm_mode": BASELINE_ARM_MODE,
                "step_surrogate_stop_rule": STEP_SURROGATE_STOP_RULE,
                "path_pull_model": PATH_PULL_MODEL,
            },
            # "algorithms": {
            #     "gica": ALG_GICA, "case": ALG_CASE, "mlingape": ALG_MLINGAPE,
            #     "gifa": ALG_GIFA, "xtreme": ALG_XTREME,
            # },
            "algorithms": {
                "gica": ALG_GICA, "case": ALG_CASE, "mlingape": ALG_MLINGAPE,
                "gifa": ALG_GIFA,
            },
        }
        seeds_path = outpath(f"seeds_run{run_idx}", seed_tag=seed_tag, out_dir=args.plots_dir, ext=".json")
        with open(seeds_path, "w", encoding="utf-8") as f:
            json.dump(seeds_manifest, f, indent=2)
        logger.info(f"[seeds] Saved seeds to: {seeds_path}")

        # ---------------------------
        # Factories
        # ---------------------------
        def make_algo_gica(k):
            seed_k = trial_seeds[k]
            env = ReasoningEnvironment(**ENV_CFG, seed=seed_k)

            true_scores = env.get_ground_truth()
            sorted_truth = sorted(true_scores.items(), key=lambda x: x[1], reverse=True)
            true_top_K = [pid for pid, _ in sorted_truth[: ALG_GICA["K"]]]

            fair = FairOracle(env, seed_k)
            sigma_step = sigma_for_oracle(env, oracle_kind="step", path_pull_model=PATH_PULL_MODEL)

            algo = GICA(
                paths=env.paths,
                feature_matrix=env.feature_matrix,
                K=ALG_GICA["K"],
                d=ENV_CFG["dim"],
                lambda_reg=ALG_GICA["lambda_reg"],
                epsilon=ALG_GICA["epsilon"],
                delta=ALG_GICA["delta"],
                R=sigma_step,
                S_0=ALG_GICA["S_0"],
                step_pool_mode=ALG_GICA["step_pool_mode"],
            )
            algo._env = env
            algo._oracle = fair.step
            algo._true_top_K = true_top_K
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
                sigma_path = sigma_for_oracle(env, oracle_kind="path", path_pull_model=PATH_PULL_MODEL)
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
                algo._oracle = lambda path_id, fp=fair.path: fp(path_id, PATH_PULL_MODEL)
                algo._true_top_m = true_top_m
                algo._name = "CASE(path-arms)"
                if PATH_PULL_MODEL == "single":
                    algo._oracle_cost = lambda _path_id: 1
                else:
                    algo._oracle_cost = lambda path_id, paths=env.paths: len(paths[int(path_id)])
                return algo

            elif BASELINE_ARM_MODE == "step_surrogate":
                sigma_step = sigma_for_oracle(env, oracle_kind="step", path_pull_model=PATH_PULL_MODEL)
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
                    step_surrogate_stop_rule=STEP_SURROGATE_STOP_RULE,
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
                sigma_path = sigma_for_oracle(env, oracle_kind="path", path_pull_model=PATH_PULL_MODEL)

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
                algo._oracle = lambda path_id, fp=fair.path: fp(path_id, PATH_PULL_MODEL)
                algo._true_top_m = true_top_m
                algo._name = f"m-LinGapE({ALG_MLINGAPE['selection_rule']}; path-arms)"

                if PATH_PULL_MODEL == "single":
                    algo._oracle_cost = lambda _path_id: 1
                else:
                    algo._oracle_cost = lambda path_id, paths=env.paths: len(paths[int(path_id)])
                return algo

            elif BASELINE_ARM_MODE == "step_surrogate":
                sigma_step = sigma_for_oracle(env, oracle_kind="step", path_pull_model=PATH_PULL_MODEL)
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
                    step_surrogate_stop_rule=STEP_SURROGATE_STOP_RULE,
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
                sigma_path = sigma_for_oracle(env, oracle_kind="path", path_pull_model=PATH_PULL_MODEL)

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
                algo._oracle = lambda path_id, fp=fair.path: fp(path_id, PATH_PULL_MODEL)
                algo._true_top_m = true_top_m
                algo._name = f"LinGIFA({ALG_GIFA['selection_rule']}; path-arms)"

                if PATH_PULL_MODEL == "single":
                    algo._oracle_cost = lambda _path_id: 1
                else:
                    algo._oracle_cost = lambda path_id, paths=env.paths: len(paths[int(path_id)])
                return algo

            elif BASELINE_ARM_MODE == "step_surrogate":
                sigma_step = sigma_for_oracle(env, oracle_kind="step", path_pull_model=PATH_PULL_MODEL)
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
                    step_surrogate_stop_rule=STEP_SURROGATE_STOP_RULE,
                )
                algo._env = env
                algo._oracle = fair.step
                algo._true_top_m = true_top_m
                algo._name = f"LinGIFA({ALG_GIFA['selection_rule']}; step-arms→rank-paths)"
                algo._oracle_cost = lambda _a: 1
                return algo

            else:
                raise ValueError("Unknown BASELINE_ARM_MODE")

        # def make_algo_xtreme(k):
        #     seed_k = trial_seeds[k]
        #     env = ReasoningEnvironment(**ENV_CFG, seed=seed_k)

        #     true_scores = env.get_ground_truth()
        #     sorted_truth = sorted(true_scores.items(), key=lambda x: x[1], reverse=True)
        #     true_top_m = [pid for pid, _ in sorted_truth[: ALG_XTREME["m"]]]

        #     fair = FairOracle(env, seed_k)
        #     oracle_xtreme = lambda path_id, fp=fair.path: fp(path_id, PATH_PULL_MODEL)

        #     algo = XtremeAlg3OnPaths(
        #         num_paths=ENV_CFG["num_paths"],
        #         m=ALG_XTREME["m"],
        #         k=ALG_XTREME["k"],
        #         r=ALG_XTREME["r"],
        #         beam_size=ALG_XTREME["beam_size"],
        #         branching=ALG_XTREME["branching"],
        #         lam=ALG_XTREME["lam"],
        #         gamma_C=ALG_XTREME["gamma_C"],
        #         seed=seed_k,
        #         name="eXtreme(Alg3)",
        #         squash_rewards=ALG_XTREME["squash_rewards"],
        #         sigmoid_alpha=ALG_XTREME["sigmoid_alpha"],
        #         sigmoid_beta=ALG_XTREME["sigmoid_beta"],
        #         sigmoid_clip=ALG_XTREME.get("sigmoid_clip", 35.0),
        #     )
        #     algo._env = env
        #     algo._oracle = oracle_xtreme
        #     algo._true_top_m = true_top_m
        #     algo._name = "eXtreme(Alg3)"

        #     if PATH_PULL_MODEL == "single":
        #         algo._oracle_cost = lambda _path_id: 1
        #     else:
        #         algo._oracle_cost = lambda path_id, paths=env.paths: len(paths[int(path_id)])
        #     return algo

        # ---------------------------
        # Run trials (FAIR budget)
        # ---------------------------
        _RHO = dict(measure_rho=args.measure_rho, rho_every=args.rho_every,
                    rho_npairs=_rho_np, rho_nsteps=_rho_ns, rho_pairs=args.rho_pairs)
        _factory = {"gica": (make_algo_gica, "GICA"), "case": (make_algo_case, "CASE"),
                    "mlingape": (make_algo_mlingape, "m-LinGapE"), "gifa": (make_algo_gifa, "LinGIFA")}
        all_results = {}
        for _key in args.algos:
            _fn, _label = _factory[_key]
            all_results[_label] = run_trials(make_algo_fn=_fn, target_K=args.top_k,
                n_trials=args.n_trials, max_rounds=args.max_rounds, verbose=args.verbose,
                log_every=args.log_every, logger=logger, **_RHO)
        # res_xtreme = run_trials(make_algo_fn=make_algo_xtreme, n_trials=args.n_trials,
        #                         max_rounds=args.max_rounds, verbose=args.verbose, log_every=args.log_every,
        #                         logger=logger)

        # all_results = {
        #     "GICA": res_gica,
        #     "CASE": res_case,
        #     "m-LinGapE": res_mlingape,
        #     "LinGIFA": res_gifa,
        #     "eXtreme(Alg3)": res_xtreme,
        # }


        # >>> add this line <
        summarize_results(all_results, logger=logger)

        # Save all the raw data for persistence
        save_experimental_results(all_results, args.plots_dir, seed_tag)

        # Generate separate, single-plot figures for all metrics
        
        # 1. Total Pairwise Comparisons
        plot_single_metric(all_results, 0, "Average Pairwise Comparisons", "Number of Comparisons", 
                           outpath("comparisons_linear", seed_tag, args.plots_dir, f".{args.file_format}"), is_log=False)
        plot_single_metric(all_results, 0, "Average Pairwise Comparisons (Log Scale)", "Number of Comparisons", 
                           outpath("comparisons_log", seed_tag, args.plots_dir, f".{args.file_format}"), is_log=True)

        # 2. Algorithmic Runtime (excluding printing overhead)
        plot_single_metric(all_results, 1, "Algorithmic Runtime", "Runtime (seconds)", 
                           outpath("runtime_linear", seed_tag, args.plots_dir, f".{args.file_format}"), is_log=False)
        plot_single_metric(all_results, 1, "Algorithmic Runtime (Log Scale)", "Runtime (seconds)", 
                           outpath("runtime_log", seed_tag, args.plots_dir, f".{args.file_format}"), is_log=True)

        # 3. Oracle Calls
        plot_single_metric(all_results, 2, "Total Oracle Samples Evaluated", "Oracle Call Cost", 
                           outpath("oracle_linear", seed_tag, args.plots_dir, f".{args.file_format}"), is_log=False)
        plot_single_metric(all_results, 2, "Total Oracle Samples Evaluated (Log Scale)", "Oracle Call Cost", 
                           outpath("oracle_log", seed_tag, args.plots_dir, f".{args.file_format}"), is_log=True)

        # 4. GICA Specific Convergence Profile
        if "GICA" in all_results:
            plot_gica_convergence_standalone(
                all_results["GICA"], 
                args.gica_epsilon, 
                outpath("gica_convergence", seed_tag, args.plots_dir, f".{args.file_format}")
            )

        logger.info(f"[run:{run_idx}] Done. Results and 7 standalone figures written under: {args.plots_dir}")