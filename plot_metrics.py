import numpy as np
import time
import matplotlib.pyplot as plt

from P_GIHA import P_GIHA
from GIFA import LinGIFA   # or: from GIFA import GIFA
from m_LinGapE import m_LinGapE
from run import ReasoningEnvironment
from XTreme import XtremeAlg3OnPaths

from CASE import CASE


class FairOracle:
    """
    Deterministic, arm-wise noise oracle.
    - Each STEP (P-GIHA) has its own RNG stream (seeded by trial_seed + step_id)
    - Each PATH (CASE) has its own RNG stream (seeded by trial_seed + path_id)
    This makes noise independent of call order.
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
        x = (x * 0xbf58476d1ce4e5b9) & mask
        x ^= (x >> 27)
        x = (x * 0x94d049bb133111eb) & mask
        x ^= (x >> 31)
        return x & mask

    def _seed_for(self, kind: int, arm_id: int) -> int:
        # kind: 0=step, 1=path
        # combine trial_seed + kind + arm_id into a single deterministic 64-bit seed
        x = (self.trial_seed * 0x9e3779b97f4a7c15) ^ (kind * 0xbf58476d1ce4e5b9) ^ (int(arm_id) + 1)
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
        noise = float(rng.normal(0.0, self.env.noise_std))
        return clean + noise


def pad_nan(seqs):
    """Pad ragged list of 1D arrays with NaN to common length."""
    max_len = max(len(s) for s in seqs)
    out = np.full((len(seqs), max_len), np.nan, dtype=float)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s
    return out

def run_trials(make_algo_fn, n_trials=50, max_rounds=2000, verbose=True, log_every=50):
    total_comparisons = []
    runtimes = []
    oracle_costs = []     # NEW: step-equivalent oracle calls per trial
    G_traces = []
    lcb_traces = []
    acc_traces = []

    for k in range(n_trials):
        algo = make_algo_fn(k)

        oracle_base = algo._oracle
        cost_fn = getattr(algo, "_oracle_cost", lambda _a: 1)

        oracle_cost = 0  # step-equivalent cost in THIS trial

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
                print(f"[{algo._name} trial {k+1}/{n_trials}] iter={algo.t} done={done} acc={correct}/{algo.m} ({acc:.2f})")

        rt = time.perf_counter() - t0

        runtimes.append(rt)
        total_comparisons.append(algo.total_comparisons)
        oracle_costs.append(oracle_cost)  # NEW

        G_traces.append(np.array(algo.best_G_history, dtype=float))
        lcb_traces.append(np.array(algo.min_lcb_history, dtype=float))
        acc_traces.append(np.array(acc_hist, dtype=float))

        if verbose:
            correct = len(set(top).intersection(set(algo._true_top_m)))
            acc = correct / algo.m
            print(f"[{algo._name} trial {k+1}/{n_trials}] finished: iter={algo.t}, done={done}, final acc={correct}/{algo.m} ({acc:.2f}), "
                  f"oracle_cost={oracle_cost}, runtime={rt:.2f}s")

    return (
        np.array(total_comparisons),
        np.array(runtimes),
        np.array(oracle_costs),  # NEW (3rd item)
        G_traces,
        lcb_traces,
        acc_traces
    )







def plot_results(total_comparisons, runtimes, G_traces, lcb_traces, label="P-GIHA",
                 epsilon=0.05, save_path="pgihA_metrics.png"):
    avg_comp = float(np.mean(total_comparisons))
    std_comp = float(np.std(total_comparisons, ddof=1)) if len(total_comparisons) > 1 else 0.0

    # pad ragged traces (trials x time)
    G = pad_nan(G_traces)
    L = pad_nan(lcb_traces)

    # mean + std across trials (ignore NaNs)
    mean_G = np.nanmean(G, axis=0)
    std_G  = np.nanstd(G, axis=0, ddof=1)

    mean_L = np.nanmean(L, axis=0)
    std_L  = np.nanstd(L, axis=0, ddof=1)

    # how many trials contribute at each round (to avoid weird bands when only 0/1 trials exist)
    cnt_G = np.sum(~np.isnan(G), axis=0)
    cnt_L = np.sum(~np.isnan(L), axis=0)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.2))

    # (a) Avg comparisons
    ax = axes[0]
    ax.bar([0], [avg_comp], yerr=[std_comp], capsize=6)
    ax.set_xticks([0])
    ax.set_xticklabels([label], rotation=45, ha="right")
    ax.set_ylabel("Avg no. of comparisons")
    ax.set_title("(a) comparisons across\nsimulations")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # (b) Runtime boxplot
    ax = axes[1]
    ax.boxplot([runtimes], labels=[label], showfliers=True)
    ax.set_ylabel("Avg runtime (seconds)")
    ax.set_title("(b) Average runtime\n(in seconds)")

    # (c) Gap index comparison (hardest boundary) with std band
    ax = axes[2]
    xG = np.arange(len(mean_G))
    ax.plot(xG, mean_G, linewidth=1, color="tab:orange", label="Gap index (mean)")

    # shade only where >=2 trials contribute
    maskG = cnt_G >= 2
    ax.fill_between(
        xG[maskG],
        (mean_G - std_G)[maskG],
        (mean_G + std_G)[maskG],
        color="tab:orange",
        alpha=0.2,
        label="±1 std"
    )

    ax.set_xlabel("Rounds")
    ax.set_ylabel("Gap index")
    ax.set_title("(c) Gap index\ncomparison")
    ax.legend(loc="best")

    # (d) Δhat - W_t (stopping quantity) with std band
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
        label="±1 std"
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


def plot_compare(resA, resB, nameA="P-GIHA", nameB="CASE", epsilon=0.05, save_path="compare.png"):
    compsA, rtsA, orcA, G_A, L_A, Acc_A = resA
    compsB, rtsB, orcB, G_B, L_B, Acc_B = resB

    GA = pad_nan(G_A); GB = pad_nan(G_B)
    LA = pad_nan(L_A); LB = pad_nan(L_B)
    AA = pad_nan(Acc_A); AB = pad_nan(Acc_B)

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

    # ---- figure: 1x8 now (f split into 3) ----
    fig, axes = plt.subplots(1, 8, figsize=(32, 3.6))

    # (a) comparisons
    means = [np.mean(compsA), np.mean(compsB)]
    stds  = [
        np.std(compsA, ddof=1) if len(compsA) > 1 else 0.0,
        np.std(compsB, ddof=1) if len(compsB) > 1 else 0.0
    ]
    axes[0].bar([0, 1], means, yerr=stds, capsize=6)
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels([nameA, nameB], rotation=45, ha="right")
    axes[0].set_ylabel("Avg no. of comparisons")
    axes[0].set_title("(a) comparisons across\nsimulations")
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # (b) runtime
    axes[1].boxplot([rtsA, rtsB], labels=[nameA, nameB], showfliers=True)
    axes[1].set_ylabel("Runtime (seconds)")
    axes[1].set_title("(b) Average runtime\n(in seconds)")

    # (c) gap index
    ax = axes[2]
    xA = np.arange(len(muGA)); xB = np.arange(len(muGB))
    ax.plot(xA, muGA, color="tab:orange", label=f"{nameA}")
    ax.plot(xB, muGB, color="tab:green",  label=f"{nameB}")
    ax.fill_between(xA[cntGA >= 2], (muGA - sdGA)[cntGA >= 2], (muGA + sdGA)[cntGA >= 2], color="tab:orange", alpha=0.2)
    ax.fill_between(xB[cntGB >= 2], (muGB - sdGB)[cntGB >= 2], (muGB + sdGB)[cntGB >= 2], color="tab:green",  alpha=0.2)
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Gap index")
    ax.set_title("(c) Gap index\ncomparison")
    ax.legend(loc="best")

    # (d) stopping quantity
    ax = axes[3]
    xA = np.arange(len(muLA)); xB = np.arange(len(muLB))
    ax.plot(xA, muLA, color="tab:blue", label=f"{nameA}")
    ax.plot(xB, muLB, color="tab:red",  label=f"{nameB}")
    ax.fill_between(xA[cntLA >= 2], (muLA - sdLA)[cntLA >= 2], (muLA + sdLA)[cntLA >= 2], color="tab:blue", alpha=0.2)
    ax.fill_between(xB[cntLB >= 2], (muLB - sdLB)[cntLB >= 2], (muLB + sdLB)[cntLB >= 2], color="tab:red",  alpha=0.2)
    ax.axhline(-epsilon, linestyle="--", color="gray", linewidth=1, label="-epsilon")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Δhat - W")
    ax.set_title("(d) stopping quantity")
    ax.legend(loc="best")

    # (e) accuracy
    ax = axes[4]
    xA = np.arange(len(muAA)); xB = np.arange(len(muAB))
    ax.plot(xA, muAA, color="tab:purple", label=f"{nameA}")
    ax.plot(xB, muAB, color="tab:brown",  label=f"{nameB}")
    ax.fill_between(xA[cntAA >= 2], (muAA - sdAA)[cntAA >= 2], (muAA + sdAA)[cntAA >= 2], color="tab:purple", alpha=0.2)
    ax.fill_between(xB[cntAB >= 2], (muAB - sdAB)[cntAB >= 2], (muAB + sdAB)[cntAB >= 2], color="tab:brown",  alpha=0.2)
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(e) Accuracy")
    ax.legend(loc="best")

    # ---- oracle stats ----
    mean_orcA = float(np.mean(orcA))
    mean_orcB = float(np.mean(orcB))
    std_orcA  = float(np.std(orcA, ddof=1)) if len(orcA) > 1 else 0.0
    std_orcB  = float(np.std(orcB, ddof=1)) if len(orcB) > 1 else 0.0

    # Avoid log-scale issues if any 0 sneaks in
    orcA_pos = np.clip(np.array(orcA, dtype=float), 1e-12, None)
    orcB_pos = np.clip(np.array(orcB, dtype=float), 1e-12, None)

    # (f1) oracle calls - linear
    ax = axes[5]
    ax.bar([0, 1], [mean_orcA, mean_orcB], yerr=[std_orcA, std_orcB], capsize=6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([nameA, nameB], rotation=45, ha="right")
    ax.set_ylabel("Avg oracle calls\n(step-equiv.)")
    ax.set_title("(f1) oracle calls\n(linear)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # (f2) oracle calls - log
    ax = axes[6]
    ax.bar([0, 1], [np.mean(orcA_pos), np.mean(orcB_pos)],
           yerr=[
               np.std(orcA_pos, ddof=1) if len(orcA_pos) > 1 else 0.0,
               np.std(orcB_pos, ddof=1) if len(orcB_pos) > 1 else 0.0
           ],
           capsize=6)
    ax.set_yscale("log")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([nameA, nameB], rotation=45, ha="right")
    ax.set_ylabel("Avg oracle calls\n(step-equiv.)")
    ax.set_title("(f2) oracle calls\n(log)")

    # (f3) relative oracle calls = CASE / P-GIHA (mean ± std over trials)
    ax = axes[7]
    ratio = orcB_pos / orcA_pos
    mean_ratio = float(np.mean(ratio))
    std_ratio  = float(np.std(ratio, ddof=1)) if len(ratio) > 1 else 0.0
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
    """
    Paper-style comparison of MANY algorithms on one figure (no simple-regret).
    res_dict: {name: (comps, rts, orc, G_traces, L_traces, Acc_traces)}
    """
    names = list(res_dict.keys())
    nA = len(names)

    # ---- helpers ----
    def mean_std(x):
        x = np.asarray(x, dtype=float)
        mu = float(np.mean(x))
        sd = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
        return mu, sd

    def mean_std_cnt_from_traces(traces):
        M = pad_nan(traces)  # trials x time (NaN padded)
        mu = np.nanmean(M, axis=0)
        sd = np.nanstd(M, axis=0, ddof=1) if M.shape[0] > 1 else np.zeros_like(mu)
        cnt = np.sum(~np.isnan(M), axis=0)
        return mu, sd, cnt

    # ---- unpack per algorithm ----
    comps_list = []
    rts_list = []
    orc_list = []
    G_stats = {}
    L_stats = {}
    A_stats = {}

    for name in names:
        comps, rts, orc, G_tr, L_tr, Acc_tr = res_dict[name]
        comps_list.append(np.asarray(comps))
        rts_list.append(np.asarray(rts))
        orc_list.append(np.asarray(orc))

        G_stats[name] = mean_std_cnt_from_traces(G_tr)
        L_stats[name] = mean_std_cnt_from_traces(L_tr)
        A_stats[name] = mean_std_cnt_from_traces(Acc_tr)

    # ---- figure layout: (a) comparisons, (b) runtime, (c) gap index, (d) stopping, (e) accuracy, (f1) oracle linear, (f2) oracle log ----
    fig, axes = plt.subplots(1, 7, figsize=(30, 3.8))

    # (a) comparisons bar
    ax = axes[0]
    means = [mean_std(c)[0] for c in comps_list]
    stds  = [mean_std(c)[1] for c in comps_list]
    ax.bar(np.arange(nA), means, yerr=stds, capsize=6)
    ax.set_xticks(np.arange(nA))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Avg no. of comparisons")
    ax.set_title("(a) comparisons")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # (b) runtime boxplot
    ax = axes[1]
    ax.boxplot(rts_list, labels=names, showfliers=True)
    ax.set_ylabel("Runtime (seconds)")
    ax.set_title("(b) runtime")

    # (c) gap index curves (mean ± std)
    ax = axes[2]
    for name in names:
        mu, sd, cnt = G_stats[name]
        x = np.arange(len(mu))
        line, = ax.plot(x, mu, label=name)
        mask = cnt >= 2
        if np.any(mask):
            ax.fill_between(x[mask], (mu - sd)[mask], (mu + sd)[mask],
                            alpha=0.2, color=line.get_color())
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Gap index")
    ax.set_title("(c) gap index")
    ax.legend(loc="best", fontsize=8)

    # (d) stopping quantity curves (mean ± std)
    ax = axes[3]
    for name in names:
        mu, sd, cnt = L_stats[name]
        x = np.arange(len(mu))
        line, = ax.plot(x, mu, label=name)
        mask = cnt >= 2
        if np.any(mask):
            ax.fill_between(x[mask], (mu - sd)[mask], (mu + sd)[mask],
                            alpha=0.2, color=line.get_color())
    ax.axhline(-epsilon, linestyle="--", linewidth=1, color="gray", label="-epsilon")
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Δhat - W")
    ax.set_title("(d) stopping qty")
    ax.legend(loc="best", fontsize=8)

    # (e) accuracy curves (mean ± std)
    ax = axes[4]
    for name in names:
        mu, sd, cnt = A_stats[name]
        x = np.arange(len(mu))
        line, = ax.plot(x, mu, label=name)
        mask = cnt >= 2
        if np.any(mask):
            ax.fill_between(x[mask], (mu - sd)[mask], (mu + sd)[mask],
                            alpha=0.2, color=line.get_color())
    ax.set_xlabel("Rounds")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("(e) accuracy")
    ax.legend(loc="best", fontsize=8)

    # (f1) oracle calls (step-equiv) linear
    ax = axes[5]
    means = [mean_std(o)[0] for o in orc_list]
    stds  = [mean_std(o)[1] for o in orc_list]
    ax.bar(np.arange(nA), means, yerr=stds, capsize=6)
    ax.set_xticks(np.arange(nA))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Avg oracle calls\n(step-equiv.)")
    ax.set_title("(f1) oracle calls")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # (f2) oracle calls (step-equiv) log
    ax = axes[6]
    orc_pos = [np.clip(np.asarray(o, dtype=float), 1e-12, None) for o in orc_list]
    means = [mean_std(o)[0] for o in orc_pos]
    stds  = [mean_std(o)[1] for o in orc_pos]
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




if __name__ == "__main__":
    # ---------------------------
    # Config
    # ---------------------------
    SEED = 0
    ENV_CFG = dict(
        num_paths=200,
        num_total_steps=50000,
        dim=8,
        noise_std=0.1,
        path_len_min=50,
        path_len_max=300,
    )

    ALG_PGIHA = dict(
        m=10,
        lambda_reg=1.0,
        epsilon=0.1,
        delta=0.05,
        R=0.1,
        S_0=2.0,
        step_pool_mode="paths",
    )

    ALG_CASE = dict(
        m=10,
        lambda_reg=1.0,
        epsilon=0.1,
        delta=0.05,
        R=0.1,
        S_0=2.0,
        challenger_size=50,
        challenger_batch=50,
        seed=0,
    )

    ALG_MLINGAPE = dict(
        m=10, lambda_reg=1.0, epsilon=0.1, delta=0.05, R=0.1, S_0=2.0,
        selection_rule="largest_variance",   # or "greedy" or "optimized"
    )

    ALG_GIFA = dict(
        m=10, lambda_reg=1.0, epsilon=0.1, delta=0.05, R=0.1, S_0=2.0,
        selection_rule="largest_variance",   # or "greedy"
    )

    ALG_XTREME = dict(
        m=10,
        k=1,                 # keep k=1 for apples-to-apples with other top-m id baselines
        r=1,                 # IGW slots (must satisfy 1 <= r <= k)
        lam=1.0,
        gamma_C=1.0,
        branching=2,
        beam_size=ENV_CFG["num_paths"],  # safe: beam won't prune anything
        seed=0,

        # If you want reward squashing ONLY for eXtreme and NOT in the env:
        squash_rewards=True,
        sigmoid_alpha=5.0,
        sigmoid_beta=0.0,
        sigmoid_clip=35.0,
    )



    N_TRIALS = 10
    MAX_ROUNDS = 10_000
    # Deterministic trial seeds (same seeds used for BOTH algorithms)
    trial_seeds = list(range(N_TRIALS))
    assert len(trial_seeds) == N_TRIALS


    # ---------------------------
    # Factories
    # ---------------------------
    # --- NEW: cache env + truth so CASE and P-GIHA share EXACT same env per trial k ---
    # ENV_CACHE = {}
    # TRUTH_CACHE = {}

    # def get_env_and_truth(k):
    #     if k not in ENV_CACHE:
    #         env = ReasoningEnvironment(**ENV_CFG, seed=SEED + k)

    #         true_scores = env.get_ground_truth()
    #         sorted_truth = sorted(true_scores.items(), key=lambda x: x[1], reverse=True)
    #         true_top_m = [pid for pid, _ in sorted_truth[:ALG_PGIHA["m"]]]  # m is same in both configs

    #         ENV_CACHE[k] = env
    #         TRUTH_CACHE[k] = true_top_m

    #     return ENV_CACHE[k], TRUTH_CACHE[k]


    def make_algo_pgiha(k):
        seed_k = trial_seeds[k]

        # one env per trial
        env = ReasoningEnvironment(**ENV_CFG, seed=seed_k)

        # truth (same for both algorithms)
        true_scores = env.get_ground_truth()
        sorted_truth = sorted(true_scores.items(), key=lambda x: x[1], reverse=True)
        true_top_m = [pid for pid, _ in sorted_truth[:ALG_PGIHA["m"]]]

        # fair oracle for this trial/env
        fair = FairOracle(env, seed_k)

        algo = P_GIHA(
            paths=env.paths,
            feature_matrix=env.feature_matrix,
            m=ALG_PGIHA["m"],
            d=ENV_CFG["dim"],
            lambda_reg=ALG_PGIHA["lambda_reg"],
            epsilon=ALG_PGIHA["epsilon"],
            delta=ALG_PGIHA["delta"],
            R=ALG_PGIHA["R"],
            S_0=ALG_PGIHA["S_0"],
            step_pool_mode=ALG_PGIHA["step_pool_mode"],
        )

        algo._env = env
        algo._oracle = fair.step              # <<< CHANGED (fair, order-independent)
        algo._true_top_m = true_top_m
        algo._name = "P-GIHA"
        algo._oracle_cost = lambda step_idx: 1
        return algo



    def make_algo_case(k):
        seed_k = trial_seeds[k]

        # one env per trial (same seed_k as P-GIHA trial k)
        env = ReasoningEnvironment(**ENV_CFG, seed=seed_k)

        # truth (same for both algorithms)
        true_scores = env.get_ground_truth()
        sorted_truth = sorted(true_scores.items(), key=lambda x: x[1], reverse=True)
        true_top_m = [pid for pid, _ in sorted_truth[:ALG_CASE["m"]]]

        # fair oracle for this trial/env
        fair = FairOracle(env, seed_k)

        algo = CASE(
            paths=env.paths,
            feature_matrix=env.feature_matrix,
            m=ALG_CASE["m"],
            d=ENV_CFG["dim"],
            lambda_reg=ALG_CASE["lambda_reg"],
            epsilon=ALG_CASE["epsilon"],
            delta=ALG_CASE["delta"],
            R=ALG_CASE["R"],
            S_0=ALG_CASE["S_0"],
            challenger_size=ALG_CASE.get("challenger_size", 5),
            challenger_batch=ALG_CASE.get("challenger_batch", 5),
            seed=seed_k,  # keep CASE internal RNG deterministic
        )

        algo._env = env
        algo._oracle = fair.path              # <<< CHANGED (fair, order-independent)
        algo._true_top_m = true_top_m
        algo._name = "CASE"
        algo._oracle_cost = lambda path_id, paths=env.paths: len(paths[int(path_id)])
        return algo
    
    def make_algo_mlingape(k):
        seed_k = trial_seeds[k]
        env = ReasoningEnvironment(**ENV_CFG, seed=seed_k)

        true_scores = env.get_ground_truth()
        sorted_truth = sorted(true_scores.items(), key=lambda x: x[1], reverse=True)
        true_top_m = [pid for pid, _ in sorted_truth[:ALG_MLINGAPE["m"]]]

        fair = FairOracle(env, seed_k)

        algo = m_LinGapE(
            paths=env.paths,
            feature_matrix=env.feature_matrix,
            m=ALG_MLINGAPE["m"],
            d=ENV_CFG["dim"],
            lambda_reg=ALG_MLINGAPE["lambda_reg"],
            epsilon=ALG_MLINGAPE["epsilon"],
            delta=ALG_MLINGAPE["delta"],
            R=ALG_MLINGAPE["R"],
            S_0=ALG_MLINGAPE["S_0"],
            selection_rule=ALG_MLINGAPE["selection_rule"],
            seed=seed_k,
        )

        algo._env = env
        algo._oracle = fair.path
        algo._true_top_m = true_top_m
        algo._name = f"m-LinGapE({ALG_MLINGAPE['selection_rule']})"
        algo._oracle_cost = lambda path_id, paths=env.paths: len(paths[int(path_id)])
        return algo


    def make_algo_gifa(k):
        seed_k = trial_seeds[k]
        env = ReasoningEnvironment(**ENV_CFG, seed=seed_k)

        true_scores = env.get_ground_truth()
        sorted_truth = sorted(true_scores.items(), key=lambda x: x[1], reverse=True)
        true_top_m = [pid for pid, _ in sorted_truth[:ALG_GIFA["m"]]]

        fair = FairOracle(env, seed_k)

        algo = LinGIFA(
            paths=env.paths,
            feature_matrix=env.feature_matrix,
            m=ALG_GIFA["m"],
            d=ENV_CFG["dim"],
            lambda_reg=ALG_GIFA["lambda_reg"],
            epsilon=ALG_GIFA["epsilon"],
            delta=ALG_GIFA["delta"],
            R=ALG_GIFA["R"],
            S_0=ALG_GIFA["S_0"],
            selection_rule=ALG_GIFA["selection_rule"],
            seed=seed_k,
        )

        algo._env = env
        algo._oracle = fair.path
        algo._true_top_m = true_top_m
        algo._name = f"LinGIFA({ALG_GIFA['selection_rule']})"
        algo._oracle_cost = lambda path_id, paths=env.paths: len(paths[int(path_id)])
        return algo
    

    def make_algo_xtreme(k):
        seed_k = trial_seeds[k]
        env = ReasoningEnvironment(**ENV_CFG, seed=seed_k)

        # truth
        true_scores = env.get_ground_truth()
        sorted_truth = sorted(true_scores.items(), key=lambda x: x[1], reverse=True)
        true_top_m = [pid for pid, _ in sorted_truth[:ALG_XTREME["m"]]]

        fair = FairOracle(env, seed_k)

        # --- optional: sigmoid squashing ONLY for eXtreme (NOT for baselines) ---

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
        algo._oracle_cost = lambda path_id, paths=env.paths: len(paths[int(path_id)])  # step-equiv cost
        return algo






    # ---------------------------
    # Run trials (NOTE: new run_trials returns 5 outputs now)
    # ---------------------------
    res_pgiha = run_trials(
        make_algo_fn=make_algo_pgiha,
        n_trials=N_TRIALS,
        max_rounds=MAX_ROUNDS,
        verbose=True,
        log_every=50,
    )

    res_case = run_trials(
        make_algo_fn=make_algo_case,
        n_trials=N_TRIALS,
        max_rounds=MAX_ROUNDS,
        verbose=True,
        log_every=50,
    )

    res_mlingape = run_trials(make_algo_mlingape, n_trials=N_TRIALS, max_rounds=MAX_ROUNDS, verbose=True, log_every=50)
    res_gifa     = run_trials(make_algo_gifa,     n_trials=N_TRIALS, max_rounds=MAX_ROUNDS, verbose=True, log_every=50)
    res_xtreme = run_trials(make_algo_xtreme, n_trials=N_TRIALS, max_rounds=MAX_ROUNDS, verbose=True, log_every=50)


    plot_compare(res_pgiha, res_mlingape, nameA="P-GIHA", nameB="m-LinGapE", epsilon=ALG_PGIHA["epsilon"], save_path="compare_pgiha_mlingape.png")
    plot_compare(res_pgiha, res_gifa,     nameA="P-GIHA", nameB="LinGIFA",   epsilon=ALG_PGIHA["epsilon"], save_path="compare_pgiha_gifa.png")
    plot_compare(res_case,  res_mlingape, nameA="CASE",   nameB="m-LinGapE", epsilon=ALG_CASE["epsilon"],  save_path="compare_case_mlingape.png")
    plot_compare(res_case,  res_gifa,     nameA="CASE",   nameB="LinGIFA",   epsilon=ALG_CASE["epsilon"],  save_path="compare_case_gifa.png")
    plot_compare(res_case, res_xtreme, nameA="CASE", nameB="eXtreme", epsilon=ALG_CASE["epsilon"], save_path="compare_case_xtreme.png")
    plot_compare(res_pgiha, res_xtreme, nameA="P-GIHA", nameB="eXtreme", epsilon=ALG_PGIHA["epsilon"], save_path="compare_pgiha_xtreme.png")


    # ---------------------------
    # Plot comparison (NOTE: new plot_compare expects the 5-output tuples)
    # ---------------------------
    plot_compare(
        res_pgiha,
        res_case,
        nameA="P-GIHA",
        nameB="CASE",
        epsilon=ALG_PGIHA["epsilon"],
        save_path="compare_pgiha_case.png",
    )

    all_results = {
        "P-GIHA": res_pgiha,
        "CASE": res_case,
        "m-LinGapE": res_mlingape,
        "LinGIFA": res_gifa,
        "eXtreme(Alg3)": res_xtreme,
    }


    plot_compare_all(
        all_results,
        epsilon=ALG_PGIHA["epsilon"],   # (or a shared epsilon you set for all algs)
        save_path="compare_all_algorithms.png",
    )

