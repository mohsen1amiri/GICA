import numpy as np
import time
import matplotlib.pyplot as plt

from P_GIHA import P_GIHA
from run import ReasoningEnvironment

from CASE import CASE



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
    G_traces = []
    lcb_traces = []
    acc_traces = []   # NEW

    for k in range(n_trials):
        algo = make_algo_fn(k)
        oracle = algo._oracle

        t0 = time.perf_counter()
        done = False
        top = None

        acc_hist = []  # NEW: accuracy per round (aligned with algo.t)

        while (not done) and (algo.t < max_rounds):
            done, top = algo.select_and_update(oracle)

            # compute accuracy every iteration (cheap)
            correct = len(set(top).intersection(set(algo._true_top_m)))
            acc = correct / algo.m
            acc_hist.append(acc)

            if verbose and (algo.t % log_every == 0):
                print(f"[{algo._name} trial {k+1}/{n_trials}] iter={algo.t} done={done} acc={correct}/{algo.m} ({acc:.2f})")

        rt = time.perf_counter() - t0
        runtimes.append(rt)
        total_comparisons.append(algo.total_comparisons)

        G_traces.append(np.array(algo.best_G_history, dtype=float))
        lcb_traces.append(np.array(algo.min_lcb_history, dtype=float))
        acc_traces.append(np.array(acc_hist, dtype=float))   

        if verbose:
            correct = len(set(top).intersection(set(algo._true_top_m)))
            acc = correct / algo.m
            print(f"[{algo._name} trial {k+1}/{n_trials}] finished: iter={algo.t}, done={done}, final acc={correct}/{algo.m} ({acc:.2f}), runtime={rt:.2f}s")

    return (np.array(total_comparisons),
            np.array(runtimes),
            G_traces,
            lcb_traces,
            acc_traces)   






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
    compsA, rtsA, G_A, L_A, Acc_A = resA
    compsB, rtsB, G_B, L_B, Acc_B = resB

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

    fig, axes = plt.subplots(1, 5, figsize=(22, 3.6))

    # (a) comparisons
    means = [np.mean(compsA), np.mean(compsB)]
    stds  = [np.std(compsA, ddof=1), np.std(compsB, ddof=1)]
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

    N_TRIALS = 10
    MAX_ROUNDS = 5000

    # ---------------------------
    # Factories
    # ---------------------------
    # --- NEW: cache env + truth so CASE and P-GIHA share EXACT same env per trial k ---
    ENV_CACHE = {}
    TRUTH_CACHE = {}

    def get_env_and_truth(k):
        if k not in ENV_CACHE:
            env = ReasoningEnvironment(**ENV_CFG, seed=SEED + k)

            true_scores = env.get_ground_truth()
            sorted_truth = sorted(true_scores.items(), key=lambda x: x[1], reverse=True)
            true_top_m = [pid for pid, _ in sorted_truth[:ALG_PGIHA["m"]]]  # m is same in both configs

            ENV_CACHE[k] = env
            TRUTH_CACHE[k] = true_top_m

        return ENV_CACHE[k], TRUTH_CACHE[k]


    def make_algo_pgiha(k):
        env, true_top_m = get_env_and_truth(k)

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

        # attach env + truth for logging
        algo._env = env
        algo._true_top_m = true_top_m
        algo._name = "P-GIHA"

        # --- NEW: oracle with its own RNG (so it doesn't depend on CASE run order) ---
        rng = np.random.default_rng(10_000 + k)

        def oracle_step(step_idx, env=env, rng=rng):
            feat = env.feature_matrix[step_idx]
            clean = float(feat @ env.true_theta)
            return clean + rng.normal(0, env.noise_std)

        algo._oracle = oracle_step
        return algo


    def make_algo_case(k):
        env, true_top_m = get_env_and_truth(k)

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
            challenger_size=ALG_CASE["challenger_size"],
            challenger_batch=ALG_CASE["challenger_batch"],
            seed=ALG_CASE["seed"],
        )

        algo._env = env
        algo._true_top_m = true_top_m
        algo._name = "CASE"

        # --- NEW: oracle with its own RNG (independent of P-GIHA run order) ---
        rng = np.random.default_rng(20_000 + k)

        def oracle_path(path_id, env=env, rng=rng):
            steps = env.paths[path_id]
            g = np.mean(env.feature_matrix[steps], axis=0)
            clean = float(g @ env.true_theta)
            return clean + rng.normal(0, env.noise_std)

        algo._oracle = oracle_path
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

    # ---------------------------
    # Plot comparison (NOTE: new plot_compare expects the 5-output tuples)
    # ---------------------------
    plot_compare(
        res_pgiha,
        res_case,
        nameA="P-GIHA",
        nameB="CASE",
        epsilon=ALG_PGIHA["epsilon"],
        save_path="compare_pgiha_case_with_acc.png",
    )
