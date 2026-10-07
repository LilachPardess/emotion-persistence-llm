"""
Summary plot for the steered decay experiment (08_steered_decay_experiment.py).

Effect: for every neutral set, seed, emotion E and turn t,
    diff = cosine(steered with E, vs E) - cosine(unsteered, vs E)
using the steered and unsteered runs with the same sentences and the same seed.
Each (neutral set, seed) pair is one replicate; error bars are 95% CIs
(1.96 x SEM) across replicates.

Noise band: the same kind of difference between two UNSTEERED runs with
different seeds, i.e. what pure sampling luck produces when steering does
nothing. The gray band is +-1.96 x (std of those differences) / sqrt(n
replicates): if steering had no effect at turn t, the mean effect would fall
inside the band 95% of the time. Needs at least 2 seeds.

Outputs:
  - steered_decay_summary.csv
  - steered_decay_summary_plot.png      (headline: averaged over emotions within each
                                         replicate, then mean +- 95% CI, with the noise band)
  - steered_decay_per_emotion_plot.png  (one small panel per emotion, same y axis)

Usage:
    python 09_plot_decay_summary.py
Requires the steered_neutral_sentences_<neutral_set>.csv files written by 08.
"""
import glob
from itertools import combinations

import matplotlib.pyplot as plt
import pandas as pd

CSV_GLOB = "steered_neutral_sentences_*.csv"
SUMMARY_CSV = "steered_decay_summary.csv"
PLOT_PATH = "steered_decay_summary_plot.png"
PER_EMOTION_PLOT_PATH = "steered_decay_per_emotion_plot.png"
Y_LABEL = "extra emotion signal\ncosine(steered) - cosine(unsteered)"
Z95 = 1.96

COLORS = {
    "joy": "#f4a259", "admiration": "#8cb369", "optimism": "#5b8e7d",
    "sadness": "#4059ad", "anger": "#d1495b", "fear": "#6b2737",
}


def load(csv_glob):
    paths = sorted(glob.glob(csv_glob))
    if not paths:
        raise FileNotFoundError(f"No files match {csv_glob}; run 08_steered_decay_experiment.py first.")
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    if "seed" not in df.columns:
        df["seed"] = 0
    return df, len(paths)


def effect_diffs(df):
    keys = ["neutral_set", "seed", "measured_against", "t"]
    steered = df[df["condition"] == df["measured_against"]][keys + ["cosine_sim"]]
    unsteered = df[df["condition"] == "unsteered"][keys + ["cosine_sim"]]
    merged = steered.merge(unsteered, on=keys, suffixes=("_steered", "_unsteered"))
    merged["diff"] = merged["cosine_sim_steered"] - merged["cosine_sim_unsteered"]
    return merged[keys + ["diff"]]


def null_diffs(df):
    """unsteered(seed a) - unsteered(seed b) for every pair of seeds a < b."""
    unsteered = df[df["condition"] == "unsteered"]
    wide = unsteered.pivot_table(index=["neutral_set", "measured_against", "t"],
                                 columns="seed", values="cosine_sim")
    parts = []
    for a, b in combinations(wide.columns, 2):
        part = (wide[a] - wide[b]).rename("diff").reset_index()
        part["pair"] = f"{a}-{b}"
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else None


def null_overall(df):
    """Null for the emotion-averaged effect, one value per (neutral set, seed a, t).

    The emotion vectors sum to ~0, so averaging ONE conversation's cosines over
    emotions cancels its noise. The real effect averages a different steered
    conversation per emotion, so the null does too: emotion k is read from
    unsteered seed b_k != a, minus the mean of unsteered seed a.
    """
    unsteered = df[df["condition"] == "unsteered"]
    wide = unsteered.pivot_table(index=["neutral_set", "t", "seed"],
                                 columns="measured_against", values="cosine_sim")
    seeds = sorted(unsteered["seed"].unique())
    if len(seeds) < 2:
        return None
    emotions = list(wide.columns)
    rows = []
    for (neutral_set, t), block in wide.groupby(level=["neutral_set", "t"]):
        block = block.droplevel(["neutral_set", "t"])
        for i, a in enumerate(seeds):
            others = [s for s in seeds if s != a]
            null_mean = sum(block.loc[others[k % len(others)], e] for k, e in enumerate(emotions)) / len(emotions)
            rows.append({"neutral_set": neutral_set, "seed": a, "t": t,
                         "diff": null_mean - block.loc[a].mean()})
    return pd.DataFrame(rows)


def mean_ci(frame, by):
    out = frame.groupby(by)["diff"].agg(["mean", "std", "count"]).reset_index()
    out["ci95"] = Z95 * out["std"] / out["count"] ** 0.5
    return out


def main():
    df, n_sets = load(CSV_GLOB)
    n_seeds = df["seed"].nunique()
    effects = effect_diffs(df)
    nulls = null_diffs(df)

    per_emotion = mean_ci(effects, ["measured_against", "t"])
    per_rep = effects.groupby(["neutral_set", "seed", "t"])["diff"].mean().reset_index()
    overall = mean_ci(per_rep, ["t"])

    if nulls is not None:
        null_sd_emotion = nulls.groupby(["measured_against", "t"])["diff"].std().rename("null_sd")
        per_emotion = per_emotion.merge(null_sd_emotion.reset_index(), on=["measured_against", "t"])
        null_rep = null_overall(df)
        overall = overall.merge(null_rep.groupby("t")["diff"].std().rename("null_sd").reset_index(), on="t")
        for table in (per_emotion, overall):
            table["noise_band95"] = Z95 * table["null_sd"] / table["count"] ** 0.5
            table["beyond_noise"] = table["mean"].abs() > table["noise_band95"]
    else:
        print("Only one seed found: no noise band. Run 08 with several SEEDS to get one.")

    cols = ["emotion", "t", "mean", "ci95", "count"] + (["noise_band95", "beyond_noise"] if nulls is not None else [])
    summary = pd.concat([
        per_emotion.rename(columns={"measured_against": "emotion"}),
        overall.assign(emotion="all"),
    ], ignore_index=True)[cols]
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Saved {SUMMARY_CSV}  ({n_sets} neutral sets x {n_seeds} seeds = {n_sets * n_seeds} replicates)")
    print(summary[summary["emotion"] == "all"].drop(columns="emotion").round(4).to_string(index=False))

    replicates = f"{n_sets} sentence sets x {n_seeds} seed{'s' if n_seeds > 1 else ''}"
    plot_headline(overall.sort_values("t"), nulls is not None, replicates)
    plot_per_emotion(per_emotion, nulls is not None, replicates)


def draw_curve(ax, sub, color, has_noise, label=None):
    ts = sub["t"]
    if has_noise:
        ax.fill_between(ts, -sub["noise_band95"], sub["noise_band95"],
                        color="gray", alpha=0.2, linewidth=0, label="sampling noise (95%)")
    ax.fill_between(ts, sub["mean"] - sub["ci95"], sub["mean"] + sub["ci95"],
                    color=color, alpha=0.25, linewidth=0)
    ax.plot(ts, sub["mean"], marker="o", color=color, linewidth=2, label=label)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(list(ts))
    ax.grid(axis="y", alpha=0.3)


def plot_headline(overall, has_noise, replicates):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    draw_curve(ax, overall, "#1f4e79", has_noise, label="mean over 6 emotions (95% CI)")
    t0 = overall.iloc[0]
    ax.annotate("steered reply", xy=(t0["t"], t0["mean"]), xytext=(0.6, t0["mean"]),
                fontsize=9, va="center", arrowprops=dict(arrowstyle="->", color="gray"))
    ax.set_xlabel("turn (t): reply to neutral sentence t+1  (steering only at t=0)")
    ax.set_ylabel(Y_LABEL)
    ax.set_title(f"Does a steered emotion persist into later replies?\n({replicates})", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved {PLOT_PATH}")


def plot_per_emotion(per_emotion, has_noise, replicates):
    emotions = [e for e in COLORS if e in set(per_emotion["measured_against"])]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, sharey=True)
    for ax, emotion in zip(axes.ravel(), emotions):
        sub = per_emotion[per_emotion["measured_against"] == emotion].sort_values("t")
        draw_curve(ax, sub, COLORS[emotion], has_noise)
        ax.set_title(emotion)
    for ax in axes[1]:
        ax.set_xlabel("turn (t)")
    for ax in axes[:, 0]:
        ax.set_ylabel(Y_LABEL, fontsize=8)
    fig.suptitle(f"Per emotion: steered minus unsteered, mean and 95% CI ({replicates}); steering only at t=0")
    fig.tight_layout()
    fig.savefig(PER_EMOTION_PLOT_PATH, dpi=150)
    plt.close(fig)
    print(f"Saved {PER_EMOTION_PLOT_PATH}")


if __name__ == "__main__":
    main()
