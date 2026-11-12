"""Rubric-based comparison of eight AI agents in three classes.

Scores are the authors' rubric judgements (0-5 sub-scores scaled to 0-10), informed by public pricing pages and
indicative public benchmarks. They are NOT measured by us. Ranking uncertainty is quantified by Monte Carlo
perturbation of the scores. Run: python -m hotel_agent.compare_agents
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "reports"

CLASSES = {
    "Frontier assistants": ["ChatGPT", "Claude", "Gemini"],
    "Autonomous task agents": ["Manus", "Devin"],
    "Workflow agent platforms": ["n8n", "Zapier", "Copilot Studio"],
}
TOOLS = [t for ts in CLASSES.values() for t in ts]
CLASS_OF = {t: c for c, ts in CLASSES.items() for t in ts}
MARKERS = {"Frontier assistants": "o", "Autonomous task agents": "s", "Workflow agent platforms": "^"}
SHADES = {"Frontier assistants": "#0072B2", "Autonomous task agents": "#D55E00", "Workflow agent platforms": "#009E73"}  # Okabe-Ito
NAVY = "#1F3864"
PROFILE_COLORS = ["#0072B2", "#E69F00", "#009E73"]
UNCERTAINTY = 1.0  # +/- points on the 0-10 scale; sources conflict and scores are judgements


def _row(*v):
    return dict(zip(TOOLS, v))


#                                    ChatGPT Claude Gemini Manus Devin n8n  Zapier Copilot
CAPABILITY = {
    "Autonomy on multi-step tasks": _row(4, 4.5, 3.5, 4.5, 4, 3, 3, 3),
    "Breadth of task types": _row(4.5, 4.5, 4.5, 4, 2, 4.5, 4, 3.5),
    "Tool and integration breadth": _row(4, 4, 4.5, 3, 3, 5, 5, 4),
    "Reasoning, coding and data analysis": _row(4.5, 5, 4.5, 4, 4.5, 3, 2.5, 3),
    "Customizability and extensibility": _row(3.5, 4, 3.5, 2.5, 2.5, 4.5, 3, 3.5),
}
PRECISION = {  # reliability and controllability, not answer accuracy measured by us
    "Control and reproducibility of the flow": _row(3, 3.5, 3, 2, 2.5, 4.5, 4, 3.5),
    "Tool-use reliability (indicative evidence)": _row(4, 4.5, 3.5, 3, 3.5, 3.5, 3.5, 3),
    "Transparency and observability": _row(3, 3.5, 3, 3, 3.5, 4.5, 4, 3.5),
}
COST = {  # higher = more expensive
    "Entry price": _row(2.5, 2.5, 2.0, 3, 2.5, 0.5, 1.5, 4),
    "Price unpredictability (usage or credit billing)": _row(1.5, 1.5, 1.5, 3.5, 4, 2, 3, 3.5),
    "Cost at scale / top tier": _row(3.5, 3, 3, 3.5, 4, 2.5, 3.5, 4),
}
NEW_USER = {
    "Onboarding ease": _row(5, 4, 4.5, 4.5, 3, 3.5, 4.5, 3),
    "No-code access": _row(4.5, 4, 4.5, 4.5, 3.5, 4, 5, 4),
}
DIMS = {"capability": CAPABILITY, "precision": PRECISION, "cost": COST, "new_user": NEW_USER}
PROFILES = {  # weights: capability, precision, (10 - cost), new user
    "Balanced": (0.30, 0.25, 0.25, 0.20),
    "Reliability first": (0.25, 0.45, 0.15, 0.15),
    "Budget and ease first": (0.15, 0.15, 0.35, 0.35),
}
MAIN = "Balanced"

# Illustrative workload for the LLM assistants: 1,000 questions x (3,000 input + 800 output tokens).
# Mid-tier API prices (USD per 1M tokens, in/out), third-party sources that partly conflict.
WORKLOAD_Q, IN_TOK, OUT_TOK = 1000, 3000, 800
API_PRICES = {"ChatGPT": (2.0, 10.0), "Claude": (2.0, 10.0), "Gemini": (2.0, 12.0)}


def workload_cost() -> pd.Series:
    return pd.Series({t: WORKLOAD_Q * (IN_TOK * i + OUT_TOK * o) / 1e6 for t, (i, o) in API_PRICES.items()})


def scale10(table: dict) -> pd.Series:
    return pd.DataFrame(table).loc[TOOLS].mean(axis=1) * 2


def dims() -> pd.DataFrame:
    return pd.DataFrame({k: scale10(v) for k, v in DIMS.items()})


def overall(d: pd.DataFrame, w) -> pd.Series:
    wc, wp, wk, wn = w
    return wc * d.capability + wp * d.precision + wk * (10 - d.cost) + wn * d.new_user


def build() -> pd.DataFrame:
    d = dims()
    for name, w in PROFILES.items():
        d[f"score[{name}]"] = overall(d, w)
    return d.round(2)


def ranks(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith("score[")]
    return df[cols].rank(ascending=False, method="min").astype(int)


def monte_carlo(profile=MAIN, n=10000, seed=0) -> pd.DataFrame:
    """Rank distribution when every dimension score is perturbed by U(-UNCERTAINTY, +UNCERTAINTY)."""
    rng = np.random.default_rng(seed)
    d = dims().to_numpy()  # tools x 4
    w = np.array(PROFILES[profile])
    sign = np.array([1, 1, -1, 1])  # cost enters as (10 - cost)
    out = np.zeros((n, len(TOOLS)), dtype=int)
    for i in range(n):
        x = np.clip(d + rng.uniform(-UNCERTAINTY, UNCERTAINTY, d.shape), 0, 10)
        s = (x * sign * w).sum(axis=1) + w[2] * 10
        out[i] = (-s).argsort().argsort() + 1
    return pd.DataFrame(out, columns=TOOLS)


def rank_summary(mc: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "median": mc.median(),
            "p05": mc.quantile(0.05),
            "p95": mc.quantile(0.95),
            "P(rank 1)": (mc == 1).mean(),
            "P(top 3)": (mc <= 3).mean(),
        }
    ).sort_values("median")


# ---------------------------------------------------------------- figures (article style, black text)
def _style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "text.color": "black",
            "axes.labelcolor": "black",
            "axes.edgecolor": "black",
            "xtick.color": "black",
            "ytick.color": "black",
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "font.size": 10,
            "axes.titlesize": 11,
        }
    )


# label offsets (points) and alignment tuned so the clustered points stay readable
OFFS = {
    ("capability", "ChatGPT"): (8, 5, "left"), ("capability", "Gemini"): (-8, 5, "right"), ("capability", "Claude"): (0, 10, "center"),
    ("capability", "n8n"): (0, -16, "center"), ("capability", "Zapier"): (-8, -4, "right"), ("capability", "Manus"): (8, 6, "left"),
    ("capability", "Copilot Studio"): (0, 10, "center"), ("capability", "Devin"): (-8, -12, "right"),
    ("precision", "ChatGPT"): (8, 5, "left"), ("precision", "Gemini"): (-8, -12, "right"), ("precision", "Claude"): (-8, 6, "right"),
    ("precision", "n8n"): (8, 5, "left"), ("precision", "Zapier"): (8, 6, "left"), ("precision", "Manus"): (8, -12, "left"),
    ("precision", "Copilot Studio"): (0, 10, "center"), ("precision", "Devin"): (-8, -12, "right"),
}


def _scatter(d, y, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    for cls, ts in CLASSES.items():
        for t in ts:
            ax.errorbar(d.loc[t, "cost"], d.loc[t, y], xerr=UNCERTAINTY, yerr=UNCERTAINTY, fmt="none", ecolor=SHADES[cls], alpha=0.35, lw=1.1, capsize=2, zorder=1)
    ax.axvspan(1, 5, ymin=(7.5 - 3.5) / 6.5, ymax=1, color="#E3F4EA", zorder=0)  # cheap-and-strong region
    for cls, ts in CLASSES.items():
        xs = [d.loc[t, "cost"] for t in ts]
        ys = [d.loc[t, y] for t in ts]
        ax.scatter(xs, ys, marker=MARKERS[cls], s=95, color=SHADES[cls], edgecolor="white", linewidth=1.2, zorder=3, label=cls)
        for t in ts:
            dx, dy, ha = OFFS[(y, t)]
            ax.annotate(t, (d.loc[t, "cost"], d.loc[t, y]), textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=9.5, color=SHADES[cls], fontweight="bold")
    ax.axvline(5, color=NAVY, lw=0.7, ls=":")
    ax.axhline(7.5, color=NAVY, lw=0.7, ls=":")
    ax.set_xlim(1, 9.5)
    ax.set_ylim(3.5, 10)
    ax.set_xlabel("Cost  (0 = cheap, 10 = expensive)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, color=NAVY, fontweight="bold")
    ax.text(1.15, 9.65, "cheap and strong", fontsize=8.5, style="italic", color="#2E7D4F")
    ax.legend(loc="lower left", fontsize=8.5, frameon=True, edgecolor="#999999")
    ax.text(0.99, 0.01, "axes zoomed; faint bars: ±1 uncertainty", transform=ax.transAxes, ha="right", fontsize=7.5, color="#555555")
    ax.grid(alpha=0.3, color="#9bb0c8", lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def fig_heatmap(path):
    rows, names = [], []
    for dim, spec in DIMS.items():
        for crit in spec:
            rows.append([spec[crit][t] for t in TOOLS])
            names.append(f"[{dim[:3].upper()}] {crit}")
    m = np.array(rows)
    fig, ax = plt.subplots(figsize=(7.6, 6.4))
    im = ax.imshow(m, cmap="YlGnBu", vmin=0, vmax=6.0, aspect="auto")
    ax.set_xticks(range(len(TOOLS)))
    ax.set_xticklabels(TOOLS, rotation=35, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8.5)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:g}", ha="center", va="center", fontsize=8.5, color="white" if m[i, j] > 3.6 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def fig_ranking(df, mc, path):
    summ = rank_summary(mc)
    order = df["score[Balanced]"].sort_values().index.tolist()  # bottom -> top
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    for i, t in enumerate(order):
        cls = CLASS_OF[t]
        ax.barh(i, df.loc[t, "score[Balanced]"], color=SHADES[cls], edgecolor="white", height=0.62)
        ax.text(df.loc[t, "score[Balanced]"] + 0.05, i, f"{df.loc[t, 'score[Balanced]']:.2f}   (rank {int(summ.loc[t, 'p05'])}-{int(summ.loc[t, 'p95'])})", va="center", fontsize=8.5)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_xlim(4.5, 9.2)
    ax.set_xlabel("Overall score, balanced profile (0-10); in brackets: 90% rank interval under ±1 score noise")
    ax.xaxis.label.set_fontsize(8.5)
    handles = [plt.Rectangle((0, 0), 1, 1, color=SHADES[c]) for c in CLASSES]
    ax.legend(handles, list(CLASSES), loc="lower right", fontsize=8.5, edgecolor="#999999")
    ax.grid(axis="x", alpha=0.3, color="#9bb0c8", lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def fig_rank_prob(mc, path):
    summ = rank_summary(mc)
    order = summ.index.tolist()
    k = len(TOOLS)
    m = np.array([[(mc[t] == r).mean() for r in range(1, k + 1)] for t in order])
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    ax.imshow(m, cmap="Blues", vmin=0, vmax=max(0.6, m.max()), aspect="auto")
    ax.set_xticks(range(k))
    ax.set_xticklabels(range(1, k + 1))
    ax.set_yticks(range(k))
    ax.set_yticklabels(order)
    ax.set_xlabel("Rank (1 = best)")
    for i in range(k):
        for j in range(k):
            if m[i, j] >= 0.03:
                ax.text(j, i, f"{100 * m[i, j]:.0f}", ha="center", va="center", fontsize=8.5, color="white" if m[i, j] > 0.3 else "black")
    ax.set_title("Probability (%) of each rank under ±1 score noise, balanced profile")
    ax.title.set_fontsize(9.5)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def fig_profiles(df, path):
    cols = [c for c in df.columns if c.startswith("score[")]
    order = df["score[Balanced]"].sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    w = 0.26
    for i, c in enumerate(cols):
        ax.bar([j + (i - 1) * w for j in range(len(order))], df.loc[order, c], w, label=c[6:-1], color=PROFILE_COLORS[i], edgecolor="white")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=25, ha="right")
    ax.set_ylabel("Overall score (0-10)")
    ax.set_ylim(3, 9.2)
    ax.legend(ncol=3, loc="upper right", fontsize=8.5, edgecolor="#999999")
    ax.grid(axis="y", alpha=0.3, color="#9bb0c8", lw=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plots(df, mc):
    _style()
    d = dims()
    _scatter(d, "capability", "Capability  (0-10)", "Capability versus cost", OUT / "capability_cost.png")
    _scatter(d, "precision", "Precision / reliability  (0-10)", "Precision versus cost", OUT / "precision_cost.png")
    fig_heatmap(OUT / "subscores_heatmap.png")
    fig_ranking(df, mc, OUT / "ranking.png")
    fig_rank_prob(mc, OUT / "rank_probability.png")
    fig_profiles(df, OUT / "profiles.png")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    df = build()
    mc = monte_carlo()
    df.to_csv(OUT / "agent_comparison.csv", encoding="utf-8-sig")
    plots(df, mc)
    print(df.to_string())
    print()
    print(ranks(df).to_string())
    print()
    print(rank_summary(mc).round(2).to_string())
    print()
    print(workload_cost().to_string())
