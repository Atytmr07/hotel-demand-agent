"""Rubric-based comparison of three AI agents: ChatGPT (OpenAI), Claude (Anthropic), Gemini (Google).

Scores are the authors' rubric judgements (0-5 sub-scores, scaled to 0-10), informed by public pricing pages
and indicative public benchmarks. They are NOT measured by us. Run: python -m hotel_agent.compare_agents
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

OUT = Path(__file__).resolve().parents[2] / "reports"
TOOLS = ["ChatGPT", "Claude", "Gemini"]
COLORS = {"ChatGPT": "#10a37f", "Claude": "#d97757", "Gemini": "#4285f4"}
UNCERTAINTY = 1.0  # +/- points on the 0-10 scale; sources conflict and scores are judgements

CAPABILITY = {
    "Agentic tool use / computer use": {"ChatGPT": 4.5, "Claude": 4.5, "Gemini": 3.5},
    "Coding and data analysis (code execution)": {"ChatGPT": 4.5, "Claude": 5, "Gemini": 4},
    "Multimodal input and long context": {"ChatGPT": 4, "Claude": 3.5, "Gemini": 5},
    "Web research / deep research": {"ChatGPT": 4.5, "Claude": 4, "Gemini": 4.5},
    "Ecosystem and integrations (connectors, MCP, Workspace)": {"ChatGPT": 4.5, "Claude": 4.5, "Gemini": 4.5},
}
PRECISION = {  # reliability on agentic tasks (not answer accuracy measured by us)
    "Tool-use policy adherence (tau-bench style, indicative)": {"ChatGPT": 4, "Claude": 4.5, "Gemini": 3.5},
    "Multi-step task success (GAIA style, indicative)": {"ChatGPT": 4, "Claude": 4.5, "Gemini": 3.5},
    "Structured output / function calling support": {"ChatGPT": 4.5, "Claude": 4.5, "Gemini": 4.5},
}
COST = {  # higher = more expensive
    "Cheapest paid subscription": {"ChatGPT": 2.5, "Claude": 2.5, "Gemini": 2.0},
    "Mid-tier API price per token": {"ChatGPT": 2.0, "Claude": 2.0, "Gemini": 2.2},
    "Cheapest API model price floor": {"ChatGPT": 1.0, "Claude": 2.5, "Gemini": 1.5},
    "Top subscription tier price": {"ChatGPT": 4.0, "Claude": 3.0, "Gemini": 3.0},
}
NEW_USER = {
    "Onboarding ease": {"ChatGPT": 5, "Claude": 4, "Gemini": 4.5},
    "Access to agent features on low-cost plans": {"ChatGPT": 3.5, "Claude": 3.5, "Gemini": 4},
}
PROFILES = {  # weights: capability, precision, (10 - cost), new user
    "balanced (main)": (0.30, 0.25, 0.25, 0.20),
    "reliability first (our project)": (0.25, 0.45, 0.15, 0.15),
    "budget and ease first": (0.15, 0.15, 0.35, 0.35),
}

# Illustrative workload cost: 1,000 report/chat queries, 3,000 input + 800 output tokens each.
# Mid-tier API prices (USD per 1M tokens, input/output), third-party sources that partly conflict.
WORKLOAD_Q, IN_TOK, OUT_TOK = 1000, 3000, 800
API_PRICES = {"ChatGPT": (2.0, 10.0), "Claude": (2.0, 10.0), "Gemini": (2.0, 12.0)}


def workload_cost() -> pd.Series:
    return pd.Series(
        {t: WORKLOAD_Q * (IN_TOK * i + OUT_TOK * o) / 1e6 for t, (i, o) in API_PRICES.items()}, name="usd_per_1000_queries"
    )


def scale10(table: dict) -> pd.Series:
    return pd.DataFrame(table).loc[TOOLS].mean(axis=1) * 2


def build() -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "capability": scale10(CAPABILITY),
            "precision": scale10(PRECISION),
            "cost": scale10(COST),
            "new_user": scale10(NEW_USER),
        }
    )
    for name, (wc, wp, wk, wn) in PROFILES.items():
        df[f"score[{name}]"] = wc * df.capability + wp * df.precision + wk * (10 - df.cost) + wn * df.new_user
    return df.round(2)


def ranks(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith("score[")]
    return df[cols].rank(ascending=False, method="min").astype(int)


def _scatter(df, y, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(6.4, 5))
    offsets = {"ChatGPT": (0, 22), "Claude": (24, -30), "Gemini": (-58, -30)}  # keeps labels apart when points cluster
    for t in TOOLS:
        ax.errorbar(
            df.loc[t, "cost"], df.loc[t, y], xerr=UNCERTAINTY, yerr=UNCERTAINTY, fmt="none", ecolor=COLORS[t], alpha=0.5, capsize=3, zorder=2
        )
        ax.scatter(df.loc[t, "cost"], df.loc[t, y], s=60 + 20 * df.loc[t, "new_user"], color=COLORS[t], edgecolor="k", alpha=0.95, zorder=3)
        ax.annotate(t, (df.loc[t, "cost"], df.loc[t, y]), textcoords="offset points", xytext=offsets[t], ha="center", fontsize=11, color=COLORS[t], fontweight="bold")
    ax.axhline(7.5, color="grey", lw=0.8, ls="--")
    ax.axvline(5, color="grey", lw=0.8, ls="--")
    ax.set_xlim(2, 8)
    ax.set_ylim(5, 10)
    ax.set_xlabel("Cost (0 = cheap, 10 = expensive)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.text(2.1, 9.7, "cheap and strong", fontsize=9, color="grey")
    ax.text(0.99, 0.01, "axes zoomed (full scale 0-10); marker size = new-user friendliness; bars = ±1", transform=ax.transAxes, ha="right", fontsize=7.5, color="grey")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plots(df: pd.DataFrame):
    _scatter(df, "capability", "Capability (0-10)", "Capability vs Cost", OUT / "capability_cost.png")
    _scatter(df, "precision", "Precision / reliability (0-10)", "Precision vs Cost", OUT / "precision_cost.png")
    cols = [c for c in df.columns if c.startswith("score[")]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    w = 0.26
    for i, t in enumerate(TOOLS):
        ax.bar([j + (i - 1) * w for j in range(len(cols))], df.loc[t, cols], w, label=t, color=COLORS[t], edgecolor="k")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[6:-1] for c in cols], fontsize=9)
    ax.set_ylabel("Overall score (0-10)")
    ax.set_ylim(0, 10)
    ax.set_title("Ranking under three weighting profiles")
    ax.legend(loc="upper center", ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "ranking.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    df = build()
    df.to_csv(OUT / "agent_comparison.csv", encoding="utf-8-sig")
    plots(df)
    print(df.to_string())
    print()
    print(ranks(df).to_string())
    print()
    print(workload_cost().to_string())
