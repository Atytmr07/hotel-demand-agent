"""2nd report (docx): project summary + comparison of three AI agents. Run: python -m hotel_agent.build_report"""
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

from hotel_agent import compare_agents as ca

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports"


def table(doc, header, rows):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Light Grid Accent 1"
    for i, h in enumerate(header):
        t.rows[0].cells[i].text = str(h)
        for p in t.rows[0].cells[i].paragraphs:
            for r in p.runs:
                r.bold = True
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
    for row in t.rows:
        for c in row.cells:
            for p in c.paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
    doc.add_paragraph()


def subscores(spec):
    return [[crit] + [spec[crit][t] for t in ca.TOOLS] for crit in spec]


def bullets(doc, items):
    for t in items:
        doc.add_paragraph(t, style="List Bullet")


def figure(doc, name, caption, width=11.5):
    doc.add_picture(str(REP / name), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = doc.add_paragraph(caption)
    c.alignment = WD_ALIGN_PARAGRAPH.CENTER


def main():
    df = ca.build()
    rk = ca.ranks(df)
    wl = ca.workload_cost()
    score_cols = [c for c in df.columns if c.startswith("score[")]
    main_col, proj_col = score_cols[0], score_cols[1]

    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)
    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(2.2)

    doc.add_heading("IE 4023 - 2nd Report", 0)
    doc.add_paragraph("Hotel Demand Forecasting and Capacity/Overbooking Planning Agent, and a Comparison of Three AI Agents")
    doc.add_paragraph("Süeda Yurdakonar (230203006) / Emre Atay Tümer (240203054)")

    # ---------------- 1 ----------------
    doc.add_heading("1. Project summary and status", 1)
    doc.add_paragraph(
        "The goal is a multi-agent system that forecasts hotel occupancy, selects the forecasting model from the data, and uses the forecast "
        "to recommend an overbooking (booking limit) plan. We use the open 'Hotel booking demand' dataset (Antonio, De Almeida and Nunes, 2019; "
        "119,390 bookings, one resort hotel and one city hotel). Code and tests: github.com/Atytmr07/hotel-demand-agent."
    )
    doc.add_paragraph("Completed agents (Python, 12 automated tests):")
    bullets(
        doc,
        [
            "Data agent: turns bookings into a daily occupied-rooms series, removes cancellations and trims unreliable series edges (779 days).",
            "Forecast agent: compares seasonal-naive, ARIMA, SARIMA, Random Forest and XGBoost on a 28-day horizon with 4-fold forward-chaining "
            "validation and selects the model with the lowest MAPE. Model selection is done by code, not by an LLM.",
            "Optimization agent: chooses a daily booking limit with a mixed-integer linear program (MILP, PuLP/CBC). Demand and cancellation "
            "uncertainty are modelled with scenarios, and the solver output was verified against brute-force enumeration of all limits.",
            "Backtest and sensitivity analysis: realized revenue over the last 112 days against three baseline policies.",
        ],
    )

    doc.add_heading("1.1 Results", 2)
    table(
        doc,
        ["Model (MAPE %, 28-day horizon)", "Resort", "City"],
        [
            ["Random Forest", "2.29", "3.56"],
            ["XGBoost", "2.37", "3.32"],
            ["ARIMA(2,1,2)", "3.17", "3.48"],
            ["SARIMA(1,1,1)x(0,1,1,7)", "3.81", "3.99"],
            ["Seasonal naive", "4.68", "3.59"],
        ],
    )
    doc.add_paragraph(
        "No single model wins at both hotels (Resort: Random Forest, City: XGBoost), which motivates selecting the model from the data."
    )
    table(
        doc,
        ["Realized revenue vs. no overbooking", "Resort", "City"],
        [
            ["Fixed 10% overbooking", "+9.8%", "+9.9%"],
            ["Rule: capacity / mean show-up rate", "+37.5%", "+60.4%"],
            ["MILP (walk cost = 1.5 x room price)", "+32.4%", "+57.6%"],
        ],
    )
    table(
        doc,
        ["MILP revenue vs. rule, by walk cost (x room price)", "0.5", "1", "1.5", "3", "5", "10"],
        [
            ["Resort", "+3.3%", "-1.2%", "-3.7%", "-8.2%", "-12.0%", "-14.3%"],
            ["City", "+2.6%", "-0.1%", "-1.7%", "-6.6%", "-9.7%", "-14.4%"],
        ],
    )
    doc.add_paragraph(
        "Finding: the MILP beats the simple 'capacity / show-up rate' rule only when the walk cost is very low (about 0.5 x room price), "
        "breaks even near 1 x, and loses revenue at higher costs. The conclusion is sensitive to this assumption."
    )
    doc.add_heading("1.2 Limitations", 2)
    bullets(
        doc,
        [
            "The dataset has no hotel capacity. Capacity is assumed as the 90th percentile of net occupancy; the walk cost is assumed to be 1.5 x the mean room price; demand is assumed to be observed independently of capacity.",
            "The dataset's cancellation rate is about 40%, so the revenue gains are large and must not be read as absolute values for a real hotel.",
            "Only two hotels and about two years of data are used, so results do not generalize.",
            "A natural-language (LLM) query and reporting layer is not yet connected; at the time of writing the agents run as Python functions.",
        ],
    )

    # ---------------- 2 ----------------
    doc.add_heading("2. Comparison of three AI agents: ChatGPT, Claude and Gemini", 1)
    doc.add_heading("2.1 Goal and method", 2)
    doc.add_paragraph(
        "We compare three widely used AI agents, ChatGPT (OpenAI), Claude (Anthropic) and Gemini (Google), on four dimensions: Capability, "
        "Precision, Cost and New-user friendliness. Each dimension is the mean of 0-5 sub-scores, scaled to 0-10 (for Cost, higher means more expensive). "
        "Capability and Cost are plotted against each other, and Precision against Cost, as requested, and the agents are then ranked."
    )
    p = doc.add_paragraph()
    r = p.add_run(
        "Important: the scores are our rubric-based judgements informed by public pricing pages and indicative public benchmarks. We did not run "
        "controlled experiments on these agents (no API access was used), and 'Precision' means reliability on agentic tasks as reported in public "
        "benchmarks, not answer accuracy measured by us. Sources partly conflict, so every score carries about ±1 point of uncertainty, shown as error bars."
    )
    r.bold = True

    doc.add_heading("2.2 Sub-scores (0-5)", 2)
    hdr = ["Criterion"] + ca.TOOLS
    for label, spec in [
        ("Capability:", ca.CAPABILITY),
        ("Precision (reliability):", ca.PRECISION),
        ("Cost (higher = more expensive):", ca.COST),
        ("New-user friendliness:", ca.NEW_USER),
    ]:
        doc.add_paragraph(label)
        table(doc, hdr, subscores(spec))
    doc.add_paragraph(
        "Evidence behind the Precision scores is indicative only: public sources report Claude models leading on GAIA and tau-bench-style tool-use "
        "evaluations in most cases and GPT models leading on OSWorld in several, with Gemini less often at the top, but vendor-reported and independent "
        "numbers differ and a 2026 study reported that automated agents can exploit several major agent benchmarks. We therefore keep the score spread small."
    )

    doc.add_heading("2.3 Cost evidence (third-party sources, to be checked on official pages)", 2)
    table(
        doc,
        ["", "ChatGPT", "Claude", "Gemini"],
        [
            ["Subscriptions (USD/month)", "Free; Go about 8; Plus 20; Pro 100-500", "Free; Pro 20; Max 100-200", "Free; AI Plus about 5; AI Pro about 20; Ultra 100-200"],
            ["Mid-tier API (USD per 1M tokens, in/out)", "about 2 / 10 (sources vary: up to 5 / 30)", "about 2 / 10", "about 2 / 12 (a promotional 2 / 10 until Nov 30, 2026)"],
            ["Flagship API (in/out)", "about 10 / 50", "about 10 / 50", "not confirmed"],
            ["Cheapest API model (in/out)", "about 0.10 / 0.50", "about 1 / 5", "about 0.25 input (output not confirmed)"],
        ],
    )
    doc.add_paragraph(
        f"Illustrative workload for our reporting agent: {ca.WORKLOAD_Q:,} questions, each with {ca.IN_TOK:,} input and {ca.OUT_TOK} output tokens "
        f"(a forecast and limit table plus a short answer). At the mid-tier prices above this costs about "
        f"USD {wl['ChatGPT']:.1f} (ChatGPT), USD {wl['Claude']:.1f} (Claude) and USD {wl['Gemini']:.1f} (Gemini). "
        "Mid-tier API costs are therefore almost identical, and the cost differences come mainly from price floors, free tiers and top-tier plans."
    )

    doc.add_heading("2.4 Overall scores, charts and ranking", 2)
    table(
        doc,
        ["Agent", "Capability", "Precision", "Cost", "New user"],
        [[t, df.loc[t, "capability"], df.loc[t, "precision"], df.loc[t, "cost"], df.loc[t, "new_user"]] for t in ca.TOOLS],
    )
    figure(doc, "capability_cost.png", "Figure 1. Capability vs Cost (marker size: new-user friendliness; bars: ±1 uncertainty). The top-left is cheap and strong.")
    figure(doc, "precision_cost.png", "Figure 2. Precision vs Cost.")
    doc.add_paragraph(
        "Overall score = w1 x Capability + w2 x Precision + w3 x (10 - Cost) + w4 x New-user. To avoid depending on one weighting, the ranking "
        "is computed under three weight profiles (weights shown as capability, precision, cost, new user):"
    )
    rows = []
    for t in ca.TOOLS:
        rows.append([t] + [f"{df.loc[t, c]} (#{rk.loc[t, c]})" for c in score_cols])
    table(doc, ["Agent"] + [f"{c[6:-1]} {ca.PROFILES[c[6:-1]]}" for c in score_cols], rows)
    figure(doc, "ranking.png", "Figure 3. Overall score under three weighting profiles.", width=12)

    main_order = df[main_col].sort_values(ascending=False)
    proj_order = df[proj_col].sort_values(ascending=False)
    doc.add_heading("2.5 Ranking and interpretation", 2)
    doc.add_paragraph(
        "Main (balanced) ranking: " + ", ".join(f"{i + 1}. {t} ({v:.2f})" for i, (t, v) in enumerate(main_order.items())) + ". "
        "Reliability-first ranking, which matches our project (tool calls must be correct and reproducible): "
        + ", ".join(f"{i + 1}. {t} ({v:.2f})" for i, (t, v) in enumerate(proj_order.items())) + "."
    )
    spread = main_order.max() - main_order.min()
    doc.add_paragraph(
        f"The three agents are close. In the main profile the whole spread is {spread:.2f} points, far below the ±1 uncertainty, and each agent "
        "ranks first in a different profile (ChatGPT in the balanced profile, Claude when reliability dominates, Gemini when cost and ease dominate). "
        "So the ranking should be read as 'no clear overall winner, with different strengths': ChatGPT is the broadest, Claude is the most reliable on "
        "tool use in the sources we found, and Gemini is the cheapest and easiest to start with. The one fairly consistent pattern is that Gemini scores "
        "lower on Precision in every profile that weights it."
    )
    doc.add_paragraph(
        "For this project, the reporting agent only needs reliable function calling on a few of our own tools, and the budget should be near zero. "
        "A free-tier API model is therefore sufficient to start, and the model can be swapped later because the heavy computation (forecasting and the MILP) runs in code."
    )
    doc.add_heading("2.6 Limitations", 2)
    bullets(
        doc,
        [
            "Scores are subjective rubric judgements; no independent second rater scored the sub-criteria.",
            "Prices and model names come from third-party sites that conflict with each other and change quickly; they were not verified on official pages.",
            "We did not run the three agents on the same task to measure accuracy, latency or token cost; this is the natural next step.",
        ],
    )

    doc.add_heading("3. Conclusion and next steps", 1)
    doc.add_paragraph(
        "The project delivers a working, tested pipeline from data to forecast to optimization. Its main finding is that the MILP's advantage over a simple "
        "rule depends on the assumed walk cost. In the agent comparison the three agents are close; ChatGPT ranks first in the balanced profile, "
        "Claude in the reliability-first profile and Gemini in the budget profile. Next steps: connect an LLM reporting agent, run the three agents on the same "
        "task set to measure accuracy and cost, and source a defensible walk cost."
    )

    doc.add_heading("References", 1)
    bullets(
        doc,
        [
            "Antonio, N., De Almeida, A., Nunes, L. (2019). Hotel booking demand datasets. Data in Brief, 22, 41-49. https://doi.org/10.1016/j.dib.2018.11.126",
            "ChatGPT plans and API pricing (third-party): https://aitoolsreview.co.uk/insights/chatgpt-pricing ; https://www.morphllm.com/openai-api-pricing",
            "Claude plans and API pricing (third-party): https://www.lindy.ai/blog/claude-pricing ; https://screenapp.io/blog/claude-ai-pricing",
            "Gemini plans and API pricing (third-party): https://www.eesel.ai/blog/gemini-pricing ; https://appscribed.com/gemini-pricing/",
            "Agent benchmark summaries (third-party, indicative): https://visionary.life/claude-vs-gpt-vs-gemini-for-agentic-work-2026-benchmark/ ; https://rapidclaw.dev/blog/ai-agent-benchmarks-2026",
            "Yao et al. (2024). tau-bench: A benchmark for tool-agent-user interaction in real-world domains. https://arxiv.org/abs/2406.12045",
            "Project code: https://github.com/Atytmr07/hotel-demand-agent",
        ],
    )
    out_path = REP / "second_report.docx"
    doc.save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()
