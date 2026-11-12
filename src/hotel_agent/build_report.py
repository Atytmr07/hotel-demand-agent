"""2nd report as an article-style docx. Run: python -m hotel_agent.build_report"""
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

from hotel_agent import compare_agents as ca

ROOT = Path(__file__).resolve().parents[2]
REP = ROOT / "reports"
FONT = "Times New Roman"
BLACK = RGBColor(0, 0, 0)


def _font(run, size=None, bold=None, italic=None):
    run.font.name = FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    run.font.color.rgb = BLACK
    if size:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if italic is not None:
        run.italic = italic


def para(doc, text, size=11, bold=False, italic=False, align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=6, style=None):
    p = doc.add_paragraph(style=style)
    p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.08
    _font(p.add_run(text), size, bold, italic)
    return p


def head(doc, text, level=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12 if level == 1 else 8)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    _font(p.add_run(text), 13 if level == 1 else 11.5, bold=True, italic=(level == 2))
    return p


def bullets(doc, items):
    for t in items:
        p = doc.add_paragraph(style="List Bullet")
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        _font(p.add_run(t), 11)


def table(doc, header, rows, caption=None, size=9):
    if caption:
        c = para(doc, caption, size=10, bold=True, align=WD_ALIGN_PARAGRAPH.LEFT, space_after=3)
        c.paragraph_format.keep_with_next = True
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    for i, h in enumerate(header):
        cell = t.rows[0].cells[i]
        cell.text = ""
        _font(cell.paragraphs[0].add_run(str(h)), size, bold=True)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            _font(cells[i].paragraphs[0].add_run(str(v)), size)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def figure(doc, name, caption, width=14.5):
    doc.add_picture(str(REP / name), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.paragraphs[-1].paragraph_format.keep_with_next = True
    para(doc, caption, size=10, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10)


def main():
    df = ca.build()
    rk = ca.ranks(df)
    mc = ca.monte_carlo()
    summ = ca.rank_summary(mc)
    wl = ca.workload_cost()
    score_cols = [c for c in df.columns if c.startswith("score[")]
    main_s = df[f"score[{ca.MAIN}]"].sort_values(ascending=False)
    top5, bottom3 = main_s.index[:5].tolist(), main_s.index[5:].tolist()
    p_split = float((mc[bottom3] >= 6).all(axis=1).mean())
    gap = main_s.iloc[4] - main_s.iloc[5]

    doc = Document()
    for s in doc.sections:
        s.left_margin = s.right_margin = Cm(2.3)
        s.top_margin = s.bottom_margin = Cm(2.2)

    t = para(doc, "Comparing Eight AI Agents on Capability, Precision and Cost, with a Hotel Overbooking Case Study",
             size=17, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
    para(doc, "Süeda Yurdakonar (230203006) and Emre Atay Tümer (240203054)", size=11, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=0)
    para(doc, "IE 4023, 2nd Report, November 2026", size=10, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=10)

    # ---------------- abstract ----------------
    head(doc, "Abstract", 1)
    para(
        doc,
        "Many AI agent products now compete for the same jobs, yet they differ in kind: general assistants, autonomous task agents and workflow "
        "platforms. We compare eight of them (ChatGPT, Claude, Gemini, Manus, Devin, n8n, Zapier and Microsoft Copilot Studio) on four dimensions: "
        "capability, precision, cost and ease for new users. Scores come from an explicit rubric of 13 sub-criteria that we filled in from public "
        "pricing pages and indicative benchmarks, not from our own experiments. To show how fragile any ranking of this kind is, we perturb every "
        f"score by ±{ca.UNCERTAINTY:g} point in 10,000 Monte Carlo draws. The result has two tiers: {', '.join(top5)} cannot be separated "
        f"reliably (overall scores {main_s.iloc[4]:.2f} to {main_s.iloc[0]:.2f}), while {', '.join(bottom3)} stay behind them in "
        f"{100 * p_split:.0f}% of draws. We also report a small case study, a tested pipeline that forecasts hotel occupancy and chooses overbooking "
        "limits with a mixed-integer program, and show that its main conclusion depends on one assumed cost.",
    )
    p = para(doc, "", space_after=8)
    _font(p.add_run("Keywords: "), 10.5, bold=True)
    _font(p.add_run("AI agents, agent comparison, capability, cost, ranking uncertainty, demand forecasting, overbooking, MILP"), 10.5, italic=True)

    # ---------------- 1 introduction ----------------
    head(doc, "1. Introduction", 1)
    para(
        doc,
        "Choosing an AI agent is hard because the products are not alike. ChatGPT, Claude and Gemini are assistants that can also act; Manus and Devin "
        "are sold as agents that work on a task for a long time with little supervision; n8n, Zapier and Copilot Studio let people wire models into "
        "business workflows. Reviews usually give each a score out of ten and a rank, as if the numbers were measurements. They rarely say how "
        "sure they are."
    )
    para(
        doc,
        "This report asks three questions. First, how do eight well-known agents compare on capability, precision and cost? Second, does their "
        "ranking survive plausible changes in the scores and in the weights? Third, how do these questions look in a small working project of our own? "
        "For the third we built a hotel demand forecasting and overbooking agent. Section 2 describes the eight agents, Section 3 the method, Section 4 "
        "the results, Section 5 the case study, and Sections 6 to 8 the discussion, limitations and conclusion."
    )

    # ---------------- 2 background ----------------
    head(doc, "2. The agents and their classes", 1)
    para(
        doc,
        "We chose agents that are widely known and that cover three classes. The descriptions and prices below come from third-party reviews and "
        "price trackers, which often disagree with one another, so they should be read as approximate."
    )
    table(
        doc,
        ["Class", "Agent", "What it is", "Reported price (approximate)"],
        [
            ["Frontier assistants", "ChatGPT (OpenAI)", "General assistant with an agent mode; reported as included in the Plus plan [2]", "Free; Go about 8; Plus 20; Pro 100-500 USD/month"],
            ["", "Claude (Anthropic)", "General assistant; Claude Code is widely rated a strong coding agent [5]", "Free; Pro 20; Max 100-200 USD/month [3]"],
            ["", "Gemini (Google)", "General assistant integrated with Google services", "Free; AI Plus about 5; Pro about 20; Ultra 100-200 USD/month [4]"],
            ["Autonomous task agents", "Manus", "General agent for research and finished deliverables, credit-based", "About 20-39 USD/month, credits; higher tiers up to 200 [5]"],
            ["", "Devin (Cognition)", "Agent for autonomous software engineering", "Core 20 USD plus usage units (about 2.25 USD each); Team about 500 USD/month [5]"],
            ["Workflow agent platforms", "n8n", "Workflow automation with AI agent nodes; self-hostable under a fair-code licence", "Self-hosted free; cloud about 20-50 EUR/month and up [6, 7]"],
            ["", "Zapier", "No-code automation with an agents feature", "Free tier (about 100 tasks, or 400 agent activities, per month) [6]"],
            ["", "Copilot Studio (Microsoft)", "Builder for agents inside the Microsoft 365 ecosystem", "About 200 USD/month for 25,000 credits, or included with Microsoft 365 Copilot at about 30 USD/user/month [6]"],
        ],
        caption="Table 1. The eight agents, their classes and reported prices.",
    )
    para(
        doc,
        "The classes matter. A workflow platform runs a fixed graph of steps and calls a model where needed; an autonomous agent decides its own steps. "
        "The first is easier to control and to price, the second can do open-ended work. Comparing them on one scale mixes different jobs, so the "
        "scores say how well each agent suits a general, small agentic project, not which product is best in absolute terms."
    )

    # ---------------- 3 method ----------------
    head(doc, "3. Method", 1)
    head(doc, "3.1 Dimensions and scoring", 2)
    para(
        doc,
        "Every agent is scored on four dimensions. Capability (five sub-criteria), precision (three), cost (three) and new-user friendliness (two) are "
        "each the mean of 0 to 5 sub-scores, doubled to a 0 to 10 scale. Precision means reliability and controllability: how deterministic the flow is, "
        "how reliably tools are called according to public benchmarks, and how visible the steps are. It is not the answer accuracy of a model, which we did "
        "not measure. For cost, a higher score means more expensive and it enters the total as (10 minus cost)."
    )
    para(
        doc,
        "The sub-scores are our own judgements, assigned with the same rubric to all agents (Figure 3). The public benchmark evidence is only indicative: "
        "reports credit Claude models with leading GAIA and tau-bench style tool-use tests in most cases and GPT models with leading OSWorld in several, "
        "but vendor-reported and independent numbers differ, and a 2026 study reported that automated agents can exploit several major agent benchmarks "
        "[8, 9]. We therefore kept differences between agents small unless a structural reason (for example, a fixed workflow graph) supports them."
    )
    head(doc, "3.2 Weighting profiles and uncertainty", 2)
    para(
        doc,
        "The overall score is w1 x capability + w2 x precision + w3 x (10 - cost) + w4 x new-user. We use three weighting profiles: Balanced "
        f"{ca.PROFILES['Balanced']}, Reliability first {ca.PROFILES['Reliability first']} and Budget and ease first {ca.PROFILES['Budget and ease first']}, "
        "given as (capability, precision, cost, new user). To express how little the scores can be trusted, we add independent uniform noise between "
        f"-{ca.UNCERTAINTY:g} and +{ca.UNCERTAINTY:g} to every dimension score of every agent, recompute the Balanced ranking, and repeat this 10,000 times."
    )

    # ---------------- 4 results ----------------
    head(doc, "4. Results", 1)
    head(doc, "4.1 Capability, precision and cost", 2)
    d = ca.dims().round(2)
    table(
        doc,
        ["Agent", "Capability", "Precision", "Cost (high = expensive)", "New user"],
        [[t_, d.loc[t_, "capability"], d.loc[t_, "precision"], d.loc[t_, "cost"], d.loc[t_, "new_user"]] for t_ in ca.TOOLS],
        caption="Table 2. Dimension scores on the 0-10 scale.",
    )
    figure(doc, "capability_cost.png",
           "Figure 1. Capability versus cost. Grey bars show ±1 point of uncertainty. Markers show the class of each agent; the top left is cheap and strong.")
    para(
        doc,
        "Figure 1 shows the frontier assistants and n8n together in the cheap and strong corner. Claude has the highest capability (8.8), mainly through "
        "coding and reasoning. n8n has the lowest cost (3.3) because it can be self-hosted at no licence fee, and a high integration breadth. Manus, Devin and "
        "Copilot Studio sit in the expensive, less capable part of the plot, driven mostly by credit or usage billing and, for Devin, a narrow task range."
    )
    figure(doc, "precision_cost.png", "Figure 2. Precision versus cost, with the same conventions as Figure 1.")
    para(
        doc,
        "Precision (Figure 2) is where the classes differ most. Workflow platforms score high because their flows are explicit and logged: n8n 8.3 and "
        "Zapier 7.7. Claude (7.7) leads the assistants on indicative tool-use evidence. The autonomous agents score lowest: Manus has the lowest "
        "reproducibility score of the set, and for both Manus and Devin we found less independent evidence than for the large assistants."
    )
    figure(doc, "subscores_heatmap.png",
           "Figure 3. All 13 sub-scores (0-5). CAP: capability, PRE: precision, COS: cost (high = expensive), NEW: new-user friendliness. Darker is higher.",
           width=14)

    head(doc, "4.2 Overall ranking and how far to trust it", 2)
    figure(doc, "ranking.png",
           "Figure 4. Overall score in the Balanced profile. In brackets: the 90% interval of the rank when every score is perturbed by ±1 point.",
           width=14)
    para(
        doc,
        "In the Balanced profile the order is " + ", ".join(f"{i + 1}. {t_} ({v:.2f})" for i, (t_, v) in enumerate(main_s.items())) + ". "
        f"The first five agents lie within {main_s.iloc[0] - main_s.iloc[4]:.2f} points of each other, while the gap between the fifth and sixth is "
        f"{gap:.2f} points. Under noise, {summ.index[0]} is ranked first in {100 * summ.loc[summ.index[0], 'P(rank 1)']:.0f}% of draws and "
        f"{summ.index[1]} in {100 * summ.loc[summ.index[1], 'P(rank 1)']:.0f}%; the three frontier assistants occupy places 1 to 5 with wide intervals."
    )
    figure(doc, "rank_probability.png",
           "Figure 5. Probability of each rank (%) in 10,000 noise draws, Balanced profile. The block structure shows two tiers.", width=13)
    para(
        doc,
        f"Figure 5 makes the central point. The agents form two tiers: {', '.join(top5)} in the upper tier, and {', '.join(bottom3)} in the lower. The "
        f"lower three stay below all of the upper five in {100 * p_split:.0f}% of draws, whereas the order inside the upper tier is mostly noise."
    )
    figure(doc, "profiles.png", "Figure 6. Overall score under three weighting profiles, agents sorted by Balanced score.", width=14)
    rows = [[t_] + [f"{df.loc[t_, c]:.2f} (#{rk.loc[t_, c]})" for c in score_cols] for t_ in main_s.index]
    table(doc, ["Agent"] + [c[6:-1] for c in score_cols], rows, caption="Table 3. Overall score and rank under each weighting profile.")
    para(
        doc,
        "The weights change the order inside the upper tier but not the tiers. n8n is first in all three profiles, Claude is second in two of them, and "
        "Devin is last in all three. Weights favouring reliability lift Zapier from fifth to third place; weights favouring cost and ease lift Gemini from "
        "fourth to second place and push Claude down to fifth."
    )
    head(doc, "4.3 Cost of an illustrative workload", 2)
    para(
        doc,
        f"For the three assistants we also computed a workload cost: {ca.WORKLOAD_Q:,} questions with {ca.IN_TOK:,} input and {ca.OUT_TOK} output tokens "
        f"each, which is about the size of a forecast table plus a short answer. With mid-tier API prices of about 2/10 USD per million input/output tokens "
        f"(2/12 for Gemini), this costs about {wl['ChatGPT']:.1f} USD for ChatGPT, {wl['Claude']:.1f} for Claude and {wl['Gemini']:.1f} for Gemini. "
        "Mid-tier token costs are therefore almost identical; the price differences between products come from subscription tiers, free tiers, and how "
        "usage is billed."
    )

    # ---------------- 5 case study ----------------
    head(doc, "5. Case study: a hotel demand and overbooking agent", 1)
    para(
        doc,
        "To ground the comparison we built a small agent system and tested it. We used the open Hotel booking demand dataset [1] (119,390 bookings from one resort "
        "and one city hotel). The code and tests (12 automated tests) are at github.com/Atytmr07/hotel-demand-agent. The system has three agents written in Python."
    )
    bullets(
        doc,
        [
            "Data agent: turns bookings into a daily occupied-rooms series, removes cancellations and trims unreliable edges (779 days).",
            "Forecast agent: compares seasonal-naive, ARIMA, SARIMA, Random Forest and XGBoost on a 28-day horizon with 4-fold forward-chaining validation and selects the model with the lowest MAPE. Model selection is done by code, not by a language model.",
            "Optimization agent: chooses a daily booking limit with a mixed-integer linear program (PuLP, CBC). Demand and cancellation uncertainty enter as scenarios, and the solver output was checked against brute-force enumeration.",
        ],
    )
    table(
        doc,
        ["Model (MAPE %, 28 days)", "Resort", "City"],
        [["Random Forest", "2.29", "3.56"], ["XGBoost", "2.37", "3.32"], ["ARIMA(2,1,2)", "3.17", "3.48"],
         ["SARIMA(1,1,1)x(0,1,1,7)", "3.81", "3.99"], ["Seasonal naive", "4.68", "3.59"]],
        caption="Table 4. Forecast error by model. No single model wins at both hotels.",
    )
    table(
        doc,
        ["Realized revenue vs. no overbooking", "Resort", "City"],
        [["Fixed 10% overbooking", "+9.8%", "+9.9%"], ["Rule: capacity / mean show-up rate", "+37.5%", "+60.4%"],
         ["MILP (walk cost = 1.5 x room price)", "+32.4%", "+57.6%"]],
        caption="Table 5. Backtest over the last 112 days.",
    )
    table(
        doc,
        ["MILP vs. rule, by walk cost (x room price)", "0.5", "1", "1.5", "3", "5", "10"],
        [["Resort", "+3.3%", "-1.2%", "-3.7%", "-8.2%", "-12.0%", "-14.3%"], ["City", "+2.6%", "-0.1%", "-1.7%", "-6.6%", "-9.7%", "-14.4%"]],
        caption="Table 6. Sensitivity of the MILP's advantage to the assumed walk cost.",
    )
    para(
        doc,
        "The optimization beats a simple rule (capacity divided by the average show-up rate) only when the cost of walking a guest is very low, about half the room "
        "price. It is about even at one room price and loses revenue above that. The conclusion therefore rests on one assumption, which the dataset cannot "
        "supply. The dataset also has no hotel capacity (we assumed the 90th percentile of net occupancy) and a cancellation rate near 40%, so the percentage gains "
        "are large and should not be read as absolute values for a real hotel. Only two hotels and about two years of data are used."
    )
    para(
        doc,
        "The three agents are Python functions; a language-model layer for natural-language questions is not yet connected. This is where the comparison "
        "applies: the heavy computation (forecasting and the optimization) is deterministic code, so the choice of assistant matters little for correctness. It "
        "matters for the reporting layer, where reliable tool calling and a near-zero budget are the main requirements. A workflow platform such as n8n could "
        "orchestrate the same functions through HTTP endpoints."
    )

    # ---------------- 6 discussion ----------------
    head(doc, "6. Discussion", 1)
    para(
        doc,
        "The most reliable finding is the two-tier structure, not the order within the top tier. Treating a close ranking as exact would mislead a reader, which "
        "is why Figures 4 and 5 report intervals. Three further points follow."
    )
    bullets(
        doc,
        [
            "Price structure matters more than price level. Flat subscriptions (the three assistants) are predictable; credit and usage billing (Manus, Devin, Copilot Studio) are not, and sources warn that usage-based overage can multiply a plan's price.",
            "n8n's lead comes from two sources: a fixed, logged workflow gives it a high precision score, and self-hosting gives it a low cost score. Both depend on someone running and maintaining the system, which the cost score does not capture, and on which language model is plugged into it, which the precision score cannot guarantee.",
            "The frontier assistants are close to each other on every dimension we could score. Differences that matter in practice, such as a particular tool integration or a coding workflow, are task-specific and need a direct test.",
        ],
    )

    # ---------------- 7 limitations ----------------
    head(doc, "7. Threats to validity and limitations", 1)
    bullets(
        doc,
        [
            "The scores are subjective. One rubric was applied by the two authors, with no independent second rater. The Monte Carlo noise expresses our uncertainty but is not a measured error.",
            "We did not run the agents on a common task. No accuracy, latency or token cost was measured, and we used no paid access or API key for the comparison.",
            "Prices and descriptions come from third-party sites that conflict with each other and change quickly; none was verified on an official page. One source questions Manus's future, which we could not confirm.",
            "The eight agents cover three classes that do not do the same job. A ranking across classes depends on the weights, and the case-study use shaped our choice of criteria.",
            "The case study uses two hotels, an assumed capacity and an assumed walk cost, as discussed in Section 5.",
        ],
    )

    # ---------------- 8 conclusion ----------------
    head(doc, "8. Conclusion", 1)
    para(
        doc,
        f"Eight popular AI agents separate into two tiers on capability, precision, cost and ease: {', '.join(top5)} above, {', '.join(bottom3)} below. "
        "Inside the upper tier the order is not reliable, and each agent leads in some profile. The practical advice is to pick the class that fits the job, "
        "then test two or three candidates on your own task. Our hotel case study shows the same caution in another setting: its main result flips with one assumed "
        "cost. Next steps are to connect a language-model reporting layer, to run the candidate agents on one shared task set so that accuracy and cost are measured, "
        "and to find a defensible value for the walk cost."
    )

    # ---------------- references ----------------
    head(doc, "References", 1)
    refs = [
        "[1] Antonio, N., De Almeida, A., Nunes, L. (2019). Hotel booking demand datasets. Data in Brief, 22, 41-49. https://doi.org/10.1016/j.dib.2018.11.126",
        "[2] ChatGPT plans (third-party). https://aitoolsreview.co.uk/insights/chatgpt-pricing",
        "[3] Claude plans (third-party). https://screenapp.io/blog/claude-ai-pricing",
        "[4] Gemini plans (third-party). https://www.eesel.ai/blog/gemini-pricing",
        "[5] Roundup of autonomous agents (third-party). https://resources.rework.com/tools/ai-agents/best-autonomous-ai-agents-2026",
        "[6] Roundup of agent builders and automation platforms (third-party). https://paretotalent.com/guides/best-ai-agent-builders/",
        "[7] n8n pricing and licence (third-party). https://dupple.com/learn/n8n-pricing",
        "[8] Agent benchmark summaries (third-party, indicative). https://rapidclaw.dev/blog/ai-agent-benchmarks-2026",
        "[9] Yao, S., Shinn, N., Razavi, P., Narasimhan, K. (2024). tau-bench: A benchmark for tool-agent-user interaction in real-world domains. arXiv:2406.12045.",
        "[10] Project code and tests. https://github.com/Atytmr07/hotel-demand-agent",
    ]
    for r_ in refs:
        para(doc, r_, size=10, align=WD_ALIGN_PARAGRAPH.LEFT, space_after=2)

    out = REP / "AI_agents_comparison_article.docx"
    doc.save(out)
    print(out)


if __name__ == "__main__":
    main()
