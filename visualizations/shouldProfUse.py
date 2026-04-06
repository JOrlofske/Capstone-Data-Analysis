# visualizations/shouldprofsuse.py
#
# visualizes manually-coded qualitative data from the open-ended question:
#   "What are your thoughts on instructors using AI in your courses?
#    (for example: using AI for feedback, assessment, or course design)"
#
# this question only exists in fall 2024. the coding was done by hand in excel.
# each response was tagged with:
#   - should/shouldn't/not sure (whether professors should use AI)
#   - sentiment (positive, negative, nuanced, not sure)
#   - up to 5 thematic categories
#   - feedback ok? / assessment ok? / course design ok? (yes/no per domain)
#
# the feedback/assessment/course design columns are interesting because many
# students draw sharp lines between acceptable and unacceptable uses. someone
# might be fine with AI for course design but strongly opposed to AI feedback.
# blanks in those columns mean the respondent didnt explicitly address that
# domain — we exclude them from the denominator, not count them as "no".
#
# outputs go in output/should_profs_use/
#   2024fall_should_bar.png            - should vs shouldn't vs not sure
#   2024fall_sentiment_bar.png         - sentiment distribution
#   2024fall_category_bar.png          - category frequency
#   2024fall_domain_comparison.png     - feedback vs assessment vs course design acceptance

from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


VIZ_NAME = "Should professors use AI? Coded open-ended responses (Fall 2024)"
VIZ_SLUG = "should_profs_use"
SHOW_PLOTS = False


# ======= small helpers =======

def is_blank(cell):
    if cell is None:
        return True
    try:
        if pd.isna(cell):
            return True
    except Exception:
        pass
    return str(cell).strip() == ""


def semester_prefix(sheet_name):
    parts = sheet_name.strip().split()
    if len(parts) < 2:
        return "".join(ch.lower() for ch in sheet_name if ch.isalnum())
    season = parts[0].lower()
    year   = "".join(ch for ch in parts[1] if ch.isdigit())
    return f"{year}{season}"


# ======= normalization =======

def normalize_should(raw):
    if is_blank(raw):
        return None
    txt = str(raw).strip().lower()
    if txt == "should":
        return "Should"
    if txt == "shouldn't" or txt == "should not":
        return "Shouldn't"
    if "not sure" in txt:
        return "Not sure"
    return None

SHOULD_ORDER = ["Should", "Not sure", "Shouldn't"]
SHOULD_COLORS = {
    "Should":    "#4575b4",
    "Not sure":  "#fee08b",
    "Shouldn't": "#d73027",
}


def normalize_sentiment(raw):
    if is_blank(raw):
        return None
    txt = str(raw).strip().lower()
    if "not strictly" in txt:
        return "Mixed"
    if "not sure" in txt:
        return "Not sure"
    if "positive" in txt:
        return "Positive"
    if "negative" in txt:
        return "Negative"
    return None

SENTIMENT_ORDER = ["Positive", "Mixed", "Not sure", "Negative"]
SENTIMENT_COLORS = {
    "Positive": "#4575b4",
    "Mixed":    "#fc8d59",
    "Not sure": "#fee08b",
    "Negative": "#d73027",
}


def normalize_category(raw):
    if is_blank(raw):
        return None
    txt = str(raw).strip()
    # fix the typo in the reference column if it leaks through
    if txt.lower() == "fairness/hypocracy":
        return "Fairness/Hypocrisy"
    return txt

CATEGORY_ORDER = [
    "AI as Assistant",
    "Impersonal",
    "Professional Role",
    "Fairness/Hypocrisy",
    "Unreliable",
    "Useful",
    "Saves Time",
    "Benefits Students",
    "Depends on Course",
    "Transparency",
]


# ======= data extraction =======

def read_professor_sheet(xlsx_path, sheet_name):
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, dtype=str)

    should_counts    = Counter()
    sentiment_counts = Counter()
    category_counts  = Counter()
    n_responses      = 0

    # domain acceptance: only count rows that have an explicit yes/no,
    # skip blanks and N/A entirely — they didnt address that domain
    domain_counts = {
        "Feedback":      {"Yes": 0, "No": 0},
        "Assessment":    {"Yes": 0, "No": 0},
        "Course Design": {"Yes": 0, "No": 0},
    }
    domain_col_map = {
        "Feedback":      "Feedback OK?",
        "Assessment":    "Assessment OK?",
        "Course Design": "Course Design OK?",
    }

    for _, row in df.iterrows():
        resp = str(row.get("Response:", "")).strip()
        if resp in ("", "nan", "-", "N/A", "none", "None"):
            continue

        should = normalize_should(row.get("Should/Shouldn't"))
        if should is None:
            continue

        n_responses += 1
        should_counts[should] += 1

        sent = normalize_sentiment(row.get("Sentiment"))
        if sent:
            sentiment_counts[sent] += 1

        for col in ["Category 1", "Category 2 ", "Category 3", "Category 4", "Category 5"]:
            cat = normalize_category(row.get(col))
            if cat:
                category_counts[cat] += 1

        # domain acceptance columns
        for domain, excel_col in domain_col_map.items():
            val = str(row.get(excel_col, "")).strip().lower()
            if val == "yes":
                domain_counts[domain]["Yes"] += 1
            elif val == "no":
                domain_counts[domain]["No"] += 1
            # blank or "n/a" -> skip, doesnt count toward either

    return should_counts, sentiment_counts, category_counts, n_responses, domain_counts


# ======= plot helpers =======

def plot_distribution(counts, order, colors, n_total, title, xlabel, output_png):
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, max(3, len(order) * 0.7 + 2)))
    ax.set_title(title)

    labels = order
    vals   = [counts.get(lbl, 0) for lbl in labels]
    pcts   = [v / n_total * 100.0 if n_total > 0 else 0 for v in vals]
    bar_colors = [colors.get(lbl, "#999999") for lbl in labels]

    bars = ax.barh(labels, pcts, color=bar_colors)

    for i, (bar, cnt, pct) in enumerate(zip(bars, vals, pcts)):
        ax.text(pct + 1, i, f"{pct:.0f}% ({cnt})", va="center", fontsize=10)

    ax.set_xlabel(xlabel)
    ax.set_xlim(0, max(pcts) * 1.35 if pcts and max(pcts) > 0 else 50)

    fig.text(0.5, 0.01, f"N = {n_total} coded responses", ha="center", fontsize=9, style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_category_bar(counts, n_total, title, output_png):
    output_png.parent.mkdir(parents=True, exist_ok=True)

    present = [(cat, counts.get(cat, 0)) for cat in CATEGORY_ORDER if counts.get(cat, 0) > 0]
    present.sort(key=lambda x: x[1])

    if not present:
        fig = plt.figure(figsize=(10, 5))
        plt.title(title)
        plt.text(0.5, 0.5, "No responses", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(11, max(4, len(present) * 0.55 + 2)))
    ax.set_title(title)

    labels = [c for c, _ in present]
    vals   = [v for _, v in present]
    pcts   = [v / n_total * 100.0 if n_total > 0 else 0 for v in vals]

    bars = ax.barh(labels, pcts)

    for i, (bar, cnt, pct) in enumerate(zip(bars, vals, pcts)):
        ax.text(pct + 0.8, i, f"{pct:.0f}% ({cnt})", va="center", fontsize=9)

    ax.set_xlabel("% of responses citing this theme")
    ax.set_xlim(0, max(pcts) * 1.3 if pcts else 50)

    fig.text(
        0.5, 0.01,
        f"N = {n_total} coded responses. One response can cite multiple themes.",
        ha="center", fontsize=8, style="italic",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_domain_comparison(domain_counts, title, output_png):
    # grouped bar showing yes vs no for each domain (feedback, assessment, course design).
    # each domain has its own n because not everyone explicitly addressed each one.
    # this is the interesting chart — it shows where students draw the line on
    # acceptable instructor AI use
    output_png.parent.mkdir(parents=True, exist_ok=True)

    domains = ["Feedback", "Assessment", "Course Design"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title)

    x = np.arange(len(domains))
    width = 0.3

    yes_pcts = []
    no_pcts  = []
    ns       = []

    for domain in domains:
        yes_n = domain_counts[domain]["Yes"]
        no_n  = domain_counts[domain]["No"]
        total = yes_n + no_n
        ns.append(total)
        yes_pcts.append(yes_n / total * 100.0 if total > 0 else 0)
        no_pcts.append(no_n / total * 100.0 if total > 0 else 0)

    bars_yes = ax.bar(x - width/2, yes_pcts, width, label="OK / Acceptable",
                      color="#4575b4")
    bars_no  = ax.bar(x + width/2, no_pcts,  width, label="Not OK / Opposed",
                      color="#d73027")

    for bar, pct, domain_idx in zip(bars_yes, yes_pcts, range(len(domains))):
        yes_n = domain_counts[domains[domain_idx]]["Yes"]
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{pct:.0f}%\n({yes_n})", ha="center", va="bottom", fontsize=9)

    for bar, pct, domain_idx in zip(bars_no, no_pcts, range(len(domains))):
        no_n = domain_counts[domains[domain_idx]]["No"]
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{pct:.0f}%\n({no_n})", ha="center", va="bottom", fontsize=9)

    # show per-domain n below the x labels
    domain_labels = [f"{d}\nn={n}" for d, n in zip(domains, ns)]
    ax.set_xticks(x)
    ax.set_xticklabels(domain_labels)
    ax.set_ylabel("% of respondents who addressed this domain")
    ax.set_ylim(0, max(max(yes_pcts), max(no_pcts)) * 1.25)
    ax.legend(frameon=False)

    fig.text(
        0.5, 0.01,
        "Only respondents who explicitly stated a position on each domain are counted.\n"
        "Those who did not address a domain are excluded from its denominator.",
        ha="center", fontsize=8, style="italic",
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ======= semester config =======

SHEET_NAME = 'Fall 2024 "Should Professors Us'
SEM_LABEL  = "Fall 2024"


# ======= main =======

def run(xlsx_path: str) -> None:
    out_dir = Path("output") / VIZ_SLUG
    out_dir.mkdir(parents=True, exist_ok=True)

    # same fallback logic as shouldbetaught.py — check if the coded sheets
    # exist in the passed file, otherwise look for the coded excel nearby
    xls = pd.ExcelFile(xlsx_path)
    coded_path = xlsx_path

    has_coded = any("Should Professors" in s for s in xls.sheet_names)
    if not has_coded:
        alt = Path(xlsx_path).parent / "AI_survey_free_response_data.xlsx"
        if alt.exists():
            coded_path = str(alt)
        else:
            print(f"Warning: {xlsx_path} does not contain the professors sheet. "
                  f"Pass the AI_survey_free_response_data.xlsx file instead.")
            return

    prefix = semester_prefix(SEM_LABEL)

    should_counts, sentiment_counts, category_counts, n_resp, domain_counts = \
        read_professor_sheet(coded_path, SHEET_NAME)

    # should/shouldn't bar
    plot_distribution(
        should_counts, SHOULD_ORDER, SHOULD_COLORS, n_resp,
        f"{SEM_LABEL}. Should professors use AI in courses?\n"
        f"Coded from open-ended responses (N = {n_resp})",
        "% of responses",
        out_dir / f"{prefix}_should_bar.png",
    )

    # sentiment bar
    plot_distribution(
        sentiment_counts, SENTIMENT_ORDER, SENTIMENT_COLORS, n_resp,
        f"{SEM_LABEL}. Sentiment toward professors using AI\n"
        f"(N = {n_resp})",
        "% of responses",
        out_dir / f"{prefix}_sentiment_bar.png",
    )

    # category bar
    plot_category_bar(
        category_counts, n_resp,
        f"{SEM_LABEL}. Why should/shouldn't professors use AI?\n"
        f"Theme frequency from coded responses (N = {n_resp})",
        out_dir / f"{prefix}_category_bar.png",
    )

    # usage comparison
    plot_domain_comparison(
        domain_counts,
        f"{SEM_LABEL}. Is it acceptable for professors to use AI for...?\n"
        f"Respondents who explicitly addressed each domain",
        out_dir / f"{prefix}_domain_comparison.png",
    )