# visualizations/shouldbetaught.py
#
# visualizes manually-coded qualitative data from the open-ended question:
#   "Why do you think students should or should not learn about AI-based
#    text generating programs (such as ChatGPT) in their courses at Pitt?"
#
# the coding was done by hand in excel. each response was tagged with:
#   - should/shouldn't/not sure
#   - sentiment (positive, negative, nuanced, not sure)
#   - up to 5 thematic categories explaining their reasoning
#
# this file reads the coded excel directly — no keyword matching.
# spring 2024 (n~50) and fall 2024 (n~149).
#
# outputs go in output/should_be_taught/
#   per semester:
#     {prefix}_should_bar.png              - should vs shouldn't vs not sure
#     {prefix}_sentiment_bar.png           - sentiment distribution
#     {prefix}_category_should_bar.png     - reasons AI should be taught
#     {prefix}_category_shouldnt_bar.png   - reasons AI should not be taught
#   combined:
#     combined_should_bar.png              - should/shouldn't side by side
#     combined_sentiment_bar.png           - sentiment side by side
#     combined_category_should_bar.png     - "should" themes grouped by semester
#     combined_category_shouldnt_bar.png   - "shouldn't" themes grouped by semester

from __future__ import annotations
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


VIZ_NAME = "Should students be taught about AI? Coded open-ended responses (Spring 2024-Fall 2024)"
VIZ_SLUG = "should_be_taught"
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

# the excel has some case inconsistencies ("Not sure" vs "Not Sure", "inevitable"
# vs "Inevitable") so we normalize everything here

def normalize_should(raw):
    if is_blank(raw):
        return None
    txt = str(raw).strip().lower()
    if txt == "yes":
        return "Yes"
    elif txt == "no":
        return "No"
    elif txt == "neutral":
        return "Neutral"
    return None

SHOULD_ORDER = ["Yes", "Neutral", "No"]
SHOULD_COLORS = {
    "Yes":     "#4575b4",
    "Neutral": "#fee08b",
    "No":      "#d73027",
}


def normalize_sentiment(raw):
    if is_blank(raw):
        return None
    txt = str(raw).strip().lower()
    if "not strictly" in txt or "not sure" in txt.replace("positive", "").replace("negative", ""):
        pass
    if "not strictly" in txt:
        return "Mixed"
    if "uncertain" in txt:
        return "Uncertain"
    if "positive" in txt:
        return "Positive"
    if "negative" in txt:
        return "Negative"
    return None

SENTIMENT_ORDER = ["Positive", "Mixed", "Uncertain", "Negative"]
SENTIMENT_COLORS = {
    "Positive": "#4575b4",
    "Mixed":    "#fc8d59",
    "Uncertain": "#fee08b",
    "Negative": "#d73027",
}


def normalize_category(raw):
    if is_blank(raw):
        return None
    txt = str(raw).strip()
    # fix case on "inevitable"
    if txt.lower() == "inevitable":
        return "Inevitable"
    # fix case on "saves time"
    if txt.lower() == "saves time":
        return "Saves Time"
    return txt

# categories split by whether they argue for or against teaching AI
SHOULD_CATEGORIES = [
    "Useful",
    "Understand the Tech",
    "Prepare for Future",
    "Inevitable",
    "Saves Time",
]

SHOULDNT_CATEGORIES = [
    "Over-reliance",
    "Cheating",
    "Already known about",
    "Unreliable",
    "Uncreative Output",
    "Replaces professor's role",
    "Scared",
]

# combined list for legacy/combined charts
CATEGORY_ORDER = SHOULD_CATEGORIES + SHOULDNT_CATEGORIES


# ======= data extraction =======

def read_coded_sheet(xlsx_path, sheet_name):
    # reads one coded sheet and returns normalized counters
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, dtype=str)

    should_counts    = Counter()
    sentiment_counts = Counter()
    category_counts  = Counter()
    should_cat_counts   = Counter()  # categories arguing AI should be taught
    shouldnt_cat_counts = Counter()  # categories arguing AI shouldn't be taught
    n_responses      = 0

    for _, row in df.iterrows():
        resp = str(row.get("Response:", "")).strip()
        if resp in ("", "nan", "-", "N/A"):
            continue

        should_raw = row.get("Should/Shouldn't")
        should = normalize_should(should_raw)
        if should is None:
            continue

        n_responses += 1
        should_counts[should] += 1

        sent_raw = row.get("Sentiment")
        sent = normalize_sentiment(sent_raw)
        if sent:
            sentiment_counts[sent] += 1

        # pull categories from up to 5 columns
        for col in ["Category 1", "Category 2 ", "Category 3", "Category 4", "Category 5"]:
            cat = normalize_category(row.get(col))
            if cat:
                category_counts[cat] += 1
                if cat in SHOULD_CATEGORIES:
                    should_cat_counts[cat] += 1
                elif cat in SHOULDNT_CATEGORIES:
                    shouldnt_cat_counts[cat] += 1

    return (should_counts, sentiment_counts, category_counts,
            should_cat_counts, shouldnt_cat_counts, n_responses)


# ======= plot helpers =======

def plot_distribution(counts, order, colors, n_total, title, xlabel, output_png):
    # generic horizontal bar chart for a single distribution
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


def plot_category_bar(counts, n_total, title, output_png, category_order=None,
                      bar_color=None, xlim_max=None):
    # horizontal bar chart for categories, sorted by frequency descending.
    # only shows categories that actually appeared (no empty bars)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    if category_order is None:
        category_order = CATEGORY_ORDER

    # filter to categories that have at least 1 hit, sorted descending
    present = [(cat, counts.get(cat, 0)) for cat in category_order if counts.get(cat, 0) > 0]
    present.sort(key=lambda x: x[1])  # ascending for barh (top = highest)

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

    bars = ax.barh(labels, pcts, color=bar_color)

    for i, (bar, cnt, pct) in enumerate(zip(bars, vals, pcts)):
        ax.text(pct + 0.8, i, f"{pct:.0f}% ({cnt})", va="center", fontsize=9)

    ax.set_xlabel("% of responses citing this theme")
    if xlim_max is not None:
        ax.set_xlim(0, xlim_max)
    else:
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


def plot_combined_grouped(all_data, order, colors, title, xlabel, output_png):
    # grouped bar chart: one group per label, bars = semesters side by side
    # all_data: list of (semester_label, counts, n_total)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_title(title)

    x = np.arange(len(order))
    n_sems = len(all_data)
    width = 0.7 / n_sems

    for si, (sem_label, counts, n_total) in enumerate(all_data):
        pcts = [counts.get(lbl, 0) / n_total * 100.0 if n_total > 0 else 0 for lbl in order]
        offset = (si - (n_sems - 1) / 2) * width
        bars = ax.bar(x + offset, pcts, width,
                      label=f"{sem_label} (n={n_total})",
                      color=[colors.get(lbl, "#999999") for lbl in order],
                      edgecolor="white", linewidth=0.5, alpha=0.85 if si == 0 else 1.0)

        for bar, pct in zip(bars, pcts):
            if pct >= 3:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                        f"{pct:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=15, ha="right")
    ax.set_ylabel(xlabel)
    ax.legend(frameon=False)

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_combined_categories(all_data, title, output_png, category_order=None,
                             ylim_max=None):
    # grouped bar comparing category frequencies across semesters
    # all_data: list of (semester_label, category_counts, n_total)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    if category_order is None:
        category_order = CATEGORY_ORDER

    # only show categories that appear in at least one semester
    present_cats = []
    for cat in category_order:
        if any(counts.get(cat, 0) > 0 for _, counts, _ in all_data):
            present_cats.append(cat)

    if not present_cats:
        fig = plt.figure(figsize=(12, 6))
        plt.title(title)
        plt.text(0.5, 0.5, "No responses", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_title(title)

    x = np.arange(len(present_cats))
    n_sems = len(all_data)
    width = 0.7 / n_sems

    for si, (sem_label, counts, n_total) in enumerate(all_data):
        pcts = [counts.get(cat, 0) / n_total * 100.0 if n_total > 0 else 0 for cat in present_cats]
        offset = (si - (n_sems - 1) / 2) * width
        bars = ax.bar(x + offset, pcts, width, label=f"{sem_label} (n={n_total})")

        for bar, pct in zip(bars, pcts):
            if pct >= 3:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{pct:.0f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(present_cats, rotation=30, ha="right")
    ax.set_ylabel("% of responses citing this theme")
    if ylim_max is not None:
        ax.set_ylim(0, ylim_max)
    ax.legend(frameon=False)

    fig.text(0.5, 0.01, "One response can cite multiple themes. Percentages are per-semester.",
             ha="center", fontsize=8, style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ======= semester configs =======

SHEETS = [
    ("Spring 2024", 'Spring 2024 "Should be Taught"'),
    ("Fall 2024",   'Fall 2024 "Should be Taught"'),
]

# fixed x-axis limits per semester so charts are visually comparable
# within each semester's should vs shouldn't pair
SEMESTER_XLIM = {
    "Spring 2024": 50,
    "Fall 2024":   40,
}
COMBINED_YLIM = 50


# ======= main =======

def run(xlsx_path: str) -> None:
    out_dir = Path("output") / VIZ_SLUG
    out_dir.mkdir(parents=True, exist_ok=True)

    # we need the coded excel, not the survey excel
    xls = pd.ExcelFile(xlsx_path)
    coded_path = xlsx_path

    # check if the coded sheets exist in this file
    has_coded = any("Should be Taught" in s for s in xls.sheet_names)
    if not has_coded:
        # try looking for the coded file alongside the survey file
        alt = Path(xlsx_path).parent / "AI_survey_free_response_data.xlsx"
        if alt.exists():
            coded_path = str(alt)
        else:
            print(f"Warning: {xlsx_path} does not contain coded sheets. "
                  f"Pass the AI_survey_free_response_data.xlsx file instead.")
            return

    all_should    = []
    all_sentiment = []
    all_category  = []
    all_should_cat   = []
    all_shouldnt_cat = []

    for sem_label, sheet_name in SHEETS:
        prefix = semester_prefix(sem_label)

        (should_counts, sentiment_counts, category_counts,
         should_cat_counts, shouldnt_cat_counts, n_resp) = read_coded_sheet(
            coded_path, sheet_name,
        )

        all_should.append((sem_label, should_counts, n_resp))
        all_sentiment.append((sem_label, sentiment_counts, n_resp))
        all_category.append((sem_label, category_counts, n_resp))
        all_should_cat.append((sem_label, should_cat_counts, n_resp))
        all_shouldnt_cat.append((sem_label, shouldnt_cat_counts, n_resp))

        # per-semester should bar
        plot_distribution(
            should_counts, SHOULD_ORDER, SHOULD_COLORS, n_resp,
            f"{sem_label}. Should students be taught about AI?\n"
            f"Coded from open-ended responses (N = {n_resp})",
            "% of responses",
            out_dir / f"{prefix}_should_bar.png",
        )

        # per-semester sentiment bar
        plot_distribution(
            sentiment_counts, SENTIMENT_ORDER, SENTIMENT_COLORS, n_resp,
            f"{sem_label}. General sentiment in responses\n"
            f"(N = {n_resp})",
            "% of responses",
            out_dir / f"{prefix}_sentiment_bar.png",
        )

        # per-semester category bars — split by should vs shouldn't
        xlim = SEMESTER_XLIM.get(sem_label)

        plot_category_bar(
            should_cat_counts, n_resp,
            f"{sem_label}. Reasons AI should be taught\n"
            f"Theme frequency from coded responses (N = {n_resp})",
            out_dir / f"{prefix}_category_should_bar.png",
            category_order=SHOULD_CATEGORIES,
            bar_color="#4575b4",
            xlim_max=xlim,
        )

        plot_category_bar(
            shouldnt_cat_counts, n_resp,
            f"{sem_label}. Reasons AI should not be taught\n"
            f"Theme frequency from coded responses (N = {n_resp})",
            out_dir / f"{prefix}_category_shouldnt_bar.png",
            category_order=SHOULDNT_CATEGORIES,
            bar_color="#d73027",
            xlim_max=xlim,
        )

    # combined charts
    if len(all_should) >= 2:
        plot_combined_grouped(
            all_should, SHOULD_ORDER, SHOULD_COLORS,
            "Should students learn about AI? (Spring 2024 vs Fall 2024)",
            "% of responses",
            out_dir / "combined_should_bar.png",
        )

        plot_combined_grouped(
            all_sentiment, SENTIMENT_ORDER, SENTIMENT_COLORS,
            "Sentiment in open-ended responses (Spring 2024 vs Fall 2024)",
            "% of responses",
            out_dir / "combined_sentiment_bar.png",
        )

        plot_combined_categories(
            all_should_cat,
            "Reasons AI should be taught (Spring 2024 vs Fall 2024)",
            out_dir / "combined_category_should_bar.png",
            category_order=SHOULD_CATEGORIES,
            ylim_max=COMBINED_YLIM,
        )

        plot_combined_categories(
            all_shouldnt_cat,
            "Reasons AI should not be taught (Spring 2024 vs Fall 2024)",
            out_dir / "combined_category_shouldnt_bar.png",
            category_order=SHOULDNT_CATEGORIES,
            ylim_max=COMBINED_YLIM,
        )