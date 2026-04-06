# visualizations/motivationAttitudes.py
#
# how does the existence of AI tools impact students' motivation and feelings
# about writing? the survey asks 4 core attitude questions across 4 semesters
# (spring 2023 - fall 2024) plus a 5th question added in spring 2024.
#
# the tricky part: the response scale changed between semesters.
# spring/fall 2023 used a 5-point likert (strongly disagree .. strongly agree)
# with "uncertain" as the middle option (no "neutral" existed). spring/fall 2024
# collapsed this to yes/no/neutral. to compare across semesters we collapse
# the likert into 3 bins:
#   strongly disagree + disagree -> "No"
#   uncertain                    -> "Neutral"
#   agree + strongly agree       -> "Yes"
# per-semester charts still show the full original distributions.
#
# outputs go in output/motivation_attitudes/
#   per semester:
#     {prefix}_attitudes.png                - stacked bar with full original response scale
#   combined:
#     combined_trend_dotplot.png             - % who agree/say yes across all semesters
#     combined_collapsed_stacked.png         - 3-category stacked bars, all semesters per question
#     combined_collapsed_diverging.png       - diverging bars centered on neutral
#     should_learn_comparison.png            - 2024-only comparison for the "should learn" item
#
# column locations (from earlier survey analysis):
#   spring 2023: AD/AE (motivated), AF/AG (excited), AH/AI (easier), AJ/AK (future)
#   fall 2023:   AD/AE (motivated), AF/AG (excited), AH/AI (easier), AJ/AK (future)
#   spring 2024: BS/BT (motivated), BU/BV (excited), BW/BX (easier), BY/BZ (future), CA/CB (should learn)
#   fall 2024:   BH/BI (motivated), BJ/BK (excited), BL/BM (easier), BN/BO (future), BP/BQ (should learn)

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


VIZ_NAME = "Motivation & attitudes toward AI writing tools (Spring 2023-Fall 2024)"
VIZ_SLUG = "motivation_attitudes"
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
    if len(parts) != 2:
        return "".join(ch.lower() for ch in sheet_name if ch.isalnum())
    season = parts[0].lower()
    year   = "".join(ch for ch in parts[1] if ch.isdigit())
    return f"{year}{season}"


def col_letter_to_index(col):
    col = col.strip().upper()
    n = 0
    for ch in col:
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


def get_col_name_by_letter(df, col_letter):
    idx = col_letter_to_index(col_letter)
    if idx < 0 or idx >= len(df.columns):
        raise ValueError(f"Column {col_letter} (index {idx}) out of range.")
    return str(df.columns[idx])


def parse_int_code(cell):
    if is_blank(cell):
        return None
    try:
        return int(float(str(cell).strip()))
    except Exception:
        return None


# ======= response scale handling =======

# the 5-point likert labels from 2023. there was no "neutral" option — "uncertain"
# served as the middle/unsure response. code 1-5 maps to these.
LIKERT_LABELS = ["Strongly disagree", "Disagree", "Uncertain", "Agree", "Strongly agree"]
LIKERT_ORDER  = ["Strongly disagree", "Disagree", "Uncertain", "Agree", "Strongly agree"]

# the 2024 scale is just yes/no/neutral. mapping varies by question but the
# label column tells us directly
YNN_LABELS = ["No", "Neutral", "Yes"]

# collapsed 3-bin mapping for cross-semester comparison
COLLAPSED_ORDER = ["No", "Neutral", "Yes"]

COLLAPSED_COLORS = {
    "No":      "#d73027",
    "Neutral": "#fee08b",
    "Yes":     "#4575b4",
}

# for the full likert charts. uncertain sits in the middle where neutral would be
LIKERT_COLORS = {
    "Strongly disagree": "#d73027",
    "Disagree":          "#fc8d59",
    "Uncertain":         "#fee08b",
    "Agree":             "#91bfdb",
    "Strongly agree":    "#4575b4",
}

YNN_COLORS = {
    "No":      "#d73027",
    "Neutral": "#fee08b",
    "Yes":     "#4575b4",
}


def normalize_likert_label(s):
    # takes raw label text from the 2023 semesters and normalizes it.
    # note: there was no "neutral" option in 2023, only "uncertain"
    if is_blank(s):
        return None
    txt = str(s).strip().lower()
    mapping = {
        "strongly disagree": "Strongly disagree",
        "disagree":          "Disagree",
        "uncertain":         "Uncertain",
        "agree":             "Agree",
        "strongly agree":    "Strongly agree",
    }
    return mapping.get(txt)


def normalize_ynn_label(s):
    # takes raw label text from the 2024 semesters
    if is_blank(s):
        return None
    txt = str(s).strip().lower()
    mapping = {
        "yes": "Yes",
        "no": "No",
        "neutral": "Neutral",
    }
    return mapping.get(txt)


def collapse_likert(label):
    # maps a 2023 likert label into the 3-bin scheme for cross-semester comparison.
    # there was no "neutral" option in 2023 — "uncertain" was the middle ground
    if label in ("Strongly disagree", "Disagree"):
        return "No"
    elif label in ("Agree", "Strongly agree"):
        return "Yes"
    elif label == "Uncertain":
        return "Neutral"
    return None


# ======= data extraction =======

@dataclass
class AttitudeItem:
    short_label:  str   # e.g. "Less motivated"
    code_col:     str   # excel column letter for numeric code
    label_col:    str   # excel column letter for text label


@dataclass
class AttitudeResult:
    short_label:  str
    counts:       Counter   # full-scale label -> count
    collapsed:    Counter   # "Yes"/"No"/"Neutral" -> count
    n_valid:      int
    scale_type:   str       # "likert" or "ynn"


def analyze_item(df, code_letter, label_letter, short_label, scale_type):
    # reads one attitude question column and tallies responses
    code_col  = get_col_name_by_letter(df, code_letter)
    label_col = get_col_name_by_letter(df, label_letter)
    df2 = df.iloc[2:]

    counts    = Counter()
    collapsed = Counter()
    n_valid   = 0

    for _, row in df2.iterrows():
        raw_label = row.get(label_col)
        raw_code  = parse_int_code(row.get(code_col))

        if scale_type == "likert":
            resolved = normalize_likert_label(raw_label)
            # fallback to code if label is blank
            if resolved is None and raw_code is not None and 1 <= raw_code <= 5:
                resolved = LIKERT_LABELS[raw_code - 1]
            if resolved is None:
                continue
            n_valid += 1
            counts[resolved] += 1
            c = collapse_likert(resolved)
            if c:
                collapsed[c] += 1
        else:
            resolved = normalize_ynn_label(raw_label)
            if resolved is None:
                continue
            n_valid += 1
            counts[resolved] += 1
            collapsed[resolved] += 1

    return AttitudeResult(
        short_label=short_label,
        counts=counts,
        collapsed=collapsed,
        n_valid=n_valid,
        scale_type=scale_type,
    )


# ======= plot 1: per-semester stacked bars (full original scale) =======

def plot_semester_attitudes(results, title, output_png):
    # shows the full response distribution for each question in a single semester.
    # uses the original scale (5-point for 2023, 3-point for 2024)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        fig = plt.figure(figsize=(13, 6))
        plt.title(title)
        plt.text(0.5, 0.5, "No responses", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    scale_type = results[0].scale_type
    if scale_type == "likert":
        labels_order = LIKERT_ORDER
        colors       = LIKERT_COLORS
    else:
        labels_order = YNN_LABELS
        colors       = YNN_COLORS

    fig = plt.figure(figsize=(13, max(4, len(results) * 1.0 + 2)))
    plt.title(title)
    ax = plt.gca()

    y_labels = [r.short_label for r in results]
    left = [0.0] * len(results)

    for bucket in labels_order:
        vals = []
        for r in results:
            denom = r.n_valid if r.n_valid > 0 else 1
            vals.append(r.counts.get(bucket, 0) / denom * 100.0)

        bars = ax.barh(y_labels, vals, left=left, label=bucket, color=colors[bucket])

        for i, (bar, pct) in enumerate(zip(bars, vals)):
            if pct >= 5:
                x_center = left[i] + pct / 2
                text_color = "black" if bucket in ("Neutral", "Uncertain") else "white"
                ax.text(x_center, i, f"{pct:.0f}%", ha="center", va="center",
                        fontsize=9, color=text_color)

        left = [l + v for l, v in zip(left, vals)]

    for i, r in enumerate(results):
        ax.text(101, i, f"n={r.n_valid}", va="center", fontsize=9)

    ax.set_xlabel("% of respondents")
    ax.set_xlim(0, 100)
    ax.legend(title="Response", loc="center left", bbox_to_anchor=(1.08, 0.5), frameon=False)

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ======= plot 2: combined trend dot plot =======

def plot_trend_dotplot(trend_data, title, output_png):
    # trend_data: dict of { short_label: [(semester_label, pct_yes), ...] }
    # shows % who agree/say yes across semesters for each question
    output_png.parent.mkdir(parents=True, exist_ok=True)

    if not trend_data:
        fig = plt.figure(figsize=(12, 6))
        plt.title(title)
        plt.text(0.5, 0.5, "No responses", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title(title)

    # each question gets its own line across semesters
    question_labels = list(trend_data.keys())
    # all semesters that appear (ordered chronologically)
    all_sems = []
    for pts in trend_data.values():
        for sem, _ in pts:
            if sem not in all_sems:
                all_sems.append(sem)

    x = list(range(len(all_sems)))
    colors_cycle = ["#4575b4", "#d73027", "#91bfdb", "#fc8d59", "#66c2a5"]

    for qi, q_label in enumerate(question_labels):
        pts = trend_data[q_label]
        sem_to_pct = {sem: pct for sem, pct in pts}
        y = []
        x_pts = []
        for xi, sem in enumerate(all_sems):
            if sem in sem_to_pct:
                y.append(sem_to_pct[sem])
                x_pts.append(xi)

        color = colors_cycle[qi % len(colors_cycle)]
        ax.plot(x_pts, y, marker="o", markersize=8, label=q_label, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(all_sems, rotation=20, ha="right")
    ax.set_ylabel("% who agree / say Yes")
    ax.set_ylim(-2, 102)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ======= plot 3: combined collapsed stacked bars =======

def plot_collapsed_stacked(all_collapsed, title, output_png):
    # all_collapsed: list of (row_label, Counter(Yes/No/Neutral), n_valid)
    # each row_label is like "Less motivated (Spring 2023)"
    output_png.parent.mkdir(parents=True, exist_ok=True)

    if not all_collapsed:
        fig = plt.figure(figsize=(13, 6))
        plt.title(title)
        plt.text(0.5, 0.5, "No responses", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    fig = plt.figure(figsize=(14, max(5, len(all_collapsed) * 0.45 + 2)))
    plt.title(title)
    ax = plt.gca()

    row_labels = [r[0] for r in all_collapsed]
    left = [0.0] * len(all_collapsed)

    for bucket in COLLAPSED_ORDER:
        vals = []
        for _, counts, n_valid in all_collapsed:
            denom = n_valid if n_valid > 0 else 1
            vals.append(counts.get(bucket, 0) / denom * 100.0)

        bars = ax.barh(row_labels, vals, left=left, label=bucket,
                       color=COLLAPSED_COLORS[bucket])

        for i, (bar, pct) in enumerate(zip(bars, vals)):
            if pct >= 6:
                x_center = left[i] + pct / 2
                text_color = "black" if bucket == "Neutral" else "white"
                ax.text(x_center, i, f"{pct:.0f}%", ha="center", va="center",
                        fontsize=8, color=text_color)

        left = [l + v for l, v in zip(left, vals)]

    for i, (_, _, n) in enumerate(all_collapsed):
        ax.text(101, i, f"n={n}", va="center", fontsize=8)

    ax.set_xlabel("% of respondents")
    ax.set_xlim(0, 100)
    ax.legend(title="Response", loc="center left", bbox_to_anchor=(1.08, 0.5), frameon=False)

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ======= plot 4: combined collapsed diverging bars =======

def plot_collapsed_diverging(all_collapsed, title, output_png):
    # same data as plot 3 but centered on the neutral boundary.
    # no fans left, yes fans right, neutral sits at center
    output_png.parent.mkdir(parents=True, exist_ok=True)

    if not all_collapsed:
        fig = plt.figure(figsize=(13, 6))
        plt.title(title)
        plt.text(0.5, 0.5, "No responses", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    fig = plt.figure(figsize=(14, max(5, len(all_collapsed) * 0.45 + 2)))
    plt.title(title)
    ax = plt.gca()

    row_labels = [r[0] for r in all_collapsed]
    y_pos = range(len(row_labels))

    for i, (label, counts, n_valid) in enumerate(all_collapsed):
        denom = n_valid if n_valid > 0 else 1
        pct_no      = counts.get("No", 0) / denom * 100.0
        pct_neutral = counts.get("Neutral", 0) / denom * 100.0
        pct_yes     = counts.get("Yes", 0) / denom * 100.0

        # neutral straddles the center line: half left, half right
        half_neutral = pct_neutral / 2.0

        # no goes left from -half_neutral
        ax.barh(i, -pct_no, left=-half_neutral, color=COLLAPSED_COLORS["No"], height=0.7)
        # neutral centered
        ax.barh(i, pct_neutral, left=-half_neutral, color=COLLAPSED_COLORS["Neutral"], height=0.7)
        # yes goes right from +half_neutral
        ax.barh(i, pct_yes, left=half_neutral, color=COLLAPSED_COLORS["Yes"], height=0.7)

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("← Disagree / No          % of responses          Agree / Yes →")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")

    handles = [
        mpatches.Patch(color=COLLAPSED_COLORS[lbl], label=lbl)
        for lbl in COLLAPSED_ORDER
    ]
    ax.legend(handles=handles, title="Response", loc="center left",
              bbox_to_anchor=(1.02, 0.5), frameon=False)

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ======= plot 6: vertical grouped stacked bars =======

def plot_vertical_grouped_stacked(collapsed_by_question, title, output_png):
    """Vertical stacked bar chart: x-axis = 4 categories, 4 touching bars per category
    (one per semester). Lighter shades for earlier semesters, darker for later."""
    output_png.parent.mkdir(parents=True, exist_ok=True)

    semester_labels = ["Spring 2023", "Fall 2023", "Spring 2024", "Fall 2024"]

    # Color ramps: light -> dark for each response type
    no_ramp      = ["#f4a5a0", "#e87872", "#d73027", "#a01a10"]
    neutral_ramp = ["#fff8c4", "#fee08b", "#e6c350", "#c9a520"]
    yes_ramp     = ["#a8cce0", "#7baed0", "#4575b4", "#2a4f8a"]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title(title, fontsize=13, pad=14)

    n_categories = len(CORE_QUESTIONS)
    n_semesters = len(semester_labels)
    bar_width = 0.18
    group_width = n_semesters * bar_width  # bars touching

    for qi, q_label in enumerate(CORE_QUESTIONS):
        entries = collapsed_by_question[q_label]  # list of (sem_label, counts, n_valid)
        for si, (sem_label, counts, n_valid) in enumerate(entries):
            denom = n_valid if n_valid > 0 else 1
            pct_no      = counts.get("No", 0) / denom * 100.0
            pct_neutral = counts.get("Neutral", 0) / denom * 100.0
            pct_yes     = counts.get("Yes", 0) / denom * 100.0

            x = qi * (group_width + 0.25) + si * bar_width

            ax.bar(x, pct_no, bar_width, bottom=0, color=no_ramp[si])
            ax.bar(x, pct_neutral, bar_width, bottom=pct_no, color=neutral_ramp[si])
            ax.bar(x, pct_yes, bar_width, bottom=pct_no + pct_neutral, color=yes_ramp[si])

            # percentage labels centered on each segment
            for pct, bot, bucket in [
                (pct_no, 0, "No"),
                (pct_neutral, pct_no, "Neutral"),
                (pct_yes, pct_no + pct_neutral, "Yes"),
            ]:
                if pct >= 5:
                    text_color = "black" if bucket == "Neutral" else "white"
                    ax.text(x, bot + pct / 2, f"{pct:.0f}%",
                            ha="center", va="center", fontsize=6.5,
                            color=text_color, fontweight="bold")

            # n label on top
            ax.text(x, pct_no + pct_neutral + pct_yes + 1, f"n={n_valid}",
                    ha="center", va="bottom", fontsize=7, rotation=90)

    # x-axis tick at center of each group
    group_centers = [qi * (group_width + 0.25) + (n_semesters - 1) * bar_width / 2
                     for qi in range(n_categories)]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(CORE_QUESTIONS, fontsize=10)
    ax.set_ylabel("% of respondents")
    ax.set_ylim(0, 135)

    # Legend above the plot
    sem_handles = [mpatches.Patch(facecolor=f"#{v:02x}{v:02x}{v:02x}", label=s)
                   for s, v in zip(semester_labels, [180, 130, 80, 40])]
    resp_handles = [mpatches.Patch(facecolor=COLLAPSED_COLORS[lbl], label=lbl)
                    for lbl in COLLAPSED_ORDER]
    ax.legend(handles=resp_handles + sem_handles, loc="upper center",
              bbox_to_anchor=(0.5, 0.98), fontsize=8, frameon=True,
              ncol=7, title="Response / Semester")

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ======= plot 5: should-learn comparison (2024 only) =======

def plot_should_learn(results_by_sem, title, output_png):
    # results_by_sem: list of (semester_label, AttitudeResult)
    # simple grouped bar showing yes/no/neutral side by side
    output_png.parent.mkdir(parents=True, exist_ok=True)

    if not results_by_sem:
        fig = plt.figure(figsize=(8, 5))
        plt.title(title)
        plt.text(0.5, 0.5, "No responses", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(title)

    sem_labels = [s for s, _ in results_by_sem]
    x = np.arange(len(COLLAPSED_ORDER))
    width = 0.35
    offsets = np.linspace(-width/2 * (len(sem_labels)-1), width/2 * (len(sem_labels)-1), len(sem_labels))

    for si, (sem_label, result) in enumerate(results_by_sem):
        denom = result.n_valid if result.n_valid > 0 else 1
        vals = [result.collapsed.get(lbl, 0) / denom * 100.0 for lbl in COLLAPSED_ORDER]
        bars = ax.bar(x + offsets[si], vals, width, label=f"{sem_label} (n={result.n_valid})")
        for bar, val in zip(bars, vals):
            if val >= 3:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{val:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(COLLAPSED_ORDER)
    ax.set_ylabel("% of respondents")
    ax.legend(frameon=False)

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ======= semester configs =======

# short labels for the 4 core questions. these stay the same across semesters
# even though the exact question wording shifted slightly in 2024
CORE_QUESTIONS = [
    "Less motivated",
    "Excited to write",
    "Easier to write",
    "Important to future",
]

@dataclass(frozen=True)
class AttitudeItemConfig:
    code_letter:  str
    label_letter: str


@dataclass(frozen=True)
class SemesterConfig:
    sheet_name:   str
    scale_type:   str   # "likert" or "ynn"
    # core 4 questions in order matching CORE_QUESTIONS
    core_items:   Tuple[AttitudeItemConfig, ...]
    # "should learn" item, None if not asked this semester
    learn_item:   Optional[AttitudeItemConfig]


SPRING_2023 = SemesterConfig(
    sheet_name="Spring 2023",
    scale_type="likert",
    core_items=(
        AttitudeItemConfig("AD", "AE"),  # less motivated
        AttitudeItemConfig("AF", "AG"),  # excited
        AttitudeItemConfig("AH", "AI"),  # easier
        AttitudeItemConfig("AJ", "AK"),  # future
    ),
    learn_item=None,
)

FALL_2023 = SemesterConfig(
    sheet_name="Fall 2023",
    scale_type="likert",
    core_items=(
        AttitudeItemConfig("AD", "AE"),
        AttitudeItemConfig("AF", "AG"),
        AttitudeItemConfig("AH", "AI"),
        AttitudeItemConfig("AJ", "AK"),
    ),
    learn_item=None,
)

SPRING_2024 = SemesterConfig(
    sheet_name="Spring 2024",
    scale_type="ynn",
    core_items=(
        AttitudeItemConfig("BS", "BT"),  # less motivated
        AttitudeItemConfig("BU", "BV"),  # excited
        AttitudeItemConfig("BW", "BX"),  # easier
        AttitudeItemConfig("BY", "BZ"),  # future
    ),
    learn_item=AttitudeItemConfig("CA", "CB"),
)

FALL_2024 = SemesterConfig(
    sheet_name="Fall 2024",
    scale_type="ynn",
    core_items=(
        AttitudeItemConfig("BH", "BI"),  # less motivated
        AttitudeItemConfig("BJ", "BK"),  # excited
        AttitudeItemConfig("BL", "BM"),  # easier
        AttitudeItemConfig("BN", "BO"),  # future
    ),
    learn_item=AttitudeItemConfig("BP", "BQ"),
)

SEMESTERS = (SPRING_2023, FALL_2023, SPRING_2024, FALL_2024)


# ======= main =======

def run(xlsx_path: str) -> None:
    out_dir = Path("output") / VIZ_SLUG
    out_dir.mkdir(parents=True, exist_ok=True)

    # collectors for combined charts
    # trend_data[question_label] = [(semester_label, pct_yes), ...]
    trend_data = {q: [] for q in CORE_QUESTIONS}

    # for collapsed stacked/diverging: one row per question per semester
    # grouped by question so semesters appear together
    collapsed_by_question = {q: [] for q in CORE_QUESTIONS}

    # "should learn" results for the 2024 comparison
    learn_results = []

    for sem in SEMESTERS:
        sheet  = sem.sheet_name
        prefix = semester_prefix(sheet)
        df     = pd.read_excel(xlsx_path, sheet_name=sheet, dtype=str)

        # analyze the 4 core items
        core_results = []
        for qi, (q_label, item_cfg) in enumerate(zip(CORE_QUESTIONS, sem.core_items)):
            r = analyze_item(
                df, item_cfg.code_letter, item_cfg.label_letter,
                q_label, sem.scale_type,
            )
            core_results.append(r)

            # collect for trend dot plot - pct who agree/say yes
            denom = r.n_valid if r.n_valid > 0 else 1
            pct_yes = r.collapsed.get("Yes", 0) / denom * 100.0
            trend_data[q_label].append((sheet, pct_yes))

            # collect for collapsed stacked/diverging
            collapsed_by_question[q_label].append((sheet, r.collapsed.copy(), r.n_valid))

        # per-semester full distribution chart
        plot_semester_attitudes(
            core_results,
            f"{sheet}. Attitudes toward AI writing tools\n"
            f"({'5-point Likert (Uncertain as middle)' if sem.scale_type == 'likert' else 'Yes / No / Neutral'})",
            out_dir / f"{prefix}_attitudes.png",
        )

        # analyze "should learn" if present
        if sem.learn_item is not None:
            learn_r = analyze_item(
                df, sem.learn_item.code_letter, sem.learn_item.label_letter,
                "Should learn about AI", sem.scale_type,
            )
            learn_results.append((sheet, learn_r))

    # --- combined trend dot plot ---
    # note: "less motivated" is a negatively-worded item. a high "yes" means
    # students ARE less motivated, which is the opposite sentiment from the
    # other 3 questions. we dont flip it here - the chart shows what the data
    # says and the report can discuss the direction
    plot_trend_dotplot(
        trend_data,
        "Attitude trends across semesters (% who agree / say Yes)\n"
        "Note: 2023 Likert collapsed to Yes/No/Neutral for comparison\n"
        "\"Less motivated\" is reverse-coded (high = negative sentiment)",
        out_dir / "combined_trend_dotplot.png",
    )

    # --- combined collapsed stacked bars ---
    # build the row list grouped by question, semesters within each question
    # ordered chronologically
    all_collapsed = []
    for q_label in CORE_QUESTIONS:
        for sem_label, counts, n in collapsed_by_question[q_label]:
            row_label = f"{q_label} ({sem_label})"
            all_collapsed.append((row_label, counts, n))

    plot_collapsed_stacked(
        all_collapsed,
        "Attitudes toward AI writing tools (collapsed to Yes/No/Neutral)\n"
        "All semesters, grouped by question",
        out_dir / "combined_collapsed_stacked.png",
    )

    # --- vertical grouped stacked bars ---
    plot_vertical_grouped_stacked(
        collapsed_by_question,
        "Attitudes toward AI writing tools\n"
        "Vertical grouped stacked bars by category & semester",
        out_dir / "combined_vertical_grouped_stacked.png",
    )

    # --- combined collapsed diverging bars ---
    plot_collapsed_diverging(
        all_collapsed,
        "Attitudes toward AI writing tools (diverging)\n"
        "Centered on Neutral, all semesters grouped by question\n"
        "2023 Likert collapsed: SD+D → No, Uncertain → Neutral, A+SA → Yes",
        out_dir / "combined_collapsed_diverging.png",
    )

    # --- should learn comparison (2024 only) ---
    if learn_results:
        plot_should_learn(
            learn_results,
            "Should students learn about AI in composition courses?\n"
            "Spring 2024 vs Fall 2024",
            out_dir / "should_learn_comparison.png",
        )