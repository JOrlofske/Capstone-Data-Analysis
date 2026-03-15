# visualizations/beliefVsBehavior.py
#
# the core question here: do students actually follow their own stated beliefs
# about what should be allowed? someone who says brainstorming with AI is fine
# but never uses it would land in "allows & doesnt use". someone who says its
# not allowed but uses it anyway lands in "doesnt allow & uses anyway" (the
# interesting quadrant).
#
# only spring 2024 and fall 2024 have both the permitted-usage select-all question
# AND the per-use-case frequency questions, so thats all we can do this with.
#
# outputs go in output/belief_vs_behavior/

from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


VIZ_NAME = "Belief vs. behavior: Do students practice what they preach? (Spring 2024-Fall 2024)"
VIZ_SLUG = "belief_vs_behavior"
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


def normalize_freq_label(s):
    if is_blank(s):
        return None
    txt = str(s).strip().lower()
    mapping = {"often": "Often", "sometimes": "Sometimes", "rarely": "Rarely", "never": "Never"}
    return mapping.get(txt)


def parse_multiselect_codes(cell):
    # returns a set of string codes for fast membership check against PERMITTED_CODE_MAP keys
    if is_blank(cell):
        return set()
    return {c.strip() for c in str(cell).split(",") if c.strip()}


RAW_CODE_TO_LABEL = {1: "Often", 2: "Sometimes", 3: "Rarely", 4: "Never"}


# ======= permitted usage map and matched pairs =======

# same code->label mapping as permittedUsage.py, both semesters use it
PERMITTED_CODE_MAP = {
    "1":  "Check and correct spelling",
    "2":  "Check and correct grammar",
    "3":  "Brainstorm ideas",
    "4":  "Produce outlines or summaries of texts",
    "5":  "Improve writing style",
    "6":  "Write full sentences",
    "7":  "Expand word choice (thesaurus)",
    "8":  "Complete sentences",
    "9":  "Write full paragraphs",
    "10": "Write full essays",
    "11": "Generate counter-arguments",
}


@dataclass(frozen=True)
class MatchedPair:
    label:                      str  # readable label for charts
    permitted_code:             str  # the code string in the select-all column
    freq_code_col_letter:       str  # spring 2024 num col
    freq_label_col_letter:      str  # spring 2024 label col
    freq_code_col_letter_fall:  str  # fall 2024 num col
    freq_label_col_letter_fall: str  # fall 2024 label col


# only 4 use cases have a direct equivalent on both the permitted and frequency sides.
# other use cases exist in frequency but dont have a clean match in the permitted map
MATCHED_PAIRS = (
    MatchedPair(
        label="Brainstorming",
        permitted_code="3",
        freq_code_col_letter="AY", freq_label_col_letter="AZ",
        freq_code_col_letter_fall="AJ", freq_label_col_letter_fall="AK",
    ),
    MatchedPair(
        label="Grammar edits",
        permitted_code="2",
        freq_code_col_letter="BK", freq_label_col_letter="BL",
        freq_code_col_letter_fall="AV", freq_label_col_letter_fall="AW",
    ),
    MatchedPair(
        label="Outlining / summaries",
        permitted_code="4",
        freq_code_col_letter="BG", freq_label_col_letter="BH",
        freq_code_col_letter_fall="AR", freq_label_col_letter_fall="AS",
    ),
    MatchedPair(
        label="Rewording / style",
        permitted_code="5",
        freq_code_col_letter="BI", freq_label_col_letter="BJ",
        freq_code_col_letter_fall="AT", freq_label_col_letter_fall="AU",
    ),
)


# ======= quadrant labels and colors =======

QUADRANT_ORDER = [
    "Doesn't allow & Doesn't use",
    "Doesn't allow & Uses anyway",
    "Allows & Doesn't use",
    "Allows & Uses",
]

QUADRANT_COLORS = {
    "Doesn't allow & Doesn't use": "#91bfdb",  # consistent non-adopter
    "Doesn't allow & Uses anyway":  "#d73027",  # the interesting one - uses despite not believing its ok
    "Allows & Doesn't use":         "#fc8d59",  # permissive but not acting on it
    "Allows & Uses":                "#4575b4",  # consistent adopter
}


# ======= per-respondent classification =======

@dataclass
class PairResult:
    label:           str
    quadrant_counts: Counter  # quadrant label -> count
    n_valid:         int      # respondents with both answers present
    pct_allows:      float    # % who selected this in the permitted question
    pct_uses:        float    # % who report any use (not never)


def classify_pair(df, permitted_code_col, permitted_code, freq_num_col, freq_label_col, label):
    df2      = df.iloc[2:]
    counts   = Counter()
    n_valid  = 0
    n_allows = 0
    n_uses   = 0

    for _, row in df2.iterrows():
        perm_raw = row.get(permitted_code_col)
        if is_blank(perm_raw):
            continue
        selected = parse_multiselect_codes(perm_raw)
        allows   = permitted_code in selected

        freq_label = normalize_freq_label(row.get(freq_label_col))
        if freq_label is None:
            code = parse_int_code(row.get(freq_num_col))
            freq_label = RAW_CODE_TO_LABEL.get(code) # pyright: ignore[reportArgumentType]
        if freq_label is None:
            continue

        uses = freq_label != "Never"
        n_valid += 1

        if allows:
            n_allows += 1
        if uses:
            n_uses += 1

        if allows and uses:
            counts["Allows & Uses"] += 1
        elif allows and not uses:
            counts["Allows & Doesn't use"] += 1
        elif not allows and uses:
            counts["Doesn't allow & Uses anyway"] += 1
        else:
            counts["Doesn't allow & Doesn't use"] += 1

    denom = n_valid if n_valid > 0 else 1
    return PairResult(
        label=label,
        quadrant_counts=counts,
        n_valid=n_valid,
        pct_allows=n_allows / denom * 100.0,
        pct_uses=n_uses   / denom * 100.0,
    )


def compute_scores(df, permitted_code_col, freq_items):
    # produces a per-respondent (permissiveness_score, usage_breadth) pair for the scatter.
    # permissiveness = how many of the 11 permitted items they selected
    # usage = how many freq items they answered something other than never
    df2          = df.iloc[2:]
    perm_scores  = []
    usage_scores = []
    n_valid      = 0

    for _, row in df2.iterrows():
        perm_raw = row.get(permitted_code_col)
        if is_blank(perm_raw):
            continue
        selected   = parse_multiselect_codes(perm_raw)
        perm_score = len(selected.intersection(PERMITTED_CODE_MAP.keys()))

        n_answered = 0
        n_uses     = 0
        for num_col, label_col in freq_items:
            freq_label = normalize_freq_label(row.get(label_col))
            if freq_label is None:
                code = parse_int_code(row.get(num_col))
                freq_label = RAW_CODE_TO_LABEL.get(code) # pyright: ignore[reportArgumentType]
            if freq_label is None:
                continue
            n_answered += 1
            if freq_label != "Never":
                n_uses += 1

        if n_answered == 0:
            continue

        n_valid += 1
        perm_scores.append(perm_score)
        usage_scores.append(n_uses)

    return perm_scores, usage_scores, n_valid


# ======= plot functions =======

def plot_concordance(pair_results, title, output_png):
    output_png.parent.mkdir(parents=True, exist_ok=True)

    labels = [r.label for r in pair_results]

    fig = plt.figure(figsize=(12, max(5, len(labels) * 1.0 + 2)))
    plt.title(title)
    ax   = plt.gca()
    left = [0.0] * len(labels)

    for quad in QUADRANT_ORDER:
        vals = []
        for r in pair_results:
            denom = r.n_valid if r.n_valid > 0 else 1
            vals.append(r.quadrant_counts.get(quad, 0) / denom * 100.0)

        bars = ax.barh(labels, vals, left=left, label=quad, color=QUADRANT_COLORS[quad])

        # only annotate segments that are wide enough to actually read
        for i, (bar, pct) in enumerate(zip(bars, vals)):
            if pct >= 5:
                x_center   = left[i] + pct / 2
                text_color = "white" if quad in ("Allows & Uses", "Doesn't allow & Uses anyway") else "black"
                ax.text(x_center, i, f"{pct:.0f}%",
                        ha="center", va="center", fontsize=9, color=text_color)

        left = [l + v for l, v in zip(left, vals)]

    for i, r in enumerate(pair_results):
        ax.text(101, i, f"n={r.n_valid}", va="center", fontsize=9)

    ax.set_xlabel("% of respondents")
    ax.set_xlim(0, 100)
    ax.legend(title="Quadrant", loc="center left", bbox_to_anchor=(1.08, 0.5), frameon=False)

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_grouped_bar(pair_results, title, output_png):
    output_png.parent.mkdir(parents=True, exist_ok=True)

    labels     = [r.label for r in pair_results]
    pct_allows = [r.pct_allows for r in pair_results]
    pct_uses   = [r.pct_uses   for r in pair_results]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title)

    bars1 = ax.bar(x - width / 2, pct_allows, width,
                   label="% who say it should be allowed", color="#4575b4")
    bars2 = ax.bar(x + width / 2, pct_uses,   width,
                   label="% who actually use AI for this", color="#d73027")

    for bar, val in zip(bars1, pct_allows):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=9)
    for bar, val in zip(bars2, pct_uses):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("% of respondents")
    ax.set_ylim(0, max(max(pct_allows), max(pct_uses)) * 1.2)
    ax.legend(loc="upper right", frameon=False)

    for i, r in enumerate(pair_results):
        ax.text(i, -0.08, f"n={r.n_valid}", ha="center", va="top",
                fontsize=8, transform=ax.get_xaxis_transform())

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_scatter(perm_scores, usage_scores, n_valid, title, output_png):
    # keeping the empty guard here because np.corrcoef will blow up on empty input
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_title(title)

    if not perm_scores:
        plt.text(0.5, 0.5, "No responses", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    # jitter so overlapping dots dont stack into a solid blob
    rng = np.random.default_rng(42)
    jx  = rng.uniform(-0.25, 0.25, len(perm_scores))
    jy  = rng.uniform(-0.25, 0.25, len(usage_scores))

    ax.scatter(
        np.array(perm_scores) + jx,
        np.array(usage_scores) + jy,
        alpha=0.45, s=40, edgecolors="none",
    )

    ax.set_xlabel("Permissiveness score\n(# of use-cases selected as should be allowed, out of 11)")
    ax.set_ylabel("Usage breadth\n(# of AI use-case items answered not Never)")
    ax.set_xlim(-0.5, 11.5)
    max_use = int(max(usage_scores)) + 1 if usage_scores else 10
    ax.set_ylim(-0.5, max_use + 0.5)
    ax.set_xticks(range(0, 12))
    ax.set_yticks(range(0, max_use + 1))
    ax.grid(True, alpha=0.2)

    if len(perm_scores) >= 3:
        corr = np.corrcoef(perm_scores, usage_scores)[0, 1]
        ax.text(
            0.02, 0.98,
            f"r = {corr:.2f}   n = {n_valid}",
            transform=ax.transAxes, va="top", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ======= averaging helper =======

def average_pair_results(all_results):
    # same approach as the other files - average the percentages across semesters,
    # then back-compute fake counts so we can reuse the same plot functions
    by_label   = defaultdict(list)
    for sem_res in all_results:
        for r in sem_res:
            by_label[r.label].append(r)

    # preserve the order from the first semester
    label_order = [r.label for r in all_results[0]]
    averaged    = []

    for label in label_order:
        entries = by_label[label]
        total_n = sum(r.n_valid for r in entries)

        avg_counts = Counter()
        for quad in QUADRANT_ORDER:
            pcts = []
            for r in entries:
                denom = r.n_valid if r.n_valid > 0 else 1
                pcts.append(r.quadrant_counts.get(quad, 0) / denom * 100.0)
            avg_pct          = sum(pcts) / len(pcts)
            avg_counts[quad] = avg_pct * total_n / 100.0 # pyright: ignore[reportArgumentType]

        averaged.append(PairResult(
            label=label,
            quadrant_counts=avg_counts,
            n_valid=total_n,
            pct_allows=sum(r.pct_allows for r in entries) / len(entries),
            pct_uses=  sum(r.pct_uses   for r in entries) / len(entries),
        ))

    return averaged


# ======= semester configs =======

@dataclass(frozen=True)
class SemesterConfig:
    sheet_name:                  str
    permitted_code_col_letter:   str
    permitted_label_col_letter:  str
    is_fall:                     bool  # determines which freq col letters to pull from MatchedPair


SPRING_2024 = SemesterConfig("Spring 2024", "AO", "AP", is_fall=False)
FALL_2024   = SemesterConfig("Fall 2024",   "AB", "AC", is_fall=True)

SEMESTERS = (SPRING_2024, FALL_2024)


# ======= main =======

def run(xlsx_path: str) -> None:
    out_dir = Path("output") / VIZ_SLUG
    out_dir.mkdir(parents=True, exist_ok=True)

    all_pair_results = []

    for sem in SEMESTERS:
        sheet  = sem.sheet_name
        prefix = semester_prefix(sheet)

        df            = pd.read_excel(xlsx_path, sheet_name=sheet, dtype=str)
        perm_code_col = get_col_name_by_letter(df, sem.permitted_code_col_letter)

        pair_results = []
        for mp in MATCHED_PAIRS:
            if sem.is_fall:
                freq_num = get_col_name_by_letter(df, mp.freq_code_col_letter_fall)
                freq_lbl = get_col_name_by_letter(df, mp.freq_label_col_letter_fall)
            else:
                freq_num = get_col_name_by_letter(df, mp.freq_code_col_letter)
                freq_lbl = get_col_name_by_letter(df, mp.freq_label_col_letter)

            r = classify_pair(df, perm_code_col, mp.permitted_code, freq_num, freq_lbl, mp.label)
            pair_results.append(r)

        all_pair_results.append(pair_results)

        resp_n = pair_results[0].n_valid if pair_results else 0

        plot_concordance(
            pair_results,
            f"{sheet}. Belief vs. behavior concordance\n"
            f"Per matched use-case: does should be allowed match actual usage?\n"
            f"N approx {resp_n}",
            out_dir / f"{prefix}_concordance.png",
        )

        plot_grouped_bar(
            pair_results,
            f"{sheet}. % who allow vs. % who use (matched use-cases)\nN approx {resp_n}",
            out_dir / f"{prefix}_grouped_bar.png",
        )

        # build the full list of freq item cols for the scatter - includes items
        # beyond the 4 matched pairs so the usage breadth score covers everything
        if sem.is_fall:
            freq_letters = [
                ("AJ", "AK"),  # brainstorming
                ("AL", "AM"),  # readings
                ("AN", "AO"),  # concepts
                ("AP", "AQ"),  # research
                ("AR", "AS"),  # outlining
                ("AT", "AU"),  # rewording
                ("AV", "AW"),  # grammar
                ("AX", "AY"),  # bibliography
                ("AZ", "BA"),  # lowstakes
                ("BB", "BC"),  # translation
                ("BD", "BE"),  # programming
            ]
        else:
            freq_letters = [
                ("AY", "AZ"),  # brainstorming
                ("BA", "BB"),  # readings
                ("BC", "BD"),  # concepts
                ("BE", "BF"),  # research
                ("BG", "BH"),  # outlining
                ("BI", "BJ"),  # rewording
                ("BK", "BL"),  # grammar
                ("BM", "BN"),  # bibliography
                ("BO", "BP"),  # lowstakes
            ]

        freq_items = [
            (get_col_name_by_letter(df, n), get_col_name_by_letter(df, l))
            for n, l in freq_letters
        ]

        perm_sc, usage_sc, n_scatter = compute_scores(df, perm_code_col, freq_items)

        plot_scatter(
            perm_sc, usage_sc, n_scatter,
            f"{sheet}. Permissiveness vs. usage breadth\n"
            f"Each dot = one respondent   (jittered to reduce overlap)\n"
            f"N = {n_scatter}",
            out_dir / f"{prefix}_scatter.png",
        )

    # combined averaged charts
    if len(all_pair_results) >= 2:
        averaged  = average_pair_results(all_pair_results)
        approx_n  = averaged[0].n_valid if averaged else 0
        sem_names = " + ".join(sem.sheet_name for sem in SEMESTERS)

        plot_concordance(
            averaged,
            f"Combined average. Belief vs. behavior concordance\n"
            f"Mean of per-semester percentages ({sem_names})\n"
            f"Total respondents approx {approx_n}",
            out_dir / "combined_concordance.png",
        )

        plot_grouped_bar(
            averaged,
            f"Combined average. % who allow vs. % who use\n"
            f"Mean of per-semester percentages ({sem_names})\n"
            f"Total respondents approx {approx_n}",
            out_dir / "combined_grouped_bar.png",
        )