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
#   per semester:
#     {prefix}_concordance.png              - concordance stacked bar (4 quadrants per use-case)
#     {prefix}_grouped_bar.png              - side-by-side: % allowed vs % who use
#     {prefix}_full_scatter.png             - scatter+OLS: all 11 permitted × all 9/11 freq items
#     {prefix}_full_density_heatmap.png     - 2d histogram heatmap of the full scatter data
#     {prefix}_matched_scatter.png          - scatter+OLS: all 11 permitted × 4 matched freq items only
#     {prefix}_matched_density_heatmap.png  - 2d histogram heatmap of the matched scatter data
#     {prefix}_matched_significance.png     - chi-squared test on the 4 matched pairs
#   combined:
#     combined_concordance.png              - averaged concordance across semesters
#     combined_grouped_bar.png              - averaged grouped bar across semesters
#     combined_full_scatter.png             - pooled scatter+OLS, 9 shared freq items (no translation/programming)
#     combined_full_density_heatmap.png     - pooled density heatmap, 9 shared freq items
#     combined_matched_scatter.png          - pooled scatter+OLS, 4 matched items on both axes
#     combined_matched_density_heatmap.png  - pooled density heatmap, 4 matched items

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
from scipy import stats


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


def compute_scores(df, permitted_code_col, freq_items, permitted_keys=None):
    # produces a per-respondent (permissiveness_score, usage_breadth) pair for the scatter.
    # permissiveness = how many of the permitted_keys the respondent selected.
    # usage = how many of the passed-in freq items they answered something other than never.
    # permitted_keys defaults to all 11 codes from PERMITTED_CODE_MAP, but callers can
    # pass a smaller set (e.g. just the 4 matched codes) to restrict the X axis too.
    if permitted_keys is None:
        permitted_keys = set(PERMITTED_CODE_MAP.keys())

    df2          = df.iloc[2:]
    perm_scores  = []
    usage_scores = []
    n_valid      = 0

    for _, row in df2.iterrows():
        perm_raw = row.get(permitted_code_col)
        if is_blank(perm_raw):
            continue
        selected   = parse_multiselect_codes(perm_raw)
        perm_score = len(selected.intersection(permitted_keys))

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
    ax.set_ylabel("Matched usage breadth\n(# of 4 matched AI use-cases answered != Never)")
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


def plot_density_heatmap(perm_scores, usage_scores, n_valid, max_usage_items, title, output_png, max_perm=11, matched_key=None):
    # 2d histogram as a heatmap. the scatter plot jitters dots to avoid overlap,
    # which means it hides where the actual clusters are. this fixes that by
    # binning into a grid where color = how many respondents land at each
    # (permissiveness, usage) coordinate. since both axes are already integers
    # we can just count exact (x, y) pairs instead of doing real binning.
    # max_perm controls the X axis ceiling — 11 for full, 4 for matched.
    # matched_key: if provided, draw a legend box listing the matched use-cases
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(title)

    if not perm_scores:
        plt.text(0.5, 0.5, "No responses", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    # build a count grid — both axes are integer-valued so no binning needed
    max_use   = max_usage_items
    grid      = np.zeros((max_use + 1, max_perm + 1))

    for p, u in zip(perm_scores, usage_scores):
        pi = int(round(p))
        ui = int(round(u))
        # clamp just in case something weird got through
        pi = max(0, min(pi, max_perm))
        ui = max(0, min(ui, max_use))
        grid[ui, pi] += 1

    # mask zeros so they show as blank/white instead of the lowest color
    masked = np.ma.masked_where(grid == 0, grid)

    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="white")

    im = ax.imshow(
        masked,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        extent=[-0.5, max_perm + 0.5, -0.5, max_use + 0.5],
        vmin=1,
    )

    # annotate cells that have people in them - helpful for small clusters
    for ui in range(max_use + 1):
        for pi in range(max_perm + 1):
            cnt = int(grid[ui, pi])
            if cnt > 0:
                color = "white" if cnt > grid.max() * 0.55 else "black"
                ax.text(pi, ui, str(cnt), ha="center", va="center",
                        fontsize=7, color=color)

    ax.set_xlabel(f"Permissiveness score (# of use-cases selected as allowed, out of {max_perm})")
    ax.set_ylabel(f"Usage breadth (# of {max_usage_items} items answered != Never)")
    ax.set_xticks(range(0, max_perm + 1))
    ax.set_yticks(range(0, max_use + 1))

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("# of respondents")

    # stick the n in the corner so its always visible
    ax.text(0.98, 0.02, f"n = {n_valid}", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # if this is a matched chart, show which 4 use-cases are being counted
    if matched_key:
        key_text = "Matched use-cases (each adds +1):\n"
        key_text += "  X: +1 if respondent permitted it\n"
        key_text += "  Y: +1 if respondent uses it (≠ Never)\n\n"
        for i, label in enumerate(matched_key, 1):
            key_text += f"  {i}. {label}\n"
        fig.text(
            0.98, 0.50, key_text.rstrip(),
            transform=fig.transFigure, ha="left", va="center",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.9),
        )
        # nudge the figure to make room for the key on the right
        fig.subplots_adjust(right=0.68)

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_scatter_regression(perm_scores, usage_scores, n_valid, title, output_png, max_perm=11, matched_key=None):
    # same data as the regular scatter but this one actually fits a line through
    # it and reports the full regression stats (slope, r, r^2, p-value). the
    # regular scatter only shows r in a text box - this makes the relationship
    # (or lack thereof) more visually obvious with the line drawn in.
    # max_perm controls the X axis ceiling — 11 for full, 4 for matched.
    # matched_key: if provided, draw a legend box listing the matched use-cases
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_title(title)

    if len(perm_scores) < 3:
        plt.text(0.5, 0.5, "No responses", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    px = np.array(perm_scores, dtype=float)
    uy = np.array(usage_scores, dtype=float)

    # jitter for the dots (same seed as the regular scatter so they match)
    rng = np.random.default_rng(42)
    jx  = rng.uniform(-0.25, 0.25, len(px))
    jy  = rng.uniform(-0.25, 0.25, len(uy))

    ax.scatter(px + jx, uy + jy, alpha=0.35, s=35, edgecolors="none", zorder=2)

    # OLS regression on the raw (unjittered) values
    slope, intercept, r_val, p_val, std_err = stats.linregress(px, uy)

    # draw the fit line across the full x range
    x_line = np.linspace(-0.5, max_perm + 0.5, 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color="#d73027", linewidth=2, linestyle="--",
            label="OLS fit", zorder=3)

    # stats box in the corner - put everything useful in one place
    # p < .001 gets printed as "< .001" since exact tiny floats arent helpful
    p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "< .0001"
    stat_text = (
        f"r = {r_val:.3f}\n"
        f"r² = {r_val**2:.3f}\n"
        f"slope = {slope:.3f}\n"
        f"p = {p_str}\n"
        f"n = {n_valid}"
    )
    ax.text(
        0.02, 0.98, stat_text,
        transform=ax.transAxes, va="top", fontsize=10, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85),
    )

    ax.set_xlabel(f"Permissiveness score (# allowed, out of {max_perm})")
    ax.set_ylabel("Usage breadth (# of AI items used)")
    ax.set_xlim(-0.5, max_perm + 0.5)
    max_use = int(max(usage_scores)) + 1
    ax.set_ylim(-0.5, max_use + 0.5)
    ax.set_xticks(range(0, max_perm + 1))
    ax.set_yticks(range(0, max_use + 1))
    ax.grid(True, alpha=0.2)
    ax.legend(loc="lower right", frameon=False)

    # if this is a matched chart, show which 4 use-cases are being counted
    if matched_key:
        key_text = "Matched use-cases (each adds +1):\n"
        key_text += "  X: +1 if respondent permitted it\n"
        key_text += "  Y: +1 if respondent uses it (≠ Never)\n\n"
        for i, label in enumerate(matched_key, 1):
            key_text += f"  {i}. {label}\n"
        fig.text(
            0.98, 0.50, key_text.rstrip(),
            transform=fig.transFigure, ha="left", va="center",
            fontsize=8, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.9),
        )
        fig.subplots_adjust(right=0.68)

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_significance(pair_results, title, output_png):
    # for each matched pair, build a 2x2 contingency table and run chi-squared.
    # the table is:
    #                    Uses    Doesnt use
    #   Doesnt allow  [  a         b      ]
    #   Allows        [  c         d      ]
    #
    # if belief and behavior are independent (null hypothesis), then the
    # proportion who use should be the same regardless of whether they allow it.
    # a significant result means beliefs DO predict behavior to some extent.
    #
    # also reports the specific observed vs expected count for the "doesnt allow
    # & uses anyway" cell, which is the one we actually care about most.
    output_png.parent.mkdir(parents=True, exist_ok=True)

    if not pair_results:
        fig = plt.figure(figsize=(12, 6))
        plt.title(title)
        plt.text(0.5, 0.5, "No responses", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    labels      = []
    chi2_vals   = []
    p_vals      = []
    obs_counts  = []   # observed "doesnt allow & uses" count
    exp_counts  = []   # expected "doesnt allow & uses" under independence
    ns          = []
    sig_markers = []   # *, **, ***, or ns

    for r in pair_results:
        # pull the four quadrant counts
        a = r.quadrant_counts.get("Doesn't allow & Uses anyway", 0)
        b = r.quadrant_counts.get("Doesn't allow & Doesn't use", 0)
        c = r.quadrant_counts.get("Allows & Uses", 0)
        d = r.quadrant_counts.get("Allows & Doesn't use", 0)

        table = np.array([[a, b], [c, d]])

        # need at least some data in every margin for chi2 to make sense.
        # cant use any(table.sum(axis=x) == 0) here because numpy returns an
        # array from the comparison, and python's any() chokes on that with a
        # "truth value of an array is ambiguous" error. so we check explicitly.
        row_sums = table.sum(axis=1)
        col_sums = table.sum(axis=0)
        if table.sum() < 5 or 0 in row_sums or 0 in col_sums:
            labels.append(r.label)
            chi2_vals.append(float("nan"))
            p_vals.append(float("nan"))
            obs_counts.append(a)
            exp_counts.append(float("nan"))
            ns.append(r.n_valid)
            sig_markers.append("n/a")
            continue

        chi2, p, dof, expected = stats.chi2_contingency(table, correction=True)

        # expected[0,0] corresponds to the "doesnt allow & uses" cell
        exp_a = expected[0, 0]

        labels.append(r.label)
        chi2_vals.append(chi2)
        p_vals.append(p)
        obs_counts.append(a)
        exp_counts.append(exp_a)
        ns.append(r.n_valid)

        # significance markers using standard thresholds
        if   p < 0.001: sig_markers.append("***")
        elif p < 0.01:  sig_markers.append("**")
        elif p < 0.05:  sig_markers.append("*")
        else:            sig_markers.append("ns")

    # plot as a table-style figure. tried making this a bar chart initially but
    # it was hard to read - a formatted text table is actually clearer here
    # since the interesting part is the numbers not the shape
    fig, ax = plt.subplots(figsize=(14, max(3.5, len(labels) * 0.8 + 3)))
    ax.set_title(title, pad=20)
    ax.axis("off")

    col_labels = [
        "Use case",
        "Obs.\n(disallow\n& use)",
        "Exp.\n(under\nindep.)",
        "Obs/Exp\nratio",
        "χ²",
        "p-value",
        "Sig.",
        "n",
    ]

    cell_text = []
    cell_colors = []
    for i in range(len(labels)):
        obs_v = obs_counts[i]
        exp_v = exp_counts[i]

        if np.isnan(exp_v) or exp_v == 0:
            ratio_str = "-"
        else:
            ratio_str = f"{obs_v / exp_v:.2f}x"

        chi2_str = f"{chi2_vals[i]:.2f}" if not np.isnan(chi2_vals[i]) else "-"

        if np.isnan(p_vals[i]):
            p_str = "-"
        elif p_vals[i] < 0.0001:
            p_str = "< .0001"
        else:
            p_str = f"{p_vals[i]:.4f}"

        exp_str = f"{exp_v:.1f}" if not np.isnan(exp_v) else "-"

        row = [
            labels[i],
            str(obs_v),
            exp_str,
            ratio_str,
            chi2_str,
            p_str,
            sig_markers[i],
            str(ns[i]),
        ]
        cell_text.append(row)

        # color the significance column for visual pop
        row_colors = ["white"] * len(col_labels)
        sig = sig_markers[i]
        if sig == "***":   row_colors[6] = "#fee0d2"
        elif sig == "**":  row_colors[6] = "#fee0d2"
        elif sig == "*":   row_colors[6] = "#fef0e6"
        else:              row_colors[6] = "#e8f4e8"
        cell_colors.append(row_colors)

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        colColours=["#d5e8f0"] * len(col_labels),
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)

    # make header row bold
    for j in range(len(col_labels)):
        tbl[0, j].set_text_props(fontweight="bold")

    # footnote explaining what the test means
    fig.text(
        0.5, 0.02,
        "χ² test of independence (Yates correction). H₀: belief and behavior are independent.\n"
        "Obs/Exp > 1 means MORE people use-despite-disallowing than chance predicts.\n"
        "* p < .05   ** p < .01   *** p < .001   ns = not significant",
        ha="center", va="bottom", fontsize=9, style="italic",
    )

    plt.tight_layout(rect=[0, 0.08, 1, 1])
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

    # collectors for combined scatter/heatmap - we concatenate respondent-level
    # scores across semesters rather than averaging, since the scatter needs
    # individual data points not summary stats
    all_full_perm    = []
    all_full_usage   = []
    all_matched_perm  = []
    all_matched_usage = []

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

        # use ALL frequency items for the scatter, not just the 4 matched pairs.
        # the x-axis (permissiveness) already covers all 11 permitted items, so
        # the y-axis should cover all frequency items too. the axes arent measuring
        # the exact same conceptual space this way (permitted has things like
        # "write full essays" that have no frequency counterpart, and frequency has
        # "understanding readings" that has no permitted counterpart) but thats fine -
        # the question is whether broadly permissive people are also broad users,
        # not whether specific item pairs line up. thats what the concordance is for.
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

        # scatter with OLS regression line overlaid - reports slope, r, r^2, p
        plot_scatter_regression(
            perm_sc, usage_sc, n_scatter,
            f"{sheet}. FULL permissiveness vs. usage breadth (with OLS fit)\n"
            f"X = all 11 permitted items, Y = all {len(freq_letters)} frequency items\n"
            f"N = {n_scatter}",
            out_dir / f"{prefix}_full_scatter.png",
        )

        # heatmap version of the scatter - shows where the actual density clusters
        # are, since the jittering in the scatter hides overlapping points
        max_use_items = len(freq_letters)
        plot_density_heatmap(
            perm_sc, usage_sc, n_scatter, max_use_items,
            f"{sheet}. FULL permissiveness vs. usage (density)\n"
            f"Cell value = # of respondents at that coordinate\n"
            f"X = all 11 permitted items, Y = all {len(freq_letters)} frequency items\n"
            f"N = {n_scatter}",
            out_dir / f"{prefix}_full_density_heatmap.png",
        )

        # for the combined full scatter we need both semesters on the same Y axis.
        # translation and programming only exist in fall, so the shared set is the
        # 9 items that appear in both semesters. for spring this is already what we
        # computed above; for fall we recompute without the last two items.
        if sem.is_fall:
            shared_freq_letters = freq_letters[:9]  # everything except translation + programming
            shared_freq_items = [
                (get_col_name_by_letter(df, n), get_col_name_by_letter(df, l))
                for n, l in shared_freq_letters
            ]
            shared_perm, shared_usage, _ = compute_scores(df, perm_code_col, shared_freq_items)
        else:
            # spring already uses 9 items
            shared_perm, shared_usage = perm_sc, usage_sc

        all_full_perm.extend(shared_perm)
        all_full_usage.extend(shared_usage)

        # --- matched-only versions of the scatter and heatmap ---
        # these restrict BOTH axes to just the 4 use-cases that have a direct
        # counterpart in both questions. the full versions above answer "are
        # broadly permissive people also broad users?" while these answer the
        # tighter question "does allowing these 4 specific tasks predict using
        # AI for those exact 4 tasks?"
        matched_freq_letters = []
        matched_perm_codes   = set()
        for mp in MATCHED_PAIRS:
            matched_perm_codes.add(mp.permitted_code)
            if sem.is_fall:
                matched_freq_letters.append((mp.freq_code_col_letter_fall, mp.freq_label_col_letter_fall))
            else:
                matched_freq_letters.append((mp.freq_code_col_letter, mp.freq_label_col_letter))

        matched_freq_items = [
            (get_col_name_by_letter(df, n), get_col_name_by_letter(df, l))
            for n, l in matched_freq_letters
        ]

        # restrict both axes - permitted_keys limits X to the 4 matched codes,
        # matched_freq_items limits Y to the 4 matched frequency items
        n_matched   = len(MATCHED_PAIRS)
        matched_key = [mp.label for mp in MATCHED_PAIRS]

        matched_perm_sc, matched_usage_sc, matched_n = compute_scores(
            df, perm_code_col, matched_freq_items, permitted_keys=matched_perm_codes,
        )

        all_matched_perm.extend(matched_perm_sc)
        all_matched_usage.extend(matched_usage_sc)

        plot_scatter_regression(
            matched_perm_sc, matched_usage_sc, matched_n,
            f"{sheet}. MATCHED permissiveness vs. usage breadth (with OLS fit)\n"
            f"Both axes restricted to {n_matched} matched use-cases\n"
            f"N = {matched_n}",
            out_dir / f"{prefix}_matched_scatter.png",
            max_perm=n_matched,
            matched_key=matched_key,
        )

        plot_density_heatmap(
            matched_perm_sc, matched_usage_sc, matched_n, n_matched,
            f"{sheet}. MATCHED permissiveness vs. usage (density)\n"
            f"Cell value = # of respondents at that coordinate\n"
            f"Both axes restricted to {n_matched} matched use-cases\n"
            f"N = {matched_n}",
            out_dir / f"{prefix}_matched_density_heatmap.png",
            max_perm=n_matched,
            matched_key=matched_key,
        )

        # chi-squared test per matched pair - is the "disallows but uses" group
        # bigger than youd expect under independence? this is inherently a
        # matched-only analysis since it needs the 1:1 pair between permitted
        # items and frequency items
        plot_significance(
            pair_results,
            f"{sheet}. Statistical significance of belief-behavior mismatch (MATCHED)\n"
            f"Chi-squared test: is usage independent of stated beliefs?",
            out_dir / f"{prefix}_matched_significance.png",
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

    # combined scatter and heatmap charts - concatenate respondent-level data
    # across semesters. for the full version we exclude translation + programming
    # since those only exist in fall and averaging one semester isnt meaningful.
    n_combined_full = len(all_full_perm)
    if n_combined_full >= 3:
        plot_scatter_regression(
            all_full_perm, all_full_usage, n_combined_full,
            f"Combined. FULL permissiveness vs. usage breadth (with OLS fit)\n"
            f"X = all 11 permitted items, Y = 9 shared frequency items (no translation/programming)\n"
            f"All respondents pooled, N = {n_combined_full}",
            out_dir / "combined_full_scatter.png",
        )

        plot_density_heatmap(
            all_full_perm, all_full_usage, n_combined_full, 9,
            f"Combined. FULL permissiveness vs. usage (density)\n"
            f"X = all 11 permitted items, Y = 9 shared frequency items (no translation/programming)\n"
            f"All respondents pooled, N = {n_combined_full}",
            out_dir / "combined_full_density_heatmap.png",
        )

    n_combined_matched = len(all_matched_perm)
    n_matched = len(MATCHED_PAIRS)
    matched_key = [mp.label for mp in MATCHED_PAIRS]
    if n_combined_matched >= 3:
        plot_scatter_regression(
            all_matched_perm, all_matched_usage, n_combined_matched,
            f"Combined. MATCHED permissiveness vs. usage breadth (with OLS fit)\n"
            f"Both axes restricted to {n_matched} matched use-cases\n"
            f"All respondents pooled, N = {n_combined_matched}",
            out_dir / "combined_matched_scatter.png",
            max_perm=n_matched,
            matched_key=matched_key,
        )

        plot_density_heatmap(
            all_matched_perm, all_matched_usage, n_combined_matched, n_matched,
            f"Combined. MATCHED permissiveness vs. usage (density)\n"
            f"Both axes restricted to {n_matched} matched use-cases\n"
            f"All respondents pooled, N = {n_combined_matched}",
            out_dir / "combined_matched_density_heatmap.png",
            max_perm=n_matched,
            matched_key=matched_key,
        )