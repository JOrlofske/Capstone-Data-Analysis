# visualizations/aiUseCaseFrequency.py
#
# how often are students using AI for specific writing tasks? this only covers
# spring 2024 and fall 2024 because earlier semesters didnt ask per-use-case
# frequency questions at all - just whether they used AI at all.
#
# spring 2024 has 9 use cases, fall 2024 added translation and programming on top.
# that asymmetry shows up in the dot plot where those two items only have one point.
#
# outputs go in output/ai_usecase_frequency/

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


VIZ_NAME = "AI use-case frequency comparison (Spring 2024-Fall 2024)"
VIZ_SLUG = "ai_usecase_frequency"
SHOW_PLOTS = False


# ======= response scale =======

# codes come in as 1-4, but 1=Often and 4=Never which is backwards for intuitive reading.
# keeping the raw order here because thats what the spreadsheet actually stores
FREQ_LABELS_RAW_ORDER = ["Often", "Sometimes", "Rarely", "Never"]

# for all the charts we flip to low->high so never is on the left
FREQ_LABELS_PLOT_ORDER = ["Never", "Rarely", "Sometimes", "Often"]

RAW_CODE_TO_LABEL = {
    1: "Often",
    2: "Sometimes",
    3: "Rarely",
    4: "Never",
}

# inverted score so higher number = more frequent use, makes the mean more intuitive
# (1.2 = barely anyone uses it, 3.8 = almost everyone uses it often)
INVERTED_SCORE = {
    1: 4,  # Often
    2: 3,  # Sometimes
    3: 2,  # Rarely
    4: 1,  # Never
}


# ======= use case labels and ordering =======

USE_CASE_LABELS = {
    "brainstorming": "Brainstorming / developing ideas",
    "readings":      "Understanding readings",
    "concepts":      "Understanding class concepts",
    "research":      "Research / gathering sources",
    "outlining":     "Outlining / early drafts",
    "rewording":     "Rewording for tone or style",
    "grammar":       "Final grammar edits",
    "bibliography":  "Formatting bibliographies",
    "lowstakes":     "Low-stakes short writing",
    "translation":   "Translation between languages",
    "programming":   "Computer programming",
}

# ordered roughly from idea-level help on the left to production-level output on the right.
# this gives the dot plot a shape thats easier to read than alphabetical
USE_CASE_ORDER = [
    "brainstorming",
    "readings",
    "concepts",
    "research",
    "outlining",
    "rewording",
    "grammar",
    "bibliography",
    "lowstakes",
    "translation",
    "programming",
]


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
    # turns spring 2024 into 2024spring etc
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
        if not ("A" <= ch <= "Z"):
            raise ValueError(f"Invalid Excel column letter: {col}")
        n = n * 26 + (ord(ch) - ord("A") + 1)
    return n - 1


def get_col_name_by_letter(df, col_letter):
    idx = col_letter_to_index(col_letter)
    return str(df.columns[idx])


def parse_int_code(cell):
    if is_blank(cell):
        return None
    try:
        return int(float(str(cell).strip()))
    except Exception:
        return None


def normalize_freq_label(s):
    # just lowercases and looks up - the labels in the sheet are pretty clean
    if is_blank(s):
        return None
    txt = str(s).strip().lower()
    mapping = {
        "often":     "Often",
        "sometimes": "Sometimes",
        "rarely":    "Rarely",
        "never":     "Never",
    }
    return mapping.get(txt)


def compute_respondent_n(df, cols):
    # skip rows 0 and 1 (metadata), count rows not entirely blank across given cols
    df2  = df.iloc[2:]
    mask = None
    for c in cols:
        b    = df2[c].apply(is_blank)
        mask = b if mask is None else (mask & b)
    return int((~mask).sum()) if mask is not None else 0


# ======= per-item analysis =======

@dataclass(frozen=True)
class UseCaseResult:
    key:           str      # canonical key e.g. brainstorming
    label:         str      # readable label for charts
    counts:        Counter  # label -> count
    n_nonblank:    int      # how many actually answered this item
    mean_inverted: float    # 1-4 where 4=Often
    pct_any_use:   float    # % who said anything other than never


def analyze_single_item(df, num_col, label_col, key, label):
    df2        = df.iloc[2:]
    counts     = Counter()
    scores     = []
    n_nonblank = 0

    for _, row in df2.iterrows():
        num_code   = parse_int_code(row.get(num_col))
        label_text = normalize_freq_label(row.get(label_col))

        resolved = None
        if label_text is not None:
            resolved = label_text
        elif num_code in RAW_CODE_TO_LABEL:
            resolved = RAW_CODE_TO_LABEL[num_code]

        if resolved is None:
            continue

        n_nonblank += 1
        counts[resolved] += 1

        # build the inverted score for mean calculation - prefer the numeric code
        # since label_text can sometimes be ambiguous
        if num_code in INVERTED_SCORE:
            scores.append(INVERTED_SCORE[num_code])
        else:
            lbl_to_inv = {"Never": 1, "Rarely": 2, "Sometimes": 3, "Often": 4}
            if resolved in lbl_to_inv:
                scores.append(lbl_to_inv[resolved])

    mean_inv  = sum(scores) / len(scores) if scores else 0.0
    n_not_never = sum(v for k, v in counts.items() if k != "Never")
    pct_any   = n_not_never / n_nonblank * 100.0 if n_nonblank > 0 else 0.0

    return UseCaseResult(
        key=key,
        label=label,
        counts=counts,
        n_nonblank=n_nonblank,
        mean_inverted=mean_inv,
        pct_any_use=pct_any,
    )


# ======= plot functions =======

def plot_mean_bars(results, title, output_png):
    output_png.parent.mkdir(parents=True, exist_ok=True)

    items  = sorted(results, key=lambda r: r.mean_inverted)
    labels = [r.label for r in items]
    means  = [r.mean_inverted for r in items]

    fig = plt.figure(figsize=(13, 7))
    plt.title(title)
    ax = plt.gca()
    ax.barh(labels, means)
    ax.set_xlabel("Mean frequency score (1=Never  2=Rarely  3=Sometimes  4=Often)")
    ax.set_xlim(0.8, 4.4)

    pad = 0.03
    for i, r in enumerate(items):
        ax.text(r.mean_inverted + pad, i, f"{r.mean_inverted:.2f}  (n={r.n_nonblank})", va="center")

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_stacked_bars(results, title, output_png):
    output_png.parent.mkdir(parents=True, exist_ok=True)

    items  = sorted(results, key=lambda r: r.mean_inverted)
    labels = [r.label for r in items]

    fig = plt.figure(figsize=(13, 7))
    plt.title(title)
    ax   = plt.gca()
    left = [0.0] * len(items)

    for bucket in FREQ_LABELS_PLOT_ORDER:
        vals = []
        for r in items:
            denom = r.n_nonblank if r.n_nonblank > 0 else 1
            vals.append(r.counts.get(bucket, 0) / denom * 100.0)
        ax.barh(labels, vals, left=left, label=bucket)
        left = [l + v for l, v in zip(left, vals)]

    for i, r in enumerate(items):
        ax.text(101, i, f"n={r.n_nonblank}", va="center")

    ax.set_xlabel("Percent of responses")
    ax.set_xlim(0, 100)
    ax.legend(title="Frequency", loc="center left", bbox_to_anchor=(1.10, 0.5), frameon=False)

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_dot_plot(series, title, output_png, n_by_semester=None):
    output_png.parent.mkdir(parents=True, exist_ok=True)

    # only include use cases that appear in at least one semester
    all_keys = []
    for key in USE_CASE_ORDER:
        for _, vals in series:
            if key in vals:
                if key not in all_keys:
                    all_keys.append(key)
                break

    axis_labels = [USE_CASE_LABELS[k] for k in all_keys]
    x = list(range(len(all_keys)))

    fig = plt.figure(figsize=(14, 7))
    plt.title(title)
    ax = plt.gca()

    for sem_label, values in series:
        y = [values.get(k, float("nan")) for k in all_keys]
        ax.plot(x, y, marker="o", linestyle="None", markersize=8, label=sem_label)

    ax.set_xticks(x)
    ax.set_xticklabels(axis_labels, rotation=35, ha="right")
    ax.set_ylabel("% who use AI for this task (answered Often, Sometimes, or Rarely)")
    ax.set_ylim(-2, 102)
    ax.grid(True, axis="y", alpha=0.25)

    legend1 = ax.legend(title="Semester", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

    if n_by_semester:
        from matplotlib.lines import Line2D

        ordered = [sl for sl, _ in series if sl in n_by_semester]
        handles2 = [
            Line2D([0], [0], linestyle="none", marker="", color="none",
                   label=f"{sl}: n={n_by_semester[sl]}")
            for sl in ordered
        ]
        ax.legend(handles=handles2, title="Sample size", loc="lower left",
                  bbox_to_anchor=(1.02, 0.0), frameon=False, handlelength=0, handletextpad=0.0)
        # gotta re-add legend1 or matplotlib clobbers it
        ax.add_artist(legend1)

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_heatmap(results, title, output_png):
    # using imshow so we need actual data - keep the empty guard here
    output_png.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        fig = plt.figure(figsize=(10, 6))
        plt.title(title)
        plt.text(0.5, 0.5, "No responses", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    items      = sorted(results, key=lambda r: r.mean_inverted, reverse=True)
    row_labels = [r.label for r in items]
    col_labels = FREQ_LABELS_PLOT_ORDER

    data = []
    for r in items:
        denom = r.n_nonblank if r.n_nonblank > 0 else 1
        data.append([r.counts.get(lbl, 0) / denom * 100.0 for lbl in col_labels])

    data_arr = np.array(data)

    fig, ax = plt.subplots(figsize=(10, max(5, len(items) * 0.55 + 2)))
    ax.set_title(title)

    im = ax.imshow(data_arr, cmap=plt.cm.YlOrRd, aspect="auto", vmin=0, vmax=100)  # pyright: ignore[reportAttributeAccessIssue]
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = data_arr[i, j]
            ax.text(j, i, f"{val:.0f}%", ha="center", va="center",
                    color="white" if val > 50 else "black", fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("% of respondents")

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_diverging_stacked(results, title, output_png):
    output_png.parent.mkdir(parents=True, exist_ok=True)

    items  = sorted(results, key=lambda r: r.mean_inverted)
    labels = [r.label for r in items]

    pct = {lbl: [] for lbl in FREQ_LABELS_PLOT_ORDER}
    for r in items:
        denom = r.n_nonblank if r.n_nonblank > 0 else 1
        for lbl in FREQ_LABELS_PLOT_ORDER:
            pct[lbl].append(r.counts.get(lbl, 0) / denom * 100.0)

    # diverging layout: never/rarely go left (negative), sometimes/often go right (positive).
    # rarely anchors at 0 on the left and never extends further left from there
    colors = {
        "Never":     "#d73027",
        "Rarely":    "#fc8d59",
        "Sometimes": "#91bfdb",
        "Often":     "#4575b4",
    }

    y_pos      = range(len(labels))
    never_vals = pct["Never"]
    rarely_vals  = pct["Rarely"]

    rarely_left = [-r for r in rarely_vals]
    ax_left = [-r for r in rarely_vals]
    ax_left2 = [-(n + r) for n, r in zip(never_vals, rarely_vals)]

    fig = plt.figure(figsize=(13, max(5, len(items) * 0.5 + 2)))
    plt.title(title)
    ax = plt.gca()

    ax.barh(y_pos, [-r for r in rarely_vals],  left=[0]*len(labels),  color=colors["Rarely"],    label="Rarely")
    ax.barh(y_pos, [-n for n in never_vals],   left=rarely_left,       color=colors["Never"],     label="Never")
    ax.barh(y_pos, pct["Sometimes"],           left=[0]*len(labels),   color=colors["Sometimes"], label="Sometimes")
    ax.barh(y_pos, pct["Often"],               left=pct["Sometimes"],  color=colors["Often"],     label="Often")

    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels)
    ax.set_xlabel("<- Less frequent use          % of responses          More frequent use ->")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="-")

    handles_ordered = [
        matplotlib.patches.Patch(color=colors[lbl], label=lbl)  # pyright: ignore[reportAttributeAccessIssue]
        for lbl in FREQ_LABELS_PLOT_ORDER
    ]
    ax.legend(handles=handles_ordered, title="Frequency", loc="center left",
              bbox_to_anchor=(1.02, 0.5), frameon=False)

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_native_translation_heatmap(df, native_label_col, translation_num_col,
                                    translation_label_col, title, output_png):
    # cross-tab: native english speaker (yes/no) x how often they use AI for translation.
    # only fall 2024 has the native language question so this is a one-off
    output_png.parent.mkdir(parents=True, exist_ok=True)

    df2       = df.iloc[2:].copy()
    row_data  = []

    for _, row in df2.iterrows():
        native_raw = row.get(native_label_col)
        if is_blank(native_raw):
            continue
        native = str(native_raw).strip()
        if native not in ("Yes", "No"):
            continue

        freq_label = normalize_freq_label(row.get(translation_label_col))
        if freq_label is None:
            code = parse_int_code(row.get(translation_num_col))
            freq_label = RAW_CODE_TO_LABEL.get(code) # pyright: ignore[reportArgumentType]
        if freq_label is None:
            continue

        row_data.append((native, freq_label))

    if not row_data:
        fig = plt.figure(figsize=(8, 4))
        plt.title(title)
        plt.text(0.5, 0.5, "No responses", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    row_labels    = ["Yes", "No"]
    col_labels    = FREQ_LABELS_PLOT_ORDER
    grp_counts    = {g: Counter() for g in row_labels}
    grp_totals    = {g: 0 for g in row_labels}

    for native, freq in row_data:
        grp_counts[native][freq] += 1
        grp_totals[native] += 1

    data  = []
    annot = []
    for g in row_labels:
        total    = grp_totals[g] if grp_totals[g] > 0 else 1
        row_pct  = []
        row_ann  = []
        for lbl in col_labels:
            cnt = grp_counts[g].get(lbl, 0)
            pct = cnt / total * 100.0
            row_pct.append(pct)
            row_ann.append(f"{pct:.0f}%\n({cnt}/{grp_totals[g]})")
        data.append(row_pct)
        annot.append(row_ann)

    data_arr = np.array(data)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_title(title)

    im = ax.imshow(data_arr, cmap=plt.cm.YlOrRd, aspect="auto", vmin=0, vmax=100) # pyright: ignore[reportAttributeAccessIssue]
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels([f"Native English: {g}" for g in row_labels])

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = data_arr[i, j]
            ax.text(j, i, annot[i][j], ha="center", va="center",
                    color="white" if val > 50 else "black", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("% of group")

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ======= averaging helper (for combined charts) =======

def average_results(all_results):
    # takes a list of per-semester result lists and averages the pcts across them.
    # n_nonblank in the output is the sum (just for display) not a true pooled n
    key_data = defaultdict(list)
    for sem_results in all_results:
        for r in sem_results:
            key_data[r.key].append(r)

    averaged = []
    for key in USE_CASE_ORDER:
        if key not in key_data:
            continue
        entries = key_data[key]

        avg_pcts = {}
        for lbl in FREQ_LABELS_PLOT_ORDER:
            pcts = []
            for r in entries:
                denom = r.n_nonblank if r.n_nonblank > 0 else 1
                pcts.append(r.counts.get(lbl, 0) / denom * 100.0)
            avg_pcts[lbl] = sum(pcts) / len(pcts)

        avg_mean    = sum(r.mean_inverted for r in entries) / len(entries)
        avg_pct_any = sum(r.pct_any_use   for r in entries) / len(entries)
        total_n     = sum(r.n_nonblank    for r in entries)

        # back-compute fake counts so counts[lbl] / total_n * 100 == avg_pcts[lbl].
        # this lets us reuse the same plot functions without a separate code path
        avg_counts = Counter({lbl: avg_pcts[lbl] * total_n / 100.0 for lbl in FREQ_LABELS_PLOT_ORDER})

        averaged.append(UseCaseResult(
            key=key,
            label=USE_CASE_LABELS[key],
            counts=avg_counts,
            n_nonblank=total_n,
            mean_inverted=avg_mean,
            pct_any_use=avg_pct_any,
        ))

    return averaged


# ======= never-to-all and adoption tier plots =======

def plot_never_to_all(xlsx_path, out_dir, n_by_semester):
    # what % of respondents said never to literally every single use case item
    out_dir.mkdir(parents=True, exist_ok=True)

    sem_labels  = []
    pct_vals    = []
    count_vals  = []
    total_vals  = []

    for sem in SEMESTERS:
        df = pd.read_excel(xlsx_path, sheet_name=sem.sheet_name, dtype=str)

        item_cols = []
        for item in sem.items:
            num_col   = get_col_name_by_letter(df, item.code_col_letter)
            label_col = get_col_name_by_letter(df, item.label_col_letter)
            item_cols.append((num_col, label_col))

        df2          = df.iloc[2:]
        n_all_never  = 0
        n_any_resp   = 0

        for _, row in df2.iterrows():
            has_any     = False
            is_all_never = True

            for num_col, label_col in item_cols:
                freq_label = normalize_freq_label(row.get(label_col))
                if freq_label is None:
                    code = parse_int_code(row.get(num_col))
                    freq_label = RAW_CODE_TO_LABEL.get(code) # pyright: ignore[reportArgumentType]

                if freq_label is not None:
                    has_any = True
                    if freq_label != "Never":
                        is_all_never = False
                        break

            if has_any:
                n_any_resp += 1
                if is_all_never:
                    n_all_never += 1

        pct = n_all_never / n_any_resp * 100.0 if n_any_resp > 0 else 0.0
        sem_labels.append(sem.sheet_name)
        pct_vals.append(pct)
        count_vals.append(n_all_never)
        total_vals.append(n_any_resp)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_title(
        "Respondents who answered Never to all AI use-case items\n"
        "(i.e. report zero AI usage for any writing task)"
    )

    bars = ax.bar(sem_labels, pct_vals, color=["#4575b4", "#d73027"])
    ax.set_ylabel("% of respondents")
    ax.set_ylim(0, max(pct_vals) * 1.4 if pct_vals else 50)

    for i, (bar, cnt, tot, pct) in enumerate(zip(bars, count_vals, total_vals, pct_vals)):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            f"{pct:.1f}%\n({cnt} / {tot})",
            ha="center", va="bottom", fontsize=11,
        )

    plt.tight_layout()
    fig.savefig(out_dir / "never_to_all.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# tier labels and colors for the adoption tier stacked bar
TIER_ORDER = ["Never to all", "Max: Rarely", "Max: Sometimes", "At least one Often"]
TIER_COLORS = {
    "Never to all":       "#d73027",
    "Max: Rarely":        "#fc8d59",
    "Max: Sometimes":     "#91bfdb",
    "At least one Often": "#4575b4",
}

FREQ_RANK = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3}


def plot_adoption_tiers(xlsx_path, out_dir, n_by_semester):
    out_dir.mkdir(parents=True, exist_ok=True)

    sem_labels  = []
    tier_pcts   = {t: [] for t in TIER_ORDER}
    tier_counts = {t: [] for t in TIER_ORDER}
    totals      = []

    for sem in SEMESTERS:
        df = pd.read_excel(xlsx_path, sheet_name=sem.sheet_name, dtype=str)

        item_cols = []
        for item in sem.items:
            num_col   = get_col_name_by_letter(df, item.code_col_letter)
            label_col = get_col_name_by_letter(df, item.label_col_letter)
            item_cols.append((num_col, label_col))

        df2    = df.iloc[2:]
        counts = Counter()
        n_any  = 0

        for _, row in df2.iterrows():
            max_rank = -1
            has_any  = False

            for num_col, label_col in item_cols:
                freq_label = normalize_freq_label(row.get(label_col))
                if freq_label is None:
                    code = parse_int_code(row.get(num_col))
                    freq_label = RAW_CODE_TO_LABEL.get(code) # pyright: ignore[reportArgumentType]

                if freq_label is not None:
                    has_any = True
                    rank = FREQ_RANK.get(freq_label, -1)
                    if rank > max_rank:
                        max_rank = rank

            if not has_any:
                continue

            n_any += 1

            if max_rank <= 0:
                counts["Never to all"] += 1
            elif max_rank == 1:
                counts["Max: Rarely"] += 1
            elif max_rank == 2:
                counts["Max: Sometimes"] += 1
            else:
                counts["At least one Often"] += 1

        sem_labels.append(sem.sheet_name)
        totals.append(n_any)
        denom = n_any if n_any > 0 else 1
        for tier in TIER_ORDER:
            tier_counts[tier].append(counts.get(tier, 0))
            tier_pcts[tier].append(counts.get(tier, 0) / denom * 100.0)

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.set_title(
        "AI adoption tiers. respondents grouped by their highest frequency\n"
        "reported across all use-case items"
    )

    x      = range(len(sem_labels))
    bottom = [0.0] * len(sem_labels)

    for tier in TIER_ORDER:
        vals = tier_pcts[tier]
        bars = ax.bar(x, vals, bottom=bottom, label=tier, color=TIER_COLORS[tier], width=0.5)

        for i, (bar, pct, cnt, tot) in enumerate(zip(bars, vals, tier_counts[tier], totals)):
            if pct >= 4:  # dont label segments too small to read
                y_center = bottom[i] + pct / 2
                ax.text(
                    bar.get_x() + bar.get_width() / 2, y_center,
                    f"{pct:.1f}%\n({cnt})",
                    ha="center", va="center", fontsize=9,
                    color="white" if tier in ("Never to all", "At least one Often") else "black",
                )

        bottom = [b + v for b, v in zip(bottom, vals)]

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{s}\n(n={t})" for s, t in zip(sem_labels, totals)])
    ax.set_ylabel("% of respondents")
    ax.set_ylim(0, 105)
    ax.legend(title="Adoption tier", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    plt.tight_layout()
    fig.savefig(out_dir / "adoption_tiers.png", dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ======= semester configs =======

@dataclass(frozen=True)
class UseCaseItem:
    key:              str  # canonical key
    code_col_letter:  str
    label_col_letter: str


@dataclass(frozen=True)
class SemesterConfig:
    sheet_name: str
    items:      Tuple[UseCaseItem, ...]


SPRING_2024 = SemesterConfig(
    sheet_name="Spring 2024",
    items=(
        UseCaseItem("brainstorming", "AY", "AZ"),
        UseCaseItem("readings",      "BA", "BB"),
        UseCaseItem("concepts",      "BC", "BD"),
        UseCaseItem("research",      "BE", "BF"),
        UseCaseItem("outlining",     "BG", "BH"),
        UseCaseItem("rewording",     "BI", "BJ"),
        UseCaseItem("grammar",       "BK", "BL"),
        UseCaseItem("bibliography",  "BM", "BN"),
        UseCaseItem("lowstakes",     "BO", "BP"),
    ),
)

FALL_2024 = SemesterConfig(
    sheet_name="Fall 2024",
    items=(
        UseCaseItem("brainstorming", "AJ", "AK"),
        UseCaseItem("readings",      "AL", "AM"),
        UseCaseItem("concepts",      "AN", "AO"),
        UseCaseItem("research",      "AP", "AQ"),
        UseCaseItem("outlining",     "AR", "AS"),
        UseCaseItem("rewording",     "AT", "AU"),
        UseCaseItem("grammar",       "AV", "AW"),
        UseCaseItem("bibliography",  "AX", "AY"),
        UseCaseItem("lowstakes",     "AZ", "BA"),
        UseCaseItem("translation",   "BB", "BC"),  # these two are new in fall 2024
        UseCaseItem("programming",   "BD", "BE"),
    ),
)

SEMESTERS = (SPRING_2024, FALL_2024)


# ======= main =======

def run(xlsx_path: str) -> None:
    out_dir = Path("output") / VIZ_SLUG
    out_dir.mkdir(parents=True, exist_ok=True)

    dot_series   = []
    n_by_semester = {}
    all_results  = []

    for sem in SEMESTERS:
        sheet  = sem.sheet_name
        prefix = semester_prefix(sheet)

        df = pd.read_excel(xlsx_path, sheet_name=sheet, dtype=str)

        results  = []
        all_cols = []

        for item in sem.items:
            num_col   = get_col_name_by_letter(df, item.code_col_letter)
            label_col = get_col_name_by_letter(df, item.label_col_letter)
            all_cols.extend([num_col, label_col])

            r = analyze_single_item(
                df, num_col, label_col,
                key=item.key,
                label=USE_CASE_LABELS[item.key],
            )
            results.append(r)

        resp_n = compute_respondent_n(df, all_cols)
        n_by_semester[prefix] = resp_n

        # pull the question stem from the first item's column header
        q_raw   = str(df.loc[0, get_col_name_by_letter(df, sem.items[0].code_col_letter)]).strip()
        stem_end = q_raw.find(" - ")
        q_stem  = q_raw[:stem_end] if stem_end > 0 else q_raw

        plot_mean_bars(
            results,
            f"{sheet} AI use-case frequency (mean score)\nQuestion: {q_stem}\nN = {resp_n}   (1=Never ... 4=Often)",
            out_dir / f"{prefix}_usecase_bars.png",
        )

        plot_stacked_bars(
            results,
            f"{sheet} AI use-case frequency (distribution)\nQuestion: {q_stem}\nN = {resp_n}",
            out_dir / f"{prefix}_stacked_bars.png",
        )

        plot_heatmap(
            results,
            f"{sheet} AI use-case frequency (heatmap)\nN = {resp_n}",
            out_dir / f"{prefix}_heatmap.png",
        )

        plot_diverging_stacked(
            results,
            f"{sheet} AI use-case frequency (diverging)\nCentre = boundary between Rarely and Sometimes\nN = {resp_n}",
            out_dir / f"{prefix}_diverging_stacked.png",
        )

        dot_vals = {r.key: r.pct_any_use for r in results}
        dot_series.append((prefix, dot_vals))
        all_results.append(results)

    if dot_series:
        plot_dot_plot(
            dot_series,
            "AI use-case adoption over time (% who report any use)\n"
            "Use-cases ordered from idea-level assistance -> production-level output\n"
            "Fall 2024 adds translation and programming items",
            out_dir / "combined_dot_plot.png",
            n_by_semester=n_by_semester,
        )

    # combined averaged charts across both semesters
    if len(all_results) >= 2:
        averaged  = average_results(all_results)
        total_n   = sum(n_by_semester.values())
        sem_names = " + ".join(sem.sheet_name for sem in SEMESTERS)

        plot_stacked_bars(
            averaged,
            f"Combined average AI use-case frequency (distribution)\nMean of per-semester percentages ({sem_names})\nTotal respondents = {total_n}",
            out_dir / "combined_stacked_bars.png",
        )

        plot_heatmap(
            averaged,
            f"Combined average. AI use-case frequency (heatmap)\nMean of per-semester percentages ({sem_names})\nTotal respondents = {total_n}",
            out_dir / "combined_heatmap.png",
        )

        plot_diverging_stacked(
            averaged,
            f"Combined average. AI use-case frequency (diverging)\nCentre = boundary between Rarely and Sometimes\nMean of per-semester percentages ({sem_names})\nTotal respondents = {total_n}",
            out_dir / "combined_diverging_stacked.png",
        )

    # fall 2024 only: cross-tab of native english speaker vs translation use
    f24_df = pd.read_excel(xlsx_path, sheet_name="Fall 2024", dtype=str)
    plot_native_translation_heatmap(
        f24_df,
        native_label_col=get_col_name_by_letter(f24_df, "E"),       # Q3.1
        translation_num_col=get_col_name_by_letter(f24_df, "BB"),   # Q9_10
        translation_label_col=get_col_name_by_letter(f24_df, "BC"), # Q9_10.1
        title=(
            "Fall 2024. AI for translation by native English status\n"
            "% within each group who use AI for translation between languages"
        ),
        output_png=out_dir / "2024fall_native_x_translation.png",
    )

    plot_never_to_all(xlsx_path, out_dir, n_by_semester)
    plot_adoption_tiers(xlsx_path, out_dir, n_by_semester)