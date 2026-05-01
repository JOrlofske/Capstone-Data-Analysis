# visualizations/instructorPermissionImpact.py
#
# does instructor permission to use AI actually impact whether students use it?
# spring 2024 and fall 2024 each ask three yes/no questions:
#   - used AI when instructor explicitly gave option to use it
#   - used AI when instructor didn't specify whether it was OK
#   - used AI when instructor said not to use it
#
# earlier semesters (fall 2022 - fall 2023) did not ask these questions.
#
# outputs go in output/instructor_permission_impact/
#   per semester:
#     {prefix}_permission_bars.png              - grouped bar chart (counts)
#     {prefix}_permission_stacked_pct.png       - stacked percentage bar chart
#   combined:
#     combined_permission_bars.png              - grouped bars, both semesters
#     combined_permission_stacked_pct.png       - stacked percentage, both semesters
#
# column locations (from survey analysis):
#   spring 2024: AS/AT (explicitly allowed), AU/AV (didn't specify), AW/AX (said not to)
#   fall 2024:   AD/AE (explicitly allowed), AF/AG (didn't specify), AH/AI (said not to)

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


VIZ_NAME = "Instructor permission impact on student AI usage (Spring 2024-Fall 2024)"
VIZ_SLUG = "instructor_permission_impact"
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


# ======= response handling =======

RESPONSE_ORDER  = ["Yes", "No"]
RESPONSE_COLORS = {
    "Yes": "#4575b4",
    "No":  "#d73027",
}


def normalize_yn_label(s):
    if is_blank(s):
        return None
    txt = str(s).strip().lower()
    mapping = {
        "yes": "Yes",
        "no":  "No",
    }
    return mapping.get(txt)


# ======= data extraction =======

PERMISSION_SCENARIOS = [
    "Instructor allowed AI",
    "Instructor didn't specify",
    "Instructor said no AI",
]


@dataclass
class PermissionResult:
    scenario_label: str
    counts:         Counter   # "Yes"/"No" -> count
    n_valid:        int


def analyze_permission_item(df, code_letter, label_letter, scenario_label):
    """Reads one permission question column and tallies Yes/No responses."""
    code_col  = get_col_name_by_letter(df, code_letter)
    label_col = get_col_name_by_letter(df, label_letter)
    df2 = df.iloc[2:]  # skip header rows

    counts  = Counter()
    n_valid = 0

    for _, row in df2.iterrows():
        raw_label = row.get(label_col)
        resolved  = normalize_yn_label(raw_label)
        if resolved is None:
            continue
        n_valid += 1
        counts[resolved] += 1

    return PermissionResult(
        scenario_label=scenario_label,
        counts=counts,
        n_valid=n_valid,
    )


# ======= plot 1: per-semester grouped bar chart (counts) =======

def plot_semester_bars(results, title, output_png):
    """Simple grouped bar chart showing Yes/No counts for each scenario."""
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

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title, fontsize=12, pad=12)

    scenario_labels = [r.scenario_label for r in results]
    x = np.arange(len(scenario_labels))
    width = 0.35

    yes_vals = [r.counts.get("Yes", 0) for r in results]
    no_vals  = [r.counts.get("No", 0) for r in results]

    bars_yes = ax.bar(x - width / 2, yes_vals, width, label="Yes",
                      color=RESPONSE_COLORS["Yes"])
    bars_no  = ax.bar(x + width / 2, no_vals, width, label="No",
                      color=RESPONSE_COLORS["No"])

    # count labels on top of bars
    for bars in [bars_yes, bars_no]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                        str(int(h)), ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, fontsize=10)
    ax.set_ylabel("Number of respondents")

    # show n on x-axis
    for i, r in enumerate(results):
        ax.text(i, -0.08, f"(n={r.n_valid})", ha="center", va="top",
                fontsize=9, transform=ax.get_xaxis_transform())

    ax.legend(frameon=False, fontsize=10)
    ax.set_ylim(0, max(max(yes_vals), max(no_vals)) * 1.15)

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ======= plot 2: per-semester stacked percentage bar chart =======

def plot_semester_stacked_pct(results, title, output_png):
    """Horizontal stacked percentage bar chart for one semester."""
    output_png.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        fig = plt.figure(figsize=(10, 5))
        plt.title(title)
        plt.text(0.5, 0.5, "No responses", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    fig = plt.figure(figsize=(12, max(3, len(results) * 1.0 + 2)))
    plt.title(title, fontsize=12, pad=12)
    ax = plt.gca()

    y_labels = [r.scenario_label for r in results]
    left = [0.0] * len(results)

    for bucket in RESPONSE_ORDER:
        vals = []
        for r in results:
            denom = r.n_valid if r.n_valid > 0 else 1
            vals.append(r.counts.get(bucket, 0) / denom * 100.0)

        bars = ax.barh(y_labels, vals, left=left, label=bucket,
                       color=RESPONSE_COLORS[bucket])

        for i, (bar, pct) in enumerate(zip(bars, vals)):
            if pct >= 5:
                x_center = left[i] + pct / 2
                ax.text(x_center, i, f"{pct:.0f}%", ha="center", va="center",
                        fontsize=10, color="white", fontweight="bold")

        left = [l + v for l, v in zip(left, vals)]

    for i, r in enumerate(results):
        ax.text(101, i, f"n={r.n_valid}", va="center", fontsize=9)

    ax.set_xlabel("% of respondents")
    ax.set_xlim(0, 100)
    ax.legend(title="Used AI?", loc="center left", bbox_to_anchor=(1.08, 0.5),
              frameon=False)

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ======= plot 3: combined grouped bar chart =======

def plot_combined_bars(results_by_sem, title, output_png):
    """Grouped bar chart comparing both semesters side by side."""
    output_png.parent.mkdir(parents=True, exist_ok=True)

    if not results_by_sem:
        fig = plt.figure(figsize=(12, 6))
        plt.title(title)
        plt.text(0.5, 0.5, "No responses", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_title(title, fontsize=12, pad=12)

    n_scenarios = len(PERMISSION_SCENARIOS)
    n_semesters = len(results_by_sem)

    # for each scenario we have n_semesters * 2 bars (yes and no per semester)
    # layout: groups of bars per scenario, with sub-groups per semester
    bar_width = 0.18
    group_gap = 0.4

    # color shading: lighter for Spring 2024, darker for Fall 2024
    yes_shades = ["#7baed0", "#4575b4"]
    no_shades  = ["#fc8d59", "#d73027"]

    positions = []
    tick_positions = []

    pos = 0
    for si in range(n_scenarios):
        scenario_start = pos
        for semi, (sem_label, results) in enumerate(results_by_sem):
            r = results[si]
            yes_val = r.counts.get("Yes", 0)
            no_val  = r.counts.get("No", 0)

            b1 = ax.bar(pos, yes_val, bar_width, color=yes_shades[semi])
            b2 = ax.bar(pos + bar_width, no_val, bar_width, color=no_shades[semi])

            # count labels
            for bar in [b1[0], b2[0]]:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                            str(int(h)), ha="center", va="bottom", fontsize=9)

            # semester label below
            mid = pos + bar_width / 2
            ax.text(mid, -0.06, sem_label.replace(" ", "\n"), ha="center",
                    va="top", fontsize=7.5, transform=ax.get_xaxis_transform())

            pos += bar_width * 2 + 0.05  # small gap between semesters

        tick_positions.append((scenario_start + pos - bar_width * 2 - 0.05) / 2)
        pos += group_gap

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(PERMISSION_SCENARIOS, fontsize=10)
    ax.set_ylabel("Number of respondents")

    # build legend
    handles = [
        mpatches.Patch(color=yes_shades[0], label="Yes – Spring 2024"),
        mpatches.Patch(color=yes_shades[1], label="Yes – Fall 2024"),
        mpatches.Patch(color=no_shades[0],  label="No – Spring 2024"),
        mpatches.Patch(color=no_shades[1],  label="No – Fall 2024"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ======= plot 4: combined stacked percentage bar chart =======

def plot_combined_stacked_pct(results_by_sem, title, output_png):
    """Horizontal stacked percentage bars with all semesters grouped by scenario."""
    output_png.parent.mkdir(parents=True, exist_ok=True)

    if not results_by_sem:
        fig = plt.figure(figsize=(12, 6))
        plt.title(title)
        plt.text(0.5, 0.5, "No responses", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    # build row list: one row per (scenario, semester)
    all_rows = []  # (row_label, counts, n_valid)
    for si, scenario in enumerate(PERMISSION_SCENARIOS):
        for sem_label, results in results_by_sem:
            r = results[si]
            row_label = f"{scenario} ({sem_label})"
            all_rows.append((row_label, r.counts, r.n_valid))

    fig = plt.figure(figsize=(13, max(4, len(all_rows) * 0.7 + 2)))
    plt.title(title, fontsize=12, pad=12)
    ax = plt.gca()

    row_labels = [r[0] for r in all_rows]
    left = [0.0] * len(all_rows)

    for bucket in RESPONSE_ORDER:
        vals = []
        for _, counts, n_valid in all_rows:
            denom = n_valid if n_valid > 0 else 1
            vals.append(counts.get(bucket, 0) / denom * 100.0)

        bars = ax.barh(row_labels, vals, left=left, label=bucket,
                       color=RESPONSE_COLORS[bucket])

        for i, (bar, pct) in enumerate(zip(bars, vals)):
            if pct >= 5:
                x_center = left[i] + pct / 2
                ax.text(x_center, i, f"{pct:.0f}%", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")

        left = [l + v for l, v in zip(left, vals)]

    for i, (_, _, n) in enumerate(all_rows):
        ax.text(101, i, f"n={n}", va="center", fontsize=9)

    ax.set_xlabel("% of respondents")
    ax.set_xlim(0, 100)
    ax.legend(title="Used AI?", loc="center left", bbox_to_anchor=(1.08, 0.5),
              frameon=False)

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ======= semester configs =======

@dataclass(frozen=True)
class PermissionItemConfig:
    code_letter:  str
    label_letter: str


@dataclass(frozen=True)
class SemesterConfig:
    sheet_name: str
    # three permission scenario items in order matching PERMISSION_SCENARIOS
    items: Tuple[PermissionItemConfig, ...]


SPRING_2024 = SemesterConfig(
    sheet_name="Spring 2024",
    items=(
        PermissionItemConfig("AS", "AT"),  # instructor explicitly allowed
        PermissionItemConfig("AU", "AV"),  # instructor didn't specify
        PermissionItemConfig("AW", "AX"),  # instructor said not to
    ),
)

FALL_2024 = SemesterConfig(
    sheet_name="Fall 2024",
    items=(
        PermissionItemConfig("AD", "AE"),  # instructor explicitly allowed
        PermissionItemConfig("AF", "AG"),  # instructor didn't specify
        PermissionItemConfig("AH", "AI"),  # instructor said not to
    ),
)

SEMESTERS = (SPRING_2024, FALL_2024)


# ======= main =======

def run(xlsx_path: str) -> None:
    out_dir = Path("output") / VIZ_SLUG
    out_dir.mkdir(parents=True, exist_ok=True)

    # collector for combined charts: list of (semester_label, [PermissionResult, ...])
    all_semester_results = []

    for sem in SEMESTERS:
        sheet  = sem.sheet_name
        prefix = semester_prefix(sheet)
        df     = pd.read_excel(xlsx_path, sheet_name=sheet, dtype=str)

        # analyze the 3 permission items
        results = []
        for scenario_label, item_cfg in zip(PERMISSION_SCENARIOS, sem.items):
            r = analyze_permission_item(
                df, item_cfg.code_letter, item_cfg.label_letter,
                scenario_label,
            )
            results.append(r)

        all_semester_results.append((sheet, results))

        # --- per-semester grouped bar chart (counts) ---
        plot_semester_bars(
            results,
            f"{sheet}: Student AI usage by instructor permission\n"
            f"\"Have you used AI-based text generators in your coursework at Pitt when...\"",
            out_dir / f"{prefix}_permission_bars.png",
        )

        # --- per-semester stacked percentage chart ---
        plot_semester_stacked_pct(
            results,
            f"{sheet}: Student AI usage by instructor permission (%)\n"
            f"\"Have you used AI-based text generators in your coursework at Pitt when...\"",
            out_dir / f"{prefix}_permission_stacked_pct.png",
        )

    # --- combined grouped bar chart ---
    plot_combined_bars(
        all_semester_results,
        "Student AI usage by instructor permission (Spring & Fall 2024)\n"
        "\"Have you used AI-based text generators in your coursework at Pitt when...\"",
        out_dir / "combined_permission_bars.png",
    )

    # --- combined stacked percentage chart ---
    plot_combined_stacked_pct(
        all_semester_results,
        "Student AI usage by instructor permission (Spring & Fall 2024, %)\n"
        "\"Have you used AI-based text generators in your coursework at Pitt when...\"",
        out_dir / "combined_permission_stacked_pct.png",
    )


if __name__ == "__main__":
    run("All_AI_Surveys.xlsx")