# visualizations/techUsage.py
#
# tracks what writing tools students actually use, both for pitt coursework
# and outside of it. covers word processors, grammar checkers, citation tools,
# ai stuff, etc. fall 2022 through fall 2024.
#
# annoying wrinkle: the earlier semesters used select-all questions (just pick
# everything you use) while 2024 switched to frequency scales (never/rarely/sometimes/often).
# so theres two pretty different code paths depending on the semester.
#
# outputs go in output/tech_usage/

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt


VIZ_NAME = "Tech usage: coursework vs outside vs other (Fall 2022-Fall 2024)"
VIZ_SLUG = "tech_usage"
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
    # turns fall 2022 into 2022fall for use in filenames
    parts = sheet_name.strip().split()
    if len(parts) != 2:
        return "".join(ch.lower() for ch in sheet_name if ch.isalnum())
    season = parts[0].lower()
    year   = "".join(ch for ch in parts[1] if ch.isdigit())
    return f"{year}{season}"


def parse_multiselect_codes(cell):
    # cells come in as things like "1,2,3" or "2.0,8.0" depending on the semester
    if is_blank(cell):
        return []
    codes = []
    for p in str(cell).strip().split(","):
        p = p.strip()
        if not p:
            continue
        try:
            codes.append(int(float(p)))
        except ValueError:
            continue
    return codes


def count_codes(series, code_to_label):
    counter = Counter()
    for cell in series:
        for c in parse_multiselect_codes(cell):
            counter[code_to_label.get(c, f"Unknown ({c})")] += 1
    return counter


def compute_respondent_n(df, value_cols):
    # skip first 2 rows (metadata), count rows that arent all-blank across the given cols
    df2  = df.iloc[2:]
    mask = None
    for c in value_cols:
        b    = df2[c].apply(is_blank)
        mask = b if mask is None else (mask & b)
    return int((~mask).sum()) # pyright: ignore[reportOptionalOperand]


# ======= frequency scale helpers (2024 semesters) =======

# never/rarely/sometimes/often maps to 0-3 for mean calculations
FREQ_SCORE = {
    "never":     0,
    "rarely":    1,
    "sometimes": 2,
    "often":     3,
}

FREQ_BUCKETS = ["Never", "Rarely", "Sometimes", "Often"]


def label_to_freq_score(label):
    if is_blank(label):
        return None
    s = " ".join(str(label).strip().lower().split())
    if s in FREQ_SCORE:
        return FREQ_SCORE[s]
    # fuzzy match in case there's extra wording around the keyword
    for k, v in FREQ_SCORE.items():
        if k in s:
            return v
    return None


def extract_tech_label(qtext):
    # question text comes in as like "How often do you use X - Google Docs"
    # we just want the part after the dash
    if is_blank(qtext):
        return "Unknown technology"
    s = str(qtext).strip()
    if " - " in s:
        return s.split(" - ")[-1].strip()
    return s


def extract_question_stem(qtext):
    # opposite of extract_tech_label - grab the part before the dash for the chart title
    if is_blank(qtext):
        return ""
    s = str(qtext).strip()
    if " - " in s:
        return s.split(" - ")[0].strip()
    return s


def freq_means_for_items(df, item_cols):
    means  = []
    base_q = extract_question_stem(df.loc[0, item_cols[0]]) if item_cols else ""

    for col in item_cols:
        label_col = f"{col}.1"
        if col not in df.columns or label_col not in df.columns:
            continue

        tech = extract_tech_label(df.loc[0, col])
        scores = []
        for v in df.iloc[2:][label_col]:
            sc = label_to_freq_score(v)
            if sc is not None:
                scores.append(sc)

        if scores:
            means.append((tech, float(sum(scores)) / len(scores), len(scores)))

    return means, base_q


def freq_adoption_for_items(df, item_cols):
    # yes = anything above never. used to make the side-by-side yesno chart
    items  = []
    base_q = extract_question_stem(df.loc[0, item_cols[0]]) if item_cols else ""

    for col in item_cols:
        label_col = f"{col}.1"
        if col not in df.columns or label_col not in df.columns:
            continue

        tech = extract_tech_label(df.loc[0, col])
        yes  = 0
        n    = 0

        for v in df.iloc[2:][label_col]:
            sc = label_to_freq_score(v)
            if sc is None:
                continue
            n += 1
            if sc > 0:
                yes += 1

        if n > 0:
            items.append((tech, yes, n))

    return items, base_q


def freq_breakdown_for_items(df, item_cols):
    out    = []
    base_q = extract_question_stem(df.loc[0, item_cols[0]]) if item_cols else ""

    for col in item_cols:
        label_col = f"{col}.1"
        if col not in df.columns or label_col not in df.columns:
            continue

        tech = extract_tech_label(df.loc[0, col])
        c    = Counter()

        for v in df.iloc[2:][label_col]:
            sc = label_to_freq_score(v)
            if sc is None:
                continue
            # just convert the score back to the bucket label for the stacked chart
            c[FREQ_BUCKETS[sc]] += 1

        if sum(c.values()) > 0:
            out.append((tech, c))

    return out, base_q


# ======= plot functions =======

def plot_selectall_counts(counter, title, output_png, resp_n):
    output_png.parent.mkdir(parents=True, exist_ok=True)

    items    = sorted(counter.items(), key=lambda kv: kv[1])
    labels   = [k for k, _ in items]
    counts   = [v for _, v in items]
    percents = [c / resp_n * 100 for c in counts]

    fig = plt.figure(figsize=(12, 7))
    plt.title(title)
    ax = plt.gca()
    ax.barh(labels, counts)
    ax.set_xlabel("Selections (count)")
    ax.set_ylabel("")

    pad = max(1, int(max(counts) * 0.01))
    for i, (c, p) in enumerate(zip(counts, percents)):
        ax.text(c + pad, i, f"{c} ({p:.1f}%)", va="center")

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_freq_means(means, title, output_png, x_label):
    output_png.parent.mkdir(parents=True, exist_ok=True)

    srt    = sorted(means, key=lambda t: t[1])
    labels = [t[0] for t in srt]
    xs     = [t[1] for t in srt]
    ns     = [t[2] for t in srt]

    fig = plt.figure(figsize=(12, 7))
    plt.title(title)
    ax = plt.gca()
    ax.barh(labels, xs)
    ax.set_xlabel(x_label)
    ax.set_ylabel("")

    pad = max(0.02, max(xs) * 0.01)
    for i, (m, n) in enumerate(zip(xs, ns)):
        ax.text(m + pad, i, f"{m:.2f} (n={n})", va="center")

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_adoption_counts(items, title, output_png):
    # yes/no chart - never = no, anything else = yes
    output_png.parent.mkdir(parents=True, exist_ok=True)

    srt        = sorted(items, key=lambda t: t[1])
    labels     = [t[0] for t in srt]
    yes_counts = [t[1] for t in srt]
    ns         = [t[2] for t in srt]

    fig = plt.figure(figsize=(12, 7))
    plt.title(title)
    ax = plt.gca()
    ax.barh(labels, yes_counts)
    ax.set_xlabel("Respondents using (count)  [Never=No; >Never=Yes]")
    ax.set_ylabel("")

    pad = max(1, int(max(yes_counts) * 0.01))
    for i, (yes, n) in enumerate(zip(yes_counts, ns)):
        pct = yes / n * 100 if n else 0.0
        ax.text(yes + pad, i, f"{yes} ({pct:.1f}%, n={n})", va="center")

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_freq_breakdown(tech_counts, title, output_png):
    output_png.parent.mkdir(parents=True, exist_ok=True)

    # sort by total answered so most-answered techs end up near the bottom
    srt      = sorted(tech_counts, key=lambda t: sum(t[1].get(b, 0) for b in FREQ_BUCKETS))
    labels   = [t[0] for t in srt]
    counters = [t[1] for t in srt]

    fig = plt.figure(figsize=(13, 8))
    plt.title(title)
    ax   = plt.gca()
    left = [0] * len(labels)

    for bucket in FREQ_BUCKETS:
        vals = [c.get(bucket, 0) for c in counters]
        ax.barh(labels, vals, left=left, label=bucket)
        left = [l + v for l, v in zip(left, vals)]

    ax.set_xlabel("Respondents (count)")
    ax.set_ylabel("")
    ax.legend(title="Frequency", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ======= semester configs =======

@dataclass(frozen=True)
class SemesterConfig:
    sheet_name: str
    mode: str  # select_all or frequency

    # select_all fields
    q2q3_map: Dict[int, str] | None = None
    q6_map:   Dict[int, str] | None = None
    q2_col:   str | None = None
    q3_col:   str | None = None
    q6_col:   str | None = None

    # frequency fields
    in_cols:    List[str] | None = None
    out_cols:   List[str] | None = None
    other_cols: List[str] | None = None


FALL_2022 = SemesterConfig(
    sheet_name="Fall 2022",
    mode="select_all",
    q2_col="Q2",
    q3_col="Q3",
    q6_col="Q6",
    q2q3_map={
        1: "Handwriting (pencil, pen, paper)",
        2: "Handwriting (on a tablet or digital notepad)",
        3: "Word processor (like Word or Google docs)",
        4: "Grammar-checking built into the word processor",
        5: "Grammar-checking in a separate program (like Grammarly)",
        6: "Spell-checking built into the word processor",
        7: "Citation generators (like EasyBib or Quillbot)",
        8: "AI-based text generators (like GPT-3, ChatGPT, Sudowrite)",
    },
    q6_map={
        1: "ChatGPT GPT-3 or GPT-2",
        2: "Sudowrite",
        3: "Jasper.ai",
        4: "Paragraph.ai",
        5: "Notion.ai",
        6: "None of these",
    },
)

SPRING_2023 = SemesterConfig(
    sheet_name="Spring 2023",
    mode="select_all",
    q2_col="Q2",
    q3_col="Q3",
    q6_col="Q6",
    q2q3_map={
        1: "Handwriting (pencil, pen, paper)",
        2: "Handwriting (on a tablet or digital notepad)",
        3: "Word processor (like Word or Google docs)",
        4: "Grammar-checking built into the word processor",
        5: "Grammar-checking in a separate program (like Grammarly)",
        6: "Spell-checking built into the word processor",
        7: "Citation generators (like EasyBib or Quillbot)",
        8: "AI-based text generators (like GPT-3, GPT-4, ChatGPT, Sudowrite)",
    },
    q6_map={
        1: "ChatGPT",
        2: "Sudowrite",
        3: "Jasper.ai",
        4: "Paragraph.ai",
        5: "Notion.ai",
        6: "None of these",
        7: "Grammarly",
        8: "GPT-2, GPT-3 or GPT-4",
    },
)

FALL_2023 = SemesterConfig(
    sheet_name="Fall 2023",
    mode="select_all",
    q2_col="Q2",
    q3_col="Q3",
    q6_col="Q6",
    q2q3_map={
        1: "Handwriting (pencil, pen, paper)",
        2: "Handwriting (on a tablet or digital notepad)",
        3: "Word processor (like Word or Google docs)",
        4: "Grammar-checking built into the word processor",
        5: "Grammar-checking in a separate program (like Grammarly)",
        6: "Spell-checking built into the word processor",
        7: "Citation generators (like EasyBib or Quillbot)",
        8: "AI-based text generators (like GPT-3, GPT-4, ChatGPT, Sudowrite)",
    },
    q6_map={
        1: "ChatGPT",
        2: "Sudowrite",
        3: "Jasper.ai",
        4: "Paragraph.ai",
        5: "Notion.ai",
        6: "None of these",
        7: "Grammarly",
        8: "GPT-2, GPT-3 or GPT-4",
    },
)

SPRING_2024 = SemesterConfig(
    sheet_name="Spring 2024",
    mode="frequency",
    in_cols=[f"Q4_{i}" for i in range(1, 7)],    # Q4_1..Q4_6
    out_cols=[f"Q5_{i}" for i in range(1, 7)],   # Q5_1..Q5_6
    other_cols=[f"Q6_{i}" for i in range(1, 6)], # Q6_1..Q6_5
)

FALL_2024 = SemesterConfig(
    sheet_name="Fall 2024",
    mode="frequency",
    in_cols=[f"Q5_{i}" for i in range(1, 6)],    # Q5_1..Q5_5
    out_cols=[f"Q6_{i}" for i in range(1, 6)],   # Q6_1..Q6_5
    other_cols=[],  # they removed the other tech section this semester
)

SEMESTERS = (FALL_2022, SPRING_2023, FALL_2023, SPRING_2024, FALL_2024)


# ======= main =======

def run(xlsx_path: str) -> None:
    out_dir = Path("output") / VIZ_SLUG
    out_dir.mkdir(parents=True, exist_ok=True)

    for sem in SEMESTERS:
        sheet  = sem.sheet_name
        prefix = semester_prefix(sheet)

        # load everything as strings so row 0/1 metadata and code columns dont get mangled
        df = pd.read_excel(xlsx_path, sheet_name=sheet, dtype=str)

        if sem.mode == "select_all":
            assert sem.q2_col and sem.q3_col and sem.q6_col
            assert sem.q2q3_map and sem.q6_map

            # quick sanity check - if these columns arent there something is very wrong
            for c in [sem.q2_col, sem.q3_col, sem.q6_col]:
                if c not in df.columns:
                    raise ValueError(f"Sheet {sheet}: missing expected column {c}.")

            q2_text = str(df.loc[0, sem.q2_col]).strip()
            q3_text = str(df.loc[0, sem.q3_col]).strip()
            q6_text = str(df.loc[0, sem.q6_col]).strip()

            df2  = df.iloc[2:].copy()
            mask = (
                df2[sem.q2_col].apply(is_blank)
                & df2[sem.q3_col].apply(is_blank)
                & df2[sem.q6_col].apply(is_blank)
            )
            rows     = df2[~mask].copy()
            resp_n   = len(rows)

            # print(f"{sheet}: resp_n={resp_n}")  # sanity check

            q2_counts = count_codes(rows[sem.q2_col], sem.q2q3_map)
            q3_counts = count_codes(rows[sem.q3_col], sem.q2q3_map)
            q6_counts = count_codes(rows[sem.q6_col], sem.q6_map)

            plot_selectall_counts(
                q2_counts,
                f"{sheet}. In coursework\nQuestion: {q2_text}\nN = {resp_n}",
                out_dir / f"{prefix}_in_school.png",
                resp_n,
            )
            plot_selectall_counts(
                q3_counts,
                f"{sheet}. Outside coursework\nQuestion: {q3_text}\nN = {resp_n}",
                out_dir / f"{prefix}_outside_school.png",
                resp_n,
            )
            plot_selectall_counts(
                q6_counts,
                f"{sheet}. Other technologies\nQuestion: {q6_text}\nN = {resp_n}",
                out_dir / f"{prefix}_other_tech.png",
                resp_n,
            )

        elif sem.mode == "frequency":
            in_cols    = sem.in_cols or []
            out_cols   = sem.out_cols or []
            other_cols = sem.other_cols or []

            all_cols = [c for c in (in_cols + out_cols + other_cols) if c in df.columns]
            if not all_cols:
                raise ValueError(f"Sheet {sheet}: no expected frequency columns found.")

            resp_n = compute_respondent_n(df, all_cols)

            # in-coursework charts
            in_item_cols = [c for c in in_cols if c in df.columns]

            in_means, in_q = freq_means_for_items(df, in_item_cols)
            plot_freq_means(
                in_means,
                f"{sheet}. In coursework (mean)\nQuestion: {in_q}\nN = {resp_n}\n(0=Never ... 3=Often)",
                out_dir / f"{prefix}_in_school_mean.png",
                x_label="Average frequency score",
            )

            in_yesno, _ = freq_adoption_for_items(df, in_item_cols)
            plot_adoption_counts(
                in_yesno,
                f"{sheet}. In coursework (yes/no)\nQuestion: {in_q}\nN = {resp_n}\n(Never=No; >Never=Yes)",
                out_dir / f"{prefix}_in_school_yesno.png",
            )

            in_breakdown, _ = freq_breakdown_for_items(df, in_item_cols)
            plot_freq_breakdown(
                in_breakdown,
                f"{sheet}. In coursework (frequency breakdown)\nQuestion: {in_q}\nN = {resp_n}",
                out_dir / f"{prefix}_in_school_breakdown.png",
            )

            # outside-coursework charts
            out_item_cols = [c for c in out_cols if c in df.columns]

            out_means, out_q = freq_means_for_items(df, out_item_cols)
            plot_freq_means(
                out_means,
                f"{sheet}. Outside coursework (mean)\nQuestion: {out_q}\nN = {resp_n}\n(0=Never ... 3=Often)",
                out_dir / f"{prefix}_outside_school_mean.png",
                x_label="Average frequency score",
            )

            out_yesno, _ = freq_adoption_for_items(df, out_item_cols)
            plot_adoption_counts(
                out_yesno,
                f"{sheet}. Outside coursework (yes/no)\nQuestion: {out_q}\nN = {resp_n}\n(Never=No; >Never=Yes)",
                out_dir / f"{prefix}_outside_school_yesno.png",
            )

            out_breakdown, _ = freq_breakdown_for_items(df, out_item_cols)
            plot_freq_breakdown(
                out_breakdown,
                f"{sheet}. Outside coursework (frequency breakdown)\nQuestion: {out_q}\nN = {resp_n}",
                out_dir / f"{prefix}_outside_school_breakdown.png",
            )

            # other tech - spring 2024 has it, fall 2024 dropped it
            if other_cols:
                other_item_cols = [c for c in other_cols if c in df.columns]

                other_means, other_q = freq_means_for_items(df, other_item_cols)
                plot_freq_means(
                    other_means,
                    f"{sheet}. Other technologies (mean)\nQuestion: {other_q}\nN = {resp_n}\n(0=Never ... 3=Often)",
                    out_dir / f"{prefix}_other_tech_mean.png",
                    x_label="Average frequency score",
                )

                other_yesno, _ = freq_adoption_for_items(df, other_item_cols)
                plot_adoption_counts(
                    other_yesno,
                    f"{sheet}. Other technologies (yes/no)\nQuestion: {other_q}\nN = {resp_n}\n(Never=No; >Never=Yes)",
                    out_dir / f"{prefix}_other_tech_yesno.png",
                )

                other_breakdown, _ = freq_breakdown_for_items(df, other_item_cols)
                plot_freq_breakdown(
                    other_breakdown,
                    f"{sheet}. Other technologies (frequency breakdown)\nQuestion: {other_q}\nN = {resp_n}",
                    out_dir / f"{prefix}_other_tech_breakdown.png",
                )

            else:
                print(f"Note (This is NOT an error): no other tech columns configured for {sheet}, skipping that section.")

        else:
            raise ValueError(f"Unknown semester mode: {sem.mode}")