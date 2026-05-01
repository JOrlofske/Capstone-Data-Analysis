# visualizations/reasons_not_using.py
#
# visualizes manually-coded qualitative data from a workbook containing
# semester sheets for the open-ended "reasons for not using" questions.
#
# this version is intentionally flexible:
#   - it scans all workbook sheets and keeps any sheet that matches the
#     expected coded layout instead of relying on a hardcoded sheet-name list
#   - it treats "Indeterminable" and "Not Enough Info" like uncodable / N/A
#     values rather than substantive categories
#
# outputs per compatible semester sheet:
#   - used ai? distribution (yes/no only; indeterminable excluded)
#   - q1 how themes
#   - q1 why-not / restraint themes
#   - q2 situations themes
#   - q2 why themes
#
# it also outputs pooled versions across all compatible sheets combined.
#
# denominator note:
#   - used ai? uses only q1 responses with a codable yes/no determination
#   - q1 "how" uses an applicable-use denominator:
#       respondents who said yes OR have at least one substantive how code
#   - q1 "why not" uses an applicable-restraint denominator:
#       respondents who said no OR have at least one substantive why-not code
#   - q2 situation chart uses only responses with at least one substantive
#     situation code
#   - q2 why chart uses only responses with at least one substantive why code
#
# outputs go in output/reasons_not_using/

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import matplotlib.pyplot as plt


VIZ_NAME = "Reasons for not using AI: coded open-ended responses"
VIZ_SLUG = "reasons_not_using"
SHOW_PLOTS = False


# ======= fixed workbook structure =======

Q1_RESPONSE_COL = 0
Q1_USED_COL = 1
Q1_HOW_COLS = [2, 3, 4]
Q1_WHYNOT_COLS = [5, 6, 7]

Q2_RESPONSE_COL = 11
Q2_SITUATION_COLS = [12, 13, 14, 15]
Q2_WHY_COLS = [16, 17, 18]

DATA_START_ROW = 2  # first two rows are titles / headers


# ======= small helpers =======


def is_blank(cell) -> bool:
    if cell is None:
        return True
    try:
        if pd.isna(cell):
            return True
    except Exception:
        pass
    txt = str(cell).strip()
    return txt == "" or txt.lower() == "nan"



def clean_text(cell):
    if is_blank(cell):
        return None
    return str(cell).strip()



def semester_prefix(sheet_name: str) -> str:
    parts = sheet_name.strip().split()
    if len(parts) < 2:
        return "".join(ch.lower() for ch in sheet_name if ch.isalnum())

    season = parts[0].lower()
    year = "".join(ch for ch in parts[1] if ch.isdigit())
    return f"{year}{season}" if year else "".join(ch.lower() for ch in sheet_name if ch.isalnum())



def semester_label(sheet_name: str) -> str:
    parts = sheet_name.strip().split()
    if len(parts) < 2:
        return sheet_name.strip()
    year = "".join(ch for ch in parts[1] if ch.isdigit())
    return f"{parts[0].title()} {year}" if year else sheet_name.strip()



def merge_counts_in_place(target: Counter, source: Counter) -> None:
    for key, value in source.items():
        target[key] += value



def row_has_any_text(row: pd.Series, cols: Iterable[int]) -> bool:
    for col in cols:
        if clean_text(row.get(col)):
            return True
    return False



def sheet_is_compatible(df: pd.DataFrame) -> bool:
    """Detect the expected coded layout without relying on specific sheet names."""
    if df.shape[1] < 19 or df.shape[0] < 2:
        return False

    def norm(v) -> str:
        txt = clean_text(v)
        return txt.lower() if txt else ""

    row0 = df.iloc[0]
    row1 = df.iloc[1]

    checks = [
        norm(row1.get(Q1_RESPONSE_COL)) == "response:",
        norm(row1.get(Q1_USED_COL)) == "used ai?",
        norm(row1.get(Q1_HOW_COLS[0])) == "how?",
        norm(row1.get(Q1_WHYNOT_COLS[0])) == "why not?",
        norm(row1.get(Q2_RESPONSE_COL)) == "response:",
        norm(row1.get(Q2_SITUATION_COLS[0])) == "situations",
        norm(row1.get(Q2_WHY_COLS[0])) == "why?",
    ]

    q1_title = norm(row0.get(Q1_RESPONSE_COL))
    q2_title = norm(row0.get(Q2_RESPONSE_COL))
    title_check = ("question 1" in q1_title) and ("question 2" in q2_title)

    return sum(checks) >= 6 and title_check


# ======= normalization =======

USED_AI_ORDER = ["Yes", "No"]
USED_AI_COLORS = {
    "Yes": "#4575b4",
    "No": "#d73027",
}

Q1_HOW_ORDER = [
    "Brainstorming",
    "Proofreading",
    "Rephrasing",
    "Summarizing",
    "Tutoring/Conceptualizing",
    "Studying Tool",
    "Content Generation/Drafting",
    "Research",
    "Debugging/Coding",
    "Personal Use",
]

Q1_WHYNOT_ORDER = [
    "It's Cheating",
    "Fear Of Consequences",
    "Desire For Learning",
    "AI Inaccuracy/Distrust",
    "Desire For Originality",
    "Ethical Concerns",
    "Fear Of Dependency",
    "Unfamiliar With Tools",
    "No Perceived Need",
]

Q2_SITUATION_ORDER = [
    "Graded Writing",
    "Creative/Personal Writing",
    "When Prohibited",
    "Research",
    "Exams",
    "Math & STEM",
    "Applications/Formal Documents",
    "Communications Writing",
    "All Situations",
]

Q2_WHY_ORDER = [
    "It's Cheating",
    "Fear Of Consequences",
    "Desire For Learning",
    "AI Inaccuracy/Distrust",
    "AI Writing is Unnatural",
    "AI Fails at Being Personal",
    "No Perceived Need",
]


# uncodable / exclude-from-analysis values
UNCODABLE_USED_AI = {"indeterminable"}
UNCODABLE_THEME = {"not enough info"}



def normalize_used_ai(raw):
    txt = clean_text(raw)
    if txt is None:
        return None

    low = txt.lower()
    if low == "yes":
        return "Yes"
    if low == "no":
        return "No"
    if low in UNCODABLE_USED_AI or "indeterminable" in low:
        return None
    return None



def normalize_q1_how(raw):
    txt = clean_text(raw)
    if txt is None:
        return None

    low = txt.strip().lower()
    if low in UNCODABLE_THEME:
        return None

    mapping = {
        "brainstorming": "Brainstorming",
        "proofreading": "Proofreading",
        "rephrasing": "Rephrasing",
        "summarizing": "Summarizing",
        "tutoring/conceptualizing": "Tutoring/Conceptualizing",
        "studying tool": "Studying Tool",
        "content generation/drafting": "Content Generation/Drafting",
        "research": "Research",
        "debugging/coding": "Debugging/Coding",
        "personal use": "Personal Use",
    }
    return mapping.get(low, txt.strip())



def normalize_q1_whynot(raw):
    txt = clean_text(raw)
    if txt is None:
        return None

    low = txt.strip().lower()
    if low in UNCODABLE_THEME:
        return None

    mapping = {
        "it's cheating": "It's Cheating",
        "fear of consequences": "Fear Of Consequences",
        "desire for learning": "Desire For Learning",
        "ai inaccuracy/distrust": "AI Inaccuracy/Distrust",
        "ai innacuracy/distrust": "AI Inaccuracy/Distrust",
        "desire for originality": "Desire For Originality",
        "ethical concerns": "Ethical Concerns",
        "fear of dependency": "Fear Of Dependency",
        "fear of dependency ": "Fear Of Dependency",
        "unfamiliar with tools": "Unfamiliar With Tools",
        "no perceived need": "No Perceived Need",
    }
    return mapping.get(low, txt.strip())



def normalize_q2_situation(raw):
    txt = clean_text(raw)
    if txt is None:
        return None

    low = txt.strip().lower()
    if low in UNCODABLE_THEME:
        return None

    mapping = {
        "graded writing": "Graded Writing",
        "creative/personal writing": "Creative/Personal Writing",
        "when prohibited": "When Prohibited",
        "research": "Research",
        "exams": "Exams",
        "math/stem": "Math & STEM",
        "math & stem": "Math & STEM",
        "applications/formal documents": "Applications/Formal Documents",
        "communications writing": "Communications Writing",
        "all situations": "All Situations",
    }
    return mapping.get(low, txt.strip())



def normalize_q2_why(raw):
    txt = clean_text(raw)
    if txt is None:
        return None

    low = txt.strip().lower()
    if low in UNCODABLE_THEME:
        return None

    mapping = {
        "it's cheating": "It's Cheating",
        "fear of consequences": "Fear Of Consequences",
        "desire for learning": "Desire For Learning",
        "desire for learning ": "Desire For Learning",
        "ai inaccuracy/distrust": "AI Inaccuracy/Distrust",
        "ai innacuracy/distrust": "AI Inaccuracy/Distrust",
        "ai writing is unnatural": "AI Writing is Unnatural",
        "ai fails at being personal": "AI Fails at Being Personal",
        "no perceived need": "No Perceived Need",
    }
    return mapping.get(low, txt.strip())


# ======= data extraction =======


def read_sheet(xlsx_path: str, sheet_name: str) -> Dict[str, object]:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, dtype=str)

    used_ai_counts = Counter()
    q1_how_counts = Counter()
    q1_whynot_counts = Counter()
    q2_situation_counts = Counter()
    q2_why_counts = Counter()

    n_q1_used_ai = 0
    n_q1_how = 0
    n_q1_whynot = 0
    n_q2_situations = 0
    n_q2_why = 0

    for _, row in df.iloc[DATA_START_ROW:].iterrows():
        q1_resp = clean_text(row.get(Q1_RESPONSE_COL))
        q2_resp = clean_text(row.get(Q2_RESPONSE_COL))

        if q1_resp:
            used_ai = normalize_used_ai(row.get(Q1_USED_COL))
            if used_ai:
                used_ai_counts[used_ai] += 1
                n_q1_used_ai += 1

            substantive_how_codes = []
            for col in Q1_HOW_COLS:
                code = normalize_q1_how(row.get(col))
                if code:
                    substantive_how_codes.append(code)
                    q1_how_counts[code] += 1

            substantive_whynot_codes = []
            for col in Q1_WHYNOT_COLS:
                code = normalize_q1_whynot(row.get(col))
                if code:
                    substantive_whynot_codes.append(code)
                    q1_whynot_counts[code] += 1

            # applicable use denominator: yes OR any substantive how code
            if used_ai == "Yes" or substantive_how_codes:
                n_q1_how += 1

            # applicable restraint denominator: no OR any substantive why-not code
            if used_ai == "No" or substantive_whynot_codes:
                n_q1_whynot += 1

        if q2_resp:
            row_situations = []
            for col in Q2_SITUATION_COLS:
                code = normalize_q2_situation(row.get(col))
                if code:
                    row_situations.append(code)
                    q2_situation_counts[code] += 1

            row_whys = []
            for col in Q2_WHY_COLS:
                code = normalize_q2_why(row.get(col))
                if code:
                    row_whys.append(code)
                    q2_why_counts[code] += 1

            if row_situations:
                n_q2_situations += 1
            if row_whys:
                n_q2_why += 1

    return {
        "sheet_name": sheet_name,
        "semester_label": semester_label(sheet_name),
        "semester_prefix": semester_prefix(sheet_name),
        "used_ai_counts": used_ai_counts,
        "q1_how_counts": q1_how_counts,
        "q1_whynot_counts": q1_whynot_counts,
        "q2_situation_counts": q2_situation_counts,
        "q2_why_counts": q2_why_counts,
        "n_q1_used_ai": n_q1_used_ai,
        "n_q1_how": n_q1_how,
        "n_q1_whynot": n_q1_whynot,
        "n_q2_situations": n_q2_situations,
        "n_q2_why": n_q2_why,
    }


# ======= plot helpers =======


def plot_distribution(counts, order, colors, n_total, title, subtitle_note, output_png):
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, max(3, len(order) * 0.7 + 2)))
    ax.set_title(title)

    labels = order
    vals = [counts.get(lbl, 0) for lbl in labels]
    pcts = [v / n_total * 100.0 if n_total > 0 else 0 for v in vals]
    bar_colors = [colors.get(lbl, "#999999") for lbl in labels]

    bars = ax.barh(labels, pcts, color=bar_colors)

    for i, (bar, cnt, pct) in enumerate(zip(bars, vals, pcts)):
        ax.text(pct + 1, i, f"{pct:.0f}% ({cnt})", va="center", fontsize=10)

    ax.set_xlabel("% of codable responses")
    ax.set_xlim(0, max(pcts) * 1.35 if pcts and max(pcts) > 0 else 50)

    fig.text(
        0.5,
        0.01,
        f"N = {n_total} codable Q1 responses. {subtitle_note}",
        ha="center",
        fontsize=8,
        style="italic",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)



def plot_theme_bar(counts, n_total, title, category_order, bar_color, note, output_png):
    output_png.parent.mkdir(parents=True, exist_ok=True)

    present = [(cat, counts.get(cat, 0)) for cat in category_order if counts.get(cat, 0) > 0]
    present.sort(key=lambda x: x[1])

    if not present:
        fig = plt.figure(figsize=(10, 5))
        plt.title(title)
        plt.text(0.5, 0.5, "No coded responses", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(11, max(4, len(present) * 0.55 + 2)))
    ax.set_title(title)

    labels = [c for c, _ in present]
    vals = [v for _, v in present]
    pcts = [v / n_total * 100.0 if n_total > 0 else 0 for v in vals]

    bars = ax.barh(labels, pcts, color=bar_color)

    for i, (bar, cnt, pct) in enumerate(zip(bars, vals, pcts)):
        ax.text(pct + 0.8, i, f"{pct:.0f}% ({cnt})", va="center", fontsize=9)

    ax.set_xlabel("% of applicable responses citing this theme")
    ax.set_xlim(0, max(pcts) * 1.3 if pcts else 50)

    fig.text(
        0.5,
        0.01,
        f"N = {n_total}. {note} One response can cite multiple themes.",
        ha="center",
        fontsize=8,
        style="italic",
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ======= chart builders =======


def make_semester_charts(stats: Dict[str, object], out_dir: Path) -> None:
    sem = stats["semester_label"]
    prefix = stats["semester_prefix"]

    plot_distribution(
        stats["used_ai_counts"],
        USED_AI_ORDER,
        USED_AI_COLORS,
        stats["n_q1_used_ai"],
        f"{sem}. Have students used AI-based text generators?\n"
        f"Coded from open-ended Question 1",
        "Indeterminable responses were excluded as uncodable rather than treated as a category.",
        out_dir / f"{prefix}_used_ai_bar.png",
    )

    plot_theme_bar(
        stats["q1_how_counts"],
        stats["n_q1_how"],
        f"{sem}. How students use AI-based text generators\n"
        f"Theme frequency from Question 1",
        Q1_HOW_ORDER,
        "#4575b4",
        "Applicable-use denominator: respondents who said Yes or were coded with at least one substantive use theme. 'Not Enough Info' was excluded as uncodable.",
        out_dir / f"{prefix}_q1_how_bar.png",
    )

    plot_theme_bar(
        stats["q1_whynot_counts"],
        stats["n_q1_whynot"],
        f"{sem}. Why students avoid or limit AI-based text generators\n"
        f"Theme frequency from Question 1",
        Q1_WHYNOT_ORDER,
        "#d73027",
        "Applicable-restraint denominator: respondents who said No or were coded with at least one substantive why-not theme. 'Not Enough Info' was excluded as uncodable.",
        out_dir / f"{prefix}_q1_whynot_bar.png",
    )

    plot_theme_bar(
        stats["q2_situation_counts"],
        stats["n_q2_situations"],
        f"{sem}. Situations where students would choose not to use AI\n"
        f"Theme frequency from Question 2",
        Q2_SITUATION_ORDER,
        "#4575b4",
        "Denominator includes only Question 2 responses with at least one substantive situation code. 'Not Enough Info' was excluded as uncodable.",
        out_dir / f"{prefix}_q2_situations_bar.png",
    )

    plot_theme_bar(
        stats["q2_why_counts"],
        stats["n_q2_why"],
        f"{sem}. Why students would choose not to use AI in those situations\n"
        f"Theme frequency from Question 2",
        Q2_WHY_ORDER,
        "#d73027",
        "Denominator includes only Question 2 responses with at least one substantive why code. 'Not Enough Info' was excluded as uncodable.",
        out_dir / f"{prefix}_q2_why_bar.png",
    )



def make_combined_stats(all_stats: List[Dict[str, object]]) -> Dict[str, object]:
    combined = {
        "semester_label": "All compatible sheets combined",
        "semester_prefix": "all_semesters",
        "used_ai_counts": Counter(),
        "q1_how_counts": Counter(),
        "q1_whynot_counts": Counter(),
        "q2_situation_counts": Counter(),
        "q2_why_counts": Counter(),
        "n_q1_used_ai": 0,
        "n_q1_how": 0,
        "n_q1_whynot": 0,
        "n_q2_situations": 0,
        "n_q2_why": 0,
    }

    for stats in all_stats:
        merge_counts_in_place(combined["used_ai_counts"], stats["used_ai_counts"])
        merge_counts_in_place(combined["q1_how_counts"], stats["q1_how_counts"])
        merge_counts_in_place(combined["q1_whynot_counts"], stats["q1_whynot_counts"])
        merge_counts_in_place(combined["q2_situation_counts"], stats["q2_situation_counts"])
        merge_counts_in_place(combined["q2_why_counts"], stats["q2_why_counts"])

        combined["n_q1_used_ai"] += stats["n_q1_used_ai"]
        combined["n_q1_how"] += stats["n_q1_how"]
        combined["n_q1_whynot"] += stats["n_q1_whynot"]
        combined["n_q2_situations"] += stats["n_q2_situations"]
        combined["n_q2_why"] += stats["n_q2_why"]

    return combined


# ======= main =======


def run(xlsx_path: str) -> None:
    out_dir = Path("output") / VIZ_SLUG
    out_dir.mkdir(parents=True, exist_ok=True)

    xls = pd.ExcelFile(xlsx_path)
    target_sheets = []
    for sheet_name in xls.sheet_names:
        preview = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None, dtype=str, nrows=3)
        if sheet_is_compatible(preview):
            target_sheets.append(sheet_name)

    if not target_sheets:
        print(
            f"Warning: {xlsx_path} does not appear to contain any compatible coded sheets."
        )
        return

    all_stats = []
    for sheet_name in target_sheets:
        stats = read_sheet(xlsx_path, sheet_name)
        all_stats.append(stats)
        make_semester_charts(stats, out_dir)

    combined_stats = make_combined_stats(all_stats)
    make_semester_charts(combined_stats, out_dir)

    print(f"Processed {len(target_sheets)} compatible sheets.")
    print(f"Wrote charts to {out_dir.resolve()}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python reasons_not_using.py <path_to_xlsx>")
    else:
        run(sys.argv[1])
