# visualizations/permittedUsage.py
#
# tracks where students draw the this is cheating line for AI/writing tools
# in pitt comp courses from fall 2022 through fall 2024. the tricky part is that
# the survey format changed pretty significantly between the older and newer semesters
# (likert scales -> select-all), so a lot of this code is just bridging that gap
# so everything can land on the same dot plot at the end.
#
# outputs go in output/permitted_usage/

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import matplotlib.pyplot as plt


VIZ_NAME = "Permitted usage of tools in composition courses (Fall 2022-Fall 2024)"
VIZ_SLUG = "permitted_usage"
SHOW_PLOTS = False


# ======= small utilities that get used everywhere =======

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
    # turns fall 2022 into 2022fall - used as a filename prefix and dict key throughout
    parts = sheet_name.strip().split()
    if len(parts) != 2:
        return "".join(ch.lower() for ch in sheet_name if ch.isalnum())
    season = parts[0].lower()
    year = "".join(ch for ch in parts[1] if ch.isdigit())
    return f"{year}{season}"


def col_letter_to_index(col):
    # A=0, Z=25, AA=26, etc. standard excel column indexing
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


def parse_multiselect_codes(cell):
    # handles cells like 1,2,3 or 2.0,10.0 - the floats come from how pandas reads mixed columns
    if is_blank(cell):
        return []
    out = []
    for p in str(cell).strip().split(","):
        p = p.strip()
        if not p:
            continue
        try:
            out.append(int(float(p)))
        except ValueError:
            continue
    return out


def compute_respondent_n(df, cols):
    # skip the first 2 rows - theyre metadata/header rows in the xlsx, not actual responses.
    # a row counts as responded if its non-blank in ANY of the relevant columns
    df2  = df.iloc[2:]
    mask = None
    for c in cols:
        b    = df2[c].apply(is_blank)
        mask = b if mask is None else (mask & b)
    return int((~mask).sum()) # pyright: ignore[reportOptionalOperand]


def parse_int_code(cell):
    if is_blank(cell):
        return None
    try:
        return int(float(str(cell).strip()))
    except Exception:
        return None


# ======= Likert scale constants =======

# order here matches the numeric codes 1-6 in the spreadsheet, so dont rearrange
LIKERT_LABELS = [
    "Definitely not",
    "Probably not",
    "Maybe",
    "Probably yes",
    "Definitely yes",
    "I have no idea",
]

# i have no idea (code 6) is intentionally left out - averaging no idea in
# with actual opinions would drag the mean in a weird direction
LIKERT_SCORE_MAP = {
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
}


# ======= plot helpers =======

def plot_likert_stacked_breakdown(items, title, output_png):
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig      = plt.figure(figsize=(13, 7))
    labels   = [q for q, _, _ in items]
    counters = [c for _, c, _ in items]

    plt.title(title)
    ax   = plt.gca()
    left = [0] * len(labels)

    for bucket in LIKERT_LABELS:
        vals = [c.get(bucket, 0) for c in counters]
        ax.barh(labels, vals, left=left, label=bucket)
        left = [l + v for l, v in zip(left, vals)]

    ax.set_xlabel("Respondents (count)")
    ax.set_ylabel("")
    ax.legend(title="Response", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_likert_means(items, title, output_png):
    output_png.parent.mkdir(parents=True, exist_ok=True)

    items  = sorted(items, key=lambda t: t[1])
    labels = [t[0] for t in items]
    means  = [t[1] for t in items]
    ns     = [t[2] for t in items]
    noidea = [t[3] for t in items]

    fig = plt.figure(figsize=(13, 7))
    plt.title(title)
    ax = plt.gca()
    ax.barh(labels, means)
    ax.set_xlabel("Average permissibility score (1=Definitely not ... 5=Definitely yes)")
    ax.set_ylabel("")
    ax.set_xlim(0.8, 5.4)

    pad = 0.03
    for i, (m, n, ni) in enumerate(zip(means, ns, noidea)):
        ax.text(m + pad, i, f"{m:.2f} (n={n}, no idea={ni})", va="center")

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


def plot_selectall_counts(counter, title, output_png, respondent_n, sort_desc=False):
    # percent labels use respondent_n as the denominator, NOT the sum of selections,
    # since each person could pick multiple options (select-all format)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    items    = sorted(counter.items(), key=lambda kv: kv[1], reverse=sort_desc)
    if sort_desc:
        items = list(reversed(items))

    labels   = [k for k, _ in items]
    counts   = [v for _, v in items]
    percents = [c / respondent_n * 100 for c in counts]

    fig = plt.figure(figsize=(13, 8))
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


# ======= 2022/2023 Likert parsing =======

def normalize_likert_value_label(s):
    # tries exact match first, then falls through to fuzzy matching in case theres
    # extra whitespace or slightly different wording in the raw data
    if is_blank(s):
        return None
    txt = " ".join(str(s).strip().lower().split())

    mapping = {
        "definitely not": "Definitely not",
        "probably not":   "Probably not",
        "maybe":          "Maybe",
        "probably yes":   "Probably yes",
        "definitely yes": "Definitely yes",
        "i have no idea": "I have no idea",
        "no idea":        "I have no idea",
    }
    if txt in mapping:
        return mapping[txt]

    # fuzzy fallback - handles things like Definitely Not or extra words
    if "definitely" in txt and "not" in txt:
        return "Definitely not"
    if "probably" in txt and "not" in txt:
        return "Probably not"
    if txt == "maybe":
        return "Maybe"
    if "probably" in txt and "yes" in txt:
        return "Probably yes"
    if "definitely" in txt and "yes" in txt:
        return "Definitely yes"
    if "no idea" in txt:
        return "I have no idea"

    return None


def analyze_paired_likert_questions(df, q_pairs):
    relevant_cols = []
    for num_col, label_col, _ in q_pairs:
        relevant_cols.extend([num_col, label_col])

    resp_n    = compute_respondent_n(df, relevant_cols)
    df2       = df.iloc[2:]
    breakdown = []
    means     = []

    for num_col, label_col, short_label in q_pairs:
        counts      = Counter()
        n_nonblank  = 0
        mean_scores = []
        n_no_idea   = 0

        for _, row in df2.iterrows():
            num_code   = parse_int_code(row.get(num_col))
            label_text = normalize_likert_value_label(row.get(label_col))

            resolved = None
            if label_text is not None:
                resolved = label_text
            elif num_code in [1, 2, 3, 4, 5, 6]:
                resolved = LIKERT_LABELS[num_code - 1]

            if resolved is None:
                continue

            n_nonblank += 1
            counts[resolved] += 1

            # i have no idea respondents still show up in the breakdown chart,
            # but they should NOT count toward the mean - tracking them separately
            # so we can at least annotate how many there were
            if resolved == "I have no idea":
                n_no_idea += 1
            else:
                if num_code in LIKERT_SCORE_MAP:
                    mean_scores.append(LIKERT_SCORE_MAP[num_code])
                else:
                    # yeah I know this basically duplicates LIKERT_SCORE_MAP but
                    # that one maps code->score; here we sometimes only have the label
                    lbl_to_score = {
                        "Definitely not": 1,
                        "Probably not":   2,
                        "Maybe":          3,
                        "Probably yes":   4,
                        "Definitely yes": 5,
                    }
                    if resolved in lbl_to_score:
                        mean_scores.append(lbl_to_score[resolved])

        breakdown.append((short_label, counts, n_nonblank))

        if mean_scores:
            means.append((short_label, sum(mean_scores) / len(mean_scores), len(mean_scores), n_no_idea))
        else:
            means.append((short_label, 0.0, 0, n_no_idea))

    means = [t for t in means if t[2] > 0 or t[3] > 0]
    return breakdown, means, resp_n


# ======= 2024 select-all parsing =======

def detect_selectall_code_to_label_mapping(df, code_col, label_col, fallback_map, valid_codes):
    # rather than hardcoding code->label entirely, we infer it from the data itself by majority vote.
    # the fallback_map is just a safety net for codes that either never appear or where the
    # code count and label count dont line up in a given row (which does happen)
    obs_pairs     = Counter()
    mapping_votes = {c: Counter() for c in valid_codes}

    for _, row in df.iloc[2:].iterrows():
        codes      = parse_multiselect_codes(row.get(code_col))
        labels_raw = str(row.get(label_col)).strip() if not is_blank(row.get(label_col)) else ""
        labels     = [x.strip() for x in labels_raw.split(",")] if labels_raw else []
        labels     = [l for l in labels if l]

        # only use rows where the code count and label count match - mismatched rows
        # are too ambiguous to infer from and just get skipped
        if codes and labels and len(codes) == len(labels):
            for c, lbl in zip(codes, labels):
                if c in mapping_votes:
                    mapping_votes[c][lbl] += 1
                    obs_pairs[(c, lbl)] += 1
        else:
            continue

    # print(f"mapping_votes: {mapping_votes}")  # handy for debugging label inference issues

    resolved = {}
    for c in valid_codes:
        if mapping_votes[c]:
            resolved[c] = mapping_votes[c].most_common(1)[0][0]
        elif c in fallback_map:
            resolved[c] = fallback_map[c]
        else:
            resolved[c] = f"Unknown ({c})"

    return resolved, obs_pairs


# ======= combined dot plot (moral boundary curve) =======

# rough ordering from least invasive help (spell check) to most offloading (write the whole thing).
# the positioning is a judgment call but it makes the curve read intuitively left-to-right
BOUNDARY_AXIS = [
    "Check and correct spelling",
    "Check and correct grammar",
    "Expand word choice (thesaurus)",
    "Brainstorm ideas",
    "Produce outlines or summaries of texts",
    "Improve writing style",
    "Generate counter-arguments",
    "Complete sentences",
    "Write full sentences",
    "Write full paragraphs",
    "Write full essays",
]

# older semesters only asked about 2 specific scenarios; mapping those onto the
# closest spot on the common axis so they show up in the dot plot at all
SCENARIO_TO_AXIS = {
    "Revise/fix/style essay":                  "Improve writing style",
    "Revise/fix/style essay (e.g., ChatGPT)":  "Improve writing style",
    "Write essay from prompts":                "Write full essays",
    "Write essay from prompts (e.g., ChatGPT)":"Write full essays",
}


def likert_allowed_percent(df, num_col, label_col):
    # allowed = probably yes (4) or definitely yes (5)
    # i have no idea (6) is excluded from the denominator entirely -
    # they didnt pick a side so they shouldnt count toward allowed OR not allowed
    df2     = df.iloc[2:]
    allowed = 0
    usable  = 0

    for _, row in df2.iterrows():
        num_code   = parse_int_code(row.get(num_col))
        label_text = normalize_likert_value_label(row.get(label_col))

        code = None
        if num_code in [1, 2, 3, 4, 5, 6]:
            code = num_code
        elif label_text is not None:
            # need this to convert label->code when the numeric column is blank;
            # the label col is sometimes more reliably populated
            lbl_to_code = {
                "Definitely not": 1,
                "Probably not":   2,
                "Maybe":          3,
                "Probably yes":   4,
                "Definitely yes": 5,
                "I have no idea": 6,
            }
            code = lbl_to_code.get(label_text)

        if code is None:
            continue
        if code == 6:
            continue  # skip i have no idea

        usable += 1
        if code in (4, 5):
            allowed += 1

    if usable == 0:
        return None
    return allowed / usable * 100.0


def plot_dot_plot(series, title, output_png, n_by_semester=None):
    output_png.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 7))
    plt.title(title)
    ax  = plt.gca()
    x   = list(range(len(BOUNDARY_AXIS)))

    for sem_label, values in series:
        y = [values.get(ax_lbl, float("nan")) for ax_lbl in BOUNDARY_AXIS]
        ax.plot(x, y, marker="o", linestyle="None", label=sem_label)

    ax.set_xticks(x)
    ax.set_xticklabels(BOUNDARY_AXIS, rotation=35, ha="right")
    ax.set_ylabel("% allowed")
    ax.set_ylim(-2, 102)
    ax.grid(True, axis="y", alpha=0.25)

    legend1 = ax.legend(title="Semester", loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)

    if n_by_semester:
        from matplotlib.lines import Line2D

        # preserve order from series so the n= list matches the dot colors above
        sem_order = []
        for sem_label, _ in series:
            if sem_label not in sem_order:
                sem_order.append(sem_label)

        handles2 = []
        for sem_label in sem_order:
            if sem_label in n_by_semester:
                handles2.append(
                    Line2D([0], [0], linestyle="none", marker="", color="none",
                           label=f"{sem_label}: n={n_by_semester[sem_label]}")
                )

        ax.legend(handles=handles2, title="Sample size", loc="lower left",
                  bbox_to_anchor=(1.02, 0.0), frameon=False, handlelength=0, handletextpad=0.0)

        # matplotlib drops the first legend when you add a second one,
        # so gotta explicitly re-add it with add_artist
        ax.add_artist(legend1)

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ======= combined Likert distribution over time =======

LIKERT_SEMESTER_ORDER = ["2022fall", "2023spring", "2023fall"]
LIKERT_SEMESTER_RANK  = {s: i for i, s in enumerate(LIKERT_SEMESTER_ORDER)}

# fall 2022 didnt include the (e.g., ChatGPT) parenthetical; later semesters did.
# normalizing so both variants collapse into the same scenario for the distribution charts.
SCENARIO_CANON = {
    "Revise/fix/style essay":                  "Revise/fix/style essay",
    "Revise/fix/style essay (e.g., ChatGPT)":  "Revise/fix/style essay",
    "Write essay from prompts":                "Write essay from prompts",
    "Write essay from prompts (e.g., ChatGPT)":"Write essay from prompts",
}


def likert_response_distribution(df, num_col, label_col):
    df2    = df.iloc[2:]
    counts = Counter()
    n      = 0

    for _, row in df2.iterrows():
        num_code   = parse_int_code(row.get(num_col))
        label_text = normalize_likert_value_label(row.get(label_col))

        resolved = None
        if label_text is not None:
            resolved = label_text
        elif num_code in [1, 2, 3, 4, 5, 6]:
            resolved = LIKERT_LABELS[num_code - 1]

        if resolved is None or resolved not in LIKERT_LABELS:
            continue

        counts[resolved] += 1
        n += 1

    return counts, n


def plot_likert_dist_over_time(records, scenario_name, title, output_png):
    output_png.parent.mkdir(parents=True, exist_ok=True)

    filtered = []
    for sem_prefix, raw_scenario, cnts, n in records:
        if SCENARIO_CANON.get(raw_scenario, raw_scenario) == scenario_name:
            filtered.append((sem_prefix, cnts, n))

    fig = plt.figure(figsize=(12, 6))
    plt.title(title)

    if not filtered:
        plt.text(0.5, 0.5, f"No records found for scenario: {scenario_name}", ha="center", va="center")
        plt.axis("off")
        plt.tight_layout()
        fig.savefig(output_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    filtered = sorted(filtered, key=lambda t: LIKERT_SEMESTER_RANK.get(t[0], 999))

    # barh draws the first item at the bottom, so reversing makes the
    # earliest semester appear at the top of the chart (more natural to read)
    filtered = list(reversed(filtered))

    bar_labels = [sem for sem, _, _ in filtered]
    ns         = [n   for _, _, n   in filtered]
    buckets    = LIKERT_LABELS[:]

    pcts = {b: [] for b in buckets}
    for _, cnts, n in filtered:
        denom = n if n > 0 else 1
        for b in buckets:
            pcts[b].append(cnts.get(b, 0) / denom * 100.0)

    ax   = plt.gca()
    left = [0.0] * len(bar_labels)

    for b in buckets:
        vals = pcts[b]
        ax.barh(bar_labels, vals, left=left, label=b)
        left = [l + v for l, v in zip(left, vals)]

    ax.set_xlabel("Percent of responses")
    ax.set_ylabel("")
    ax.set_xlim(0, 100)

    for i, n in enumerate(ns):
        ax.text(101, i, f"n={n}", va="center")

    ax.legend(title="Response", loc="center left", bbox_to_anchor=(1.12, 0.5), frameon=False)

    plt.tight_layout()
    fig.savefig(output_png, dpi=300, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close(fig)


# ======= semester configs =======

@dataclass(frozen=True)
class SemesterConfig:
    sheet_name: str
    mode: str  # paired_likert or select_all

    paired_questions: List[Tuple[str, str, str]] | None = None

    selectall_code_col_letter:  str | None = None
    selectall_label_col_letter: str | None = None
    selectall_fallback_map:     Dict[int, str] | None = None


FALL_2022 = SemesterConfig(
    sheet_name="Fall 2022",
    mode="paired_likert",
    paired_questions=[
        ("P", "Q", "Revise/fix/style essay"),
        ("R", "S", "Write essay from prompts"),
    ],
)

SPRING_2023 = SemesterConfig(
    sheet_name="Spring 2023",
    mode="paired_likert",
    paired_questions=[
        ("R", "S", "Revise/fix/style essay (e.g., ChatGPT)"),
        ("T", "U", "Write essay from prompts (e.g., ChatGPT)"),
    ],
)

FALL_2023 = SemesterConfig(
    sheet_name="Fall 2023",
    mode="paired_likert",
    paired_questions=[
        ("S", "T", "Revise/fix/style essay (e.g., ChatGPT)"),
        ("U", "V", "Write essay from prompts (e.g., ChatGPT)"),
    ],
)

# verified against the spreadsheet; generate counter-arguments being code 11 was
# a surprise - would have expected it to follow the general pattern better
ALLOWED_TOOLS_2024_MAP = {
    1:  "Check and correct spelling",
    2:  "Check and correct grammar",
    3:  "Brainstorm ideas",
    4:  "Produce outlines or summaries of texts",
    5:  "Improve writing style",
    6:  "Write full sentences",
    7:  "Expand word choice (thesaurus)",
    8:  "Complete sentences",
    9:  "Write full paragraphs",
    10: "Write full essays",
    11: "Generate counter-arguments",
}

SPRING_2024 = SemesterConfig(
    sheet_name="Spring 2024",
    mode="select_all",
    selectall_code_col_letter="AO",
    selectall_label_col_letter="AP",
    selectall_fallback_map=ALLOWED_TOOLS_2024_MAP,
)

FALL_2024 = SemesterConfig(
    sheet_name="Fall 2024",
    mode="select_all",
    selectall_code_col_letter="AB",
    selectall_label_col_letter="AC",
    selectall_fallback_map=ALLOWED_TOOLS_2024_MAP,
)

SEMESTERS = (FALL_2022, SPRING_2023, FALL_2023, SPRING_2024, FALL_2024)


# ======= main =======

def run(xlsx_path: str) -> None:
    out_dir = Path("output") / VIZ_SLUG
    out_dir.mkdir(parents=True, exist_ok=True)

    boundary_series = []
    dist_records    = []
    n_by_semester   = {}

    for sem in SEMESTERS:
        sheet  = sem.sheet_name
        prefix = semester_prefix(sheet)

        df = pd.read_excel(xlsx_path, sheet_name=sheet, dtype=str)

        if sem.mode == "paired_likert":
            assert sem.paired_questions

            resolved_pairs = []
            for num_letter, label_letter, short_label in sem.paired_questions:
                num_col   = get_col_name_by_letter(df, num_letter)
                label_col = get_col_name_by_letter(df, label_letter)
                resolved_pairs.append((num_col, label_col, short_label))

            for num_col, label_col, short_label in resolved_pairs:
                cnts, n = likert_response_distribution(df, num_col, label_col)
                dist_records.append((prefix, short_label, cnts, n))

            bdry_vals = {}
            for num_col, label_col, short_label in resolved_pairs:
                axis_label = SCENARIO_TO_AXIS.get(short_label)
                if not axis_label:
                    continue
                pct = likert_allowed_percent(df, num_col, label_col)
                if pct is not None:
                    bdry_vals[axis_label] = pct
            boundary_series.append((prefix, bdry_vals))

            breakdown, mean_data, resp_n = analyze_paired_likert_questions(df, resolved_pairs)
            n_by_semester[prefix] = resp_n

            q_text = str(df.loc[0, resolved_pairs[0][0]]).strip() if resolved_pairs else ""

            plot_likert_stacked_breakdown(
                breakdown,
                f"{sheet}. Allowed tools line (scenario breakdown)\nQuestion: {q_text}\nN = {resp_n}",
                out_dir / f"{prefix}_permitted_breakdown.png",
            )

            mean_data_for_plot = [t for t in mean_data if t[2] > 0]
            if mean_data_for_plot:
                plot_likert_means(
                    mean_data_for_plot,
                    (
                        f"{sheet}. Allowed tools line (mean permissibility)\n"
                        f"Question: {q_text}\n"
                        f"N = {resp_n}\n"
                        f"(1=Definitely not ... 5=Definitely yes; excludes i have no idea)"
                    ),
                    out_dir / f"{prefix}_permitted_mean.png",
                )
            else:
                # nothing to plot - just dump a blank with a message
                png = out_dir / f"{prefix}_permitted_mean.png"
                png.parent.mkdir(parents=True, exist_ok=True)
                fig = plt.figure(figsize=(12, 7))
                plt.title(f"{sheet}. Allowed tools line (mean permissibility)\nN = {resp_n}")
                plt.text(0.5, 0.5, "No usable responses for mean (all blank or i have no idea).",
                         ha="center", va="center")
                plt.axis("off")
                plt.tight_layout()
                fig.savefig(png, dpi=300, bbox_inches="tight")
                plt.close(fig)

        elif sem.mode == "select_all":
            assert sem.selectall_code_col_letter and sem.selectall_label_col_letter and sem.selectall_fallback_map

            code_col  = get_col_name_by_letter(df, sem.selectall_code_col_letter)
            label_col = get_col_name_by_letter(df, sem.selectall_label_col_letter)

            resp_n = compute_respondent_n(df, [code_col, label_col])
            n_by_semester[prefix] = resp_n

            valid_codes  = sorted(sem.selectall_fallback_map.keys())
            inferred_map, obs = detect_selectall_code_to_label_mapping(
                df=df,
                code_col=code_col,
                label_col=label_col,
                fallback_map=sem.selectall_fallback_map,
                valid_codes=valid_codes,
            )

            # union of codes seen in the data + codes in the fallback map,
            # so we dont miss anything that showed up unexpectedly
            seen_codes = set()
            for cell in df.iloc[2:][code_col]:
                seen_codes.update(parse_multiselect_codes(cell))
            candidate_codes = sorted(seen_codes.union(sem.selectall_fallback_map.keys()))

            code_to_label = {
                c: inferred_map.get(c, sem.selectall_fallback_map.get(c, f"Unknown ({c})"))
                for c in candidate_codes
            }

            # inline the old _count_selectall_codes - it was just a loop
            counts = Counter()
            for cell in df.iloc[2:][code_col]:
                for c in parse_multiselect_codes(cell):
                    counts[code_to_label.get(c, f"Unknown ({c})")] += 1

            q_text = str(df.loc[0, code_col]).strip()

            plot_selectall_counts(
                counts,
                f"{sheet}. Allowed tools line (select all)\nQuestion: {q_text}\nN = {resp_n}",
                out_dir / f"{prefix}_permitted_selectall.png",
                respondent_n=resp_n,
                sort_desc=False,
            )

            bdry_vals = {ax_lbl: (counts.get(ax_lbl, 0) / resp_n) * 100.0 for ax_lbl in BOUNDARY_AXIS}
            boundary_series.append((prefix, bdry_vals))
            for sem_label, vals in boundary_series:
                print(sem_label, vals)  # left in - helps verify 2024 boundary values are coming out right

        else:
            raise ValueError(f"Unknown semester mode: {sem.mode}")

    if boundary_series:
        plot_dot_plot(
            boundary_series,
            "Allowed tool boundary over time (% allowed)\n"
            "Axis ordered from low assistance -> high offloading\n"
            "Older semesters plotted as scenario points (style revision vs full essay generation)",
            out_dir / "combined_dot_plot.png",
            n_by_semester=n_by_semester,
        )

    # two distribution charts for the likert semesters only (2022fall through 2023fall),
    # ordered fall 2022 -> spring 2023 -> fall 2023 top to bottom
    if dist_records:
        plot_likert_dist_over_time(
            dist_records,
            scenario_name="Revise/fix/style essay",
            title="Distribution over time. Revise/fix/style essay (Likert scenario)\n"
                  "Stacked bars show full response distribution per semester",
            output_png=out_dir / "combined_distribution_revise.png",
        )

        plot_likert_dist_over_time(
            dist_records,
            scenario_name="Write essay from prompts",
            title="Distribution over time. Write essay from prompts (Likert scenario)\n"
                  "Stacked bars show full response distribution per semester",
            output_png=out_dir / "combined_distribution_write_essay.png",
        )