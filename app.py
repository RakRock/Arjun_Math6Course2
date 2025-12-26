import json
import os
import re
import sqlite3
import tempfile
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple

import pdfplumber
import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
UNITS_DATA_PATH = BASE_DIR / "units_data.json"
PROGRESS_DB_PATH = BASE_DIR / "progress.db"
MODEL_NAME = "grok-4-fast"
API_BASE_URL = "https://api.x.ai/v1"
PDF_FOLDER = BASE_DIR / "pdf_units"

KEYWORDS = [
    "ratio",
    "rate",
    "proportion",
    "fraction",
    "decimal",
    "percent",
    "integer",
    "equation",
    "expression",
    "inequality",
    "variable",
    "geometry",
    "area",
    "perimeter",
    "volume",
    "surface area",
    "triangle",
    "circle",
    "angle",
    "probability",
    "statistics",
    "mean",
    "median",
    "mode",
    "range",
    "box plot",
    "histogram",
    "data",
    "unit rate",
    "slope",
    "graph",
    "table",
]

# Map units to skills for revision/progress displays
UNIT_SKILL_MAP = {
    1: ["integers", "expressions"],
    2: ["equations", "inequalities"],
    3: ["ratios", "percents"],
    4: ["proportions", "unit rates"],
    5: ["geometry", "area", "perimeter"],
    6: ["volume", "surface area"],
    7: ["statistics", "data displays"],
}


def load_units_data() -> Dict[str, Dict[str, int]]:
    if not UNITS_DATA_PATH.exists():
        return {}
    try:
        with UNITS_DATA_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_units_data(data: Dict[str, Dict[str, int]]) -> None:
    with UNITS_DATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def clean_text(text: str) -> str:
    lowered = text.lower()
    cleaned = re.sub(r"[^\w\s]", " ", lowered)
    return cleaned


def extract_keywords_from_pdf(uploaded_file) -> Dict[str, int]:
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            temp_path = tmp.name

        full_text_parts: List[str] = []
        with pdfplumber.open(temp_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                full_text_parts.append(page_text)

        cleaned = clean_text("\n".join(full_text_parts))
        tokens = cleaned.split()
        counts = Counter(token for token in tokens if token in KEYWORDS)
        return dict(counts)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


def extract_keywords_from_path(path: Path) -> Dict[str, int]:
    full_text_parts: List[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            full_text_parts.append(page_text)

    cleaned = clean_text("\n".join(full_text_parts))
    tokens = cleaned.split()
    counts = Counter(token for token in tokens if token in KEYWORDS)
    return dict(counts)


def ensure_pdf_folder() -> None:
    PDF_FOLDER.mkdir(exist_ok=True)


def process_pdfs_in_folder() -> Dict[str, Dict[str, int]]:
    ensure_pdf_folder()
    updated: Dict[str, Dict[str, int]] = {}
    for pdf_path in sorted(PDF_FOLDER.glob("*.pdf")):
        counts = extract_keywords_from_path(pdf_path)
        updated[pdf_path.name] = counts
    save_units_data(updated)
    return updated


def list_available_units(units_data: Dict[str, Dict[str, int]]) -> List[int]:
    units = set()
    for filename in units_data.keys():
        match = re.search(r"unit\s*(\d+)", filename, re.IGNORECASE)
        if match:
            units.add(int(match.group(1)))
    return sorted(units)


def aggregate_keywords(
    selected_units: List[int], units_data: Dict[str, Dict[str, int]]
) -> Counter:
    aggregated = Counter()
    for filename, counts in units_data.items():
        match = re.search(r"unit\s*(\d+)", filename, re.IGNORECASE)
        if match and int(match.group(1)) in selected_units:
            aggregated.update(counts)
    return aggregated


def top_topics(counter: Counter) -> List[str]:
    if not counter:
        return [f"{kw} (emphasis: 1)" for kw in KEYWORDS[:10]]
    return [
        f"{topic} (emphasis: {count})" for topic, count in counter.most_common(10)
    ]


def topic_weights(counter: Counter) -> List[Dict[str, int]]:
    if not counter:
        return [{"topic": kw, "weight": 1} for kw in KEYWORDS[:10]]
    return [
        {"topic": topic, "weight": count}
        for topic, count in counter.most_common(10)
    ]


def unit_topics(selected_units: List[int]) -> List[str]:
    """Fallback topics derived from selected units when keyword counts are empty."""
    topics = {skill for u in selected_units if u in UNIT_SKILL_MAP for skill in UNIT_SKILL_MAP[u]}
    return list(topics) if topics else KEYWORDS[:10]


def filter_counter(counter: Counter, allowed: List[str]) -> Counter:
    if not allowed:
        return counter
    allowed_set = set(allowed)
    return Counter({k: v for k, v in counter.items() if k in allowed_set})


def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=API_BASE_URL)


def generate_questions(
    client: OpenAI,
    selected_units: List[int],
    topics: List[str],
    weighted_topics: List[Dict[str, int]],
    allowed_topics: List[str],
    difficulty: str,
) -> List[Dict]:
    prompt = (
        "Generate 20 questions for 6th-grade Springboard Course 2 math."
        f" Units: {selected_units}. Focus on: {topics}."
        f" Difficulty: {difficulty}. Mix multiple-choice and short-answer."
        " Output JSON ONLY as an object:"
        ' {"questions": [{"question": str, "options": list or null, "answer": str,'
        ' "explanation": str}, ...exactly 20 items...]}'
        " Keep questions kid-friendly with real-life hooks."
        " Do not include any text before or after the JSON. No trailing commas."
        " Output exactly 20 questions—no more, no less."
        f" Use this weighted emphasis (allocate questions roughly proportionally): {json.dumps(weighted_topics)}."
        " Ensure top 3 weighted topics each get at least 2 questions; remaining topics at least 1."
        f" Use ONLY these allowed topics and content appropriate to the selected units—avoid topics from other units: {allowed_topics}."
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=2000,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful math quiz generator."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content.strip()
    st.session_state["last_quiz_raw"] = content
    try:
        parsed = _parse_json_safely(content)
        questions = _coerce_questions(parsed)
        if questions is None and isinstance(parsed, dict):
            # Final fallback: wrap single dict as a one-item list
            questions = [parsed]
    except json.JSONDecodeError as exc:
        recovered = _extract_dicts_from_text(content)
        if recovered:
            questions = _coerce_questions(recovered) or recovered
        else:
            raise ValueError(
                f"Could not parse quiz JSON. Please retry. Details: {exc}"
            ) from exc

    if not isinstance(questions, list):
        # Store for UI debugging
        st.session_state["last_quiz_raw"] = content
        raise ValueError("Quiz response was not a list. Please retry generation.")

    # Enforce expected count strictly
    if len(questions) < 20:
        st.session_state["last_quiz_raw"] = content
        raise ValueError(f"Expected 20 questions, but got {len(questions)}. Please retry.")
    if len(questions) > 20:
        st.session_state["last_quiz_raw"] = content
        st.session_state["last_quiz_error"] = (
            f"Warning: expected 20 questions but got {len(questions)}. "
            "Trimming to the first 20."
        )
        questions = questions[:20]
    cleaned_questions = []
    for item in questions:
        if not isinstance(item, dict) or "question" not in item:
            continue
        cleaned_questions.append(
            {
                "question": str(item.get("question", "")).strip(),
                "options": item.get("options") or None,
                "answer": str(item.get("answer", "")).strip(),
                "explanation": str(item.get("explanation", "")).strip(),
            }
        )
    if not cleaned_questions:
        raise ValueError("No valid questions returned. Please retry generation.")
    return cleaned_questions


def _parse_json_safely(raw: str) -> Dict:
    """Attempt to parse JSON, retrying with trimmed braces if needed."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Remove code fences if present
        cleaned = raw
        if "```" in cleaned:
            cleaned = cleaned.strip().strip("`")
        # Try object slice
        start_obj = cleaned.find("{")
        end_obj = cleaned.rfind("}")
        if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
            candidate = cleaned[start_obj : end_obj + 1]
            # Fix common missing commas between objects (}{ -> },{)
            candidate = re.sub(r"\}\s*\{", "},{", candidate)
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass
        # Try list slice
        start_list = cleaned.find("[")
        end_list = cleaned.rfind("]")
        if start_list != -1 and end_list != -1 and end_list > start_list:
            candidate = cleaned[start_list : end_list + 1]
            candidate = re.sub(r"\}\s*\{", "},{", candidate)
            return json.loads(candidate)
        raise


def _extract_dicts_from_text(raw: str) -> List[Dict]:
    """Best-effort extraction of JSON objects from arbitrary text."""
    decoder = json.JSONDecoder()
    objs = []
    idx = 0
    length = len(raw)
    while idx < length:
        try:
            obj, end = decoder.raw_decode(raw, idx)
            if isinstance(obj, dict):
                objs.append(obj)
            idx = end
        except json.JSONDecodeError:
            idx += 1
    return objs


def _coerce_questions(parsed) -> List[Dict]:
    """Normalize various model response shapes into a question list."""

    def find_question_list(obj):
        """Recursively search for a list of dicts containing 'question' keys."""
        if isinstance(obj, list):
            if obj and all(isinstance(x, dict) for x in obj):
                # Prefer lists that look like questions
                if any("question" in x for x in obj):
                    return obj
            # Recurse into elements
            for x in obj:
                found = find_question_list(x)
                if found:
                    return found
        if isinstance(obj, dict):
            # Single question dict
            if "question" in obj:
                return [obj]
            for val in obj.values():
                found = find_question_list(val)
                if found:
                    return found
        return None

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        # Preferred key
        if isinstance(parsed.get("questions"), list):
            return parsed["questions"]
        if "question" in parsed:
            return [parsed]
        # Common alternates
        for key in ("items", "data", "quiz", "result"):
            if isinstance(parsed.get(key), list):
                return parsed[key]
        # First list-valued entry
        for val in parsed.values():
            if isinstance(val, list):
                return val
        # Dict of numbered objects
        values = list(parsed.values())
        if values and all(isinstance(v, dict) for v in values):
            return values
        # Recursive search for a list of question dicts
        found = find_question_list(parsed)
        if found:
            return found
    return None


def grade_quiz(client: OpenAI, qa_payload: List[Dict]) -> Tuple[int, List[str]]:
    prompt = (
        "Grade the following quiz answers for a 6th-grade Springboard Course 2 quiz."
        " Return JSON ONLY: {\"score\": int, \"feedback\": list of strings}."
        " In feedback, list revision/needs-improvement notes first, strengths at the end."
        " Be brief, encouraging, and specific. Do not add prose outside JSON."
        f" Data: {json.dumps(qa_payload)}"
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=500,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a concise and fair math grader."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content.strip()
    try:
        result = _parse_json_safely(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse grading JSON. Details: {exc}") from exc

    score = int(result.get("score", 0))
    feedback = result.get("feedback") or []
    if not isinstance(feedback, list):
        feedback = [str(feedback)]
    return score, [str(item) for item in feedback]


def ensure_db() -> None:
    conn = sqlite3.connect(PROGRESS_DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS quizzes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student TEXT NOT NULL,
                quiz_date TEXT NOT NULL,
                units TEXT NOT NULL,
                score INTEGER NOT NULL,
                details TEXT,
                difficulty TEXT,
                revision TEXT
            )
            """
        )
        # Lightweight migration: add difficulty column if missing
        cols = {row[1] for row in conn.execute("PRAGMA table_info(quizzes)")}
        if "difficulty" not in cols:
            conn.execute("ALTER TABLE quizzes ADD COLUMN difficulty TEXT")
            cols.add("difficulty")
        if "revision" not in cols:
            conn.execute("ALTER TABLE quizzes ADD COLUMN revision TEXT")
        conn.commit()
    finally:
        conn.close()


def save_quiz_result(
    student: str,
    units: List[int],
    score: int,
    feedback: List[str],
    difficulty: str,
    revision: str,
) -> None:
    ensure_db()
    conn = sqlite3.connect(PROGRESS_DB_PATH)
    try:
        units_str = ",".join(str(u) for u in sorted(units))
        conn.execute(
            """
            INSERT INTO quizzes (student, quiz_date, units, score, details, difficulty, revision)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student,
                date.today().isoformat(),
                units_str,
                score,
                json.dumps(feedback),
                difficulty,
                revision,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def fetch_progress(student: str) -> List[Tuple[str, int, str]]:
    if not PROGRESS_DB_PATH.exists():
        return []
    conn = sqlite3.connect(PROGRESS_DB_PATH)
    try:
        cursor = conn.execute(
            "SELECT units, score, quiz_date, details, difficulty, revision FROM quizzes WHERE student = ?",
            (student,),
        )
        return cursor.fetchall()
    finally:
        conn.close()


def render_progress(student: str) -> None:
    records = fetch_progress(student)
    if not records:
        st.info("No quizzes yet.")
        return

    unit_scores: defaultdict[int, List[int]] = defaultdict(list)
    daily: Dict[str, Dict] = {}
    table_rows = []
    # Map internal difficulty codes to labels
    difficulty_labels_inv = {
        "easy": "Rookie Quest",
        "medium": "Adventurer Quest",
        "complex": "Boss Battle",
    }
    all_quizzes_count = len(records)
    last_score = records[-1][1] if records else None

    for units_str, score, quiz_day, details_json, difficulty, revision in records:
        quiz_day = quiz_day or ""
        diff_label = difficulty or ""
        table_rows.append(
            {
                "Date": quiz_day,
                "Units": units_str,
                "Score": score,
                "Mode": difficulty_labels_inv.get(diff_label, diff_label),
                "Needs Revision": revision or "",
                "Feedback": ", ".join(json.loads(details_json)[:2]) if details_json else "",
            }
        )
        if quiz_day not in daily:
            daily[quiz_day] = {
                "count": 0,
                "units": set(),
                "score_sum": 0,
            }
        daily[quiz_day]["count"] += 1
        daily[quiz_day]["score_sum"] += score
        for unit in units_str.split(","):
            unit = unit.strip()
            if unit.isdigit():
                unit_int = int(unit)
                unit_scores[unit_int].append(score)
                daily[quiz_day]["units"].add(unit_int)

    if not daily:
        st.info("No quizzes yet.")
        return

    # Build heatmap data for last 365 days (or available range)
    today = date.today()
    dates = [date.fromisoformat(d) for d in daily.keys() if d]
    if not dates:
        st.info("No quizzes yet.")
        return
    min_date = min(dates, default=today)
    start_date = max(min_date, today.replace(year=today.year - 1))

    all_days = pd.date_range(start=start_date, end=today, freq="D")
    heat_rows = []
    for day in all_days:
        day_str = day.date().isoformat()
        meta = daily.get(day_str, {"count": 0, "units": set(), "score_sum": 0})
        count = meta["count"]
        units_list = sorted(meta["units"]) if meta["units"] else []
        skills = sorted(
            {skill for u in units_list for skill in UNIT_SKILL_MAP.get(u, [])}
        )
        tooltip = (
            f"Date: {day_str}<br>"
            f"Quizzes: {count}<br>"
            f"Units: {units_list}<br>"
            f"Total Score: {meta['score_sum']}/20 per quiz<br>"
            f"Skills: {skills or '—'}"
        )
        week = (day.date() - start_date).days // 7
        heat_rows.append(
            {
                "week": week,
                "weekday": day.day_name(),
                "count": count,
                "tooltip": tooltip,
            }
        )

    heat_df = pd.DataFrame(heat_rows)
    # Order weekdays to mimic GitHub style (Sunday on top)
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat_df["weekday"] = pd.Categorical(heat_df["weekday"], categories=weekday_order, ordered=True)

    fig = px.density_heatmap(
        heat_df,
        x="week",
        y="weekday",
        z="count",
        color_continuous_scale="Greens",
        hover_data={"tooltip": True},
    )
    fig.update_traces(hovertemplate="%{customdata}")
    fig.update_layout(
        title="Quiz Activity (last 365 days or available dates)",
        xaxis_title="Week",
        yaxis_title="Day of Week",
        coloraxis_colorbar=dict(title="Quizzes"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
    fig.update_traces(customdata=heat_df["tooltip"])

    st.subheader("Progress Dashboard")
    st.caption("Activity heatmap (darker = more quizzes that day)")
    st.plotly_chart(fig, width="stretch")

    # Summary metrics
    quiz_days = {d for d in daily.keys() if d}
    col1, col2, col3 = st.columns(3)
    col1.metric("Unique Days", len(quiz_days))
    col2.metric("Total Quizzes", all_quizzes_count)
    col3.metric("Last Score", f"{last_score}/20" if last_score is not None else "—")

    st.divider()
    st.subheader("Unit Averages")
    units_sorted = sorted(unit_scores.items())
    if units_sorted:
        cols = st.columns(min(3, len(units_sorted)))
        for idx, (unit, scores) in enumerate(units_sorted):
            with cols[idx % len(cols)]:
                avg_score = sum(scores) / len(scores)
                st.write(f"**Unit {unit}**")
                st.write(f"Avg {avg_score:.1f}/20 • {len(scores)} quiz(es)")
    else:
        st.info("No unit data yet.")

    st.divider()
    st.subheader("Quiz History")
    df = pd.DataFrame(table_rows).sort_values("Date", ascending=False)

    # Filters
    all_units = sorted({u for row in df["Units"] for u in row.split(",") if u.strip().isdigit()}) if not df.empty else []
    all_modes = sorted({m for m in df["Mode"] if m})
    f1, f2 = st.columns(2)
    sel_units = f1.multiselect("Filter by unit", options=all_units, default=all_units)
    sel_modes = f2.multiselect("Filter by mode", options=all_modes, default=all_modes or None)

    if sel_units:
        df = df[df["Units"].apply(lambda x: any(u.strip() in sel_units for u in x.split(",")))]
    if sel_modes:
        df = df[df["Mode"].isin(sel_modes)]

    recent = df.head(10)
    st.dataframe(recent, width="stretch", height=320)
    if len(df) > len(recent):
        with st.expander(f"Show all ({len(df)} rows)"):
            st.dataframe(df, width="stretch", height=500)


def init_session_state() -> None:
    st.session_state.setdefault("questions", [])
    st.session_state.setdefault("selected_units", [])
    st.session_state.setdefault("last_quiz_raw", "")
    st.session_state.setdefault("last_quiz_error", "")
    st.session_state.setdefault("quiz_locked", False)
    st.session_state.setdefault("show_progress", False)


def reset_quiz_state(clear_questions: bool = True) -> None:
    """Reset quiz interaction state and optionally clear questions."""
    for key in list(st.session_state.keys()):
        if key.startswith("answer_"):
            del st.session_state[key]
    if clear_questions:
        st.session_state["questions"] = []
    st.session_state["answers_submitted"] = False
    st.session_state["quiz_locked"] = False
    st.session_state["last_quiz_error"] = ""


def escape_markdown(text: str) -> str:
    """Escape characters that Streamlit markdown treats specially (including $)."""
    if text is None:
        return ""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>$])", r"\\\1", str(text))


def clean_question_text(text: str, has_options: bool) -> str:
    """Remove embedded option text if options are rendered separately."""
    if text is None:
        return ""
    cleaned = str(text)
    cleaned = cleaned.replace("(Multiple choice)", "").replace("(multiple choice)", "")
    if has_options:
        # Drop lines after the first line to avoid inline options
        cleaned = cleaned.splitlines()[0]
        # Trim at common option prefixes
        for marker in [" A)", " A.", "A)", "A."]:
            if marker in cleaned:
                cleaned = cleaned.split(marker)[0]
                break
    return cleaned.strip()


def summarize_revision(feedback: List[str], units: List[int], max_items: int = 3) -> str:
    """Friendly revision summary: concepts by unit + top feedback notes."""
    skill_set = {
        skill for u in units if u in UNIT_SKILL_MAP for skill in UNIT_SKILL_MAP[u]
    }
    parts = []
    if skill_set:
        parts.append("Concepts to review: " + ", ".join(sorted(skill_set)))
    if feedback:
        parts.append("Key notes: " + "; ".join(feedback[:max_items]))
    return " | ".join(parts)
    st.session_state.setdefault("answers_submitted", False)


def reset_quiz_state(clear_questions: bool = True) -> None:
    """Reset state for starting a fresh quiz."""
    if clear_questions:
        st.session_state["questions"] = []
    st.session_state["quiz_locked"] = False
    st.session_state["answers_submitted"] = False
    st.session_state["last_quiz_error"] = ""
    # Clear any stored answers
    for key in list(st.session_state.keys()):
        if key.startswith("answer_"):
            del st.session_state[key]


def main() -> None:
    st.set_page_config(page_title="6th Grade Springboard Course 2 Quiz Generator")
    init_session_state()
    ensure_db()
    ensure_pdf_folder()
    units_data = load_units_data()

    api_key = os.getenv("XAI_API_KEY") or st.secrets.get("XAI_API_KEY", "")
    if not api_key:
        st.info(
            "Set environment variable XAI_API_KEY for Grok access. "
            "Current session uses no live API key."
        )

    st.title("6th Grade Springboard Course 2 Quiz Generator")
    student_name = st.text_input("Student Name (required)")
    debug_raw = st.checkbox("Show raw quiz response (debug)", value=False)
    st.session_state["last_quiz_error"] = ""

    if st.button("Start New Quiz"):
        reset_quiz_state(clear_questions=True)
        st.success("Quiz state cleared. Select units and generate a new quiz.")

    st.info(
        f"Copy Springboard Course 2 PDFs into the folder: {PDF_FOLDER} "
        "and click Refresh to load/update keyword data."
    )
    if st.button("Refresh PDFs from folder"):
        with st.spinner("Processing PDFs for keywords..."):
            units_data = process_pdfs_in_folder()
        st.success("PDF folder processed and keyword data updated.")

    available_units = list_available_units(units_data)
    selected_units = st.multiselect(
        "Select units to include",
        options=available_units,
        default=st.session_state.get("selected_units", []),
    )
    st.session_state["selected_units"] = selected_units

    difficulty_labels = {
        "Rookie Quest": "easy",
        "Adventurer Quest": "medium",
        "Boss Battle": "complex",
    }
    difficulty_label = st.selectbox(
        "Pick your challenge mode",
        options=list(difficulty_labels.keys()),
        help="Kid-friendly game modes",
    )
    difficulty = difficulty_labels[difficulty_label]

    if st.sidebar.button("View Progress"):
        if student_name:
            st.session_state["show_progress"] = True
        else:
            st.sidebar.warning("Enter the student name to view progress.")

    if st.session_state.get("show_progress"):
        st.subheader("Progress")
        if student_name:
            render_progress(student_name)
        else:
            st.warning("Enter the student name to view progress.")
        if st.button("Back to Quiz"):
            st.session_state["show_progress"] = False
        return

    if st.button("Generate Quiz"):
        generation_error = None
        # Starting a new quiz unlocks submission and clears previous answers
        reset_quiz_state(clear_questions=False)
        if not student_name:
            generation_error = "Please enter the student name before generating a quiz."
        if not selected_units:
            generation_error = generation_error or "Select at least one unit to generate a quiz."
        if not api_key:
            generation_error = generation_error or "Missing XAI_API_KEY. Set it and retry."

        if generation_error:
            st.session_state["last_quiz_error"] = generation_error
            st.session_state["questions"] = []
        else:
            aggregated = aggregate_keywords(selected_units, units_data)
            allowed = unit_topics(selected_units)
            filtered = filter_counter(aggregated, allowed) if aggregated else Counter()
            topics = top_topics(filtered) if filtered else allowed
            weighted_topics = topic_weights(filtered) if filtered else [
                {"topic": t, "weight": 1} for t in allowed
            ]
            client = get_client(api_key)

            with st.spinner("Generating quiz with Grok..."):
                try:
                    questions = generate_questions(
                        client=client,
                        selected_units=selected_units,
                        topics=topics,
                        weighted_topics=weighted_topics,
                        allowed_topics=allowed,
                        difficulty=difficulty,
                    )
                except Exception as exc:
                    generation_error = f"Quiz generation failed: {exc}"
                    st.session_state["last_quiz_error"] = generation_error
                    st.session_state["questions"] = []
            if not generation_error:
                st.session_state["questions"] = questions
                st.session_state["answers_submitted"] = False
                st.session_state["last_quiz_error"] = ""
                st.success("Quiz ready!")

    if st.session_state.get("last_quiz_error"):
        st.error(st.session_state["last_quiz_error"])

    if debug_raw and st.session_state.get("last_quiz_raw"):
        st.text_area("Last raw quiz response", st.session_state["last_quiz_raw"], height=200)

    questions = st.session_state.get("questions", [])
    if questions:
        st.subheader("Quiz")
        client = get_client(api_key) if api_key else None
        disable_inputs = st.session_state.get("quiz_locked", False)
        with st.form("quiz_form"):
            for idx, q in enumerate(questions):
                has_opts = bool(q.get("options"))
                q_text = escape_markdown(clean_question_text(q.get("question", ""), has_opts))
                st.markdown(f"**Q{idx + 1}.** {q_text}")
                options = q.get("options")
                key = f"answer_{idx}"
                if options and isinstance(options, list):
                    safe_opts = [escape_markdown(opt) for opt in options]
                    st.radio(
                        "Choose an answer:",
                        safe_opts,
                        key=key,
                        index=None,
                        disabled=disable_inputs,
                    )
                else:
                    st.text_input("Your answer:", key=key, disabled=disable_inputs)
                st.markdown("---")

            submitted = st.form_submit_button("Submit Answers", disabled=disable_inputs)

        if submitted and disable_inputs:
            st.info("Quiz locked. Start a new test to answer again.")
        elif submitted:
            # Input guard: ensure all answers provided
            missing = []
            for idx, q in enumerate(questions):
                ans = st.session_state.get(f"answer_{idx}", None)
                if ans is None or (isinstance(ans, str) and not ans.strip()):
                    missing.append(idx + 1)
            if missing:
                st.error(f"Please answer all questions before submitting. Missing: {missing}")
                return

            if not student_name:
                st.error("Student name is required to grade and save results.")
                return
            if not client:
                st.error("Missing XAI_API_KEY. Set it and retry.")
                return

            qa_payload = []
            for idx, q in enumerate(questions):
                user_answer = st.session_state.get(f"answer_{idx}", "")
                qa_payload.append(
                    {
                        "question": q.get("question", ""),
                        "user_answer": user_answer,
                        "correct_answer": q.get("answer", ""),
                        "explanation": q.get("explanation", ""),
                    }
                )

            with st.spinner("Grading with Grok..."):
                try:
                    score, feedback = grade_quiz(client, qa_payload)
                except Exception as exc:
                    st.error(f"Grading failed: {exc}")
                    return

            st.success("Quiz graded")
            st.metric("Score (out of 20)", score)
            st.caption(f"Units: {selected_units} | Mode: {difficulty_label}")
            if feedback:
                st.markdown("**Feedback (needs work first, strengths last):**")
                for item in feedback:
                    st.write(f"• {item}")

            # Map label back to internal difficulty code
            difficulty_code = difficulty
            revision = summarize_revision(feedback, selected_units, max_items=3)
            save_quiz_result(
                student_name,
                selected_units,
                score,
                feedback,
                difficulty_code,
                revision,
            )
            st.session_state["answers_submitted"] = True
            st.session_state["quiz_locked"] = True
        elif disable_inputs:
            st.info("Quiz locked. Start a new test to answer again.")


if __name__ == "__main__":
    main()

