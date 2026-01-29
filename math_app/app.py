import json
import os
import re
import smtplib
import sqlite3
import ssl
import tempfile
from io import StringIO
from collections import Counter, defaultdict
from datetime import date, datetime
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Tuple

import pdfplumber
import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI
from streamlit_autorefresh import st_autorefresh


BASE_DIR = Path(__file__).resolve().parent
db_dir_env = os.getenv("DB_DIR")
DB_DIR = Path(db_dir_env) if db_dir_env else BASE_DIR / ".data"
try:
    DB_DIR.mkdir(parents=True, exist_ok=True)
except PermissionError:
    DB_DIR = BASE_DIR / ".data"
    DB_DIR.mkdir(parents=True, exist_ok=True)
UNITS_DATA_PATH = BASE_DIR / "units_data.json"
UNITS_CONCEPTS_AI_PATH = BASE_DIR / "units_concepts_ai.json"
PROGRESS_DB_PATH = DB_DIR / "progress.db"
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


# ============ AI-Powered Concept Extraction ============

def load_ai_concepts() -> Dict[str, Dict[str, int]]:
    """Load AI-generated concepts from JSON file."""
    if UNITS_CONCEPTS_AI_PATH.exists():
        try:
            with UNITS_CONCEPTS_AI_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_ai_concepts(data: Dict[str, Dict[str, int]]) -> None:
    """Save AI-generated concepts to JSON file."""
    try:
        with UNITS_CONCEPTS_AI_PATH.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass


def extract_text_from_pdf(path: Path) -> str:
    """Extract full text from a PDF file."""
    full_text_parts: List[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            full_text_parts.append(page_text)
    return "\n".join(full_text_parts)


def extract_concepts_with_llm(client, unit_name: str, unit_text: str) -> Dict[str, int]:
    """Use LLM to extract meaningful math concepts from unit text."""
    # Truncate text if too long (keep first ~4000 chars for context)
    truncated_text = unit_text[:4000] if len(unit_text) > 4000 else unit_text
    
    prompt = f"""Analyze this 6th-grade math unit and identify the KEY MATHEMATICAL CONCEPTS being taught.

Unit: {unit_name}
Content excerpt:
{truncated_text}

Return a JSON object where:
- Keys are specific math concepts/topics (e.g., "solving two-step equations", "unit rates", "area of composite figures")
- Values are importance scores from 1-100 (higher = more central to the unit)

Focus on:
- What mathematical skills are being taught
- What formulas or methods are covered  
- What types of problems students will solve
- Key vocabulary and definitions

Return ONLY valid JSON, nothing else. Example format:
{{"solving two-step equations": 95, "combining like terms": 85, "distributive property": 80, "variables on both sides": 75}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=600,
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a math curriculum analyzer. Extract key concepts from unit content."},
                {"role": "user", "content": prompt},
            ],
        )
        content = response.choices[0].message.content.strip()
        parsed = json.loads(content)
        
        # Ensure all values are integers
        if isinstance(parsed, dict):
            return {str(k): int(v) for k, v in parsed.items() if isinstance(v, (int, float))}
        return {}
    except Exception as exc:
        st.warning(f"LLM concept extraction failed for {unit_name}: {exc}")
        return {}


def process_pdfs_with_llm(client, progress_callback=None) -> Dict[str, Dict[str, int]]:
    """Process all PDFs in folder using LLM to extract concepts."""
    ensure_pdf_folder()
    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))
    concepts = {}
    total = len(pdf_files)
    
    for idx, pdf_path in enumerate(pdf_files):
        unit_name = pdf_path.stem
        if progress_callback:
            progress_callback(idx + 1, total, unit_name)
        
        unit_text = extract_text_from_pdf(pdf_path)
        concepts[unit_name] = extract_concepts_with_llm(client, unit_name, unit_text)
    
    save_ai_concepts(concepts)
    return concepts


def get_ai_topics_for_units(selected_units: List[int], ai_concepts: Dict[str, Dict[str, int]]) -> List[Dict[str, int]]:
    """Get AI-generated topics for selected units."""
    aggregated: Counter = Counter()
    
    for unit_name, concepts in ai_concepts.items():
        # Check if this unit matches any selected unit number
        match = re.search(r"unit\s*(\d+)", unit_name, re.IGNORECASE)
        if match and int(match.group(1)) in selected_units:
            for concept, weight in concepts.items():
                aggregated[concept] = max(aggregated[concept], weight)
    
    if not aggregated:
        return []
    
    # Return top 10 concepts as list of dicts
    return [{"topic": topic, "weight": weight} for topic, weight in aggregated.most_common(10)]


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
        timeout=90,
        messages=[
            {"role": "system", "content": "You are a helpful math quiz generator."},
            {"role": "user", "content": prompt},
        ],
    )
    content = (response.choices[0].message.content or "").strip()
    st.session_state["last_quiz_raw"] = content
    if not content:
        raise ValueError("Quiz generation returned an empty response.")
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


def generate_boss_set(client: OpenAI, unit: int) -> List[Dict]:
    prompt = (
        "Generate 20 challenging 'Boss Battle' questions for 6th-grade Springboard Course 2 math."
        f" Unit: {unit}. Difficulty: complex. Each question must include the correct answer and explanation."
        " Format: JSON ONLY as an object: {\"questions\": [{\"question\": str, \"options\": list or null, \"answer\": str, \"explanation\": str}, ...exactly 20 items...]}"
        " Mix multiple-choice and short-answer. Use kid-friendly but rigorous wording. No extra text."
        " No trailing commas. Exactly 20 items."
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=2400,
        temperature=0.3,
        response_format={"type": "json_object"},
        timeout=90,
        messages=[
            {"role": "system", "content": "You are a precise math item writer."},
            {"role": "user", "content": prompt},
        ],
    )
    content = (response.choices[0].message.content or "").strip()
    st.session_state["last_quiz_raw"] = content
    parsed = _parse_json_safely(content)
    questions = _coerce_questions(parsed)
    if not isinstance(questions, list):
        raise ValueError("Boss set response was not a list.")
    if len(questions) < 20:
        raise ValueError(f"Expected 20 boss questions, got {len(questions)}.")
    cleaned: List[Dict] = []
    for item in questions[:20]:
        if not isinstance(item, dict) or "question" not in item or "answer" not in item:
            continue
        cleaned.append(
            {
                "question": str(item.get("question", "")).strip(),
                "options": item.get("options") or None,
                "answer": str(item.get("answer", "")).strip(),
                "explanation": str(item.get("explanation", "")).strip(),
            }
        )
    if len(cleaned) < 20:
        raise ValueError("Boss set had insufficient valid questions.")
    return cleaned


def store_boss_set(unit: int, questions: List[Dict]) -> int:
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO boss_sets (unit, created_at) VALUES (?, ?)",
            (unit, datetime.now().isoformat(timespec="seconds")),
        )
        boss_set_id = cur.lastrowid
        for q in questions:
            cur.execute(
                """
                INSERT INTO boss_questions (boss_set_id, question, options, answer, explanation)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    boss_set_id,
                    q.get("question", ""),
                    json.dumps(q.get("options")) if q.get("options") is not None else None,
                    q.get("answer", ""),
                    q.get("explanation", ""),
                ),
            )
        conn.commit()
        return boss_set_id
    finally:
        conn.close()


def fetch_boss_set(unit: int, student: str) -> Tuple[int | None, List[Dict]]:
    """Return next unused boss set for a student+unit, or (None, [])."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT bs.id
            FROM boss_sets bs
            WHERE bs.unit = ?
              AND bs.id NOT IN (
                  SELECT boss_set_id FROM boss_set_usage WHERE student = ?
              )
            ORDER BY bs.created_at ASC
            LIMIT 1
            """,
            (unit, student),
        )
        row = cur.fetchone()
        if not row:
            return None, []
        boss_set_id = row[0]
        cur.execute(
            """
            SELECT question, options, answer, explanation
            FROM boss_questions
            WHERE boss_set_id = ?
            ORDER BY id ASC
            """,
            (boss_set_id,),
        )
        qrows = cur.fetchall()
        questions: List[Dict] = []
        for qr in qrows:
            q_text, opts_json, ans, expl = qr
            opts = None
            if opts_json:
                try:
                    opts = json.loads(opts_json)
                except Exception:
                    opts = opts_json
            questions.append(
                {
                    "question": q_text or "",
                    "options": opts,
                    "answer": ans or "",
                    "explanation": expl or "",
                }
            )
        return boss_set_id, questions
    finally:
        conn.close()


def boss_set_counts(unit: int, student: str | None = None) -> Tuple[int, int | None]:
    """Return (total sets for unit, unused sets for student if provided)."""
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM boss_sets WHERE unit = ?", (unit,))
        total = cur.fetchone()[0] or 0
        if student is None:
            return total, None
        cur.execute(
            """
            SELECT COUNT(*)
            FROM boss_sets bs
            WHERE bs.unit = ?
              AND bs.id NOT IN (
                  SELECT boss_set_id FROM boss_set_usage WHERE student = ?
              )
            """,
            (unit, student),
        )
        unused = cur.fetchone()[0] or 0
        return total, unused
    finally:
        conn.close()


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
        timeout=60,
        messages=[
            {"role": "system", "content": "You are a concise and fair math grader."},
            {"role": "user", "content": prompt},
        ],
    )
    content = (response.choices[0].message.content or "").strip()
    st.session_state["last_grading_raw"] = content
    if not content:
        st.session_state["last_grading_error"] = "Grading returned an empty response."
        raise ValueError("Grading returned an empty response.")
    try:
        result = _parse_json_safely(content)
    except json.JSONDecodeError as exc:
        st.session_state["last_grading_error"] = f"JSON parse error: {exc}"
        raise ValueError(f"Could not parse grading JSON. Details: {exc}") from exc

    score = int(result.get("score", 0))
    feedback = result.get("feedback") or []
    if not isinstance(feedback, list):
        feedback = [str(feedback)]
    return score, [str(item) for item in feedback]


def get_conn():
    """Return a local SQLite connection."""
    return sqlite3.connect(PROGRESS_DB_PATH)


def ensure_db() -> None:
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS quizzes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student TEXT NOT NULL,
                quiz_date TEXT NOT NULL,
                units TEXT NOT NULL,
                score INTEGER NOT NULL,
                details TEXT,
                difficulty TEXT,
                revision TEXT,
                start_time TEXT,
                end_time TEXT
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS quiz_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                quiz_id INTEGER NOT NULL,
                question TEXT,
                options TEXT,
                correct_answer TEXT,
                user_answer TEXT,
                is_correct INTEGER,
                explanation TEXT,
                FOREIGN KEY (quiz_id) REFERENCES quizzes(id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS boss_sets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                unit INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS boss_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                boss_set_id INTEGER NOT NULL,
                question TEXT,
                options TEXT,
                answer TEXT,
                explanation TEXT,
                FOREIGN KEY (boss_set_id) REFERENCES boss_sets(id)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS boss_set_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                boss_set_id INTEGER NOT NULL,
                student TEXT NOT NULL,
                used_at TEXT NOT NULL,
                FOREIGN KEY (boss_set_id) REFERENCES boss_sets(id)
            )
            """
        )
        cols = {row[1] for row in cur.execute("PRAGMA table_info(quizzes)")}
        if "difficulty" not in cols:
            cur.execute("ALTER TABLE quizzes ADD COLUMN difficulty TEXT")
            cols.add("difficulty")
        if "revision" not in cols:
            cur.execute("ALTER TABLE quizzes ADD COLUMN revision TEXT")
        if "start_time" not in cols:
            cur.execute("ALTER TABLE quizzes ADD COLUMN start_time TEXT")
        if "end_time" not in cols:
            cur.execute("ALTER TABLE quizzes ADD COLUMN end_time TEXT")
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
    start_time: str,
    end_time: str,
    qa_payload: List[Dict],
    boss_set_id: int | None = None,
) -> None:
    ensure_db()
    conn = get_conn()
    try:
        cur = conn.cursor()
        units_str = ",".join(str(u) for u in sorted(units))
        cur.execute(
            """
            INSERT INTO quizzes (student, quiz_date, units, score, details, difficulty, revision, start_time, end_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student,
                date.today().isoformat(),
                units_str,
                score,
                json.dumps(feedback),
                difficulty,
                revision,
                start_time,
                end_time,
            ),
        )
        quiz_id = cur.lastrowid
        # Store per-question details and correctness
        for item in qa_payload:
            user_ans = item.get("user_answer", "")
            correct_ans = item.get("correct_answer", "")
            is_correct = None
            if correct_ans:
                is_correct = int(str(user_ans).strip().lower() == str(correct_ans).strip().lower())
            cur.execute(
                """
                INSERT INTO quiz_questions (quiz_id, question, options, correct_answer, user_answer, is_correct, explanation)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    quiz_id,
                    item.get("question", ""),
                    json.dumps(item.get("options")) if item.get("options") is not None else None,
                    correct_ans,
                    user_ans,
                    is_correct,
                    item.get("explanation", ""),
                ),
            )
        if boss_set_id is not None:
            cur.execute(
                """
                INSERT INTO boss_set_usage (boss_set_id, student, used_at)
                VALUES (?, ?, ?)
                """,
                (boss_set_id, student, datetime.now().isoformat(timespec="seconds")),
            )
        conn.commit()
    finally:
        conn.close()


def fetch_progress(student: str) -> List[Tuple[str, int, str]]:
    if PROGRESS_DB_PATH and not PROGRESS_DB_PATH.exists():
        return []
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT id, units, score, quiz_date, details, difficulty, revision, start_time, end_time FROM quizzes WHERE student = ?",
            (student,),
        )
        return cur.fetchall()
    finally:
        conn.close()


def fetch_quiz_questions(quiz_id: int) -> List[Dict]:
    """Return stored questions/answers for a quiz."""
    try:
        quiz_id_int = int(quiz_id)
    except Exception:
        return []
    conn = get_conn()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT question, options, correct_answer, user_answer, is_correct, explanation
            FROM quiz_questions
            WHERE quiz_id = ?
            ORDER BY id ASC
            """,
            (quiz_id_int,),
        )
        rows = cur.fetchall()
        result: List[Dict] = []
        for row in rows:
            q_text, opts_json, correct, user, is_correct, expl = row
            opts = None
            if opts_json:
                try:
                    opts = json.loads(opts_json)
                except Exception:
                    opts = opts_json
            result.append(
                {
                    "question": q_text or "",
                    "options": opts,
                    "correct_answer": correct or "",
                    "user_answer": user or "",
                    "is_correct": bool(is_correct) if is_correct is not None else None,
                    "explanation": expl or "",
                }
            )
        return result
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

    for (
        quiz_id,
        units_str,
        score,
        quiz_day,
        details_json,
        difficulty,
        revision,
        start_time,
        end_time,
    ) in records:
        quiz_day = quiz_day or ""
        diff_label = difficulty or ""
        table_rows.append(
            {
                "Quiz ID": quiz_id,
                "Date": quiz_day,
                "Units": units_str,
                "Score": score,
                "Mode": difficulty_labels_inv.get(diff_label, diff_label),
                "Needs Revision": revision or "",
                "Feedback": ", ".join(json.loads(details_json)[:2]) if details_json else "",
                "Start Time": start_time or "",
                "End Time": end_time or "",
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
    if "math_unit_filter" not in st.session_state:
        st.session_state["math_unit_filter"] = all_units
    if "math_mode_filter" not in st.session_state:
        st.session_state["math_mode_filter"] = all_modes

    f1, f2, f3 = st.columns([1, 1, 0.7])
    sel_units = f1.multiselect("Filter by unit", options=all_units, key="math_unit_filter")
    sel_modes = f2.multiselect("Filter by mode", options=all_modes, key="math_mode_filter")

    def clear_math_filters():
        st.session_state["math_unit_filter"] = []
        st.session_state["math_mode_filter"] = []

    with f3:
        st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
        st.button("Reset filters", on_click=clear_math_filters)

    filtered_df = df.copy()
    if sel_units:
        filtered_df = filtered_df[filtered_df["Units"].apply(lambda x: any(u.strip() in sel_units for u in x.split(",")))]
    if sel_modes:
        filtered_df = filtered_df[filtered_df["Mode"].isin(sel_modes)]

    if not sel_units and not sel_modes:
        st.dataframe(filtered_df, width="stretch", height=500)
    else:
        recent = filtered_df.head(10)
        st.dataframe(recent, width="stretch", height=320)
        if len(filtered_df) > len(recent):
            with st.expander(f"Show all ({len(filtered_df)} rows)"):
                st.dataframe(filtered_df, width="stretch", height=500)

    # Download history CSV (filtered view)
    if not filtered_df.empty:
        hist_buf = StringIO()
        filtered_df.to_csv(hist_buf, index=False)
        st.download_button(
            "Download quiz history (CSV)",
            data=hist_buf.getvalue(),
            file_name="quiz_history.csv",
            mime="text/csv",
        )

    # Detailed question/answer dump per quiz
    st.subheader("Quiz Details")
    if filtered_df.empty:
        st.info("No quizzes to show.")
        return
    for _, row in filtered_df.iterrows():
        quiz_id = row.get("Quiz ID")
        if quiz_id is None or (isinstance(quiz_id, float) and pd.isna(quiz_id)):
            continue
        label = f"{row.get('Date', '')} • Units {row.get('Units', '')} • {row.get('Score', '')}/20 • {row.get('Mode', '')}"
        with st.expander(label):
            questions = fetch_quiz_questions(int(quiz_id))
            if not questions:
                st.write("No stored questions for this quiz.")
            else:
                qa_buf = StringIO()
                qa_df = pd.DataFrame(questions)
                qa_df.to_csv(qa_buf, index=False)
                st.download_button(
                    f"Download quiz {quiz_id} Q&A (CSV)",
                    data=qa_buf.getvalue(),
                    file_name=f"quiz_{quiz_id}_qa.csv",
                    mime="text/csv",
                )
                for idx, q in enumerate(questions, start=1):
                    st.markdown(f"**Q{idx}.** {escape_markdown(clean_question_text(q.get('question',''), bool(q.get('options'))))}")
                    opts = q.get("options")
                    if opts and isinstance(opts, list):
                        st.write("Options:")
                        st.write(", ".join(escape_markdown(o) for o in opts))
                    st.write(f"Your answer: {escape_markdown(q.get('user_answer',''))}")
                    st.write(f"Correct answer: {escape_markdown(q.get('correct_answer',''))}")
                    ic = q.get("is_correct")
                    if ic is True:
                        st.success("Marked correct")
                    elif ic is False:
                        st.error("Marked incorrect")
                    if q.get("explanation"):
                        st.write(f"Why: {escape_markdown(q.get('explanation',''))}")
                    st.markdown("---")


def init_session_state() -> None:
    st.session_state.setdefault("questions", [])
    st.session_state.setdefault("selected_units", [])
    st.session_state.setdefault("last_quiz_raw", "")
    st.session_state.setdefault("last_quiz_error", "")
    st.session_state.setdefault("last_grading_raw", "")
    st.session_state.setdefault("last_grading_error", "")
    st.session_state.setdefault("quiz_locked", False)
    st.session_state.setdefault("show_progress", False)
    st.session_state.setdefault("math_mode", "none")  # none | quiz | progress
    st.session_state.setdefault("boss_set_id", None)


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
    st.session_state["boss_set_id"] = None


def escape_markdown(text: str) -> str:
    """Escape characters that Streamlit markdown treats specially (including $)."""
    if text is None:
        return ""
    return re.sub(r"([\\`*_{}\[\]()#+\-.!|>$])", r"\\\1", str(text))


def format_elapsed(_: str) -> str:
    """Compat stub for older timer code paths; returns a neutral timer string."""
    return "00:00"


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


def send_results_email(
    student: str,
    units: List[int],
    difficulty_label: str,
    score: int,
    qa_payload: List[Dict],
) -> Tuple[bool, str]:
    smtp_host = os.getenv("SMTP_HOST") or "smtp.gmail.com"
    smtp_port = os.getenv("SMTP_PORT") or "587"
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    to_addr = os.getenv("SMTP_TO") or "rmudisoo@gmail.com"
    if not (smtp_host and smtp_port and to_addr):
        return False, "SMTP config missing (host/port or recipient)"
    try:
        smtp_port_int = int(smtp_port)
    except ValueError:
        return False, "SMTP_PORT is not an integer"

    body_lines = [
        f"Student: {student}",
        f"Units: {units}",
        f"Mode: {difficulty_label}",
        f"Score: {score}/20",
        "",
        "Questions and answers:",
    ]
    for idx, qa in enumerate(qa_payload, start=1):
        body_lines.append(f"{idx}. {qa.get('question','')}")
        body_lines.append(f"Your answer: {qa.get('user_answer','')}")
        body_lines.append(f"Correct answer: {qa.get('correct_answer','')}")
        expl = qa.get("explanation", "")
        if expl:
            body_lines.append(f"Why: {expl}")
        body_lines.append("")
    msg = MIMEText("\n".join(body_lines))
    msg["Subject"] = f"Math Quiz Results - {student} - {date.today().isoformat()}"
    msg["From"] = smtp_user or to_addr
    msg["To"] = to_addr

    context = ssl.create_default_context()
    try:
        with smtplib.SMTP(smtp_host, smtp_port_int) as server:
            server.starttls(context=context)
            if smtp_user and smtp_pass:
                server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        return True, "Email sent"
    except Exception as exc:
        return False, f"Email send failed: {exc}"
    st.session_state.setdefault("answers_submitted", False)


def reset_quiz_state(clear_questions: bool = True) -> None:
    """Reset state for starting a fresh quiz."""
    if clear_questions:
        st.session_state["questions"] = []
    st.session_state["quiz_locked"] = False
    st.session_state["answers_submitted"] = False
    st.session_state["last_quiz_error"] = ""
    st.session_state["quiz_start_time"] = None
    # Clear any stored answers
    for key in list(st.session_state.keys()):
        if key.startswith("answer_"):
            del st.session_state[key]


def main() -> None:
    st.set_page_config(page_title="6th Grade Springboard Course 2 Quiz Generator")
    # Keep-alive to prevent long-idle disconnects (ping every 60 seconds)
    st_autorefresh(interval=60_000, key="keepalive_math")
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

    if st.button("Show last session logs"):
        st.subheader("Last session logs")
        st.write(f"Last quiz error: {st.session_state.get('last_quiz_error') or 'None'}")
        st.write(f"Last grading error: {st.session_state.get('last_grading_error') or 'None'}")
        st.text_area(
            "Last raw quiz response",
            st.session_state.get("last_quiz_raw", ""),
            height=200,
        )
        st.text_area(
            "Last raw grading response",
            st.session_state.get("last_grading_raw", ""),
            height=200,
        )

    # Action chooser
    st.subheader("Choose an action")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Start New Quiz"):
            if not student_name:
                st.warning("Enter the student name to start a quiz.")
            else:
                st.session_state["math_mode"] = "quiz"
                st.session_state["show_progress"] = False
                reset_quiz_state(clear_questions=True)
                st.success("Quiz state cleared. Select units and generate a new quiz.")
    with col_b:
        if st.button("View Progress"):
            if not student_name:
                st.warning("Enter the student name to view progress.")
            else:
                st.session_state["math_mode"] = "progress"
                st.session_state["show_progress"] = True

    # If no action chosen yet, stop after showing the chooser
    if st.session_state.get("math_mode") == "none":
        return

    # Progress view
    if st.session_state.get("math_mode") == "progress":
        st.subheader("Progress")
        if student_name:
            render_progress(student_name)
        else:
            st.warning("Enter the student name to view progress.")
        if st.button("Back"):
            st.session_state["show_progress"] = False
            st.session_state["math_mode"] = "none"
        return

    # Quiz view below
    st.info(
        f"Copy Springboard Course 2 PDFs into the folder: {PDF_FOLDER} "
        "and click Refresh to load/update keyword data."
    )
    
    # Load AI concepts if available
    if "ai_concepts" not in st.session_state:
        st.session_state["ai_concepts"] = load_ai_concepts()
    ai_concepts = st.session_state["ai_concepts"]
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Refresh PDFs from folder"):
            with st.spinner("Processing PDFs for keywords..."):
                units_data = process_pdfs_in_folder()
            st.success("PDF folder processed and keyword data updated.")
    
    with col2:
        if st.button("🤖 Generate Concepts (AI)"):
            if not api_key:
                st.error("API key required for AI concept extraction.")
            else:
                client = get_client(api_key)
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(current, total, name):
                    progress_bar.progress(current / total)
                    status_text.text(f"Analyzing {current}/{total}: {name}")
                
                with st.spinner("Extracting concepts with AI..."):
                    new_concepts = process_pdfs_with_llm(client, progress_callback=update_progress)
                    st.session_state["ai_concepts"] = new_concepts
                
                progress_bar.empty()
                status_text.empty()
                st.success("AI concepts generated and saved!")
                st.rerun()

    available_units = list_available_units(units_data)
    selected_units = st.multiselect(
        "Select units to include",
        options=available_units,
        default=st.session_state.get("selected_units", []),
    )
    st.session_state["selected_units"] = selected_units

    # Show AI concepts for selected units if available
    if selected_units and ai_concepts:
        ai_topics_display = get_ai_topics_for_units(selected_units, ai_concepts)
        if ai_topics_display:
            with st.expander("📚 AI-Extracted Concepts (click to expand)", expanded=False):
                for item in ai_topics_display:
                    st.markdown(f"- **{item['topic']}** ({item['weight']})")
        else:
            st.caption("No AI concepts found for selected units. Click '🤖 Generate Concepts (AI)' to analyze PDFs.")
    elif selected_units and not ai_concepts:
        st.caption("💡 Tip: Click '🤖 Generate Concepts (AI)' for smarter topic extraction.")

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

    if st.button("Generate Quiz"):
        generation_error = None
        # Starting a new quiz unlocks submission and clears previous answers
        reset_quiz_state(clear_questions=False)
        st.session_state["quiz_start_time"] = datetime.now().isoformat(timespec="seconds")
        if not student_name:
            generation_error = "Please enter the student name before generating a quiz."
        if not selected_units:
            generation_error = generation_error or "Select at least one unit to generate a quiz."
        if difficulty == "complex" and not api_key:
            generation_error = generation_error or "Missing XAI_API_KEY for Boss Battle generation."

        if generation_error:
            st.session_state["last_quiz_error"] = generation_error
            st.session_state["questions"] = []
        else:
            client = get_client(api_key) if api_key else None
            if difficulty == "complex":
                if len(selected_units) != 1:
                    generation_error = "For Boss Battle, select exactly one unit."
                else:
                    unit = selected_units[0]
                    try:
                        if not client:
                            raise ValueError("Missing XAI_API_KEY for Boss Battle generation.")
                        questions = generate_boss_set(client, unit)
                        st.session_state["boss_set_id"] = None
                        st.session_state["questions"] = questions
                        st.session_state["answers_submitted"] = False
                        st.session_state["last_quiz_error"] = ""
                        st.success("Boss Battle ready!")
                    except Exception as exc:
                        generation_error = f"Boss Battle generation failed: {exc}"
                        st.session_state["last_quiz_error"] = generation_error
                        st.session_state["questions"] = []
            else:
                # Prefer AI-generated concepts if available
                ai_topics = get_ai_topics_for_units(selected_units, ai_concepts) if ai_concepts else []
                
                if ai_topics:
                    # Use AI-extracted concepts
                    topics = [f"{t['topic']} (emphasis: {t['weight']})" for t in ai_topics]
                    weighted_topics = ai_topics
                    allowed = [t["topic"] for t in ai_topics]
                else:
                    # Fallback to keyword counting
                    aggregated = aggregate_keywords(selected_units, units_data)
                    allowed = unit_topics(selected_units)
                    filtered = filter_counter(aggregated, allowed) if aggregated else Counter()
                    topics = top_topics(filtered) if filtered else allowed
                    weighted_topics = topic_weights(filtered) if filtered else [
                        {"topic": t, "weight": 1} for t in allowed
                    ]
                
                client = get_client(api_key) if api_key else None

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
    if debug_raw:
        st.text_area(
            "Last raw quiz response",
            st.session_state.get("last_quiz_raw", ""),
            height=200,
        )
        st.text_area(
            "Last raw grading response",
            st.session_state.get("last_grading_raw", ""),
            height=200,
        )
        if st.session_state.get("last_grading_error"):
            st.write(f"Last grading error: {st.session_state['last_grading_error']}")

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
                        "options": q.get("options"),
                        "user_answer": user_answer,
                        "correct_answer": q.get("answer", ""),
                        "explanation": q.get("explanation", ""),
                    }
                )

            with st.spinner("Grading with Grok..."):
                try:
                    score, feedback = grade_quiz(client, qa_payload)
                except Exception as exc:
                    st.session_state["last_grading_error"] = str(exc)
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
            start_ts = st.session_state.get("quiz_start_time") or datetime.now().isoformat(timespec="seconds")
            end_ts = datetime.now().isoformat(timespec="seconds")
            save_quiz_result(
                student_name,
                selected_units,
                score,
                feedback,
                difficulty_code,
                revision,
                start_ts,
                end_ts,
                qa_payload,
                st.session_state.get("boss_set_id"),
            )
            # Email results if SMTP is configured
            email_ok, email_msg = send_results_email(
                student_name, selected_units, difficulty_label, score, qa_payload
            )
            if email_msg:
                if email_ok:
                    st.info(email_msg)
                else:
                    st.warning(email_msg)
            st.session_state["answers_submitted"] = True
            st.session_state["quiz_locked"] = True
        elif disable_inputs:
            st.info("Quiz locked. Start a new test to answer again.")


if __name__ == "__main__":
    main()

