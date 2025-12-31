import json
import os
import random
import re
import smtplib
import ssl
from collections import defaultdict
from datetime import date, datetime, timedelta
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
VOCAB_DB_PATH = BASE_DIR / "vocab_progress.db"
MODEL_NAME = "grok-4-fast"
API_BASE_URL = "https://api.x.ai/v1"

# 6th-grade friendly base word list (sample; expand as needed)
BASE_WORDS = [
    "abundant",
    "cautious",
    "brisk",
    "daring",
    "eager",
    "faithful",
    "gather",
    "harvest",
    "improve",
    "jolly",
    "keen",
    "lively",
    "mystery",
    "noble",
    "observe",
    "polite",
    "quiet",
    "radiant",
    "steady",
    "timid",
    "unique",
    "vivid",
    "wander",
    "youthful",
    "zealous",
    "ancient",
    "balance",
    "capture",
    "distant",
    "elevate",
    "forgive",
    "glance",
    "humble",
    "insight",
    "journey",
    "kindle",
    "lengthy",
    "mingle",
    "nurture",
    "outcome",
    "pioneer",
    "rescue",
    "sincere",
    "thrive",
    "uplift",
    "vital",
    "wisdom",
    "zephyr",
    "adapt",
    "brave",
    "curious",
    "delight",
    "elegant",
    "fable",
    "gentle",
    "honest",
    "invent",
    "jungle",
    "kindness",
    "legend",
    "mighty",
    "notice",
    "originate",
    "protect",
    "question",
    "reliable",
    "spark",
    "treasure",
    "uplift",
    "value",
    "wonder",
    "yearn",
    "zeal",
    "ally",
    "bold",
    "contrast",
    "define",
    "essence",
    "flexible",
    "glow",
    "harbor",
    "ideal",
    "jovial",
    "limit",
    "maintain",
    "nervous",
    "optimistic",
    "precise",
    "quest",
    "renew",
    "steady",
    "talent",
    "uphold",
    "verify",
    "whisper",
    "yield",
    "zenith",
    # Additional easy/moderate words
    "bright",
    "calm",
    "cheerful",
    "clever",
    "confident",
    "cozy",
    "creative",
    "curious",
    "dazzle",
    "eager",
    "fancy",
    "friendly",
    "gentle",
    "giggle",
    "happy",
    "helpful",
    "honest",
    "hopeful",
    "joyful",
    "kind",
    "lucky",
    "mild",
    "neat",
    "nice",
    "peaceful",
    "playful",
    "polite",
    "proud",
    "quiet",
    "rapid",
    "shiny",
    "silly",
    "smart",
    "soft",
    "speedy",
    "strong",
    "sweet",
    "tasty",
    "thankful",
    "tidy",
    "warm",
    "wild",
    "wise",
    "wonderful",
    "zippy",
    "ancient",
    "arrive",
    "assist",
    "attempt",
    "bold",
    "brief",
    "calmly",
    "careful",
    "caring",
    "cautious",
    "certain",
    "chief",
    "choose",
    "clearly",
    "comfort",
    "common",
    "complete",
    "consider",
    "contain",
    "control",
    "correct",
    "decide",
    "defend",
    "depend",
    "describe",
    "direct",
    "discover",
    "distant",
    "divide",
    "easily",
    "effort",
    "encourage",
    "energy",
    "explain",
    "famous",
    "fierce",
    "follow",
    "fortunate",
    "frequent",
    "future",
    "gather",
    "gigantic",
    "graceful",
    "greet",
    "healthy",
    "identify",
    "imagine",
    "improve",
    "include",
    "inform",
    "inspire",
    "involve",
    "journey",
    "leader",
    "liberty",
    "limit",
    "major",
    "manage",
    "mature",
    "mention",
    "mighty",
    "modern",
    "native",
    "notice",
    "observe",
    "organize",
    "patient",
    "permit",
    "persuade",
    "possible",
    "prefer",
    "prepare",
    "pretend",
    "prevent",
    "protect",
    "provide",
    "puzzle",
    "recall",
    "recent",
    "reflect",
    "refuse",
    "relax",
    "remain",
    "remind",
    "rescue",
    "respect",
    "review",
    "reward",
    "satisfy",
    "scarce",
    "select",
    "sincere",
    "solution",
    "steady",
    "strength",
    "struggle",
    "support",
    "talent",
    "tender",
    "travel",
    "unite",
    "value",
    "wander",
    "warn",
    "wealthy",
    "wildlife",
    "willing",
    "wonder",
    "worthy",
    # Extra easy/moderate words
    "ability",
    "accurate",
    "active",
    "admire",
    "adventure",
    "afford",
    "agree",
    "alert",
    "amazing",
    "amuse",
    "announce",
    "applaud",
    "approach",
    "arrange",
    "assist",
    "attract",
    "avoid",
    "aware",
    "balance",
    "behave",
    "belong",
    "benefit",
    "bravery",
    "briefly",
    "brighten",
    "bundle",
    "calmness",
    "careless",
    "celebrate",
    "certainly",
    "challenge",
    "cheer",
    "clarify",
    "collect",
    "combine",
    "comfort",
    "communicate",
    "compare",
    "complain",
    "complete",
    "concern",
    "confirm",
    "confuse",
    "connect",
    "consider",
    "construct",
    "contain",
    "continue",
    "contribute",
    "convince",
    "cooperate",
    "courage",
    "create",
    "curiosity",
    "decorate",
    "defend",
    "delicate",
    "dependable",
    "describe",
    "deserve",
    "design",
    "develop",
    "discover",
    "distant",
    "divide",
    "donate",
    "effort",
    "elegant",
    "encourage",
    "enjoyable",
    "enormous",
    "entire",
    "entrust",
    "essential",
    "estimate",
    "evaluate",
    "excellent",
    "excited",
    "expand",
    "explain",
    "explore",
    "fairness",
    "familiar",
    "famous",
    "fantastic",
    "fascinate",
    "flexible",
    "fortunate",
    "freedom",
    "frequent",
    "friendly",
    "frustrate",
    "genuine",
    "grateful",
    "harmony",
    "helpful",
    "honesty",
    "imagine",
    "improve",
    "include",
    "incredible",
    "independent",
    "inspire",
    "intelligent",
    "introduce",
    "inventive",
    "justice",
    "kindness",
    "knowledge",
    "laughter",
    "leadership",
    "logical",
    "lovable",
    "mature",
    "memorable",
    "mysterious",
    "natural",
    "nervous",
    "optimistic",
    "ordinary",
    "organize",
    "participate",
    "patience",
    "peaceful",
    "pleasant",
    "politeness",
    "popular",
    "positive",
    "possible",
    "powerful",
    "prepare",
    "present",
    "prevent",
    "promise",
    "protect",
    "provide",
    "puzzled",
    "quality",
    "realize",
    "refresh",
    "reliable",
    "remarkable",
    "repeat",
    "replace",
    "report",
    "request",
    "respectful",
    "responsible",
    "respond",
    "result",
    "rewarding",
    "satisfy",
    "scatter",
    "sensible",
    "serious",
    "sincere",
    "skillful",
    "succeed",
    "supportive",
    "survive",
    "talented",
    "thankful",
    "thoughtful",
    "tolerant",
    "tradition",
    "trustworthy",
    "understand",
    "upgrade",
    "valuable",
    "variety",
    "victory",
    "welcome",
    "wonderful",
    "worthwhile",
    # Slightly harder/academic (still kid-friendly)
    "adaptable",
    "ambitious",
    "analyze",
    "arrange",
    "attempt",
    "avenue",
    "balance",
    "cautious",
    "collaborate",
    "compassion",
    "comprehend",
    "conclusion",
    "consequence",
    "cooperate",
    "culture",
    "curious",
    "determine",
    "diligent",
    "discovery",
    "efficient",
    "elaborate",
    "empathy",
    "encounter",
    "endeavor",
    "engage",
    "estimate",
    "evaluate",
    "evidence",
    "example",
    "expand",
    "experience",
    "expression",
    "factor",
    "flexible",
    "focus",
    "function",
    "illustrate",
    "impact",
    "indicate",
    "influence",
    "interpret",
    "investigate",
    "maintain",
    "maximum",
    "minimum",
    "motivate",
    "multiple",
    "objective",
    "observe",
    "occur",
    "outcome",
    "pattern",
    "perspective",
    "primary",
    "process",
    "purpose",
    "reason",
    "reflect",
    "relevant",
    "require",
    "research",
    "resourceful",
    "respond",
    "result",
    "similar",
    "solution",
    "structure",
    "summarize",
    "support",
    "symbol",
    "technique",
    "theory",
    "variable",
    "verify",
]

DIFFICULTY_ORDER = ["Easy", "Medium", "Hard"]


def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=API_BASE_URL)


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


def ensure_mc_options(answer: str) -> List[str]:
    """Ensure 4 multiple-choice options including the correct answer."""
    ans = str(answer).strip()
    pool = [w for w in BASE_WORDS if w.lower() != ans.lower()]
    random.shuffle(pool)
    # add a small shuffle offset to reduce repetition patterns
    if len(pool) > 6:
        offset = random.randint(0, min(6, len(pool) - 3))
        pool = pool[offset:] + pool[:offset]
    distractors = pool[:3] if len(pool) >= 3 else pool
    opts = [ans] + distractors
    random.shuffle(opts)
    return opts


def clean_question_text(text: str) -> str:
    """Strip inline options/markers from question text."""
    if text is None:
        return ""
    cleaned = str(text)
    cleaned = cleaned.replace("(Multiple choice)", "").replace("(multiple choice)", "")
    # Keep only the first line to drop inline option lists if present
    cleaned = cleaned.splitlines()[0]
    # Trim at common option markers
    for marker in [" A)", " A.", "A)", "A."]:
        if marker in cleaned:
            cleaned = cleaned.split(marker)[0]
            break
    return cleaned.strip()


def clean_option(opt: str) -> str:
    """Strip leading choice markers like 'A)' / 'B.' from options."""
    if opt is None:
        return ""
    s = str(opt).strip()
    s = re.sub(r"^[A-Da-d][\)\.]\s*", "", s)
    return s


def ensure_vocab_db() -> None:
    import sqlite3

    conn = sqlite3.connect(VOCAB_DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                student TEXT PRIMARY KEY,
                current_difficulty TEXT DEFAULT 'Easy',
                summary TEXT DEFAULT ''
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student TEXT NOT NULL,
                session_date TEXT NOT NULL,
                session_time TEXT NOT NULL,
                exercise_type TEXT NOT NULL,
                score INTEGER NOT NULL,
                words_mastered TEXT,
                wrong_words TEXT,
                feedback TEXT,
                difficulty TEXT,
                summary TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS wrong_words (
                student TEXT NOT NULL,
                word TEXT NOT NULL,
                count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (student, word)
            )
            """
        )
        # migrations
        user_cols = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
        if "summary" not in user_cols:
            conn.execute("ALTER TABLE users ADD COLUMN summary TEXT DEFAULT ''")

        cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        if "wrong_words" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN wrong_words TEXT")
        if "session_time" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN session_time TEXT DEFAULT ''")
        if "summary" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN summary TEXT DEFAULT ''")
        conn.commit()
    finally:
        conn.close()


def get_current_difficulty(student: str) -> str:
    import sqlite3

    ensure_vocab_db()
    conn = sqlite3.connect(VOCAB_DB_PATH)
    try:
        row = conn.execute(
            "SELECT current_difficulty FROM users WHERE student = ?", (student,)
        ).fetchone()
        if row:
            return row[0]
        conn.execute(
            "INSERT INTO users (student, current_difficulty, summary) VALUES (?, ?, '')",
            (student, "Easy"),
        )
        conn.commit()
        return "Easy"
    finally:
        conn.close()


def update_difficulty(student: str, score: int, total: int = 15) -> str:
    pct = score / total
    current = get_current_difficulty(student)
    idx = DIFFICULTY_ORDER.index(current)
    if pct >= 0.9 and idx < len(DIFFICULTY_ORDER) - 1:
        idx += 1
    elif pct >= 0.8 and idx < len(DIFFICULTY_ORDER) - 1:
        idx += 1
    elif pct < 0.6 and idx > 0:
        idx -= 1
    new_level = DIFFICULTY_ORDER[idx]
    import sqlite3

    conn = sqlite3.connect(VOCAB_DB_PATH)
    try:
        conn.execute(
            "UPDATE users SET current_difficulty = ? WHERE student = ?",
            (new_level, student),
        )
        conn.commit()
    finally:
        conn.close()
    return new_level


def record_session(
    student: str,
    exercise_type: str,
    score: int,
    words_mastered: List[str],
    feedback: List[str],
    difficulty: str,
    wrong_words: List[str],
    summary: str,
) -> None:
    import sqlite3

    ensure_vocab_db()
    conn = sqlite3.connect(VOCAB_DB_PATH)
    try:
        conn.execute(
            """
            INSERT INTO sessions (student, session_date, session_time, exercise_type, score, words_mastered, wrong_words, feedback, difficulty, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student,
                date.today().isoformat(),
                datetime.now().isoformat(timespec="seconds"),
                exercise_type,
                score,
                json.dumps(words_mastered or []),
                json.dumps(wrong_words or []),
                json.dumps(feedback or []),
                difficulty,
                summary,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def fetch_sessions(student: str) -> List[Tuple]:
    import sqlite3

    ensure_vocab_db()
    conn = sqlite3.connect(VOCAB_DB_PATH)
    try:
        return conn.execute(
            "SELECT session_date, session_time, exercise_type, score, words_mastered, wrong_words, feedback, difficulty, summary FROM sessions WHERE student = ? ORDER BY session_time DESC",
            (student,),
        ).fetchall()
    finally:
        conn.close()


def compute_streak(dates: List[date]) -> int:
    if not dates:
        return 0
    uniq = sorted(set(dates))
    streak = 1
    best = 1
    for i in range(len(uniq) - 1, 0, -1):
        if uniq[i] - uniq[i - 1] == timedelta(days=1):
            streak += 1
            best = max(best, streak)
        else:
            streak = 1
    return best


def award_badges(scores: List[int]) -> List[str]:
    badges = []
    if any(s >= 12 for s in scores):
        badges.append("Vocab Hero 🏅")
    if any(s >= 14 for s in scores):
        badges.append("Synonym Star 🌟")
    if len(scores) >= 5 and sum(scores) / len(scores) >= 12:
        badges.append("Consistency Champ 🧠")
    return badges


def update_wrong_words(student: str, wrong_words: List[str]) -> None:
    import sqlite3

    if not wrong_words:
        return
    ensure_vocab_db()
    conn = sqlite3.connect(VOCAB_DB_PATH)
    try:
        for w in wrong_words:
            conn.execute(
                """
                INSERT INTO wrong_words (student, word, count)
                VALUES (?, ?, 1)
                ON CONFLICT(student, word) DO UPDATE SET count = count + 1
                """,
                (student, w),
            )
        conn.commit()
    finally:
        conn.close()


def fetch_wrong_words(student: str) -> List[Tuple[str, int]]:
    import sqlite3

    ensure_vocab_db()
    conn = sqlite3.connect(VOCAB_DB_PATH)
    try:
        return conn.execute(
            "SELECT word, count FROM wrong_words WHERE student = ? ORDER BY count DESC",
            (student,),
        ).fetchall()
    finally:
        conn.close()


def generate_vocab_questions(
    client: OpenAI, student: str, exercise_type: str, difficulty: str, prev_summary: str, retries: int = 3
) -> List[Dict]:
    theme = "animals and adventures"
    prompt = (
        f"Generate 15 vocabulary exercises for an 11-year-old on {exercise_type},"
        f" difficulty {difficulty}. ALL questions must be multiple-choice only."
        " Provide exactly 4 options per question and make them 'choose the best answer' style."
        f" Use these base words for inspiration: {BASE_WORDS[:30]}."
        f" Theme: {theme}. Output JSON list of objects:"
        ' [{"question": str, "options": list, "answer": str, "explanation": str}].'
        f" Previous performance summary to prioritize weaknesses: {prev_summary or 'None provided'}."
    )
    last_err = None
    attempts_log = []

    def coerce_questions(parsed_obj):
        if isinstance(parsed_obj, list):
            return parsed_obj
        if isinstance(parsed_obj, dict):
            if isinstance(parsed_obj.get("questions"), list):
                return parsed_obj["questions"]
            # first list-valued entry
            for val in parsed_obj.values():
                if isinstance(val, list):
                    return val
            # dict of numbered objects
            values = list(parsed_obj.values())
            if values and all(isinstance(v, dict) for v in values):
                return values
        return None

    last_count = 0
    for attempt in range(retries + 1):
        extra_prompt = ""
        if attempt > 0:
            extra_prompt = (
                f" Previous attempt returned only {last_count} items. Regenerate exactly 15 items."
            )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=1500,
            temperature=0.4 if attempt == 0 else 0.6,
            response_format={"type": "json_object"},
            timeout=60,
            messages=[
                {"role": "system", "content": "You are a fun English vocab coach for 11-year-olds."},
                {"role": "user", "content": prompt + extra_prompt},
            ],
        )
        content = response.choices[0].message.content.strip()
        st.session_state["vocab_last_raw"] = content
        attempts_log.append({"attempt": attempt + 1, "content_preview": content[:400]})

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            last_err = exc
            continue

        questions = coerce_questions(parsed)
        if not isinstance(questions, list):
            last_err = ValueError("Vocab response was not a list.")
            continue

        cleaned = []
        for item in questions:
            if not isinstance(item, dict) or "question" not in item:
                continue
            answer_val = str(item.get("answer", "")).strip()
            options_val = item.get("options") or None
            if not options_val or not isinstance(options_val, list):
                options_val = ensure_mc_options(answer_val)
            cleaned.append(
                {
                    "question": str(item.get("question", "")).strip(),
                    "options": options_val,
                    "answer": answer_val,
                    "explanation": str(item.get("explanation", "")).strip(),
                }
            )
        if len(cleaned) >= 15:
            st.session_state["vocab_attempts_log"] = attempts_log
            return cleaned[:15]
        last_count = len(cleaned)
        # Try extracting dicts from free text as a fallback
        extracted = _extract_dicts_from_text(content)
        if extracted:
            maybe = coerce_questions(extracted)
            cleaned2 = []
            if isinstance(maybe, list):
                for item in maybe:
                    if not isinstance(item, dict) or "question" not in item:
                        continue
                    cleaned2.append(
                        {
                            "question": str(item.get("question", "")).strip(),
                            "options": item.get("options") or None,
                            "answer": str(item.get("answer", "")).strip(),
                            "explanation": str(item.get("explanation", "")).strip(),
                        }
                    )
            if len(cleaned2) >= 15:
                st.session_state["vocab_attempts_log"] = attempts_log
                return cleaned2[:15]
            last_count = len(cleaned2)
            last_err = ValueError(f"Expected 15 vocab exercises, got {len(cleaned2)} after fallback.")
        else:
            last_err = ValueError(f"Expected 15 vocab exercises, got {len(cleaned)}.")

    st.session_state["vocab_attempts_log"] = attempts_log
    raise ValueError(f"Could not generate enough vocab exercises. Last error: {last_err}")


def summarize_progress(
    client: OpenAI,
    prev_summary: str,
    feedback: List[str],
    mastered: List[str],
    wrong_words: List[str],
) -> str:
    prompt = (
        "Update a running summary of a student's vocab progress."
        " Return JSON ONLY: {\"summary\": str}."
        f" Previous summary: {prev_summary or 'None'}."
        f" Feedback from latest session: {feedback}."
        f" Mastered words: {mastered}."
        f" Missed words: {wrong_words}."
        " Keep it concise (2-4 sentences), kid-friendly, highlight improvements and next focus areas."
    )
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=200,
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a concise, encouraging vocabulary coach."},
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content.strip()
    parsed = json.loads(content)
    return str(parsed.get("summary", "")).strip()


def grade_vocab(client: OpenAI, qa_payload: List[Dict]) -> Tuple[int, List[str], List[str]]:
    prompt = (
        "Grade these vocab answers. Return JSON ONLY:"
        ' {"score": int, "feedback": list of strings, "mastered_words": list of strings}.'
        " Keep feedback brief and kid-friendly. Mastered words are those answered correctly or explained well."
        f" Data: {json.dumps(qa_payload)}"
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=600,
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a concise, encouraging vocabulary grader."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content.strip()
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse grading JSON. Details: {exc}") from exc
    score = int(parsed.get("score", 0))
    feedback = parsed.get("feedback") or []
    mastered = parsed.get("mastered_words") or []
    if not isinstance(feedback, list):
        feedback = [str(feedback)]
    if not isinstance(mastered, list):
        mastered = [str(mastered)]
    return score, [str(f) for f in feedback], [str(w) for w in mastered]


def send_results_email(
    student: str,
    exercise_type: str,
    difficulty: str,
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
        f"Type: {exercise_type}",
        f"Difficulty: {difficulty}",
        f"Score: {score}/15",
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
    msg["Subject"] = f"Vocab Quiz Results - {student} - {date.today().isoformat()}"
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


def render_vocab_progress(student: str) -> None:
    records = fetch_sessions(student)
    if not records:
        st.info("Start exercising to build your vocab!")
        return
    dates = []
    scores = []
    table_rows = []
    all_words = []
    wrong_words_all = []
    last_summary = ""
    for session_date, session_time, ex_type, score, words_json, wrong_json, feedback_json, difficulty, summary in records:
        dates.append(date.fromisoformat(session_date))
        scores.append(score)
        words = json.loads(words_json) if words_json else []
        all_words.extend(words)
        wrongs = json.loads(wrong_json) if wrong_json else []
        for w in wrongs:
            wrong_words_all.append(w)
        feedback_list = json.loads(feedback_json) if feedback_json else []
        last_summary = summary or last_summary
        table_rows.append(
            {
                "Date": session_date,
                "Time": session_time or "",
                "Type": ex_type,
                "Score": score,
                "Difficulty": difficulty,
                "Mastered": ", ".join(words[:3]) + ("..." if len(words) > 3 else ""),
                "Missed": ", ".join(wrongs[:3]) + ("..." if len(wrongs) > 3 else ""),
                "Feedback": "; ".join(feedback_list[:2]),
                "Summary": (summary or "")[:120] + ("..." if summary and len(summary) > 120 else ""),
            }
        )

    streak = compute_streak(dates)
    badges = award_badges(scores)
    st.metric("Streak (days)", streak)
    st.metric("Sessions", len(records))
    st.metric("Avg Score", f"{sum(scores)/len(scores):.1f}/15")
    if badges:
        st.success("Badges: " + ", ".join(badges))

    st.subheader("Session History")
    st.dataframe(
        pd.DataFrame(table_rows).sort_values("Date", ascending=False),
        width="stretch",
    )

    if all_words:
        st.subheader("Top Words Learned")
        counts = pd.Series(all_words).value_counts().head(10)
        st.bar_chart(counts)

    wrong_hist = fetch_wrong_words(student)
    if wrong_hist:
        st.subheader("Words to Review (missed most often)")
        st.table(pd.DataFrame(wrong_hist, columns=["Word", "Missed Count"]).head(10))


def main():
    vocab_tab_content()


def vocab_tab_content():
    ensure_vocab_db()
    api_key = os.getenv("XAI_API_KEY") or st.secrets.get("XAI_API_KEY", "")
    if not api_key:
        st.warning("Set XAI_API_KEY in env or secrets for Grok access.")
    st.title("Vocab Builder: Fun English 📚✨")
    student = st.text_input("Student Name (required)", key="vocab_student")
    debug_raw = st.checkbox("Show raw vocab response (debug)", value=False)
    exercise_type = st.selectbox(
        "Pick exercise type",
        ["Synonyms", "Antonyms", "Mixed Vocab", "Advanced (Analogies & Roots)"],
    )

    if st.button("My Progress"):
        if not student:
            st.error("Enter the student name to view progress.")
        else:
            render_vocab_progress(student)
        st.stop()

    if st.button("Start Exercises"):
        if not student:
            st.error("Enter the student name first.")
            st.stop()
        current_level = get_current_difficulty(student)
        # Fetch existing summary
        prev_summary = ""
        try:
            import sqlite3

            conn = sqlite3.connect(VOCAB_DB_PATH)
            row = conn.execute("SELECT summary FROM users WHERE student = ?", (student,)).fetchone()
            prev_summary = row[0] if row else ""
        finally:
            conn.close()

        client = get_client(api_key)
        with st.spinner(f"Generating exercises at {current_level}..."):
            try:
                questions = generate_vocab_questions(
                    client, student, exercise_type, current_level, prev_summary
                )
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
                st.stop()
        st.session_state["vocab_questions"] = questions
        st.session_state["vocab_answers_submitted"] = False
        st.session_state["vocab_locked"] = False
        st.session_state["vocab_exercise_type"] = exercise_type
        st.session_state["vocab_level"] = current_level
        st.session_state["vocab_prev_summary"] = prev_summary
        st.success("Exercises ready!")
        if debug_raw and "vocab_last_raw" in st.session_state:
            st.text_area("Raw vocab response", st.session_state["vocab_last_raw"], height=200)
        if debug_raw and "vocab_attempts_log" in st.session_state:
            st.json(st.session_state["vocab_attempts_log"])

    questions = st.session_state.get("vocab_questions", [])
    if questions:
        disable_inputs = st.session_state.get("vocab_locked", False)
        with st.form("vocab_form"):
            for idx, q in enumerate(questions):
                q_text = clean_question_text(q.get("question", ""))
                st.markdown(f"**Q{idx + 1}. {q_text}**")
                options = q.get("options")
                key = f"vocab_answer_{idx}"
                if options and isinstance(options, list):
                    safe_opts = [clean_option(o) for o in options]
                    st.radio("Choose an answer:", safe_opts, key=key, index=None, disabled=disable_inputs)
                else:
                    st.text_input("Your answer:", key=key, disabled=disable_inputs)
                st.markdown("---")
            submitted = st.form_submit_button("Submit", disabled=disable_inputs)

        if submitted and disable_inputs:
            st.info("Locked. Start new exercises to answer again.")
        elif submitted:
            # Input guard
            missing = []
            for idx, q in enumerate(questions):
                ans = st.session_state.get(f"vocab_answer_{idx}", None)
                if ans is None or (isinstance(ans, str) and not ans.strip()):
                    missing.append(idx + 1)
            if missing:
                st.error(f"Please answer all questions before submitting. Missing: {missing}")
                st.stop()

            client = get_client(api_key)
            qa_payload = []
            wrong_words = []
            for idx, q in enumerate(questions):
                user_answer = st.session_state.get(f"vocab_answer_{idx}", "")
                if str(user_answer).strip() != str(q.get("answer", "")).strip():
                    wrong_words.append(str(q.get("answer", "")).strip())
                qa_payload.append(
                    {
                        "question": q.get("question", ""),
                        "user_answer": user_answer,
                        "correct_answer": q.get("answer", ""),
                        "explanation": q.get("explanation", ""),
                    }
                )

            with st.spinner("Grading..."):
                try:
                    score, feedback, mastered = grade_vocab(client, qa_payload)
                except Exception as exc:
                    st.error(f"Grading failed: {exc}")
                    st.stop()

            st.success("Graded!")
            st.metric("Score (out of 15)", score)
            if feedback:
                st.markdown("**Feedback (needs work first, strengths last):**")
                for item in feedback:
                    st.write(f"• {item}")
            if mastered:
                st.markdown("**Mastered words:** " + ", ".join(mastered))

            prev_summary = st.session_state.get("vocab_prev_summary", "")
            try:
                new_summary = summarize_progress(client, prev_summary, feedback, mastered, wrong_words)
            except Exception:
                new_summary = prev_summary

            new_level = update_difficulty(student, score, total=15)
            update_wrong_words(student, wrong_words)
            record_session(
                student,
                exercise_type,
                score,
                mastered,
                feedback,
                st.session_state.get("vocab_level", "Easy"),
                wrong_words,
                new_summary,
            )
            # Update summary in users
            try:
                import sqlite3

                conn = sqlite3.connect(VOCAB_DB_PATH)
                conn.execute("UPDATE users SET summary = ? WHERE student = ?", (new_summary, student))
                conn.commit()
            finally:
                conn.close()

            email_ok, email_msg = send_results_email(
                student,
                exercise_type,
                st.session_state.get("vocab_level", "Easy"),
                score,
                qa_payload,
            )
            if email_msg:
                if email_ok:
                    st.info(email_msg)
                else:
                    st.warning(email_msg)

            st.info(f"Current difficulty updated to: {new_level}")
            st.session_state["vocab_locked"] = True

            # Clear answers to prevent resubmit
            for key in list(st.session_state.keys()):
                if key.startswith("vocab_answer_"):
                    del st.session_state[key]


if __name__ == "__main__":
    main()

