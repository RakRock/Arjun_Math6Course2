import json
import random
import re
import os
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Tuple

import streamlit as st
from openai import OpenAI
from streamlit_autorefresh import st_autorefresh

BASE_DIR = Path(__file__).resolve().parent
TOPICS_JSON = BASE_DIR / "topics.json"
QUIZ_DB_PATH = BASE_DIR / "quiz_progress.db"
MODEL_NAME = "grok-4-fast"
API_BASE_URL = "https://api.x.ai/v1"


@st.cache_data(show_spinner=False)
def load_topics() -> Dict[str, List[str]]:
    try:
        with TOPICS_JSON.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def clean_option(opt: str) -> str:
    """Strip leading choice markers like 'A:' / 'B)' from options."""
    if opt is None:
        return ""
    s = str(opt).strip()
    s = re.sub(r"^[A-Da-d][\)\.:]\s*", "", s)
    return s


def dedup_and_shuffle(questions: List[Dict], num_questions: int) -> List[Dict]:
    seen = set()
    uniq = []
    for q in questions:
        qtext = str(q.get("question", "")).strip().lower()
        if not qtext or qtext in seen:
            continue
        seen.add(qtext)
        uniq.append(q)
    random.shuffle(uniq)
    return uniq[:num_questions]


def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=API_BASE_URL)


def ensure_quiz_db() -> None:
    import sqlite3

    conn = sqlite3.connect(QUIZ_DB_PATH)
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
                topic TEXT NOT NULL,
                difficulty TEXT NOT NULL,
                score INTEGER NOT NULL,
                total INTEGER NOT NULL,
                details TEXT,
                summary TEXT
            )
            """
        )
        # migrations
        cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        if "summary" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN summary TEXT DEFAULT ''")
        conn.commit()
    finally:
        conn.close()


def get_current_difficulty(student: str) -> str:
    import sqlite3

    ensure_quiz_db()
    conn = sqlite3.connect(QUIZ_DB_PATH)
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


def update_difficulty(student: str, score: int, total: int) -> str:
    pct = score / total
    order = ["Easy", "Medium", "Hard"]
    current = get_current_difficulty(student)
    idx = order.index(current)
    if pct >= 0.9 and idx < len(order) - 1:
        idx += 1
    elif pct >= 0.8 and idx < len(order) - 1:
        idx += 1
    elif pct < 0.6 and idx > 0:
        idx -= 1
    new_level = order[idx]
    import sqlite3

    conn = sqlite3.connect(QUIZ_DB_PATH)
    try:
        conn.execute(
            "UPDATE users SET current_difficulty = ? WHERE student = ?",
            (new_level, student),
        )
        conn.commit()
    finally:
        conn.close()
    return new_level


def fetch_last_summary(student: str) -> str:
    import sqlite3

    ensure_quiz_db()
    conn = sqlite3.connect(QUIZ_DB_PATH)
    try:
        row = conn.execute("SELECT summary FROM users WHERE student = ?", (student,)).fetchone()
        return row[0] if row else ""
    finally:
        conn.close()


def record_session(
    student: str,
    topic: str,
    difficulty: str,
    score: int,
    total: int,
    details: List[Dict],
    summary: str,
) -> None:
    import sqlite3

    ensure_quiz_db()
    conn = sqlite3.connect(QUIZ_DB_PATH)
    try:
        conn.execute(
            """
            INSERT INTO sessions (student, session_date, session_time, topic, difficulty, score, total, details, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student,
                date.today().isoformat(),
                datetime.now().isoformat(timespec="seconds"),
                topic,
                difficulty,
                score,
                total,
                json.dumps(details),
                summary,
            ),
        )
        conn.execute(
            "UPDATE users SET summary = ?, current_difficulty = ? WHERE student = ?",
            (summary, difficulty, student),
        )
        conn.commit()
    finally:
        conn.close()


def fetch_history(student: str) -> List[Tuple]:
    import sqlite3

    ensure_quiz_db()
    conn = sqlite3.connect(QUIZ_DB_PATH)
    try:
        return conn.execute(
            "SELECT session_date, topic, difficulty, score, total, summary FROM sessions WHERE student = ? ORDER BY session_time DESC",
            (student,),
        ).fetchall()
    finally:
        conn.close()


def generate_questions(
    client: OpenAI,
    num_questions: int,
    topic: str,
    difficulty: str,
    past_performance: str,
) -> List[Dict]:
    prompt = (
        f"Generate {num_questions} kid-friendly trivia questions for an 11-year-old."
        f" Topic: {topic}. Difficulty: {difficulty}."
        f" Adapt gently from past performance: {past_performance or 'None'}."
        " Keep it factual and simple. Mix what/where/why/how."
        " Each question must have 4 options labeled A-D, exactly 1 correct."
        " DO NOT include the correct answer or explanation in this response."
        ' Output JSON array only: [{"question":"...","options":["A: ...","B: ...","C: ...","D: ..."]}]'
    )
    attempts_log = []
    st.session_state["quiz_last_prompt"] = prompt

    def coerce_questions(parsed_obj):
        if isinstance(parsed_obj, list):
            return parsed_obj
        if isinstance(parsed_obj, dict):
            if isinstance(parsed_obj.get("questions"), list):
                return parsed_obj["questions"]
            for val in parsed_obj.values():
                if isinstance(val, list):
                    return val
            values = list(parsed_obj.values())
            if values and all(isinstance(v, dict) for v in values):
                return values
        return None

    for attempt in range(1):
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=900,
            temperature=0.5,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a friendly, fact-accurate trivia writer for kids."},
                {"role": "user", "content": prompt},
            ],
        )
        content = resp.choices[0].message.content.strip()
        st.session_state["quiz_last_raw"] = content
        attempts_log.append({"attempt": attempt + 1, "preview": content[:400]})

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            try:
                from json import JSONDecoder

                dec = JSONDecoder()
                objs = []
                idx = 0
                while idx < len(content):
                    try:
                        obj, end = dec.raw_decode(content, idx)
                        objs.append(obj)
                        idx = end
                    except json.JSONDecodeError:
                        idx += 1
                parsed = objs if objs else None
            except Exception:
                parsed = None

        # If the model returned a single question dict, wrap it in a list
        if isinstance(parsed, dict) and "question" in parsed and "options" in parsed:
            questions = [parsed]
        else:
            questions = coerce_questions(parsed) if parsed is not None else None

        if not isinstance(questions, list):
            continue

        cleaned = []
        for item in questions:
            if not isinstance(item, dict) or "question" not in item:
                continue
            cleaned.append(
                {
                    "question": str(item.get("question", "")).strip(),
                    "options": item.get("options") or [],
                }
            )
        deduped = dedup_and_shuffle(cleaned, num_questions)
        if len(deduped) >= num_questions:
            st.session_state["quiz_attempts_log"] = attempts_log
            return deduped[:num_questions]

        # Fallback: regex-extract object-like snippets and parse
        obj_snippets = re.findall(r"\{[^{}]*\"question\"[^{}]*\}", content, flags=re.DOTALL)
        cleaned2 = []
        for snip in obj_snippets:
            try:
                parsed_obj = json.loads(snip)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed_obj, dict) or "question" not in parsed_obj:
                continue
            cleaned2.append(
                {
                    "question": str(parsed_obj.get("question", "")).strip(),
                    "options": parsed_obj.get("options") or [],
                }
            )
        deduped2 = dedup_and_shuffle(cleaned2, num_questions)
        if len(deduped2) >= num_questions:
            st.session_state["quiz_attempts_log"] = attempts_log
            return deduped2[:num_questions]

    st.session_state["quiz_attempts_log"] = attempts_log
    raise ValueError(f"Could not generate enough questions. Last attempt preview: {attempts_log[-1] if attempts_log else 'none'}")


def summarize_progress(
    client: OpenAI,
    prev_summary: str,
    score: int,
    total: int,
    topic: str,
) -> str:
    prompt = (
        "Update a running summary of a student's trivia progress."
        " Return JSON ONLY: {\"summary\": str}."
        f" Previous summary: {prev_summary or 'None'}."
        f" Latest topic: {topic}, score: {score}/{total}."
        " Keep it concise (2-4 sentences), kid-friendly, highlight improvements and next focus areas."
    )
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=200,
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a concise, encouraging trivia coach for kids."},
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content.strip()
    parsed = json.loads(content)
    return str(parsed.get("summary", "")).strip()


def grade_answers(
    client: OpenAI, topic: str, difficulty: str, questions: List[Dict], user_answers: List[str]
) -> Tuple[int, List[Dict]]:
    def normalize(ans: str) -> str:
        return str(ans or "").strip().lower()

    def letter_from_option_text(opt: str) -> str:
        m = re.match(r"([A-Da-d])[:\)\.]\s*", str(opt))
        return m.group(1).upper() if m else ""

    def compute_is_correct(user: str, correct: str, opts: List[str]) -> bool:
        u = normalize(user)
        c = normalize(correct)
        if not u or not c:
            return False
        # Direct match
        if u == c:
            return True
        # If correct is a letter
        if len(c) == 1 and c in ["a", "b", "c", "d"]:
            # Map user choice to letter if user provided text
            for opt in opts:
                if normalize(opt) == u:
                    return True if letter_from_option_text(opt).lower() == c else False
            if len(u) == 1 and u in ["a", "b", "c", "d"]:
                return u == c
        # Compare by option text (strip leading letters)
        cleaned_opts = [normalize(clean_option(o)) for o in opts]
        if c in cleaned_opts and u == c:
            return True
        return False

    payload = []
    for q, ua in zip(questions, user_answers):
        payload.append(
            {
                "question": q.get("question", ""),
                "options": q.get("options", []),
                "user_answer": ua,
            }
        )
    prompt = (
        "Grade the multiple-choice answers. For each item, infer the correct answer letter (A/B/C/D) and brief explanation."
        ' Return JSON object: {"items":[{"question":str,"options":list,"user_answer":str,'
        '"correct_answer":str,"explanation":str,"is_correct":bool}]}'
        " Ensure is_correct is true/false. Do not add extra text."
        f" Topic: {topic}. Difficulty: {difficulty}."
        f" Data: {json.dumps(payload)}"
    )
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=800,
        temperature=0.4,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You grade concisely and factually for kids."},
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content.strip()
    st.session_state["quiz_grade_raw"] = content
    parsed = json.loads(content)
    results = []
    if isinstance(parsed, list):
        results = parsed
    elif isinstance(parsed, dict):
        results = parsed.get("items") or parsed.get("results") or []
    cleaned = []
    score = 0
    for item in results:
        if not isinstance(item, dict):
            continue
        is_correct = bool(item.get("is_correct", False))
        if is_correct:
            score += 1
        else:
            # try to compute if missing/false using returned correct_answer
            is_correct = compute_is_correct(
                item.get("user_answer", ""),
                item.get("correct_answer", ""),
                item.get("options", []),
            )
            if is_correct:
                score += 1
        cleaned.append(
            {
                "question": item.get("question", ""),
                "options": item.get("options", []),
                "user_answer": item.get("user_answer", ""),
                "correct_answer": item.get("correct_answer", ""),
                "explanation": item.get("explanation", ""),
                "is_correct": is_correct,
            }
        )
    return score, cleaned


def main():
    st.set_page_config(page_title="Trivia Quiz (Kids)")
    # Keep-alive heartbeat; safe because it does not trigger LLM calls
    st_autorefresh(interval=240_000, key="keepalive_quiz")
    topics = load_topics()
    api_key = os.getenv("XAI_API_KEY") or st.secrets.get("XAI_API_KEY", "")
    if not api_key:
        st.warning("Set XAI_API_KEY in env or secrets for Grok access.")

    st.title("Trivia Quiz for Kids 🎈")
    student = st.text_input("Student Name (required)", key="quiz_student")

    if not topics:
        st.error("topics.json missing or invalid.")
        return

    category = st.selectbox("Pick a category", list(topics.keys()))
    topic = category  # use category as the topic driver

    # Difficulty based on stored level
    difficulty = get_current_difficulty(student) if student else "Easy"

    debug_raw = st.checkbox("Show raw quiz response (debug)", value=False, key="quiz_debug_checkbox")

    if st.button("Start Quiz"):
        if not student:
            st.error("Enter the student name first.")
            st.stop()
        prev_summary = fetch_last_summary(student)
        client = get_client(api_key)
        with st.spinner("Generating quiz..."):
            try:
                questions = generate_questions(
                    client=client,
                    num_questions=7,
                    topic=topic,
                    difficulty=difficulty,
                    past_performance=prev_summary,
                )
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
                st.stop()
        st.session_state["quiz_questions"] = questions
        st.session_state["quiz_topic"] = topic
        st.session_state["quiz_difficulty"] = difficulty
        st.session_state["quiz_prev_summary"] = prev_summary
        st.session_state["quiz_locked"] = False
        st.success("Quiz ready!")
        if debug_raw and "quiz_last_raw" in st.session_state:
            st.text_area("Raw quiz response", st.session_state["quiz_last_raw"], height=200)
        if debug_raw and "quiz_last_prompt" in st.session_state:
            st.text_area(
                "Prompt used",
                st.session_state["quiz_last_prompt"],
                height=180,
            )
        if debug_raw and "quiz_attempts_log" in st.session_state:
            st.json(st.session_state["quiz_attempts_log"])

    questions = st.session_state.get("quiz_questions", [])
    if questions:
        st.subheader(f"Topic: {st.session_state.get('quiz_topic','')}")
        disable_inputs = st.session_state.get("quiz_locked", False)
        with st.form("quiz_form_kids"):
            for idx, q in enumerate(questions):
                st.markdown(f"**Q{idx + 1}. {q.get('question','')}**")
                raw_opts = q.get("options") or []
                opts = [clean_option(o) for o in raw_opts]
                key = f"quiz_answer_{idx}"
                if opts:
                    st.radio("Choose an answer:", opts, key=key, index=None, disabled=disable_inputs)
                else:
                    st.text_input("Your answer:", key=key, disabled=disable_inputs)
                st.markdown("---")
            submitted = st.form_submit_button("Submit Answers", disabled=disable_inputs)

        if submitted and disable_inputs:
            st.info("Quiz locked. Start a new quiz to answer again.")
        elif submitted:
            missing = []
            for idx, q in enumerate(questions):
                ans = st.session_state.get(f"quiz_answer_{idx}", None)
                if ans is None or (isinstance(ans, str) and not ans.strip()):
                    missing.append(idx + 1)
            if missing:
                st.error(f"Please answer all questions before submitting. Missing: {missing}")
                st.stop()

            client = get_client(api_key)
            user_answers = [st.session_state.get(f"quiz_answer_{idx}", "") for idx in range(len(questions))]
            with st.spinner("Grading..."):
                try:
                    score, qa_payload = grade_answers(
                        client,
                        st.session_state.get("quiz_topic", ""),
                        st.session_state.get("quiz_difficulty", "Easy"),
                        questions,
                        user_answers,
                    )
                except Exception as exc:
                    st.error(f"Grading failed: {exc}")
                    st.stop()

            try:
                new_summary = summarize_progress(
                    client,
                    st.session_state.get("quiz_prev_summary", ""),
                    score,
                    len(questions),
                    st.session_state.get("quiz_topic", ""),
                )
            except Exception:
                new_summary = st.session_state.get("quiz_prev_summary", "")

            new_level = update_difficulty(student, score, total=len(questions))
            record_session(
                student,
                st.session_state.get("quiz_topic", ""),
                new_level,
                score,
                len(questions),
                qa_payload,
                new_summary,
            )

            st.success(f"Score: {score}/{len(questions)}")
            st.caption(f"New difficulty: {new_level}")
            for idx, qa in enumerate(qa_payload):
                status = "✅" if qa["is_correct"] else "❌"
                st.markdown(f"**Q{idx + 1}. {status} {qa['question']}**")
                st.write(f"Your answer: {qa['user_answer'] or '—'}")
                st.write(f"Correct answer: {qa['correct_answer']}")
                if qa.get("explanation"):
                    st.caption(f"Why: {qa['explanation']}")
                st.markdown("---")

            st.session_state["quiz_locked"] = True
            for key in list(st.session_state.keys()):
                if key.startswith("quiz_answer_"):
                    del st.session_state[key]


if __name__ == "__main__":
    main()

