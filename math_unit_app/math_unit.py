import json
import os
import re
import smtplib
import ssl
from collections import Counter
from datetime import date
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Tuple

import pdfplumber
import streamlit as st
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
PDF_FOLDER = BASE_DIR / "math-unit-pdf"
CONCEPTS_JSON = BASE_DIR / "math_unit_concepts.json"
MODEL_NAME = "grok-4-fast"
API_BASE_URL = "https://api.x.ai/v1"

def ensure_pdf_folder() -> None:
    PDF_FOLDER.mkdir(exist_ok=True)


def extract_text_by_lesson(path: Path) -> Dict[str, str]:
    """Extract text from PDF, grouped by lesson headers."""
    lessons: Dict[str, List[str]] = {}
    current_lesson = "Entire Unit"

    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            header_match = re.search(r"Lesson\s+(\d+)", page_text, re.IGNORECASE)
            if header_match:
                current_lesson = f"Lesson {header_match.group(1)}"
            lessons.setdefault(current_lesson, []).append(page_text)

    # If no lesson headers were found, treat the whole file as one unit
    if list(lessons.keys()) == ["Entire Unit"]:
        return {f"{path.stem} - Entire Unit": "\n".join(lessons["Entire Unit"])}

    return {f"{path.stem} - {lesson_name}": "\n".join(pages) 
            for lesson_name, pages in lessons.items()}


def extract_concepts_with_llm(client: OpenAI, lesson_name: str, lesson_text: str) -> Dict[str, int]:
    """Use LLM to extract meaningful math concepts from lesson text."""
    # Truncate text if too long (keep first ~3000 chars for context)
    truncated_text = lesson_text[:3000] if len(lesson_text) > 3000 else lesson_text
    
    prompt = f"""Analyze this 6th-grade math lesson and identify the KEY MATHEMATICAL CONCEPTS being taught.

Lesson: {lesson_name}
Content excerpt:
{truncated_text}

Return a JSON object where:
- Keys are specific math concepts/topics (e.g., "triangle inequality", "area of circles", "complementary angles")
- Values are importance scores from 1-100 (higher = more central to the lesson)

Focus on:
- What mathematical skills are being taught
- What formulas or theorems are covered
- What types of problems students will solve

Return ONLY valid JSON, nothing else. Example format:
{{"area of circles": 90, "circumference formula": 85, "pi approximation": 70, "radius vs diameter": 60}}"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            max_tokens=500,
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a math curriculum analyzer. Extract key concepts from lesson content."},
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
        st.warning(f"LLM concept extraction failed for {lesson_name}: {exc}")
        return {}


def build_lesson_map_from_pdf(save: bool = False) -> Dict[str, str]:
    """Build a map of lesson names to their raw text (for LLM processing)."""
    ensure_pdf_folder()
    lessons = {}
    for pdf_path in sorted(PDF_FOLDER.glob("*.pdf")):
        per_lesson = extract_text_by_lesson(pdf_path)
        lessons.update(per_lesson)
    return lessons


def build_concepts_with_llm(client: OpenAI, lessons_text: Dict[str, str], progress_callback=None) -> Dict[str, Dict[str, int]]:
    """Generate concept maps for all lessons using LLM."""
    concepts = {}
    total = len(lessons_text)
    for idx, (lesson_name, text) in enumerate(lessons_text.items()):
        if progress_callback:
            progress_callback(idx + 1, total, lesson_name)
        concepts[lesson_name] = extract_concepts_with_llm(client, lesson_name, text)
    return concepts


def load_existing_concepts() -> Dict[str, Dict[str, int]]:
    """Load existing concepts from JSON file if available."""
    if CONCEPTS_JSON.exists():
        try:
            with CONCEPTS_JSON.open("r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def top_concepts(counts: Dict[str, int], n: int = 10) -> List[Tuple[str, int]]:
    if not counts:
        return []
    return Counter(counts).most_common(n)


def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=API_BASE_URL)


def save_lesson_map(lessons: Dict[str, Dict[str, int]]) -> None:
    try:
        with CONCEPTS_JSON.open("w", encoding="utf-8") as f:
            json.dump(lessons, f, indent=2)
    except OSError:
        pass  # fail silently if unable to write


def generate_questions(client: OpenAI, lesson: str, concepts: List[Tuple[str, int]]) -> List[Dict]:
    focus = [f"{c} (emphasis: {w})" for c, w in concepts] if concepts else ["core lesson topics"]
    prompt = (
        "Generate 10 questions for 6th-grade math, based on this lesson."
        f" Lesson: {lesson}. Focus on: {focus}."
        " Mix multiple-choice and short-answer."
        ' Return JSON object: {"questions": [{"question": str, "options": list or null, "answer": str, "explanation": str}, ...]}'
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=1500,
        temperature=0.4,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful 6th-grade math question writer. Always return valid JSON."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content.strip()
    st.session_state["math_unit_last_raw"] = content
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse questions JSON. Details: {exc}") from exc
    
    # Extract questions from various possible response formats
    questions = None
    if isinstance(parsed, list):
        questions = parsed
    elif isinstance(parsed, dict):
        # Try common keys: "questions", "data", "items", or first list value
        for key in ["questions", "data", "items"]:
            if key in parsed and isinstance(parsed[key], list):
                questions = parsed[key]
                break
        if questions is None:
            # Find first list value in the dict
            for value in parsed.values():
                if isinstance(value, list):
                    questions = value
                    break
    
    if not isinstance(questions, list):
        raise ValueError(f"Questions response was not a list. Got: {type(parsed).__name__}")
    
    cleaned = []
    for item in questions:
        if not isinstance(item, dict) or "question" not in item:
            continue
        cleaned.append(
            {
                "question": str(item.get("question", "")).strip(),
                "options": item.get("options") or None,
                "answer": str(item.get("answer", "")).strip(),
                "explanation": str(item.get("explanation", "")).strip(),
            }
        )
    if len(cleaned) < 5:
        raise ValueError(f"Too few questions returned ({len(cleaned)}).")
    return cleaned[:10]


def send_results_email(
    student: str,
    lesson: str,
    score: int,
    total: int,
    results: List[Dict],
) -> Tuple[bool, str]:
    """Send quiz results via email."""
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
        f"Lesson: {lesson}",
        f"Score: {score}/{total}",
        "",
        "Questions and answers:",
    ]
    for idx, res in enumerate(results, start=1):
        status = "✓" if res.get("is_correct") else "✗"
        body_lines.append(f"{idx}. [{status}] {res.get('question', '')}")
        body_lines.append(f"   Your answer: {res.get('your_answer', '') or '—'}")
        if not res.get("is_correct"):
            body_lines.append(f"   Correct answer: {res.get('correct_answer', '')}")
        if res.get("explanation"):
            body_lines.append(f"   Why: {res.get('explanation', '')}")
        body_lines.append("")
    
    msg = MIMEText("\n".join(body_lines))
    msg["Subject"] = f"Math Unit Quiz Results - {student} - {date.today().isoformat()}"
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


def main():
    st.set_page_config(page_title="Math Unit Generator")
    ensure_pdf_folder()
    api_key = os.getenv("XAI_API_KEY") or st.secrets.get("XAI_API_KEY", "")
    if not api_key:
        st.warning("Set XAI_API_KEY in env or secrets for Grok access.")

    st.title("Math Unit Lesson Questions")
    st.caption("Drop unit PDFs into math-unit-pdf/. Click Refresh to rebuild concept maps.")

    # Student name input
    student_name = st.text_input("Enter your name:", key="math_unit_student_name")
    if not student_name:
        st.info("Please enter your name to start.")
        return

    # Load existing concepts or initialize empty
    if "math_unit_concepts" not in st.session_state:
        st.session_state["math_unit_concepts"] = load_existing_concepts()
    
    # Get lesson text from PDFs
    if "math_unit_lesson_texts" not in st.session_state:
        st.session_state["math_unit_lesson_texts"] = build_lesson_map_from_pdf()
    
    lesson_texts = st.session_state["math_unit_lesson_texts"]
    concepts_map = st.session_state["math_unit_concepts"]
    lesson_names = list(lesson_texts.keys())

    if not lesson_names:
        st.info("No PDFs found in math-unit-pdf/. Add some and refresh.")
        return

    # Lesson selection
    lesson = st.selectbox("Choose a lesson", lesson_names)
    
    # Action buttons row
    btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
    with btn_col1:
        refresh_clicked = st.button("🔄 Refresh PDFs", use_container_width=True)
    with btn_col2:
        generate_ai_clicked = st.button("🤖 Generate Concepts", use_container_width=True)
    
    if refresh_clicked:
        st.session_state["math_unit_lesson_texts"] = build_lesson_map_from_pdf()
        st.success("PDFs refreshed!")
        st.rerun()
    
    if generate_ai_clicked:
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
                new_concepts = build_concepts_with_llm(
                    client, lesson_texts, progress_callback=update_progress
                )
                st.session_state["math_unit_concepts"] = new_concepts
                save_lesson_map(new_concepts)
            
            progress_bar.empty()
            status_text.empty()
            st.success("Concepts generated and saved!")
            st.rerun()

    # Display concepts for selected lesson
    lesson_concepts = concepts_map.get(lesson, {})
    concepts = top_concepts(lesson_concepts, n=10)

    if concepts:
        with st.expander("📚 Key Concepts (AI-extracted)", expanded=True):
            concept_cols = st.columns(2)
            for idx, (name, weight) in enumerate(concepts):
                with concept_cols[idx % 2]:
                    st.markdown(f"• **{name}** `{weight}`")
    else:
        st.info("No concepts found. Click '🤖 Generate Concepts' to analyze lessons.")

    st.divider()
    
    # Generate Questions button
    gen_col1, gen_col2 = st.columns([1, 3])
    with gen_col1:
        generate_clicked = st.button("📝 Generate Questions", use_container_width=True)
    with gen_col2:
        debug_raw = st.checkbox("Show debug info", value=False)
    
    if generate_clicked:
        # Persist current concepts to JSON when generating
        save_lesson_map(concepts_map)
        client = get_client(api_key)
        with st.spinner("Generating questions..."):
            try:
                qs = generate_questions(client, lesson, concepts)
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
                return
        st.session_state["math_unit_questions"] = qs
        st.session_state["math_unit_current_lesson"] = lesson  # Store for email
        st.session_state["math_unit_submitted"] = False  # Reset submission state
        st.success("Questions ready!")
        if debug_raw and "math_unit_last_raw" in st.session_state:
            st.text_area("Raw response", st.session_state["math_unit_last_raw"], height=200)

    qs = st.session_state.get("math_unit_questions", [])
    if qs:
        st.subheader("Questions")
        with st.form("math_unit_form"):
            for idx, q in enumerate(qs):
                st.markdown(f"**Q{idx + 1}. {q.get('question','')}**")
                opts = q.get("options")
                key = f"math_unit_answer_{idx}"
                if opts and isinstance(opts, list):
                    st.radio("Choose an answer:", opts, key=key, index=None)
                else:
                    st.text_input("Your answer:", key=key)
                st.markdown("---")
            submitted = st.form_submit_button("Submit Answers")

        if submitted:
            score = 0
            results = []
            for idx, q in enumerate(qs):
                user_ans = str(st.session_state.get(f"math_unit_answer_{idx}", "")).strip()
                correct = str(q.get("answer", "")).strip()
                is_correct = user_ans.lower() == correct.lower()
                if is_correct:
                    score += 1
                results.append(
                    {
                        "question": q.get("question", ""),
                        "your_answer": user_ans,
                        "correct_answer": correct,
                        "explanation": q.get("explanation", ""),
                        "is_correct": is_correct,
                    }
                )

            st.success(f"Score: {score}/{len(qs)}")
            for idx, res in enumerate(results):
                status = "✅" if res["is_correct"] else "❌"
                st.markdown(f"**Q{idx + 1}. {status} {res['question']}**")
                st.write(f"Your answer: {res['your_answer'] or '—'}")
                if not res["is_correct"]:
                    st.write(f"Correct: {res['correct_answer']}")
                if res["explanation"]:
                    st.caption(f"Why: {res['explanation']}")
                st.markdown("---")

            # Send email with results (only once per submission)
            if not st.session_state.get("math_unit_submitted", False):
                current_lesson = st.session_state.get("math_unit_current_lesson", lesson)
                email_ok, email_msg = send_results_email(
                    student=student_name,
                    lesson=current_lesson,
                    score=score,
                    total=len(qs),
                    results=results,
                )
                st.session_state["math_unit_submitted"] = True
                if email_ok:
                    st.success(f"📧 {email_msg}")
                else:
                    st.warning(f"📧 {email_msg}")


if __name__ == "__main__":
    main()

