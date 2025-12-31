import json
import os
import re
from collections import Counter
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

# Simple math concepts/keywords to surface in concept maps
KEYWORDS = [
    "integer",
    "fraction",
    "decimal",
    "percent",
    "ratio",
    "rate",
    "proportion",
    "equation",
    "expression",
    "inequality",
    "variable",
    "geometry",
    "area",
    "perimeter",
    "volume",
    "surface",
    "triangle",
    "circle",
    "angle",
    "probability",
    "statistics",
    "mean",
    "median",
    "mode",
    "range",
    "slope",
    "graph",
    "table",
    "unit",
    "rate",
]


def ensure_pdf_folder() -> None:
    PDF_FOLDER.mkdir(exist_ok=True)


def clean_text(text: str) -> str:
    lowered = text.lower()
    return re.sub(r"[^\w\s]", " ", lowered)


def extract_concepts_from_text(text: str) -> Dict[str, int]:
    cleaned = clean_text(text)
    tokens = cleaned.split()
    return dict(Counter(token for token in tokens if token in KEYWORDS))


def extract_concepts_from_pdf(path: Path) -> Dict[str, Dict[str, int]]:
    """Return a mapping of lesson -> keyword counts for a single PDF."""
    lessons: Dict[str, List[str]] = {}
    current_lesson = "Unknown"

    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            header_match = re.search(r"Lesson\s+(\d+)", page_text, re.IGNORECASE)
            if header_match:
                current_lesson = f"Lesson {header_match.group(1)}"
            lessons.setdefault(current_lesson, []).append(page_text)

    # If no lesson headers were found, treat the whole file as one lesson
    if list(lessons.keys()) == ["Unknown"]:
        full_text = "\n".join(lessons["Unknown"])
        return {path.stem: extract_concepts_from_text(full_text)}

    lesson_counts: Dict[str, Dict[str, int]] = {}
    for lesson_name, pages in lessons.items():
        lesson_text = "\n".join(pages)
        lesson_counts[f"{path.stem} - {lesson_name}"] = extract_concepts_from_text(lesson_text)
    return lesson_counts


def build_lesson_map(save: bool = False) -> Dict[str, Dict[str, int]]:
    ensure_pdf_folder()
    lessons = {}
    for pdf_path in sorted(PDF_FOLDER.glob("*.pdf")):
        per_lesson = extract_concepts_from_pdf(pdf_path)
        lessons.update(per_lesson)
    if save:
        save_lesson_map(lessons)
    return lessons


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
        " Mix multiple-choice and short-answer. Output JSON list:"
        ' [{"question": str, "options": list or null, "answer": str, "explanation": str}].'
        " Do not include text outside JSON."
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=1500,
        temperature=0.4,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": "You are a helpful 6th-grade math question writer."},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content.strip()
    st.session_state["math_unit_last_raw"] = content
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse questions JSON. Details: {exc}") from exc
    questions = parsed.get("questions") if isinstance(parsed, dict) else parsed
    if not isinstance(questions, list):
        raise ValueError("Questions response was not a list.")
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


def main():
    st.set_page_config(page_title="Math Unit Generator")
    ensure_pdf_folder()
    api_key = os.getenv("XAI_API_KEY") or st.secrets.get("XAI_API_KEY", "")
    if not api_key:
        st.warning("Set XAI_API_KEY in env or secrets for Grok access.")

    st.title("Math Unit Lesson Questions")
    st.caption("Drop unit PDFs into math-unit-pdf/. Click Refresh to rebuild concept maps.")

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Refresh lessons from folder"):
            st.session_state["math_unit_lessons"] = build_lesson_map(save=False)
            st.success("Lessons refreshed (JSON not written yet).")

    lessons = st.session_state.get("math_unit_lessons") or build_lesson_map(save=False)
    lesson_names = list(lessons.keys())

    if not lesson_names:
        st.info("No PDFs found in math-unit-pdf/. Add some and click refresh.")
        return

    with c2:
        lesson = st.selectbox("Choose a lesson", lesson_names)

    concepts = top_concepts(lessons.get(lesson, {}))

    if concepts:
        st.markdown("**Concept map (top keywords):**")
        for name, weight in concepts:
            st.write(f"- {name}: {weight}")
    else:
        st.info("No keywords detected; generation will proceed with a generic focus.")

    debug_raw = st.checkbox("Show raw API response (debug)", value=False)

    if st.button("Generate Questions"):
        # Persist current lessons to JSON when generating
        save_lesson_map(lessons)
        client = get_client(api_key)
        with st.spinner("Generating questions..."):
            try:
                qs = generate_questions(client, lesson, concepts)
            except Exception as exc:
                st.error(f"Generation failed: {exc}")
                return
        st.session_state["math_unit_questions"] = qs
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


if __name__ == "__main__":
    main()

