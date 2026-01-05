import os
from pathlib import Path
from typing import List

import streamlit as st
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
MODEL_NAME = "grok-4-fast"
API_BASE_URL = "https://api.x.ai/v1"

SYSTEM_PROMPT = (
    "You are a friendly English coach for 11-12 year olds. "
    "Keep topics light, encouraging, and age-appropriate. "
    "Start with simple questions and build confidence."
)


def get_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=API_BASE_URL)


def start_prompt(client: OpenAI) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=200,
        temperature=0.6,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": "Start a short conversation starter (1-2 sentences) for a kid about a fun topic.",
            },
        ],
    )
    return resp.choices[0].message.content.strip()


def validate_response(client: OpenAI, user_response: str) -> str:
    prompt = (
        "A child answered the prompt. Check for clarity, kindness, and correctness. "
        "Return JSON ONLY: {\"corrected\": str, \"tips\": list of str}."
        f" Response: {user_response}"
    )
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=300,
        temperature=0.4,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.choices[0].message.content.strip()


def main():
    st.set_page_config(page_title="Speech Coach")
    api_key = os.getenv("XAI_API_KEY") or st.secrets.get("XAI_API_KEY", "")
    if not api_key:
        st.warning("Set XAI_API_KEY in env or secrets for Grok access.")

    st.title("Speech Coach (Kid-Friendly)")
    st.caption("Start a fun conversation, record a 5-10 sentence voice reply, then get gentle corrections.")

    client = get_client(api_key) if api_key else None

    if st.button("Start a conversation"):
        if not client:
            st.error("Missing XAI_API_KEY")
        else:
            try:
                starter = start_prompt(client)
                st.session_state["speech_starter"] = starter
                st.session_state["speech_feedback_raw"] = ""
                st.success("Starter ready!")
            except Exception as exc:
                st.error(f"Could not start conversation: {exc}")

    starter = st.session_state.get("speech_starter")
    if starter:
        st.markdown(f"**Conversation Starter:** {starter}")

    st.markdown("### Your Response")
    audio_file = st.file_uploader("Upload a voice reply (wav/mp3/m4a)", type=["wav", "mp3", "m4a"])
    transcript = st.text_area(
        "Or type/paste the transcript (5-10 sentences)", height=120, key="speech_transcript"
    )
    st.caption("Note: Live speech-to-text is not built-in; please provide the transcript.")

    if st.button("Validate Response"):
        if not client:
            st.error("Missing XAI_API_KEY")
        else:
            if not transcript.strip():
                st.error("Please provide a transcript of the voice reply.")
            else:
                try:
                    feedback_raw = validate_response(client, transcript)
                    st.session_state["speech_feedback_raw"] = feedback_raw
                    st.success("Feedback generated!")
                except Exception as exc:
                    st.error(f"Validation failed: {exc}")

    if st.session_state.get("speech_feedback_raw"):
        st.markdown("### Suggested Correction")
        st.code(st.session_state["speech_feedback_raw"], language="json")


if __name__ == "__main__":
    main()



