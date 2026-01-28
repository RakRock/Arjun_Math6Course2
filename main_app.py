import streamlit as st

# Import the existing apps
from math_app import app as math_app
from vocab_app import vocab_app
from quiz_app import quiz_app
from math_unit_app import math_unit


def _run_without_page_config(fn):
    """Run a Streamlit app function while temporarily disabling set_page_config."""
    orig = st.set_page_config
    st.set_page_config = lambda *args, **kwargs: None  # type: ignore
    try:
        fn()
    finally:
        st.set_page_config = orig


def main():
    # Lightweight health endpoint for uptime checks (e.g., cron ping)
    params = st.query_params
    if "health" in params:
        st.write("ok")
        return

    st.set_page_config(page_title="Learning Hub")
    st.title("Learning Hub")

    # Ensure local databases exist (they are gitignored)
    try:
        math_app.ensure_db()
    except Exception:
        pass
    try:
        vocab_app.ensure_vocab_db()
    except Exception:
        pass

    math_tab, vocab_tab, quiz_tab, unit_tab = st.tabs(
        ["Math", "Vocab Builder", "Quiz", "Math Unit"]
    )

    with math_tab:
        _run_without_page_config(math_app.main)

    with vocab_tab:
        _run_without_page_config(vocab_app.vocab_tab_content)

    with quiz_tab:
        _run_without_page_config(quiz_app.main)

    with unit_tab:
        _run_without_page_config(math_unit.main)


if __name__ == "__main__":
    main()

