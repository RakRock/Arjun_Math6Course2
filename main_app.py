import streamlit as st

# Import the existing apps
import app as math_app
import vocab_app


def _run_without_page_config(fn):
    """Run a Streamlit app function while temporarily disabling set_page_config."""
    orig = st.set_page_config
    st.set_page_config = lambda *args, **kwargs: None  # type: ignore
    try:
        fn()
    finally:
        st.set_page_config = orig


def main():
    st.set_page_config(page_title="Learning Hub")
    st.title("Learning Hub")

    math_tab, vocab_tab = st.tabs(["Math", "Vocab Builder"])

    with math_tab:
        _run_without_page_config(math_app.main)

    with vocab_tab:
        _run_without_page_config(vocab_app.vocab_tab_content)


if __name__ == "__main__":
    main()

