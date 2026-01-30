"""
Wordle App - A word guessing game for students.
Guess the 5-letter word in 6 tries!
"""

import random
import streamlit as st
from pathlib import Path

# Word list - common 5-letter words suitable for students
WORD_LIST = [
    # Animals
    "horse", "mouse", "tiger", "zebra", "shark", "whale", "eagle", "snake", "goose", "sheep",
    "bunny", "puppy", "kitty", "otter", "panda", "koala", "llama", "bison", "moose", "camel",
    # Nature
    "plant", "grass", "beach", "ocean", "river", "cloud", "storm", "sunny", "rainy", "earth",
    "stone", "coral", "maple", "daisy", "tulip", "flora", "frost", "flame", "water", "field",
    # Food
    "apple", "grape", "lemon", "mango", "melon", "peach", "berry", "bread", "candy", "pizza",
    "salad", "pasta", "juice", "honey", "sugar", "flour", "cream", "toast", "bacon", "steak",
    # School
    "books", "pencil", "paper", "chalk", "chair", "table", "class", "study", "learn", "teach",
    "smart", "brain", "think", "write", "spell", "count", "solve", "grade", "score", "award",
    # Actions
    "dance", "smile", "laugh", "sleep", "dream", "climb", "swing", "throw", "catch", "build",
    "paint", "draw", "craft", "bake", "clean", "shine", "spark", "glow", "bloom", "grow",
    # Objects
    "phone", "clock", "watch", "music", "piano", "drum", "light", "lamp", "couch", "house",
    "train", "plane", "truck", "bike", "wheel", "swing", "slide", "block", "robot", "magic",
    # Feelings/Descriptors
    "happy", "brave", "proud", "calm", "kind", "sweet", "funny", "silly", "quick", "super",
    "great", "fresh", "crisp", "warm", "cool", "loud", "quiet", "soft", "hard", "round",
    # More common words
    "world", "today", "night", "light", "right", "three", "about", "first", "would", "could",
    "their", "there", "where", "which", "those", "these", "other", "after", "under", "above",
    "start", "story", "point", "place", "space", "group", "sound", "young", "along", "grand",
]


def get_random_word() -> str:
    """Get a random 5-letter word."""
    return random.choice(WORD_LIST).upper()


def check_guess(guess: str, target: str) -> list:
    """
    Check the guess against the target word.
    Returns a list of tuples: (letter, status)
    Status: 'correct' (green), 'present' (yellow), 'absent' (gray)
    """
    guess = guess.upper()
    target = target.upper()
    result = []
    target_letters = list(target)
    
    # First pass: mark correct positions
    for i, letter in enumerate(guess):
        if letter == target[i]:
            result.append((letter, "correct"))
            target_letters[i] = None  # Mark as used
        else:
            result.append((letter, None))  # Placeholder
    
    # Second pass: mark present but wrong position
    for i, (letter, status) in enumerate(result):
        if status is None:
            if letter in target_letters:
                result[i] = (letter, "present")
                target_letters[target_letters.index(letter)] = None  # Mark as used
            else:
                result[i] = (letter, "absent")
    
    return result


def render_guess_row(result: list, row_num: int):
    """Render a single guess row with colored tiles."""
    cols = st.columns(5)
    for i, (letter, status) in enumerate(result):
        with cols[i]:
            if status == "correct":
                bg_color = "#6aaa64"  # Green
                text_color = "white"
            elif status == "present":
                bg_color = "#c9b458"  # Yellow
                text_color = "white"
            else:
                bg_color = "#787c7e"  # Gray
                text_color = "white"
            
            st.markdown(
                f"""
                <div style="
                    background-color: {bg_color};
                    color: {text_color};
                    font-size: 24px;
                    font-weight: bold;
                    text-align: center;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 2px;
                    font-family: monospace;
                ">{letter}</div>
                """,
                unsafe_allow_html=True
            )


def render_empty_row():
    """Render an empty row."""
    cols = st.columns(5)
    for i in range(5):
        with cols[i]:
            st.markdown(
                """
                <div style="
                    background-color: white;
                    color: transparent;
                    font-size: 24px;
                    font-weight: bold;
                    text-align: center;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 2px;
                    border: 2px solid #d3d6da;
                ">_</div>
                """,
                unsafe_allow_html=True
            )


def render_input_row(row_num: int) -> str:
    """Render the active input row with 5 text inputs styled as tiles."""
    cols = st.columns(5)
    letters = []
    
    for i in range(5):
        with cols[i]:
            letter = st.text_input(
                f"Letter {i+1}",
                max_chars=1,
                key=f"wordle_letter_{row_num}_{i}",
                label_visibility="collapsed",
            ).upper()
            letters.append(letter if letter.isalpha() else "")
    
    return "".join(letters)


def render_keyboard(used_letters: dict):
    """Render the on-screen keyboard showing letter status."""
    keyboard_rows = [
        "QWERTYUIOP",
        "ASDFGHJKL",
        "ZXCVBNM"
    ]
    
    st.markdown("### Keyboard")
    for row in keyboard_rows:
        cols = st.columns(len(row))
        for i, letter in enumerate(row):
            with cols[i]:
                status = used_letters.get(letter, "unused")
                if status == "correct":
                    bg_color = "#6aaa64"
                    text_color = "white"
                elif status == "present":
                    bg_color = "#c9b458"
                    text_color = "white"
                elif status == "absent":
                    bg_color = "#787c7e"
                    text_color = "white"
                else:
                    bg_color = "#d3d6da"
                    text_color = "black"
                
                st.markdown(
                    f"""
                    <div style="
                        background-color: {bg_color};
                        color: {text_color};
                        font-size: 14px;
                        font-weight: bold;
                        text-align: center;
                        padding: 8px 4px;
                        border-radius: 4px;
                        margin: 1px;
                    ">{letter}</div>
                    """,
                    unsafe_allow_html=True
                )


def init_game_state():
    """Initialize or reset the game state."""
    if "wordle_target" not in st.session_state:
        st.session_state["wordle_target"] = get_random_word()
    if "wordle_guesses" not in st.session_state:
        st.session_state["wordle_guesses"] = []
    if "wordle_results" not in st.session_state:
        st.session_state["wordle_results"] = []
    if "wordle_game_over" not in st.session_state:
        st.session_state["wordle_game_over"] = False
    if "wordle_won" not in st.session_state:
        st.session_state["wordle_won"] = False
    if "wordle_used_letters" not in st.session_state:
        st.session_state["wordle_used_letters"] = {}


def reset_game():
    """Reset the game for a new round."""
    st.session_state["wordle_target"] = get_random_word()
    st.session_state["wordle_guesses"] = []
    st.session_state["wordle_results"] = []
    st.session_state["wordle_game_over"] = False
    st.session_state["wordle_won"] = False
    st.session_state["wordle_used_letters"] = {}


def main():
    st.set_page_config(page_title="Wordle Game")
    
    # Custom CSS to style the letter input boxes
    st.markdown("""
        <style>
        /* Style the text inputs to look like Wordle tiles */
        div[data-testid="stTextInput"] input {
            text-align: center !important;
            font-size: 24px !important;
            font-weight: bold !important;
            text-transform: uppercase !important;
            padding: 12px !important;
            border: 2px solid #878a8c !important;
            border-radius: 5px !important;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: #565758 !important;
            box-shadow: none !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # JavaScript for auto-advance and Enter-to-submit (injected via components)
    import streamlit.components.v1 as components
    components.html("""
        <script>
        function setupWordleInputs() {
            // Access parent document (Streamlit's main frame)
            const doc = window.parent.document;
            const inputs = doc.querySelectorAll('input[type="text"][maxlength="1"]');
            if (inputs.length < 5) return;
            
            const letterInputs = Array.from(inputs).slice(-5);
            
            letterInputs.forEach((input, index) => {
                if (input._wordleSetup) return;
                input._wordleSetup = true;
                
                input.addEventListener('input', function(e) {
                    const val = e.target.value;
                    if (val.length === 1 && /[a-zA-Z]/.test(val)) {
                        if (index < letterInputs.length - 1) {
                            setTimeout(() => {
                                letterInputs[index + 1].focus();
                                letterInputs[index + 1].select();
                            }, 10);
                        }
                    }
                });
                
                input.addEventListener('keydown', function(e) {
                    if (e.key === 'Backspace' && e.target.value === '' && index > 0) {
                        e.preventDefault();
                        letterInputs[index - 1].focus();
                        letterInputs[index - 1].value = '';
                        // Trigger input event for Streamlit
                        letterInputs[index - 1].dispatchEvent(new Event('input', { bubbles: true }));
                    }
                    if (e.key === 'Enter') {
                        const allFilled = letterInputs.every(inp => inp.value.length === 1);
                        if (allFilled) {
                            const buttons = doc.querySelectorAll('button');
                            for (const btn of buttons) {
                                if (btn.textContent.includes('Submit')) {
                                    btn.click();
                                    break;
                                }
                            }
                        }
                    }
                });
            });
        }
        
        // Run setup with retry
        let attempts = 0;
        const interval = setInterval(() => {
            setupWordleInputs();
            attempts++;
            if (attempts > 20) clearInterval(interval);
        }, 300);
        </script>
    """, height=0)
    
    st.title("🟩 Wordle")
    st.caption("Guess the 5-letter word in 6 tries!")
    
    init_game_state()
    
    max_guesses = 6
    target = st.session_state["wordle_target"]
    guesses = st.session_state["wordle_guesses"]
    results = st.session_state["wordle_results"]
    game_over = st.session_state["wordle_game_over"]
    won = st.session_state["wordle_won"]
    used_letters = st.session_state["wordle_used_letters"]
    
    # Game instructions
    with st.expander("How to Play", expanded=False):
        st.markdown("""
        **Guess the word in 6 tries!**
        
        - Type a 5-letter word and press Enter
        - 🟩 **Green** = Letter is correct and in the right spot
        - 🟨 **Yellow** = Letter is in the word but wrong spot
        - ⬜ **Gray** = Letter is not in the word
        
        Good luck!
        """)
    
    st.divider()
    
    # Display previous guesses
    for i, result in enumerate(results):
        render_guess_row(result, i)
    
    # Game over state
    if game_over:
        # Show remaining empty rows
        for i in range(len(results), max_guesses):
            render_empty_row()
        
        st.divider()
        
        if won:
            st.success(f"🎉 Congratulations! You got it in {len(guesses)} {'try' if len(guesses) == 1 else 'tries'}!")
            st.balloons()
        else:
            st.error(f"😔 Game Over! The word was **{target}**")
        
        if st.button("🔄 Play Again", use_container_width=True):
            reset_game()
            st.rerun()
    else:
        # Active input row - type directly in the boxes
        current_row = len(guesses)
        st.caption(f"Type your guess (Attempt {current_row + 1}/{max_guesses}):")
        guess_input = render_input_row(current_row)
        
        # Show remaining empty rows after input
        for i in range(current_row + 1, max_guesses):
            render_empty_row()
        
        st.divider()
        
        # Submit button
        submit_clicked = st.button("✓ Submit Guess", use_container_width=True)
        
        if submit_clicked:
            # Validate guess
            if len(guess_input) != 5:
                st.warning("Please fill in all 5 letters.")
            elif not guess_input.isalpha():
                st.warning("Please use only letters (A-Z).")
            else:
                # Process the guess
                result = check_guess(guess_input, target)
                guesses.append(guess_input)
                results.append(result)
                
                # Update used letters
                for letter, status in result:
                    current_status = used_letters.get(letter, "unused")
                    # Only upgrade status: absent -> present -> correct
                    if status == "correct":
                        used_letters[letter] = "correct"
                    elif status == "present" and current_status != "correct":
                        used_letters[letter] = "present"
                    elif status == "absent" and current_status == "unused":
                        used_letters[letter] = "absent"
                
                # Check for win
                if guess_input == target:
                    st.session_state["wordle_game_over"] = True
                    st.session_state["wordle_won"] = True
                elif len(guesses) >= max_guesses:
                    st.session_state["wordle_game_over"] = True
                    st.session_state["wordle_won"] = False
                
                st.rerun()
    
    # Show keyboard
    st.divider()
    render_keyboard(used_letters)
    
    # Stats/hints section
    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Guesses Used", f"{len(guesses)}/{max_guesses}")
    with col2:
        if st.button("🔄 New Game"):
            reset_game()
            st.rerun()


if __name__ == "__main__":
    main()
