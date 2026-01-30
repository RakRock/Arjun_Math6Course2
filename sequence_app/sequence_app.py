"""
Sequence App - Fun pattern and number sequence game for 5-6 year olds.
Find the missing item in the pattern!
"""

import random
import streamlit as st


# ============ NUMBER SEQUENCES ============

def generate_simple_sequence():
    """Generate a simple +1 counting sequence."""
    start = random.randint(1, 10)
    sequence = [start + i for i in range(5)]
    return sequence, "+1 counting", "number"


def generate_skip_count_2():
    """Generate a skip counting by 2 sequence."""
    start = random.choice([2, 4, 6, 8, 10])
    sequence = [start + (i * 2) for i in range(5)]
    return sequence, "counting by 2s", "number"


def generate_skip_count_5():
    """Generate a skip counting by 5 sequence."""
    start = random.choice([5, 10, 15])
    sequence = [start + (i * 5) for i in range(5)]
    return sequence, "counting by 5s", "number"


def generate_skip_count_10():
    """Generate a skip counting by 10 sequence."""
    start = random.choice([10, 20, 30])
    sequence = [start + (i * 10) for i in range(5)]
    return sequence, "counting by 10s", "number"


def generate_countdown():
    """Generate a countdown sequence (-1)."""
    start = random.randint(10, 20)
    sequence = [start - i for i in range(5)]
    return sequence, "counting down", "number"


def generate_plus_2():
    """Generate a +2 sequence."""
    start = random.randint(1, 10)
    sequence = [start + (i * 2) for i in range(5)]
    return sequence, "+2 pattern", "number"


def generate_odd_numbers():
    """Generate odd numbers sequence."""
    start = random.choice([1, 3, 5])
    sequence = [start + (i * 2) for i in range(5)]
    return sequence, "odd numbers", "number"


def generate_even_numbers():
    """Generate even numbers sequence."""
    start = random.choice([2, 4, 6])
    sequence = [start + (i * 2) for i in range(5)]
    return sequence, "even numbers", "number"


def generate_countdown_2():
    """Generate countdown by 2 sequence."""
    start = random.choice([20, 18, 16, 14])
    sequence = [start - (i * 2) for i in range(5)]
    return sequence, "counting down by 2s", "number"


def generate_plus_3():
    """Generate a +3 sequence."""
    start = random.randint(1, 5)
    sequence = [start + (i * 3) for i in range(5)]
    return sequence, "+3 pattern", "number"


def generate_tens():
    """Generate tens sequence (10, 20, 30...)."""
    sequence = [10, 20, 30, 40, 50]
    return sequence, "counting by 10s", "number"


def generate_ones_to_five():
    """Generate simple 1-5 sequence."""
    sequence = [1, 2, 3, 4, 5]
    return sequence, "counting 1 to 5", "number"


def generate_repeat_number():
    """Generate repeating number pattern (e.g., 2, 2, 3, 3, 4)."""
    start = random.randint(1, 5)
    sequence = []
    for i in range(3):
        sequence.extend([start + i, start + i])
    return sequence[:5], "repeat number pattern", "number"


# ============ SHAPE/EMOJI PATTERNS ============

# Pattern sets for visual sequences
PATTERN_SETS = {
    "shapes": ["🔴", "🔵", "🟢", "🟡", "🟣", "🟠"],
    "animals": ["🐶", "🐱", "🐰", "🐻", "🐼", "🦊"],
    "fruits": ["🍎", "🍊", "🍋", "🍇", "🍓", "🍌"],
    "weather": ["☀️", "🌙", "⭐", "☁️", "🌈", "❄️"],
    "nature": ["🌸", "🌺", "🌻", "🌷", "🌹", "💐"],
    "transport": ["🚗", "🚌", "🚀", "✈️", "🚂", "🚢"],
    "food": ["🍕", "🍔", "🌮", "🍩", "🧁", "🍪"],
    "sports": ["⚽", "🏀", "🎾", "⚾", "🏈", "🎱"],
    "faces": ["😀", "😊", "🥰", "😎", "🤗", "😇"],
    "sea": ["🐟", "🐠", "🐡", "🦈", "🐙", "🦑"],
    "bugs": ["🐛", "🦋", "🐝", "🐞", "🦗", "🐜"],
    "hands": ["👋", "✋", "🤚", "👍", "👎", "✌️"],
    "hearts": ["❤️", "🧡", "💛", "💚", "💙", "💜"],
    "stars": ["⭐", "🌟", "✨", "💫", "🌠", "⚡"],
}


def generate_ab_pattern():
    """Generate an AB repeating pattern (e.g., 🔴🔵🔴🔵🔴)."""
    category = random.choice(list(PATTERN_SETS.keys()))
    items = random.sample(PATTERN_SETS[category], 2)
    a, b = items[0], items[1]
    sequence = [a, b, a, b, a, b][:5]  # ABABA
    return sequence, "AB repeating pattern", "pattern"


def generate_abc_pattern():
    """Generate an ABC repeating pattern (e.g., 🔴🔵🟢🔴🔵)."""
    category = random.choice(list(PATTERN_SETS.keys()))
    items = random.sample(PATTERN_SETS[category], 3)
    a, b, c = items[0], items[1], items[2]
    sequence = [a, b, c, a, b, c][:5]  # ABCAB
    return sequence, "ABC repeating pattern", "pattern"


def generate_aab_pattern():
    """Generate an AAB pattern (e.g., 🔴🔴🔵🔴🔴)."""
    category = random.choice(list(PATTERN_SETS.keys()))
    items = random.sample(PATTERN_SETS[category], 2)
    a, b = items[0], items[1]
    sequence = [a, a, b, a, a, b][:5]  # AABAA
    return sequence, "AAB repeating pattern", "pattern"


def generate_abb_pattern():
    """Generate an ABB pattern (e.g., 🔴🔵🔵🔴🔵)."""
    category = random.choice(list(PATTERN_SETS.keys()))
    items = random.sample(PATTERN_SETS[category], 2)
    a, b = items[0], items[1]
    sequence = [a, b, b, a, b, b][:5]  # ABBAB
    return sequence, "ABB repeating pattern", "pattern"


def generate_abcd_pattern():
    """Generate an ABCD pattern (e.g., 🔴🔵🟢🟡🔴)."""
    category = random.choice(list(PATTERN_SETS.keys()))
    items = random.sample(PATTERN_SETS[category], 4)
    sequence = items + [items[0]]  # ABCDA
    return sequence, "ABCD repeating pattern", "pattern"


def generate_growing_pattern():
    """Generate a growing pattern (e.g., ⭐, ⭐⭐, ⭐⭐⭐)."""
    emoji = random.choice(["⭐", "❤️", "🌟", "💎", "🔵", "🟢"])
    sequence = [emoji * (i + 1) for i in range(5)]
    return sequence, "growing pattern", "pattern"


def generate_shrinking_pattern():
    """Generate a shrinking pattern (e.g., ⭐⭐⭐⭐, ⭐⭐⭐, ⭐⭐)."""
    emoji = random.choice(["⭐", "❤️", "🌟", "💎", "🔵", "🟢"])
    sequence = [emoji * (5 - i) for i in range(5)]
    return sequence, "shrinking pattern", "pattern"


def generate_mirror_pattern():
    """Generate a mirror/palindrome pattern (e.g., ABCBA)."""
    category = random.choice(list(PATTERN_SETS.keys()))
    items = random.sample(PATTERN_SETS[category], 3)
    a, b, c = items[0], items[1], items[2]
    sequence = [a, b, c, b, a]  # ABCBA
    return sequence, "mirror pattern", "pattern"


def generate_color_size_pattern():
    """Generate alternating big/small pattern."""
    items = random.choice([
        ["🔴", "⚫", "🔴", "⚫", "🔴"],  # red, black
        ["🌕", "🌑", "🌕", "🌑", "🌕"],  # full moon, new moon
        ["😀", "😢", "😀", "😢", "😀"],  # happy, sad
        ["👆", "👇", "👆", "👇", "👆"],  # up, down
        ["🐘", "🐁", "🐘", "🐁", "🐘"],  # big, small animal
        ["🌞", "🌚", "🌞", "🌚", "🌞"],  # sun, moon
        ["🔊", "🔇", "🔊", "🔇", "🔊"],  # loud, quiet
        ["🏃", "🧍", "🏃", "🧍", "🏃"],  # running, standing
    ])
    return items, "alternating pattern", "pattern"


def generate_letter_pattern():
    """Generate letter sequence pattern."""
    patterns = [
        ["A", "B", "C", "D", "E"],  # alphabet
        ["A", "B", "A", "B", "A"],  # AB pattern
        ["A", "A", "B", "B", "C"],  # AABB pattern
        ["X", "Y", "Z", "X", "Y"],  # XYZ pattern
    ]
    sequence = random.choice(patterns)
    return sequence, "letter pattern", "pattern"


def generate_rainbow_pattern():
    """Generate rainbow color sequence."""
    rainbow = ["🔴", "🟠", "🟡", "🟢", "🔵"]
    return rainbow, "rainbow colors", "pattern"


def generate_counting_fingers():
    """Generate finger counting pattern."""
    fingers = ["☝️", "✌️", "🤟", "🖖", "🖐️"]
    return fingers, "counting fingers", "pattern"


def generate_size_pattern():
    """Generate size progression pattern."""
    patterns = [
        ["🐜", "🐁", "🐱", "🐕", "🐘"],  # animals small to big
        [".", "•", "●", "⬤", "🔴"],  # dots small to big (using symbols)
        ["🌱", "🌿", "🌳", "🌲", "🏔️"],  # nature small to big
    ]
    sequence = random.choice(patterns)
    return sequence, "size pattern (small to big)", "pattern"


def generate_day_night_pattern():
    """Generate day/night cycle pattern."""
    patterns = [
        ["🌅", "☀️", "🌤️", "🌙", "⭐"],  # day cycle
        ["☀️", "🌙", "☀️", "🌙", "☀️"],  # day night alternating
    ]
    sequence = random.choice(patterns)
    return sequence, "day and night pattern", "pattern"


def generate_emotion_pattern():
    """Generate emotion sequence."""
    patterns = [
        ["😀", "😊", "😐", "😢", "😭"],  # happy to sad
        ["😴", "🥱", "😐", "😊", "😀"],  # sleepy to happy
        ["😀", "😢", "😀", "😢", "😀"],  # happy sad alternating
    ]
    sequence = random.choice(patterns)
    return sequence, "feelings pattern", "pattern"


def generate_arrow_pattern():
    """Generate arrow direction pattern."""
    patterns = [
        ["⬆️", "➡️", "⬇️", "⬅️", "⬆️"],  # rotating
        ["⬆️", "⬆️", "➡️", "➡️", "⬇️"],  # direction pairs
        ["⬅️", "⬆️", "➡️", "⬇️", "⬅️"],  # circular
    ]
    sequence = random.choice(patterns)
    return sequence, "arrow direction pattern", "pattern"


def generate_number_emoji_pattern():
    """Generate number emoji pattern."""
    sequence = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
    return sequence, "number emojis", "pattern"


def generate_question():
    """Generate a random sequence question (number or pattern)."""
    # Mix of number sequences and pattern sequences
    number_generators = [
        generate_simple_sequence,
        generate_simple_sequence,
        generate_skip_count_2,
        generate_skip_count_5,
        generate_skip_count_10,
        generate_countdown,
        generate_plus_2,
        generate_odd_numbers,
        generate_even_numbers,
        generate_countdown_2,
        generate_plus_3,
        generate_tens,
        generate_ones_to_five,
        generate_repeat_number,
    ]
    
    pattern_generators = [
        generate_ab_pattern,
        generate_ab_pattern,  # Weight AB more (easier)
        generate_abc_pattern,
        generate_aab_pattern,
        generate_abb_pattern,
        generate_abcd_pattern,
        generate_growing_pattern,
        generate_shrinking_pattern,
        generate_mirror_pattern,
        generate_color_size_pattern,
        generate_letter_pattern,
        generate_rainbow_pattern,
        generate_counting_fingers,
        generate_size_pattern,
        generate_day_night_pattern,
        generate_emotion_pattern,
        generate_arrow_pattern,
        generate_number_emoji_pattern,
    ]
    
    # 50% chance number, 50% chance pattern
    if random.random() < 0.5:
        generator = random.choice(number_generators)
    else:
        generator = random.choice(pattern_generators)
    
    sequence, pattern_type, seq_type = generator()
    
    # Choose which position to hide (not first or last for easier guessing)
    hide_pos = random.randint(1, 3)
    correct_answer = sequence[hide_pos]
    
    # Generate wrong answers based on sequence type
    wrong_answers = []
    
    if seq_type == "number":
        # For numbers, generate close wrong answers
        wrong_set = set()
        attempts = 0
        while len(wrong_set) < 3 and attempts < 20:
            offset = random.choice([-3, -2, -1, 1, 2, 3])
            wrong = correct_answer + offset
            if wrong > 0 and wrong != correct_answer and wrong not in wrong_set:
                wrong_set.add(wrong)
            attempts += 1
        
        while len(wrong_set) < 3:
            wrong = random.randint(max(1, correct_answer - 5), correct_answer + 5)
            if wrong != correct_answer and wrong not in wrong_set:
                wrong_set.add(wrong)
        wrong_answers = list(wrong_set)
    else:
        # For patterns, use other items from the sequence or category
        unique_items = list(set(sequence))
        
        # Get items from the same category for variety
        all_pattern_items = []
        for items in PATTERN_SETS.values():
            all_pattern_items.extend(items)
        
        wrong_set = set()
        # First, add other items from the sequence (but not the correct one)
        for item in unique_items:
            if item != correct_answer and len(wrong_set) < 3:
                wrong_set.add(item)
        
        # If we need more, add random items from pattern sets
        attempts = 0
        while len(wrong_set) < 3 and attempts < 20:
            wrong = random.choice(all_pattern_items)
            if wrong != correct_answer and wrong not in wrong_set:
                wrong_set.add(wrong)
            attempts += 1
        
        wrong_answers = list(wrong_set)
    
    # Create display sequence with blank
    display_seq = sequence.copy()
    display_seq[hide_pos] = "?"
    
    # Shuffle options
    options = [correct_answer] + wrong_answers[:3]
    random.shuffle(options)
    
    return {
        "sequence": display_seq,
        "correct_answer": correct_answer,
        "options": options,
        "pattern_type": pattern_type,
        "hide_pos": hide_pos,
        "seq_type": seq_type,
    }


def generate_questions(num_questions: int = 10):
    """Generate multiple sequence questions."""
    questions = []
    for _ in range(num_questions):
        questions.append(generate_question())
    return questions


def render_sequence(sequence: list, highlight_pos: int = -1, highlight_color: str = "green"):
    """Render a sequence as colorful boxes."""
    cols = st.columns(len(sequence))
    for i, item in enumerate(sequence):
        with cols[i]:
            if item == "?":
                # Question mark box
                st.markdown(
                    """
                    <div style="
                        background-color: #FFD700;
                        color: #333;
                        font-size: 28px;
                        font-weight: bold;
                        text-align: center;
                        padding: 20px 15px;
                        border-radius: 10px;
                        margin: 5px;
                        border: 3px dashed #FFA500;
                    ">?</div>
                    """,
                    unsafe_allow_html=True
                )
            elif i == highlight_pos:
                # Highlighted answer
                bg = "#90EE90" if highlight_color == "green" else "#FFB6C1"
                st.markdown(
                    f"""
                    <div style="
                        background-color: {bg};
                        color: #333;
                        font-size: 28px;
                        font-weight: bold;
                        text-align: center;
                        padding: 20px 15px;
                        border-radius: 10px;
                        margin: 5px;
                        border: 3px solid #333;
                    ">{item}</div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                # Regular number box
                colors = ["#87CEEB", "#98FB98", "#DDA0DD", "#F0E68C", "#FFB6C1"]
                bg_color = colors[i % len(colors)]
                st.markdown(
                    f"""
                    <div style="
                        background-color: {bg_color};
                        color: #333;
                        font-size: 28px;
                        font-weight: bold;
                        text-align: center;
                        padding: 20px 15px;
                        border-radius: 10px;
                        margin: 5px;
                        border: 2px solid #666;
                    ">{item}</div>
                    """,
                    unsafe_allow_html=True
                )


def init_session_state():
    """Initialize session state variables."""
    if "seq_questions" not in st.session_state:
        st.session_state["seq_questions"] = []
    if "seq_current" not in st.session_state:
        st.session_state["seq_current"] = 0
    if "seq_score" not in st.session_state:
        st.session_state["seq_score"] = 0
    if "seq_answered" not in st.session_state:
        st.session_state["seq_answered"] = False
    if "seq_selected" not in st.session_state:
        st.session_state["seq_selected"] = None
    if "seq_game_over" not in st.session_state:
        st.session_state["seq_game_over"] = False


def reset_game():
    """Reset the game state."""
    st.session_state["seq_questions"] = []
    st.session_state["seq_current"] = 0
    st.session_state["seq_score"] = 0
    st.session_state["seq_answered"] = False
    st.session_state["seq_selected"] = None
    st.session_state["seq_game_over"] = False


def main():
    st.set_page_config(page_title="Sequence Game")
    
    # Fun CSS for kids
    st.markdown("""
        <style>
        .stButton > button {
            font-size: 20px !important;
            padding: 15px 30px !important;
            border-radius: 15px !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("🔢 Patterns & Sequences!")
    st.caption("Find the missing item in the pattern!")
    
    init_session_state()
    
    questions = st.session_state["seq_questions"]
    current = st.session_state["seq_current"]
    score = st.session_state["seq_score"]
    answered = st.session_state["seq_answered"]
    game_over = st.session_state["seq_game_over"]
    
    # Start screen
    if not questions:
        st.markdown("### 👋 Welcome!")
        st.markdown("""
        **How to play:**
        1. Look at the pattern (numbers or shapes!)
        2. Find the missing item (the **?**)
        3. Click on the correct answer!
        
        **Pattern types:**
        - 🔢 Number patterns: 1, 2, 3, ?, 5
        - 🔴🔵 Shape patterns: 🔴, 🔵, 🔴, ?, 🔴
        - 🍎🍊 Emoji patterns: 🍎, 🍊, 🍎, ?, 🍎
        
        Ready to find some patterns? 🎯
        """)
        
        if st.button("🎮 Start Game!", use_container_width=True):
            st.session_state["seq_questions"] = generate_questions(10)
            st.rerun()
        return
    
    # Game over screen
    if game_over:
        st.markdown("---")
        st.markdown(f"## 🎉 Great Job!")
        st.markdown(f"### Your Score: **{score}/10**")
        
        if score == 10:
            st.success("🌟 PERFECT SCORE! You're a sequence superstar! 🌟")
            st.balloons()
        elif score >= 8:
            st.success("🎯 Excellent work! You're really good at patterns!")
        elif score >= 6:
            st.info("👍 Good job! Keep practicing!")
        else:
            st.info("💪 Nice try! Practice makes perfect!")
        
        if st.button("🔄 Play Again!", use_container_width=True):
            reset_game()
            st.rerun()
        return
    
    # Current question
    q = questions[current]
    
    # Progress bar
    st.progress((current) / 10)
    st.markdown(f"**Question {current + 1} of 10** | Score: {score} ⭐")
    
    st.markdown("---")
    st.markdown("### What goes in place of the **?**")
    
    # Display the sequence
    if answered:
        # Show with the answer highlighted
        full_seq = q["sequence"].copy()
        full_seq[q["hide_pos"]] = q["correct_answer"]
        is_correct = st.session_state["seq_selected"] == q["correct_answer"]
        render_sequence(full_seq, q["hide_pos"], "green" if is_correct else "red")
    else:
        render_sequence(q["sequence"])
    
    st.markdown("---")
    
    # Answer options
    if not answered:
        st.markdown("### Choose the missing number:")
        cols = st.columns(4)
        for i, option in enumerate(q["options"]):
            with cols[i]:
                if st.button(
                    f"**{option}**",
                    key=f"opt_{current}_{i}",
                    use_container_width=True
                ):
                    st.session_state["seq_selected"] = option
                    st.session_state["seq_answered"] = True
                    if option == q["correct_answer"]:
                        st.session_state["seq_score"] += 1
                    st.rerun()
    else:
        # Show result
        is_correct = st.session_state["seq_selected"] == q["correct_answer"]
        if is_correct:
            st.success(f"✅ Correct! Great job! The answer is **{q['correct_answer']}**")
        else:
            st.error(f"❌ Not quite! The answer is **{q['correct_answer']}**")
            st.info(f"💡 Hint: This pattern was {q['pattern_type']}")
        
        # Next button
        if current < 9:
            if st.button("➡️ Next Question", use_container_width=True):
                st.session_state["seq_current"] += 1
                st.session_state["seq_answered"] = False
                st.session_state["seq_selected"] = None
                st.rerun()
        else:
            if st.button("🏁 See Results!", use_container_width=True):
                st.session_state["seq_game_over"] = True
                st.rerun()
    
    # Bottom controls
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Score", f"{score} ⭐")
    with col2:
        if st.button("🔄 Restart"):
            reset_game()
            st.rerun()


if __name__ == "__main__":
    main()
