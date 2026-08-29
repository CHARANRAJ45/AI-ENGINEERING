from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

import streamlit as st

load_dotenv()

model = ChatMistralAI(
    model="mistral-small-2506",
    temperature=1,
    max_tokens=1000
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Agent",
    page_icon="✨",
    layout="centered"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Main background */

.stApp {
    background: #faf9ff;
}


/* Remove default top padding */

.block-container {
    padding-top: 2rem;
    max-width: 850px;
}


/* Header */

.header {
    text-align: center;
    margin-bottom: 25px;
}

.header h1 {
    font-size: 38px;
    font-weight: 700;
    color: #5b21b6;
    margin-bottom: 5px;
}

.header p {
    color: #6b7280;
    font-size: 15px;
}


/* Agent selector */

.selector-box {
    background: white;
    padding: 18px 22px;
    border-radius: 16px;

    border: 1px solid #e9d5ff;

    box-shadow:
        0 4px 15px rgba(91, 33, 182, 0.06);

    margin-bottom: 20px;
}


/* Mode cards */

.mode-title {
    color: #4c1d95;
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 8px;
}


/* Chat container */

.chat-container {
    background: white;

    border: 1px solid #ede9fe;

    border-radius: 18px;

    padding: 10px;

    box-shadow:
        0 5px 20px rgba(91, 33, 182, 0.06);
}


/* Streamlit chat messages */

[data-testid="stChatMessage"] {
    border-radius: 14px;
    margin-bottom: 8px;
}


/* User message */

[data-testid="stChatMessage"]:has(
    [data-testid="chatAvatarIcon-user"]
) {
    background: #f5f3ff;
}


/* Assistant message */

[data-testid="stChatMessage"]:has(
    [data-testid="chatAvatarIcon-assistant"]
) {
    background: #ffffff;
    border: 1px solid #f3e8ff;
}


/* Chat input */

[data-testid="stChatInput"] {
    border-color: #ddd6fe !important;
}


/* Buttons */

.stButton > button {
    border-radius: 10px;

    border: 1px solid #ddd6fe;

    background: white;

    color: #6d28d9;

    font-weight: 600;

    transition: 0.2s;
}

.stButton > button:hover {
    background: #f5f3ff;
    border-color: #8b5cf6;
    color: #5b21b6;
}


/* Select box */

div[data-baseweb="select"] > div {
    border-radius: 10px !important;
    border-color: #ddd6fe !important;
    background: white !important;
}


/* Footer */

.footer {
    text-align: center;
    color: #9ca3af;
    font-size: 12px;
    margin-top: 25px;
}


/* Hide unnecessary Streamlit elements */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "mode" not in st.session_state:

    st.session_state.mode = None


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="header">

    <h1>✨ AI Agent</h1>

    <p>
        Choose an agent personality and start a conversation
    </p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# AGENT SELECTION
# ============================================================

st.markdown(
    '<div class="selector-box">',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="mode-title">Choose your response style</div>',
    unsafe_allow_html=True
)

choice = st.selectbox(
    "Agent Mode",
    [
        "😂 Funny Roast Agent",
        "🧠 Serious Agent",
        "🤝 Helpful Agent"
    ],
    label_visibility="collapsed"
)

st.markdown(
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# MODE LOGIC
# ============================================================

if choice == "😂 Funny Roast Agent":

    mode = "funny the roast agent based on user input"

elif choice == "🧠 Serious Agent":

    mode = "serious agent"

else:

    mode = "helpful agent"


# ============================================================
# RESET CHAT WHEN MODE CHANGES
# ============================================================

if st.session_state.mode != mode:

    st.session_state.mode = mode

    st.session_state.messages = [
        SystemMessage(
            content=f"You are a {mode}"
        )
    ]


# ============================================================
# CURRENT MODE
# ============================================================

st.markdown(
    f"""
    <div style="
        text-align:center;
        margin-bottom:15px;
        color:#6d28d9;
        font-size:13px;
        font-weight:600;
    ">
        ● {choice}
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    if isinstance(message, HumanMessage):

        with st.chat_message(
            "user",
            avatar="👤"
        ):

            st.write(message.content)


    elif isinstance(message, AIMessage):

        with st.chat_message(
            "assistant",
            avatar="✨"
        ):

            st.write(message.content)


# ============================================================
# USER INPUT
# ============================================================

user_input = st.chat_input(
    "Message your AI agent..."
)


if user_input:

    # Add user message

    st.session_state.messages.append(
        HumanMessage(content=user_input)
    )


    # Display user message

    with st.chat_message(
        "user",
        avatar="👤"
    ):

        st.write(user_input)


    # Generate response

    with st.chat_message(
        "assistant",
        avatar="✨"
    ):

        with st.spinner("Thinking..."):

            response = model.invoke(
                st.session_state.messages
            )

        st.write(response.content)


    # Save AI response

    st.session_state.messages.append(
        AIMessage(
            content=response.content
        )
    )


# ============================================================
# CLEAR BUTTON
# ============================================================

st.markdown("")

col1, col2, col3 = st.columns([1, 1, 1])

with col2:

    if st.button(
        "↻ Clear Chat",
        use_container_width=True
    ):

        st.session_state.messages = [
            SystemMessage(
                content=f"You are a {mode}"
            )
        ]

        st.rerun()


# ============================================================
# FOOTER
# ============================================================

st.markdown("""
<div class="footer">
    Powered by Mistral AI · LangChain · Streamlit
</div>
""", unsafe_allow_html=True)