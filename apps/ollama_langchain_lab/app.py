import streamlit as st
import ollama  # native client (kept for listing)
from langchain_community.chat_models import ChatOllama  # LangChain wrapper
from langchain.schema import HumanMessage, SystemMessage

st.set_page_config(page_title="Ollama + LangChain Lab", page_icon="🤖", layout="wide")

# --- Model list helpers ---
def get_installed_models():
    try:
        data = ollama.list()  # {'models': [ { 'name': 'llama3:latest', ...}, ... ]}
        #return [m["name"] for m in data.get("models", [])]
    except Exception as e:
        st.error(f"Could not list models. Is ollama running? Error: {e}")
        return []

AVAILABLE_MODELS = get_installed_models() or [
    "llama3.2:3b", "llama3:latest", "mistral:latest", "phi3:mini", "gemma:latest"
]
DEFAULT_MODEL = "llama3.2:3b" if "llama3.2:3b" in AVAILABLE_MODELS else AVAILABLE_MODELS[0]

st.title("🤖 Simple Ollama + LangChain Lab")

# model picker
model = st.selectbox("Select model", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(DEFAULT_MODEL))

# session state
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {"user": ..., "bot": ...}

# --- Basic chat area (LLM only) ---
st.subheader("1) LLM — Basic Chat")
user_input = st.text_input("💬 You:", key="user_input")

if st.button("Send", key="send_btn") and user_input:
    # (A) LangChain Chat interface
    chat = ChatOllama(model=model)  # uses http://localhost:11434
    messages = []

    # optional: a system prompt (style/behavior)
    messages.append(SystemMessage(content="You are a helpful, concise assistant."))

    # include prior context (simple, we’ll do real memory later)
    for pair in st.session_state.history:
        messages.append(HumanMessage(content=pair["user"]))
        messages.append(SystemMessage(content=pair["bot"]))  # not ideal role, but okay for demo

    messages.append(HumanMessage(content=user_input))

    try:
        with st.spinner("🤖 Thinking..."):
            resp = chat.invoke()
            bot_message = resp.content
        st.session_state.history.append({"user": user_input, "bot": bot_message})
        st.text_area("🤖 Bot:", bot_message, height=140)
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.info("Tip: Make sure Ollama is running:  `ollama serve`")

st.markdown("---")
st.header("Chat History")
for chat in st.session_state.history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")
