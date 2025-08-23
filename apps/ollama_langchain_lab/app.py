import streamlit as st
from langchain_community.chat_models import ChatOllama  # LangChain wrapper
from langchain.schema import HumanMessage, SystemMessage

st.set_page_config(page_title="Ollama + LangChain Lab", page_icon="🤖", layout="wide")

# Static list of available models (removed dynamic listing due to error)
AVAILABLE_MODELS = [
    "llama3.2:3b", "llama3:latest", "mistral:latest", "phi3:mini", "gemma:latest"
]
DEFAULT_MODEL = "llama3.2:3b" if "llama3.2:3b" in AVAILABLE_MODELS else AVAILABLE_MODELS[0]

st.title("🤖 Simple Ollama + LangChain Lab")

# Model picker
model = st.selectbox("Select model", AVAILABLE_MODELS, index=AVAILABLE_MODELS.index(DEFAULT_MODEL))

# Session state for chat history
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {"user": ..., "bot": ...}

# Basic chat area (LLM only)
st.subheader("1) LLM — Basic Chat")
user_input = st.text_input("💬 You:", key="user_input")

if st.button("Send", key="send_btn") and user_input:
    chat = ChatOllama(model=model)  # uses http://localhost:11434

    # Compose prompt text combining system prompt, history, and current input
    prompt_text = "You are a helpful, concise assistant.\n"
    for pair in st.session_state.history:
        prompt_text += f"User: {pair['user']}\nBot: {pair['bot']}\n"
    prompt_text += f"User: {user_input}\nBot:"

    try:
        with st.spinner("🤖 Thinking..."):
            # Pass the prompt text as input argument for invoke()
            resp = chat.invoke(input=prompt_text)
            bot_message = resp.content
        st.session_state.history.append({"user": user_input, "bot": bot_message})
        st.text_area("🤖 Bot:", bot_message, height=140)
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.info("Tip: Make sure Ollama is running: `ollama serve`")

st.markdown("---")
st.header("Chat History")
for chat in st.session_state.history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")
