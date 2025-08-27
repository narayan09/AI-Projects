import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import tempfile
import os
import time

# Page Configuration
st.set_page_config(
    page_title="🚀 Advanced Prompt Engineering Lab", 
    page_icon="🤖", 
    layout="wide"
)

# Available Models
AVAILABLE_MODELS = [
    "llama3.2:3b", "llama3:latest", "mistral:latest", 
    "phi3:mini", "gemma:latest", "codellama:latest"
]

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_history" not in st.session_state:
    st.session_state.rag_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Prompt Engineering Techniques Database
PROMPT_TECHNIQUES = {
    "Zero-Shot Prompting": {
        "description": "Ask the model to perform a task without any examples",
        "examples": [
            "Classify the sentiment: 'I love this new smartphone!'",
            "Translate to French: 'Hello, how are you today?'",
            "Write a Python function to calculate factorial"
        ]
    },
    "One-Shot Prompting": {
        "description": "Provide one example to guide the model's response",
        "examples": [
            "Example: Text: 'boring movie' → Sentiment: Negative\nNow: Text: 'great book' → Sentiment:",
            "Example: Task: Add numbers → Code: def add(a,b): return a+b\nNow: Task: Multiply → Code:"
        ]
    },
    "Few-Shot Prompting": {
        "description": "Provide multiple examples to establish a pattern",
        "examples": [
            "Q: 10 apples - 3 = ?\nA: 10-3=7 apples\nQ: 2 packs × 52 cards = ?\nA: 2×52=104 cards\nQ: 25 books - 8 + 15 = ?\nA:"
        ]
    },
    "Chain-of-Thought": {
        "description": "Encourage step-by-step reasoning",
        "examples": [
            "450 students: 60% elementary, 25% middle, rest high school. How many high schoolers? Let's think step by step.",
            "All programmers drink coffee. Some coffee drinkers work late. Do some programmers work late? Let's think step by step."
        ]
    },
    "Zero-Shot CoT": {
        "description": "Add reasoning without examples",
        "examples": [
            "Train travels 180 miles in 3 hours. How long for 300 miles? Let's think step by step."
        ]
    },
    "Role Prompting": {
        "description": "Assign expertise to the AI",
        "examples": [
            "You are a senior Python developer. Explain lists vs tuples to a beginner.",
            "You are a creative writing teacher. Improve: 'The dog ran fast.'"
        ]
    },
    "Template Prompting": {
        "description": "Use structured formats",
        "examples": [
            "Review: 'Great laptop! Fast but poor battery.'\nPros: [list]\nCons: [list]\nRating: [1-5]"
        ]
    },
    "System Prompting": {
        "description": "Set behavioral guidelines",
        "examples": [
            "SYSTEM: You are helpful customer service. Be polite and solution-focused.\nUSER: My order is 2 weeks late!"
        ]
    }
}

# Sidebar
st.sidebar.title("🎯 Navigation")
page = st.sidebar.selectbox(
    "Choose section:",
    ["🏠 Home", "📚 Techniques", "🔬 Lab", "🤖 RAG", "📖 Docs"]
)

st.sidebar.title("⚙️ Settings")
selected_model = st.sidebar.selectbox("Model:", AVAILABLE_MODELS)

# Main Content
if page == "🏠 Home":
    st.title("🚀 Advanced Prompt Engineering Lab")

    st.markdown("""
    ## Welcome to the Complete Prompt Engineering Toolkit!

    This app demonstrates **8 essential prompt engineering techniques** with practical examples.

    ### 🎯 What You'll Learn:
    - Zero-Shot, One-Shot, Few-Shot Prompting
    - Chain-of-Thought reasoning techniques  
    - Role and Template prompting
    - System prompting for consistency
    - RAG implementation with documents

    ### 🚀 Quick Start:
    1. Explore techniques in **Techniques** tab
    2. Test prompts in **Lab** tab
    3. Try RAG with documents in **RAG** tab
    """)

elif page == "📚 Techniques":
    st.title("📚 Prompt Engineering Techniques")

    technique = st.selectbox("Select technique:", list(PROMPT_TECHNIQUES.keys()))

    st.markdown(f"## {technique}")
    st.markdown(f"**Description:** {PROMPT_TECHNIQUES[technique]['description']}")

    st.markdown("### Examples:")
    for i, example in enumerate(PROMPT_TECHNIQUES[technique]['examples']):
        st.code(example, language="text")

elif page == "🔬 Lab":
    st.title("🔬 Interactive Prompt Lab")

    technique = st.selectbox("Choose technique:", list(PROMPT_TECHNIQUES.keys()))

    prompt_input = st.text_area(
        "Your Prompt:",
        value=PROMPT_TECHNIQUES[technique]['examples'][0],
        height=150
    )

    if st.button("🚀 Test Prompt"):
        if prompt_input:
            try:
                with st.spinner("Generating..."):
                    chat = ChatOllama(model=selected_model)
                    start_time = time.time()
                    response = chat.invoke(input=prompt_input)
                    end_time = time.time()
                    response_time = end_time - start_time
                    st.markdown("### Response:")
                    st.markdown(response.content)
                    st.info(f"⏱️ Response time: {response_time:.2f} seconds")
                    st.session_state.chat_history.append({
                        "technique": technique,
                        "prompt": prompt_input,
                        "response": response.content,
                        "response_time": response_time
                    })

            except Exception as e:
                st.error(f"Error: {e}")
                st.info("Make sure Ollama is running: `ollama serve`")

elif page == "🤖 RAG":
    st.title("🤖 RAG Implementation")

    st.markdown("Upload documents and chat with them using RAG!")

    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=['pdf', 'txt'],
        accept_multiple_files=True
    )

    if uploaded_files and st.button("Process Documents"):
        with st.spinner("Processing..."):
            try:
                documents = []

                for file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp:
                        tmp.write(file.getvalue())
                        tmp_path = tmp.name

                    if file.name.endswith('.pdf'):
                        loader = PyPDFLoader(tmp_path)
                    else:
                        loader = TextLoader(tmp_path)

                    documents.extend(loader.load())

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = splitter.split_documents(documents)

                embeddings = OllamaEmbeddings(model=selected_model)
                vector_store = Chroma.from_documents(splits, embeddings)
                st.session_state.vector_store = vector_store

                st.success(f"Processed {len(splits)} chunks from {len(uploaded_files)} files!")

            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.vector_store:
        st.markdown("### Chat with Documents")

        rag_technique = st.selectbox(
            "RAG Technique:",
            ["Standard RAG", "CoT RAG", "Role-based RAG", "Template RAG"]
        )

        question = st.text_input("Ask about your documents:")

        if st.button("Search & Generate") and question:
            try:
                with st.spinner("Searching..."):
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 3})
                    docs = retriever.get_relevant_documents(question)

                    context = "\n\n".join([doc.page_content for doc in docs])

                    if rag_technique == "Standard RAG":
                        prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
                    elif rag_technique == "CoT RAG":
                        prompt = f"Context: {context}\n\nQuestion: {question}\n\nLet's think step by step:\nAnswer:"
                    elif rag_technique == "Role-based RAG":
                        prompt = f"You are a research assistant. Context: {context}\n\nQuestion: {question}\n\nAnswer:"
                    else:  # Template RAG
                        prompt = f"Context: {context}\n\nQuestion: {question}\n\n**Key Info:** [from context]\n**Answer:** [direct answer]"

                    chat = ChatOllama(model=selected_model)
                    start_time = time.time()
                    response = chat.invoke(input=prompt)
                    end_time = time.time()
                    response_time = end_time - start_time
                    st.markdown("### Answer:")
                    st.markdown(response.content)
                    st.info(f"⏱️ Response time: {response_time:.2f} seconds")
                    with st.expander("Source Documents"):
                        for i, doc in enumerate(docs):
                            st.markdown(f"**Source {i+1}:**")
                            st.markdown(doc.page_content[:500] + "...")

            except Exception as e:
                st.error(f"Error: {e}")

elif page == "📖 Docs":
    st.title("📖 Documentation")

    st.markdown("""
    ## Setup Guide

    ### 1. Install Ollama
    ```bash
    curl -fsSL https://ollama.ai/install.sh | sh
    ollama pull llama3.2:3b
    ollama serve
    ```

    ### 2. Install Dependencies
    ```bash
    pip install -r requirements.txt
    ```

    ### 3. Run Application
    ```bash
    streamlit run advanced_prompt_engineering_lab.py
    ```

    ## Best Practices

    ### When to Use Each Technique
    - **Zero-Shot**: Simple, direct tasks
    - **Few-Shot**: Pattern learning, complex formatting
    - **Chain-of-Thought**: Mathematical reasoning, logic
    - **Role Prompting**: Expert knowledge needed
    - **Template**: Structured outputs
    - **RAG**: Document-based questions

    ### Tips
    - Start simple, add complexity as needed
    - Use clear, specific instructions
    - Provide examples for complex tasks
    - Test and iterate on prompts
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(f"Tests: {len(st.session_state.chat_history)}")
st.sidebar.markdown("Built with Streamlit + LangChain + Ollama")
