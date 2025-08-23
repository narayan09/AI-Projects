# Ollama + LangChain Chat Application

A simple and powerful demo application that integrates **Ollama** large language models with **LangChain** framework wrapped in a clean **Streamlit** interface. This project demonstrates how to leverage Ollama’s local LLM serving capabilities combined with LangChain’s chat abstraction to build an interactive chat assistant.

---
![Chat Interface](images/demo.JPG)

## Features

- **Interactive Chat UI:** Simple and intuitive chat interface using Streamlit.
- **Ollama Integration:** Sends prompts to Ollama server (`ollama serve`) running locally for fast and private LLM inference.
- **LangChain Wrapper:** Utilizes the `ChatOllama` LangChain community model wrapper to manage conversation flow.
- **Session Memory:** Preserves chat history during the session allowing contextual conversations.
- **Static Model Selection:** Choose from multiple pre-defined Ollama models for flexible experimentation.
- **System Prompts:** Customizable assistant behavior with system-level instructions.

---

## Getting Started

### Prerequisites

- Windows, macOS, or Linux environment
- [Ollama](https://ollama.com) installed and running locally (`ollama serve`)
- Python 3.8+
- Recommended: Virtual environment for package management

### Installation

1. Clone the repository:
https://github.com/narayan09/AI-Projects.git

    cd ollama-langchain-chat

2. Install dependencies:

    pip install streamlit langchain langchain-community ollama

3. Start Ollama server in the background:
    ollama serve
4. Run the Streamlit app:
    streamlit run app.py


---

## Usage

- Select your preferred language model from the dropdown list.
- Enter your message in the input box and press **Send**.
- See the bot’s response appear below, maintaining conversation context.
- View full chat history at the bottom of the interface.

---

## Code Overview

- `app.py`: Main Streamlit app integrating Ollama and LangChain.
- Uses `ChatOllama.invoke(input=prompt_text)` to send chat prompts.
- Chat history is stored in `st.session_state.history` for session persistence.
- Static list of Ollama models to avoid runtime listing issues.

---

## Troubleshooting

- **Port in Use Error:** If you see `Only one usage of each socket address ...` when running Ollama, check for existing processes using port 11434:

    netstat -ano | findstr :11434
    Then terminate with:
    taskkill /PID <pid> /F

- Make sure Ollama server is running (`ollama serve`) before starting the app.
- Verify network connectivity on localhost to port 11434.

---

## Technologies Used

- [Ollama](https://ollama.com) — Local language model serving
- [LangChain](https://python.langchain.com/en/latest/) — Language model framework
- [Streamlit](https://streamlit.io) — Web app UI framework
- Python 3


## Acknowledgments

- Inspired by the capabilities of Ollama and LangChain community models.
- Thanks to Streamlit for seamless UI creation.

