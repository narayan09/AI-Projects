# Project 1 - 
## Ollama + LangChain Chat Application

A simple and powerful demo application that integrates **Ollama** large language models with **LangChain** framework wrapped in a clean **Streamlit** interface. This project demonstrates how to leverage Ollama’s local LLM serving capabilities combined with LangChain’s chat abstraction to build an interactive chat assistant.

---
![Chat Interface](Images/demo.JPG)

## Features

- **Interactive Chat UI:** Simple and intuitive chat interface using Streamlit.
- **Ollama Integration:** Sends prompts to Ollama server (`ollama serve`) running locally for fast and private LLM inference.
- **LangChain Wrapper:** Utilizes the `ChatOllama` LangChain community model wrapper to manage conversation flow.
- **Session Memory:** Preserves chat history during the session allowing contextual conversations.
- **Static Model Selection:** Choose from multiple pre-defined Ollama models for flexible experimentation.
- **System Prompts:** Customizable assistant behavior with system-level instructions.
