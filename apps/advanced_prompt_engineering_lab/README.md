# 🚀 Advanced Prompt Engineering Lab

A comprehensive Streamlit application that demonstrates 8 essential prompt engineering techniques with practical examples and RAG implementation.

## ✨ Features

### 🎯 Prompt Engineering Techniques
- **Zero-Shot Prompting**: Direct task requests without examples
- **One-Shot Prompting**: Single example to guide responses  
- **Few-Shot Prompting**: Multiple examples for pattern learning
- **Chain-of-Thought (CoT)**: Step-by-step reasoning for complex problems
- **Zero-Shot CoT**: Reasoning without examples
- **Role Prompting**: Assign expertise/persona to AI
- **Template Prompting**: Structured output formats
- **System Prompting**: Behavioral guidelines

### 🤖 RAG Implementation
- Upload PDF and text documents
- Vector storage with ChromaDB
- Multiple RAG prompt strategies
- Document retrieval and generation

### 🔬 Interactive Features
- Live prompt testing with Ollama models
- Example exploration and modification
- Chat history and session tracking
- Comprehensive documentation

## 🛠️ Setup Instructions

### Prerequisites
1. Install Ollama:
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. Pull models:
   ```bash
   ollama pull llama3.2:3b
   ollama pull mistral:latest
   ollama serve
   ```

### Installation
1. Clone or download the project files
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   streamlit run advanced_prompt_engineering_lab.py
   ```

## 🎯 Usage Guide

1. **Home**: Overview and quick start guide
2. **Technique Explorer**: Learn about each prompting technique
3. **Interactive Lab**: Test prompts with live models
4. **RAG Implementation**: Upload documents and chat with them
5. **Documentation**: Best practices and troubleshooting

## 🔧 Troubleshooting

- **Connection Error**: Make sure Ollama is running (`ollama serve`)
- **Model Not Found**: Pull the model first (`ollama pull model_name`)
- **Memory Issues**: Use smaller models like `phi3:mini`
- **Slow Responses**: Try smaller chunk sizes in RAG settings

## 📚 Learning Resources

### When to Use Each Technique

| Technique | Best For | Avoid When |
|-----------|----------|------------|
| Zero-Shot | Simple, general tasks | Complex reasoning needed |
| One-Shot | Quick format examples | Multiple patterns needed |
| Few-Shot | Pattern learning | Simple, direct questions |
| Chain-of-Thought | Math, logic, analysis | Simple factual queries |
| Zero-Shot CoT | No examples available | Have good examples |
| Role Prompting | Expert knowledge needed | Generic responses OK |
| Template | Structured outputs | Creative, open-ended tasks |
| System | Consistent behavior | One-off interactions |

### Combining Techniques
- **Few-Shot + CoT**: Best for complex reasoning with examples
- **Role + Template**: Structured expert responses  
- **System + CoT**: Consistent reasoning behavior
- **RAG + Any Technique**: Knowledge-grounded responses

## 🚀 Built With
- **Streamlit**: Interactive web interface
- **LangChain**: LLM framework and document processing
- **Ollama**: Local LLM inference
- **ChromaDB**: Vector database for RAG
- **PyPDF**: PDF document processing

Happy Prompting! 🤖
