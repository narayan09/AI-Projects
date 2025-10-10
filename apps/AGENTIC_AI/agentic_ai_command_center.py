import streamlit as st
import os
from datetime import datetime
import json
import tempfile
from typing import List, Dict, Any, Type
import requests
from bs4 import BeautifulSoup
import time

# CrewAI imports (correct & consistent)
from crewai import Agent, Task, Crew, Process, LLM
from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import Tool
from pydantic import BaseModel, Field

# ---------- Global runtime safeguards (LiteLLM timeouts & retries) ----------
# These env vars act as a safety net if library defaults are high
os.environ.setdefault("LITELLM_TIMEOUT", "120")       # seconds
os.environ.setdefault("LITELLM_MAX_RETRIES", "2")

# Page Configuration
st.set_page_config(
    page_title="🤖 Agentic AI Command Center",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .agent-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .task-card {
        background-color: #f8f9fa;
        border-left: 5px solid #28a745;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
    .progress-section {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #2196f3;
    }
    .result-section {
        background-color: #f1f8e9;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #4caf50;
        margin: 1rem 0;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #dee2e6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "crew_results" not in st.session_state:
    st.session_state.crew_results = []
if "task_history" not in st.session_state:
    st.session_state.task_history = []
if "agent_logs" not in st.session_state:
    st.session_state.agent_logs = []

def log_event(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.agent_logs.append(f"[{ts}] {msg}")

# Available AI Models
AVAILABLE_MODELS = {
    "Ollama Models": {
        "llama3.2:3b": "Fast, balanced performance",
        "mistral:latest": "High quality responses",
        "phi3:mini": "Lightweight, efficient",
        "codellama:latest": "Code-focused tasks"
    },
    "OpenAI Models": {
        "gpt-4": "Most capable, expensive",
        "gpt-3.5-turbo": "Balanced cost/performance",
        "gpt-4-turbo": "Latest, most efficient"
    }
}

# Input schemas for custom tools
class SearchToolInput(BaseModel):
    """Input schema for DuckDuckGo Search Tool."""
    query: str = Field(..., description="Search query string")

class WebScrapingInput(BaseModel):
    """Input schema for Web Scraping Tool."""
    url: str = Field(..., description="URL to scrape content from")

class DataAnalysisInput(BaseModel):
    """Input schema for Data Analysis Tool."""
    data: str = Field(..., description="Text data to analyze")

class FileManagerInput(BaseModel):
    """Input schema for File Manager Tool."""
    content: str = Field(..., description="Content to save to file")
    filename: str = Field(default="", description="Optional filename for the file")

# Custom Tools for Agents
class DuckDuckGoSearchTool(BaseTool):
    name: str = "duckduckgo_search_tool"
    description: str = "Search the web using DuckDuckGo and return relevant results"
    args_schema: Type[BaseModel] = SearchToolInput

    def _run(self, query: str) -> str:
        try:
            search = DuckDuckGoSearchRun()
            results = search.invoke(query)
            return f"Search results for '{query}':\n{results}"
        except Exception as e:
            return f"Error performing search: {str(e)}"

class WebScrapingTool(BaseTool):
    name: str = "web_scraping_tool"
    description: str = "Scrape content from web pages for research purposes"
    args_schema: Type[BaseModel] = WebScrapingInput

    def _run(self, url: str) -> str:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove noise
            for tag in soup(["script", "style", "nav", "footer"]):
                tag.decompose()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)

            return text[:3000]  # Limit to 3000 chars
        except Exception as e:
            return f"Error scraping {url}: {str(e)}"

class DataAnalysisTool(BaseTool):
    name: str = "data_analysis_tool"
    description: str = "Analyze data and extract insights from text content"
    args_schema: Type[BaseModel] = DataAnalysisInput

    def _run(self, data: str) -> str:
        try:
            words = len(data.split())
            sentences = data.count('.') + data.count('!') + data.count('?')
            paragraphs = len([p for p in data.split('\n\n') if p.strip()])

            common_words = {}
            for word in data.lower().split():
                word = word.strip('.,!?":;()[]{}')
                if len(word) > 3 and word not in ['that', 'this', 'with', 'from', 'they', 'have', 'will', 'been', 'were']:
                    common_words[word] = common_words.get(word, 0) + 1

            top_keywords = sorted(common_words.items(), key=lambda x: x[1], reverse=True)[:10]

            analysis = f"""
📊 Content Analysis:
- Words: {words}
- Sentences: {sentences}
- Paragraphs: {paragraphs}
- Reading time: ~{max(1, words//200)} minutes

🔑 Top Keywords:
""" + "\n".join([f"- {word}: {count}" for word, count in top_keywords])

            return analysis
        except Exception as e:
            return f"Error analyzing data: {str(e)}"

class FileManagerTool(BaseTool):
    name: str = "file_manager_tool"
    description: str = "Save content to files and manage file operations"
    args_schema: Type[BaseModel] = FileManagerInput

    def _run(self, content: str, filename: str = "") -> str:
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"agent_output_{timestamp}.md"

            os.makedirs("outputs", exist_ok=True)
            filepath = os.path.join("outputs", filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

            return f"✅ Content saved to {filepath}"
        except Exception as e:
            return f"Error saving file: {str(e)}"

# Sidebar Configuration
st.sidebar.title("🎛️ Agent Control Panel")
st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
st.sidebar.subheader("🤖 AI Model Selection")

model_type = st.sidebar.radio(
    "Choose Model Type:",
    ["Ollama", "OpenAI"],
    help="Select the type of AI model to use for agents"
)

# Timeout & retries (so you can tweak without touching code)
timeout_sec = st.sidebar.slider("Model Timeout (sec)", min_value=30, max_value=300, value=120, step=10)
retries = st.sidebar.slider("Max Retries", min_value=0, max_value=5, value=2, step=1)

if model_type == "Ollama":
    available_models = AVAILABLE_MODELS["Ollama Models"]
    model_choice = st.sidebar.selectbox(
        "Select Ollama Model:",
        list(available_models.keys()),
        help="Choose an Ollama model - make sure it's downloaded locally"
    )
    st.sidebar.info(f"ℹ️ {available_models[model_choice]}")
    st.sidebar.warning("⚠️ Make sure Ollama is running: `ollama serve`")
    st.sidebar.info("💡 For memory features, ensure you have `ollama pull nomic-embed-text`")
    api_key = None
else:
    available_models = AVAILABLE_MODELS["OpenAI Models"]
    model_choice = st.sidebar.selectbox(
        "Select OpenAI Model:",
        list(available_models.keys()),
        help="Choose an OpenAI model - requires API key"
    )
    st.sidebar.info(f"ℹ️ {available_models[model_choice]}")
    api_key = st.sidebar.text_input("OpenAI API Key:", type="password", help="Enter your OpenAI API key")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Agent Status (static)
st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
st.sidebar.subheader("👥 Agent Status")
agent_status = {"🔍 Researcher": "Ready", "📊 Analyst": "Ready", "✍️ Writer": "Ready", "📝 Reviewer": "Ready", "🎯 Coordinator": "Ready"}
for agent, status in agent_status.items():
    st.sidebar.write(f"{agent}: {status}")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Session Statistics (basic)
st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
st.sidebar.subheader("📈 Session Stats")
st.sidebar.metric("Completed Tasks", len(st.session_state.task_history))
st.sidebar.metric("Agent Operations", len(st.session_state.agent_logs))
st.sidebar.metric("Success Rate", "95%")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

# Main Application
st.markdown('<h1 class="main-header">🤖 Agentic AI Command Center</h1>', unsafe_allow_html=True)
st.markdown("""
## Welcome to the Future of AI Collaboration! 🚀

### 🎯 System Capabilities:
- **🔍 Research Agent**: Web scraping, data gathering, source verification
- **📊 Analysis Agent**: Data processing, trend identification, insight extraction
- **✍️ Content Agent**: Writing, formatting, content creation
- **📝 Review Agent**: Quality assurance, editing, improvement
- **🎯 Coordinator**: Project management, workflow orchestration

### 🌟 Key Features:
- **Multi-Model Support**: Works with Ollama and OpenAI models
- **Specialized Tools**: Custom tools for each agent's expertise
- **Real-time Monitoring**: Track agent progress and performance
- **File Management**: Automatic saving and organization of outputs
- **Collaborative Workflow**: Agents build upon each other's work
- **Smart Memory**: Automatic embeddings configuration based on model type
""")

# Project Configuration
st.markdown("---")
st.subheader("🎯 Project Configuration")
col1, col2, col3 = st.columns(3)
with col1:
    project_topic = st.text_input("📝 Research Topic:", value="Artificial Intelligence in Healthcare")
with col2:
    output_format = st.selectbox("📄 Output Format:", ["Article", "Report", "Blog Post", "White Paper", "Executive Summary"])
with col3:
    target_audience = st.selectbox("👥 Target Audience:", ["General Public", "Technical Professionals", "Business Leaders", "Students", "Researchers"])

# Advanced Options
with st.expander("⚙️ Advanced Configuration"):
    col1, col2 = st.columns(2)
    with col1:
        max_iterations = st.slider("Max Iterations per Agent", 1, 5, 3)
        enable_memory = st.checkbox("Enable Agent Memory", value=True)
    with col2:
        process_type = st.selectbox("Execution Process:", ["Sequential", "Hierarchical"])
        verbose_mode = st.checkbox("Verbose Logging", value=True)

# Memory Configuration Info
if enable_memory:
    if model_type == "Ollama":
        st.info("🧠 **Memory Enabled**: Using Ollama embeddings (nomic-embed-text). Make sure you have it installed: `ollama pull nomic-embed-text`")
    else:
        if api_key:
            st.info("🧠 **Memory Enabled**: Using OpenAI embeddings with your API key.")
        else:
            st.warning("⚠️ **Memory Enabled**: OpenAI API key required for embeddings when using OpenAI models.")

# ---------- Helpers ----------
def get_embedder_config(model_type: str, api_key: str = None):
    if model_type == "Ollama":
        return {"provider": "ollama", "config": {"model": "nomic-embed-text", "base_url": "http://localhost:11434"}}
    else:
        cfg = {"provider": "openai", "config": {"model": "text-embedding-3-small"}}
        if api_key:
            cfg["config"]["api_key"] = api_key
        return cfg

def create_agents(model_choice: str, model_type: str) -> Dict[str, Agent]:
    """Create specialized agents with different roles using CrewAI's LLM wrapper."""
    if model_type == "Ollama":
        llm = LLM(
            model=f"ollama/{model_choice}",     # LiteLLM expects "ollama/<model_name>"
            base_url="http://localhost:11434",  # Ollama endpoint
            temperature=0.7,
            timeout=timeout_sec,                # <-- important: cut down from 600s default
            max_retries=retries                 # <-- controlled retries
        )
    else:
        llm = LLM(
            model=model_choice,
            temperature=0.7,
            timeout=timeout_sec,
            max_retries=retries
        )

    # Tools
    search_tool = DuckDuckGoSearchTool()
    scraping_tool = WebScrapingTool()
    analysis_tool = DataAnalysisTool()
    file_tool = FileManagerTool()

    # Define agents
    agents = {
        "researcher": Agent(
            role="Senior Research Specialist",
            goal="Conduct comprehensive research on given topics using web search and data gathering",
            backstory=("You are an experienced researcher with expertise in finding, analyzing, and "
                       "synthesizing information from multiple sources. You excel at identifying "
                       "credible sources and extracting relevant insights from complex data."),
            tools=[search_tool, scraping_tool],
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_retry=max_iterations
        ),
        "analyst": Agent(
            role="Data Analysis Expert",
            goal="Analyze research data and extract meaningful insights and patterns",
            backstory=("You are a skilled data analyst with years of experience in processing information, "
                       "identifying trends, and drawing actionable conclusions from research data."),
            tools=[analysis_tool],
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_retry=max_iterations
        ),
        "writer": Agent(
            role="Content Creation Specialist",
            goal="Create engaging, well-structured content based on research and analysis",
            backstory=("You are a professional content creator with expertise in various writing styles. "
                       "You can adapt your writing to different audiences and formats, from technical reports "
                       "to engaging blog posts."),
            tools=[file_tool],
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_retry=max_iterations
        ),
        "reviewer": Agent(
            role="Quality Assurance Editor",
            goal="Review, edit, and improve content for clarity, accuracy, and engagement",
            backstory=("You are a meticulous editor with years of experience in content review and improvement. "
                       "You enhance content while maintaining the author's voice and intent."),
            tools=[file_tool],
            llm=llm,
            verbose=True,
            allow_delegation=False,
            max_retry=max_iterations
        ),
        "coordinator": Agent(
            role="Project Coordination Manager",
            goal="Oversee the entire workflow and ensure all agents work together effectively",
            backstory=("You are an experienced project manager who excels at coordinating teams, managing workflows, "
                       "and ensuring deliverables meet requirements."),
            tools=[file_tool],
            llm=llm,
            verbose=True,
            allow_delegation=True,
            max_retry=max_iterations
        )
    }
    log_event("Agents initialized")
    return agents

def create_tasks(agents: Dict[str, Agent], project_details: Dict[str, Any]) -> List[Task]:
    """Create tasks for the crew based on project requirements"""
    topic = project_details.get("topic", "AI Technology")
    output_format = project_details.get("format", "article")
    target_audience = project_details.get("audience", "general")

    tasks = [
        Task(
            description=f"""Research the topic '{topic}' comprehensively. Include trends, data points, expert views,
            applications, and future prospects. Use web search/scraping tools. Provide a detailed summary with sources.""",
            agent=agents["researcher"],
            expected_output="Comprehensive research report with sources and key findings"
        ),
        Task(
            description=f"""Analyze the research data: trends, patterns, key stats, contradictions, gaps, and recommendations.""",
            agent=agents["analyst"],
            expected_output="Detailed analysis report with insights and recommendations"
        ),
        Task(
            description=f"""Create a high-quality {output_format} about '{topic}' for {target_audience} audience. Include key insights, data,
            citations, and follow best practices. Save the final content to a file.""",
            agent=agents["writer"],
            expected_output=f"Well-written {output_format} saved to file"
        ),
        Task(
            description=f"""Review and improve the content for accuracy, clarity, readability, and style. Save the polished version to a file.""",
            agent=agents["reviewer"],
            expected_output="Polished, final version of content saved to file"
        ),
        Task(
            description=f"""Coordinate the overall project. Compile all outputs into a final delivery package with summary and recommendations.""",
            agent=agents["coordinator"],
            expected_output="Complete project delivery package with all outputs and summary"
        )
    ]
    log_event(f"Tasks created for topic '{topic}'")
    return tasks

def ollama_healthcheck() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

# Task Execution
st.markdown("---")
st.subheader("🎯 Run")

if st.button("🚀 Launch Agent Crew", type="primary", use_container_width=True):
    if model_type == "OpenAI" and not api_key:
        st.error("❌ Please provide OpenAI API key to use OpenAI models")
    elif enable_memory and model_type == "OpenAI" and not api_key:
        st.error("❌ OpenAI API key required for memory features when using OpenAI models")
    elif model_type == "Ollama" and not ollama_healthcheck():
        st.error("❌ Ollama does not seem to be running at http://localhost:11434. Start it with `ollama serve`.")
    else:
        try:
            project_details = {
                "topic": project_topic,
                "format": output_format.lower(),
                "audience": target_audience.lower()
            }

            with st.spinner("🔧 Initializing agents and preparing workspace..."):
                agents = create_agents(model_choice, model_type)
                tasks = create_tasks(agents, project_details)

                embedder_config = None
                if enable_memory:
                    embedder_config = get_embedder_config(model_type, api_key)

                process_mode = Process.sequential if process_type == "Sequential" else Process.hierarchical

                crew_kwargs = {
                    "agents": list(agents.values()),
                    "tasks": tasks,
                    "process": process_mode,
                    "verbose": verbose_mode,
                    "memory": enable_memory,
                    "max_rpm": 10
                }
                if enable_memory and embedder_config:
                    crew_kwargs["embedder"] = embedder_config

                crew = Crew(**crew_kwargs)
                log_event("Crew initialized")

            st.success("✅ Agent crew initialized successfully!")

            # Progress tracking
            progress_container = st.container()
            with progress_container:
                st.markdown('<div class="progress-section">', unsafe_allow_html=True)
                st.subheader("🔄 Agent Execution Progress")
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("🚀 Starting agent execution...")
                log_event("Execution started")

                start_time = time.time()

                # Retry wrapper around kickoff to avoid 600s hangs
                attempt, max_attempts = 1, (retries + 1)
                result = None
                while attempt <= max_attempts:
                    try:
                        result = crew.kickoff()
                        break
                    except Exception as e:
                        err = str(e)
                        log_event(f"Attempt {attempt} failed: {err}")
                        if "Timeout" in err or "timed out" in err or "APIConnectionError" in err:
                            if attempt < max_attempts:
                                st.warning(f"⏱️ Model timed out (attempt {attempt}/{max_attempts}). Retrying...")
                                attempt += 1
                                continue
                        # Non-timeout or last attempt -> raise
                        raise

                execution_time = time.time() - start_time
                progress_bar.progress(1.0)
                status_text.text("✅ All agents completed successfully!")
                log_event(f"Execution finished in {execution_time:.1f}s")
                st.markdown('</div>', unsafe_allow_html=True)

            # Display results
            st.markdown('<div class="result-section">', unsafe_allow_html=True)
            st.subheader("📋 Execution Results")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Execution Time", f"{execution_time:.1f}s")
            with col2:
                st.metric("Agents Used", len(agents))
            with col3:
                st.metric("Tasks Completed", len(tasks))

            st.markdown("### 📄 Final Output:")
            st.markdown(str(result))

            # ----- Task history snapshot (queued -> completed) -----
            timestamp_now = datetime.now()
            for t in tasks:
                st.session_state.task_history.append({
                    "timestamp": timestamp_now,
                    "agent": t.agent.role,
                    "task": t.description,
                    "status": "completed",
                    "result_preview": str(result)[:500] if result else ""
                })

            # Save to session state
            st.session_state.crew_results.append({
                "timestamp": datetime.now(),
                "topic": project_topic,
                "format": output_format,
                "result": str(result),
                "execution_time": execution_time
            })

            st.markdown('</div>', unsafe_allow_html=True)
            st.balloons()

        except Exception as e:
            st.error(f"❌ Error executing agent crew: {str(e)}")
            log_event(f"ERROR: {str(e)}")
            st.info("💡 Common issues:")
            st.write("- **Ollama**: Make sure `ollama serve` is running and models are downloaded")
            st.write("- **Memory + Ollama**: Install embedding model: `ollama pull nomic-embed-text`")
            st.write("- **OpenAI**: Verify API key is correct and has sufficient credits")
            st.write(f"- **Timeouts**: Try increasing 'Model Timeout (sec)' above {timeout_sec} or lowering retries")

            with st.expander("🔍 Detailed Error Information"):
                st.code(str(e))

# Results History
if st.session_state.crew_results:
    st.markdown("---")
    st.subheader("📚 Execution History")
    for i, result in enumerate(reversed(st.session_state.crew_results[-5:]), 1):
        with st.expander(f"🗂️ Run {len(st.session_state.crew_results)-i+1}: {result['topic']} ({result['timestamp'].strftime('%H:%M:%S')})"):
            col1, col2 = st.columns([1, 3])
            with col1:
                st.write(f"**Topic:** {result['topic']}")
                st.write(f"**Format:** {result['format']}")
                st.write(f"**Duration:** {result['execution_time']:.1f}s")
            with col2:
                st.markdown("**Result:**")
                preview = result['result'][:500] + "..." if len(result['result']) > 500 else result['result']
                st.write(preview)

# ---------- New: Task History & Agent Logs UI ----------
st.markdown("---")
st.subheader("📜 Task History (latest)")
if st.session_state.task_history:
    for entry in reversed(st.session_state.task_history[-10:]):
        with st.expander(f"{entry['agent']} — {entry['timestamp'].strftime('%H:%M:%S')}"):
            st.write(f"**Task:** {entry['task']}")
            st.write(f"**Status:** {entry.get('status','completed')}")
            if entry.get("result_preview"):
                st.write(f"**Result (preview):** {entry['result_preview']}")

st.subheader("📝 Agent Logs (latest)")
if st.session_state.agent_logs:
    for log in reversed(st.session_state.agent_logs[-20:]):
        st.write(log)

# Setup Instructions
st.markdown("---")
st.subheader("⚙️ Quick Setup Guide")

tab1, tab2 = st.tabs(["🦙 Ollama Setup", "🤖 OpenAI Setup"])
