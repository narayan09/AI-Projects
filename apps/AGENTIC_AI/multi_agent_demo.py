"""
multi_agent_ollama_streamlit.py

Streamlit multi-agent demo that runs locally with Ollama.
Auto-prefixes model names with "ollama/" to satisfy litellm provider requirements.
"""

import streamlit as st
import time
import os
import requests
from datetime import datetime
from typing import Dict, Any, List

# CrewAI imports (use your installed crewai package)
from crewai import Agent, Task, Crew, Process, LLM

# -------------------------
# Page & session init
# -------------------------
st.set_page_config(page_title="Local Multi-Agent (Ollama)", layout="wide", page_icon="🤖")
st.title("🤖 Local Multi-Agent Demo (Ollama)")

if "task_history" not in st.session_state:
    st.session_state.task_history: List[Dict[str, Any]] = []
if "agent_logs" not in st.session_state:
    st.session_state.agent_logs: List[str] = []
if "crew_results" not in st.session_state:
    st.session_state.crew_results: List[Dict[str, Any]] = []

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    st.session_state.agent_logs.append(entry)

# -------------------------
# Sidebar: configuration
# -------------------------
st.sidebar.header("Configuration")

# Common local Ollama model names people use; user can type custom one too
model_choice = st.sidebar.selectbox(
    "Choose Ollama model (local)",
    ["phi3:mini", "gemma:latest", "llama3.2:3b"],
    index=0
)
custom_model = st.sidebar.text_input("Or type custom model (leave blank to use selection above):", value="")
if custom_model.strip():
    model_choice = custom_model.strip()

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7, 0.1)
timeout_sec = st.sidebar.slider("Model timeout (seconds)", 30, 300, 120, 10)
retries = st.sidebar.slider("Retries on timeout/failure", 0, 5, 2)
ollama_base = st.sidebar.text_input("Ollama base URL", value="http://localhost:11434")
st.sidebar.markdown("---")
st.sidebar.markdown("Make sure `ollama serve` is running and the model is pulled locally.")

# -------------------------
# Helpers
# -------------------------
def build_model_string(user_model: str) -> str:
    """
    Ensure the model string includes the 'ollama/' provider prefix.
    This prevents litellm.BadRequestError: LLM Provider NOT provided.
    """
    m = user_model.strip()
    # if already contains provider prefix, keep as-is
    if m.startswith("ollama/") or m.startswith("huggingface/") or m.startswith("openai/"):
        return m
    return f"ollama/{m}"

def ollama_health_check(base_url: str) -> bool:
    """
    Try a few endpoints that Ollama commonly exposes.
    Returns True if any returns a 200.
    """
    endpoints = ["/api/tags", "/api/models", "/", "/api/ping"]
    for ep in endpoints:
        try:
            url = base_url.rstrip("/") + ep
            r = requests.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            continue
    return False

def record_task_event(agent_role: str, task_desc: str, status: str, preview: str = ""):
    st.session_state.task_history.append({
        "timestamp": datetime.now(),
        "agent": agent_role,
        "task": task_desc,
        "status": status,
        "preview": preview
    })

# -------------------------
# Main UI: user input
# -------------------------
st.subheader("Task input")
topic = st.text_input("Enter topic to research (example: 'Python programming language'):", value="Python programming language")
run_button = st.button("🚀 Run (Researcher → Writer)")

# -------------------------
# When user runs crew
# -------------------------
if run_button:
    if not topic.strip():
        st.warning("Please enter a topic.")
    else:
        model_str = build_model_string(model_choice)
        st.info(f"Using model: **{model_str}**  (base: {ollama_base})")
        
        # Health-check Ollama
        healthy = ollama_health_check(ollama_base)
        if not healthy:
            st.error(f"Ollama server not reachable at {ollama_base}. Start Ollama with `ollama serve` and ensure network access from this machine.")
        else:
            # Initialize LLM (crewai.LLM) -- ensure model string has provider prefix
            try:
                log(f"Initializing LLM with model={model_str}, timeout={timeout_sec}, retries={retries}")
                llm = LLM(
                    model=model_str,
                    base_url=ollama_base,
                    temperature=temperature,
                    timeout=timeout_sec,
                    max_retries=retries
                )
            except Exception as e:
                st.error("Failed initializing LLM. See logs below.")
                log(f"LLM init error: {e}")
                st.exception(e)
                st.stop()

            # Build agents
            researcher = Agent(
                role="Researcher",
                goal="Collect concise facts and key points about the topic",
                backstory="You are a focused researcher who gathers accurate information.",
                llm=llm,
                verbose=False,
                allow_delegation=False
            )

            writer = Agent(
                role="Writer",
                goal="Write a plain-language 3-line summary based on the research",
                backstory="You are a writer who creates short, clear summaries.",
                llm=llm,
                verbose=False,
                allow_delegation=False
            )

            # Create tasks (expected_output is required)
            task1 = Task(
                description=f"Research: Provide key facts and short bullets about '{topic}'",
                agent=researcher,
                expected_output="A structured list of 5-8 key facts about the topic."
            )
            task2 = Task(
                description=f"Write a 3-line plain-language summary of the research on '{topic}'",
                agent=writer,
                expected_output="Three short lines that summarize the key points in plain language."
            )

            crew = Crew(
                agents=[researcher, writer],
                tasks=[task1, task2],
                process=Process.sequential,
                verbose=False
            )

            # Pre-run logging of task starts
            log("Preparing to run crew.kickoff()")
            record_task_event(researcher.role, task1.description, "started")
            record_task_event(writer.role, task2.description, "queued")

            # Run with retry loop to handle transient timeouts (we already set max_retries for LLM)
            attempt = 1
            max_attempts = max(1, retries + 1)
            result = None
            exec_error = None
            start_time = time.time()
            progress_bar = st.progress(0.0)
            status_placeholder = st.empty()

            while attempt <= max_attempts:
                status_placeholder.info(f"Attempt {attempt}/{max_attempts} — running crew...")
                try:
                    result = crew.kickoff()
                    status_placeholder.success("Crew finished successfully")
                    log("Crew kickoff completed")
                    break
                except Exception as e:
                    exec_error = e
                    errmsg = str(e)
                    log(f"Crew attempt {attempt} failed: {errmsg}")
                    # Helpful guidance if litellm complains about provider
                    if "LLM Provider NOT provided" in errmsg or "Provider NOT provided" in errmsg or "LLM Provider" in errmsg:
                        st.error("litellm error: LLM provider not provided. This app prefixes models with 'ollama/'.")
                        st.markdown(f"**Model used:** `{model_str}`")
                        st.markdown("If you're using a hosted provider (not Ollama), set the provider accordingly.")
                        log("LLM Provider error encountered; aborting retries.")
                        break
                    if attempt < max_attempts:
                        st.warning(f"Attempt {attempt} failed. Retrying...")
                        attempt += 1
                        progress_bar.progress(attempt / max_attempts)
                        time.sleep(1.0 + attempt * 0.5)
                        continue
                    else:
                        status_placeholder.error("All attempts failed.")
                        break

            execution_time = time.time() - start_time
            progress_bar.progress(1.0)

            # Post-run: store results and task completions
            result_preview = ""
            try:
                result_preview = str(result)[:1500] if result is not None else f"Error: {exec_error}"
            except Exception:
                result_preview = f"Could not stringify result. Raw: {repr(result)}"

            # mark tasks as completed with preview (we don't have per-task outputs from crew.kickoff())
            record_task_event(researcher.role, task1.description, "completed", preview=result_preview)
            record_task_event(writer.role, task2.description, "completed", preview=result_preview)

            st.success(f"Execution finished in {execution_time:.2f} s")
            st.markdown("### Final result (raw):")
            st.code(result_preview)

            # Save run summary
            st.session_state.crew_results.append({
                "timestamp": datetime.now(),
                "topic": topic,
                "model": model_str,
                "execution_time_s": execution_time,
                "preview": result_preview
            })

# -------------------------
# Task history & logs UI
# -------------------------
st.markdown("---")
st.subheader("📜 Task history (latest entries)")
if st.session_state.task_history:
    # show newest first
    for entry in reversed(st.session_state.task_history[-20:]):
        ts = entry["timestamp"].strftime("%H:%M:%S")
        st.write(f"**{entry['agent']}** — {ts} — *{entry['status']}*")
        st.caption(entry["task"])
        if entry.get("preview"):
            st.code(entry["preview"][:400])

st.markdown("---")
st.subheader("📝 Agent logs")
if st.session_state.agent_logs:
    for l in reversed(st.session_state.agent_logs[-50:]):
        st.text(l)

# -------------------------
# Past runs
# -------------------------
st.markdown("---")
st.subheader("📚 Past runs")
if st.session_state.crew_results:
    for run in reversed(st.session_state.crew_results[-10:]):
        st.write(f"- {run['timestamp'].strftime('%Y-%m-%d %H:%M:%S')} — topic: {run['topic']} — model: {run['model']} — {run['execution_time_s']:.2f}s")
        st.caption(run["preview"][:300])

# Footer tips
st.markdown("---")
st.info("Tips: If you still get `LLM Provider NOT provided`, ensure the model name is prefixed with `ollama/` (this app does that automatically). "
        "If Ollama is slow, try a smaller model (e.g., `phi3:mini`) and increase the timeout slider if needed.")
