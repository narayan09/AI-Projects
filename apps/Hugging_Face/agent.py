# youtube_multi_agent_demos.py

import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

import streamlit as st
from dotenv import load_dotenv

# CrewAI
from crewai import LLM, Agent, Task, Crew, Process

# -------------------------
# Setup
# -------------------------
load_dotenv()
HF_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if not HF_API_TOKEN:
    raise ValueError("Missing HUGGINGFACE_API_TOKEN in .env")

st.set_page_config(page_title="YouTube Creator Multi‑Agent Demos", layout="wide")
st.title("🎬 YouTube Creator Multi‑Agent Demos (Accurate + Tracked)")

# -------------------------
# Helpers for accuracy
# -------------------------
def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract first JSON object from model output and parse it safely.
    """
    if not text:
        return None
    # Try raw parse first
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try to find fenced code block
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        candidate = text[start_idx:end_idx+1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


def validate_keys(data: Dict[str, Any], required_keys: List[str]) -> bool:
    if data is None:
        return False
    for k in required_keys:
        if k not in data or data[k] in (None, "", []):
            return False
    return True


def ensure_json_output(llm_run_fn, system_prompt: str, user_prompt: str, required_keys: List[str], retry: int = 1) -> Dict[str, Any]:
    """
    Call an agent/task runner function that returns text; enforce strict JSON with required keys.
    Retry once if JSON invalid.
    """
    out = llm_run_fn(system_prompt, user_prompt)
    data = extract_json(out)
    if not validate_keys(data, required_keys) and retry > 0:
        # tighten instruction
        tightened = user_prompt + "\n\nIMPORTANT: Output MUST be ONLY a valid JSON object with all required keys, no prose."
        out = llm_run_fn(system_prompt, tightened)
        data = extract_json(out)
    return data or {k: None for k in required_keys}


# -------------------------
# Sidebar Config
# -------------------------
st.sidebar.header("Configuration")

WORKING_MODELS = [
    "meta-llama/Llama-3.2-3B-Instruct:together",
    "Qwen/Qwen2.5-7B-Instruct:together",
    "llama-3.2-3b-preview:groq",
    "meta-llama/Llama-3.2-3B-Instruct:sambanova"
]
model_choice = st.sidebar.selectbox("Model (HF Inference Providers)", WORKING_MODELS)
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.5, 0.1)
max_tokens = st.sidebar.slider("Max tokens per step", 128, 1024, 512, 64)
enable_bhojpuri = st.sidebar.checkbox("Generate Bhojpuri localization", value=True)

# -------------------------
# Demo Selector
# -------------------------
DEMO_OPTIONS = {
    "Full Video Package": "SEO plan → script → titles → thumbnail text → description/tags/hashtags → chapters → compliance → optional Bhojpuri → scoring",
    "Idea Lab": "Generate 20 ideas with hooks & SEO angle; rank top 5",
    "Shorts Factory": "Generate 3 short scripts with strong hooks, captions, and CTA"
}
demo = st.sidebar.selectbox("Pick a demo", list(DEMO_OPTIONS.keys()))
st.caption(f"Demo: {DEMO_OPTIONS[demo]}")

topic = st.text_input("Topic / Niche / Video Seed", value="Bhagavad Gita life lessons for productivity")
run = st.button("🚀 Run Demo")

# -------------------------
# Initialize LLM
# -------------------------
def build_llm():
    return LLM(
        model=f"huggingface/{model_choice}",
        api_key=HF_API_TOKEN,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=90,
        max_retries=2
    )

# -------------------------
# Tracking
# -------------------------
if "logs" not in st.session_state:
    st.session_state["logs"] = []
if "task_outputs" not in st.session_state:
    st.session_state["task_outputs"] = []

def log(agent: str, action: str, details: str = ""):
    st.session_state["logs"].append({
        "ts": datetime.now().strftime("%H:%M:%S"),
        "agent": agent,
        "action": action,
        "details": details[:400]
    })

def show_timeline():
    st.subheader("📜 Timeline")
    for item in reversed(st.session_state["logs"][-30:]):
        st.write(f"- {item['ts']} — {item['agent']}: {item['action']} — {item['details']}")

def show_task_outputs():
    st.subheader("🧩 Task Outputs")
    for i, t in enumerate(st.session_state["task_outputs"]):
        with st.expander(f"Task {i+1}: {t['agent']} — {t['title']}"):
            st.json(t["data"])

# -------------------------
# Agent builders
# -------------------------
def make_agents(llm):
    # Accuracy priming & policy
    policy_note = (
        "Follow YouTube Community Guidelines. Avoid harmful medical/financial advice, hate, harassment, "
        "or deceptive claims. Prefer safe, neutral phrasing; do not fabricate statistics. "
        "If facts are uncertain, omit or generalize. Return strict JSON only."
    )

    # SEO Strategist
    strategist = Agent(
        role="SEO Strategist",
        goal="Design SEO‑aligned content plan with audience hook and keywords",
        backstory="A YouTube growth strategist who applies 2025 SEO best practices with measurable heuristics.",
        llm=llm, verbose=True, allow_delegation=False, max_iter=1
    )
    # Script Writer
    writer = Agent(
        role="Script Writer",
        goal="Write engaging scripts optimized for retention and clarity",
        backstory="A creator who structures content with strong opening hooks, clear segments, and a natural CTA.",
        llm=llm, verbose=True, allow_delegation=False, max_iter=1
    )
    # Titles/Thumbnail
    titler = Agent(
        role="Title & Thumbnail Expert",
        goal="Craft compelling titles and thumbnail text prompts",
        backstory="Optimizes titles for CTR and clarity; creates concise thumbnail text prompts.",
        llm=llm, verbose=True, allow_delegation=False, max_iter=1
    )
    # Description & Tags
    copywriter = Agent(
        role="Copywriter",
        goal="Write descriptions, hashtags, and tags aligned with SEO plan",
        backstory="Writes descriptions with keyword‑rich first 150 characters, relevant tags, and focused hashtags.",
        llm=llm, verbose=True, allow_delegation=False, max_iter=1
    )
    # Chapters
    chapters = Agent(
        role="Chapters Generator",
        goal="Create accurate timestamps with >=10s spacing starting at 0:00",
        backstory="Structures content into chapters that improve navigation and search snippets.",
        llm=llm, verbose=True, allow_delegation=False, max_iter=1
    )
    # Compliance
    reviewer = Agent(
        role="Compliance Reviewer",
        goal="Flag potential policy risks and suggest safe rewrites",
        backstory="Ensures content aligns with platform policy while preserving message.",
        llm=llm, verbose=True, allow_delegation=False, max_iter=1
    )
    # Localizer (Bhojpuri/Hinglish)
    localizer = Agent(
        role="Localizer",
        goal="Create Bhojpuri/Hinglish variants of titles and a short summary",
        backstory="Adapts tone and idioms while keeping meaning and SEO cues.",
        llm=llm, verbose=True, allow_delegation=False, max_iter=1
    )
    # Referee/Rater
    rater = Agent(
        role="Referee",
        goal="Score candidates by rubric and select top options",
        backstory="Ranks titles/descriptions by clarity, CTR potential, and alignment to SEO plan.",
        llm=llm, verbose=True, allow_delegation=False, max_iter=1
    )

    return {
        "strategist": strategist,
        "writer": writer,
        "titler": titler,
        "copywriter": copywriter,
        "chapters": chapters,
        "reviewer": reviewer,
        "localizer": localizer,
        "rater": rater
    }

# -------------------------
# Task runners (return text)
# -------------------------
def run_agent_task(agent: Agent, system_prompt: str, user_prompt: str) -> str:
    # Use CrewAI Task lightweight runner
    t = Task(description=user_prompt, agent=agent, expected_output="JSON only", async_execution=False)
    # Internally, kickoff will call the agent; we use Crew wrapper for consistency
    crew = Crew(agents=[agent], tasks=[t], process=Process.sequential, verbose=False, memory=False)
    result = crew.kickoff()
    # result already stringifiable
    return str(result)

# -------------------------
# Demo flows
# -------------------------

def demo_full_video_package(topic: str, llm):
    agents = make_agents(llm)

    # 1) SEO Plan (structured)
    system = "You are a YouTube SEO strategist. Return JSON only."
    user = f"""
Topic: "{topic}"

Return JSON with keys:
- primary_keyword (string; front-loadable in title)
- secondary_keywords (array of 5-10 strings)
- audience (string: who it's for)
- angle (string: positioning)
- outline (array of 4-7 strings: high-level segments)
Rules:
- Follow 2025 best practices; avoid clickbait; be precise and useful.
- JSON only.
"""
    keys = ["primary_keyword", "secondary_keywords", "audience", "angle", "outline"]
    plan = ensure_json_output(lambda sp, up: run_agent_task(agents["strategist"], sp, up), system, user, keys)
    st.session_state["task_outputs"].append({"agent": "SEO Strategist", "title": "SEO Plan", "data": plan})
    log("SEO Strategist", "Completed", "SEO Plan")

    # 2) Script
    system = "You are a YouTube script writer. Return JSON only."
    user = f"""
Write a script JSON for topic "{topic}" using this plan:
{json.dumps(plan)}

Return keys:
- hook (1-2 sentences, immediate value)
- intro (2-4 sentences)
- body (array of 4-7 short segments; each 2-4 sentences)
- cta (1-2 sentences)
Constraints:
- Clear, concise, engaging; avoid unverifiable claims; no policy violations.
- JSON only.
"""
    script_keys = ["hook", "intro", "body", "cta"]
    script = ensure_json_output(lambda sp, up: run_agent_task(agents["writer"], sp, up), system, user, script_keys)
    st.session_state["task_outputs"].append({"agent": "Script Writer", "title": "Script", "data": script})
    log("Script Writer", "Completed", "Script")

    # 3) Titles & Thumbnail prompts
    system = "You craft titles and thumbnail prompts. Return JSON only."
    user = f"""
Using primary_keyword="{plan['primary_keyword']}" and topic "{topic}", create JSON:
- titles (array of 10 strings; 5-10 words; front-load primary keyword; accurate, non-clickbait)
- thumbnail_prompts (array of 5 strings; 2-5 words each; high-contrast emotional cues)
JSON only.
"""
    tt_keys = ["titles", "thumbnail_prompts"]
    tt = ensure_json_output(lambda sp, up: run_agent_task(agents["titler"], sp, up), system, user, tt_keys)
    st.session_state["task_outputs"].append({"agent": "Title & Thumbnail Expert", "title": "Titles & Thumbnails", "data": tt})
    log("Title & Thumbnail Expert", "Completed", "Titles & Thumbnail Prompts")

    # 4) Description, hashtags, tags
    system = "You are a YouTube copywriter. Return JSON only."
    user = f"""
Write a description and metadata for "{topic}" with this plan:
{json.dumps(plan)}

JSON keys:
- description (string; first 150 chars compelling with primary keyword; total 500-1000 chars; no link spam)
- hashtags (array of 2-3 strings; relevant)
- tags (array of 8-12 keyword tags)
JSON only.
"""
    meta_keys = ["description", "hashtags", "tags"]
    meta = ensure_json_output(lambda sp, up: run_agent_task(agents["copywriter"], sp, up), system, user, meta_keys)
    st.session_state["task_outputs"].append({"agent": "Copywriter", "title": "Description/Hashtags/Tags", "data": meta})
    log("Copywriter", "Completed", "Description/Hashtags/Tags")

    # 5) Chapters
    system = "You create YouTube chapters. Return JSON only."
    user = f"""
Create chapters for the script:
{json.dumps(script)}

Rules:
- Start at "0:00".
- At least 3 chapters, ascending order.
- Each chapter >= 10 seconds.
Return JSON:
- chapters: array of objects {{ "time": "M:SS" or "H:MM:SS", "title": "..." }}
JSON only.
"""
    chap_keys = ["chapters"]
    ch = ensure_json_output(lambda sp, up: run_agent_task(agents["chapters"], sp, up), system, user, chap_keys)
    st.session_state["task_outputs"].append({"agent": "Chapters Generator", "title": "Chapters", "data": ch})
    log("Chapters Generator", "Completed", "Chapters")

    # 6) Compliance Review
    system = "You are a policy reviewer. Return JSON only."
    user = f"""
Review for policy issues. Inputs:
Plan: {json.dumps(plan)}
Script: {json.dumps(script)}
Description: {json.dumps(meta)}

Return JSON:
- risks (array of strings describing potential issues or 'none')
- suggested_rewrites (array of strings with safer phrasing)
- notes (string)
Avoid false authority claims; keep suggestions concise.
JSON only.
"""
    rv_keys = ["risks", "suggested_rewrites", "notes"]
    rv = ensure_json_output(lambda sp, up: run_agent_task(agents["reviewer"], sp, up), system, user, rv_keys)
    st.session_state["task_outputs"].append({"agent": "Compliance Reviewer", "title": "Policy Review", "data": rv})
    log("Compliance Reviewer", "Completed", "Policy Review")

    # 7) Optional Bhojpuri localization
    loc = None
    if enable_bhojpuri:
        system = "You are a localizer. Return JSON only."
        user = f"""
Create Bhojpuri/Hinglish variants. Use culturally natural tone.
Inputs:
Titles: {json.dumps(tt['titles'][:5])}
Script hook+cta: {json.dumps({'hook': script['hook'], 'cta': script['cta']})}

Return JSON:
- titles_bhojpuri (array of 5 strings)
- short_summary_bhojpuri (string; 3-4 lines)
JSON only.
"""
        loc_keys = ["titles_bhojpuri", "short_summary_bhojpuri"]
        loc = ensure_json_output(lambda sp, up: run_agent_task(agents["localizer"], sp, up), system, user, loc_keys)
        st.session_state["task_outputs"].append({"agent": "Localizer", "title": "Bhojpuri Variants", "data": loc})
        log("Localizer", "Completed", "Bhojpuri Variants")

    # 8) Rater selects top titles
    system = "You are a strict rater. Return JSON only."
    user = f"""
Score titles against rubric (front-loaded keyword, clarity, non-clickbait, CTR potential).
Titles: {json.dumps(tt['titles'])}
Primary keyword: "{plan['primary_keyword']}"

Return JSON:
- top_titles (array of 3 strings)
- rationale (string; brief scoring explanation)
JSON only.
"""
    rt_keys = ["top_titles", "rationale"]
    rating = ensure_json_output(lambda sp, up: run_agent_task(agents["rater"], sp, up), system, user, rt_keys)
    st.session_state["task_outputs"].append({"agent": "Referee", "title": "Top Titles", "data": rating})
    log("Referee", "Completed", "Top Titles")

    # Final presentation
    st.success("✅ Full Video Package Complete")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Recommended Titles")
        st.write("\n".join([f"- {t}" for t in rating.get("top_titles", [])]))
        if loc:
            st.subheader("Bhojpuri Title Variants")
            st.write("\n".join([f"- {t}" for t in loc.get("titles_bhojpuri", [])]))
    with col2:
        st.subheader("Description")
        st.write(meta.get("description", ""))
        st.subheader("Hashtags / Tags")
        st.write(", ".join(meta.get("hashtags", [])))
        st.write(", ".join(meta.get("tags", [])))

    st.subheader("Chapters")
    for chp in (ch.get("chapters") or []):
        st.write(f"- {chp.get('time','')}: {chp.get('title','')}")

    st.subheader("Policy Notes")
    st.write("Risks: " + (", ".join(rv.get("risks", [])) if rv.get("risks") else "none"))
    st.write("Notes: " + (rv.get("notes") or ""))


def demo_idea_lab(topic: str, llm):
    agents = make_agents(llm)
    # Generate 20 ideas
    system = "You are a content strategist. Return JSON only."
    user = f"""
Generate 20 video ideas for "{topic}".
Return JSON:
- ideas: array of objects {{ "title": string, "hook": string, "seo_angle": string, "primary_keyword": string }}
Titles 5-10 words; clear & accurate; no clickbait. JSON only.
"""
    ideas = ensure_json_output(lambda sp, up: run_agent_task(agents["strategist"], sp, up), system, user, ["ideas"])
    st.session_state["task_outputs"].append({"agent": "SEO Strategist", "title": "20 Ideas", "data": ideas})
    log("SEO Strategist", "Completed", "20 Ideas")

    # Rank top 5
    system = "You are a rater. Return JSON only."
    user = f"""
Select top 5 ideas maximizing clarity, CTR potential, and audience fit.
Ideas: {json.dumps(ideas['ideas'][:20])}

Return JSON:
- top5 (array of ideas objects)
- rationale (string)
JSON only.
"""
    top = ensure_json_output(lambda sp, up: run_agent_task(agents["rater"], sp, up), system, user, ["top5", "rationale"])
    st.session_state["task_outputs"].append({"agent": "Referee", "title": "Top 5 Ideas", "data": top})
    log("Referee", "Completed", "Top 5 Ideas")

    st.success("✅ Idea Lab Complete")
    st.subheader("Top 5 Ideas")
    for i, idea in enumerate(top.get("top5", []), 1):
        st.write(f"{i}. {idea.get('title','')} — Hook: {idea.get('hook','')} — Angle: {idea.get('seo_angle','')}")


def demo_shorts_factory(topic: str, llm):
    agents = make_agents(llm)
    # 3 shorts scripts
    system = "You write short‑form scripts. Return JSON only."
    user = f"""
Create 3 short video scripts for "{topic}".
Each script: hook (1 line), body (3-5 lines; concise), cta (1 line), captions (array 3-5 overlay texts).
Return JSON: {{ "shorts": [ {{...}}, {{...}}, {{...}} ] }}
No claims needing citations; keep generic and motivational. JSON only.
"""
    shorts = ensure_json_output(lambda sp, up: run_agent_task(agents["writer"], sp, up), system, user, ["shorts"])
    st.session_state["task_outputs"].append({"agent": "Script Writer", "title": "3 Shorts", "data": shorts})
    log("Script Writer", "Completed", "3 Shorts")

    # Title & thumbnail prompts for shorts
    system = "You craft titles and thumbnail prompts. Return JSON only."
    user = f"""
For each short, create:
- title (5-8 words, front-load keyword)
- thumbnail_text (2-4 words)
Return JSON: {{ "assets": [ {{ "title": str, "thumbnail_text": str }}, ... ] }} (3 items)
JSON only.
"""
    assets = ensure_json_output(lambda sp, up: run_agent_task(agents["titler"], sp, up), system, user, ["assets"])
    st.session_state["task_outputs"].append({"agent": "Title & Thumbnail Expert", "title": "Shorts Assets", "data": assets})
    log("Title & Thumbnail Expert", "Completed", "Shorts Assets")

    st.success("✅ Shorts Factory Complete")
    for i, sc in enumerate(shorts.get("shorts", []), 1):
        st.subheader(f"Short {i}")
        st.write(f"Title: {assets.get('assets', [{}]*(i))[i-1].get('title','') if assets.get('assets') else ''}")
        st.write("Hook: " + sc.get("hook",""))
        st.write("CTA: " + sc.get("cta",""))
        st.write("Captions: " + ", ".join(sc.get("captions", [])))

# -------------------------
# Run
# -------------------------
if run:
    # reset
    st.session_state["logs"] = []
    st.session_state["task_outputs"] = []
    llm = build_llm()

    if demo == "Full Video Package":
        demo_full_video_package(topic, llm)
    elif demo == "Idea Lab":
        demo_idea_lab(topic, llm)
    elif demo == "Shorts Factory":
        demo_shorts_factory(topic, llm)

    # tracking
    show_timeline()
    show_task_outputs()
