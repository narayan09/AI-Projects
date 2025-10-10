"""
Advanced Agent Implementations for Agentic AI System
This module contains specialized agent classes with enhanced capabilities
Notes:
- CrewAI expects tools to be instances of crewai.tools.BaseTool (or dicts).
- We wrap LangChain's DuckDuckGo tool into a CrewAI BaseTool subclass.
- Non-pydantic attributes (requests.Session, metadata paths) are declared as fields
  and we enable arbitrary_types_allowed in Config so pydantic accepts them.
"""
import os
import json
import logging
from typing import List, Dict, Any, Optional, Type
from datetime import datetime
from dataclasses import dataclass
import requests
from crewai import Process, Agent, Task, Crew
from crewai.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AgentConfig:
    """Configuration class for agent parameters"""
    model_name: str = "llama3.2:3b"
    model_type: str = "ollama"  # "ollama" or "openai"
    temperature: float = 0.7
    max_tokens: int = 2000
    max_retry: int = 3
    verbose: bool = True

class TaskResult(BaseModel):
    """Structured task result model"""
    agent_name: str
    task_description: str
    result: str
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = {}

#
# ---------- CrewAI-compatible tools (subclass crewai.tools.BaseTool) ----------
#

# 1) Wrapper for DuckDuckGoSearchRun (LangChain) so it's a CrewAI tool.
class DDGInput(BaseModel):
    query: str = Field(..., description="Search query")

class DuckDuckGoCrewTool(BaseTool):
    name: str = "duckduckgo_search"
    description: str = "Search the web using DuckDuckGo and return results (wrapped for CrewAI)."
    args_schema: Type[BaseModel] = DDGInput

    class Config:
        arbitrary_types_allowed = True

    def _run(self, query: str) -> str:
        try:
            ddg = DuckDuckGoSearchRun()
            result = ddg.invoke(query)
            return str(result)
        except Exception as e:
            logger.exception("DuckDuckGoCrewTool failed")
            return f"Error running DuckDuckGo search: {e}"

# 2) Advanced web scraping tool (CrewAI BaseTool)
class WebScrapingInput(BaseModel):
    url: str = Field(..., description="The URL to scrape")
    max_length: Optional[int] = Field(5000, description="Maximum length of returned text")

class AdvancedWebScrapingTool(BaseTool):
    """Enhanced web scraping tool with better error handling and content extraction"""
    name: str = "advanced_web_scraping"
    description: str = "Advanced web scraping with content cleaning and structure preservation"
    args_schema: Type[BaseModel] = WebScrapingInput

    session: Optional[requests.Session] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.session is None:
            self._setup_session()

    def _setup_session(self):
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        })

    def _run(self, url: str, max_length: int = 5000) -> str:
        try:
            from bs4 import BeautifulSoup
            import re
            logger.info(f"[Scraper] Scraping URL: {url}")
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript']):
                element.decompose()
            main_content = None
            for selector in ['main', 'article', '.content', '#content', '.main-content']:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            if not main_content:
                main_content = soup.find('body') or soup
            text_content = []
            for element in main_content.find_all(['h1','h2','h3','h4','h5','h6','p','li']):
                text = element.get_text(strip=True)
                if text and len(text) > 10:
                    if element.name.startswith('h'):
                        try:
                            level = int(element.name[1])
                        except Exception:
                            level = 2
                        text_content.append(f"{'#' * max(1, level)} {text}")
                    elif element.name == 'li':
                        text_content.append(f"- {text}")
                    else:
                        text_content.append(text)
            content = '\n\n'.join(text_content)
            content = re.sub(r'\n\s*\n', '\n\n', content)
            content = re.sub(r' +', ' ', content)
            if len(content) > max_length:
                content = content[:max_length] + "..."
            logger.info(f"[Scraper] Scraped {len(content)} characters from {url}")
            return content
        except Exception as e:
            logger.exception("Scraper encountered an error")
            return f"Error scraping {url}: {e}"

# 3) Advanced data analysis tool (CrewAI BaseTool)
class AdvancedDataAnalysisTool(BaseTool):
    name: str = "advanced_data_analysis"
    description: str = "Perform comprehensive data analysis with statistics and insights"

    class Config:
        arbitrary_types_allowed = True

    def _run(self, data: str, analysis_type: str = "comprehensive") -> str:
        try:
            import re
            from collections import Counter
            logger.info(f"[Analysis] Analyzing {len(data)} characters of data")
            words = data.split()
            sentences = re.split(r'[.!?]+', data)
            paragraphs = [p.strip() for p in data.split('\n\n') if p.strip()]
            word_count = len(words)
            sentence_count = len([s for s in sentences if s.strip()])
            paragraph_count = len(paragraphs)
            avg_words_per_sentence = word_count / max(sentence_count, 1)
            avg_words_per_paragraph = word_count / max(paragraph_count, 1)
            keywords = Counter(re.findall(r'\b[a-zA-Z]{3,}\b', data.lower())).most_common(10)
            report = f"""
COMPREHENSIVE DATA ANALYSIS
Basic Metrics:
- Word Count: {word_count}
- Sentences: {sentence_count}
- Paragraphs: {paragraph_count}
- Avg Words/Sentence: {avg_words_per_sentence:.1f}
- Avg Words/Paragraph: {avg_words_per_paragraph:.1f}
Top Keywords: {', '.join([f'{k}({c})' for k, c in keywords])}
"""
            return report.strip()
        except Exception as e:
            logger.exception("Data analysis failed")
            return f"Error in data analysis: {e}"

# 4) Smart file manager (CrewAI BaseTool)
class SmartFileManagerTool(BaseTool):
    name: str = "smart_file_manager"
    description: str = "Intelligent file management with organization and metadata tracking"
    base_dir: str = Field(default="outputs")
    metadata_file: Optional[str] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        os.makedirs(self.base_dir, exist_ok=True)
        if not self.metadata_file:
            self.metadata_file = os.path.join(self.base_dir, "file_metadata.json")

    def _run(self, content: str, filename: str = None, content_type: str = "markdown", metadata: Dict = None) -> str:
        metadata = metadata or {}
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                ext = self._get_extension(content_type)
                filename = f"agent_output_{timestamp}.{ext}"
            date_dir = os.path.join(self.base_dir, datetime.now().strftime("%Y-%m-%d"))
            os.makedirs(date_dir, exist_ok=True)
            filepath = os.path.join(date_dir, filename)
            formatted_content = self._format_content(content, content_type, metadata)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(formatted_content)
            self._update_metadata(filename, filepath, content_type, metadata)
            logger.info(f"[FileManager] Saved file: {filepath}")
            return f"✅ Content saved to {filepath}"
        except Exception as e:
            logger.exception("Saving file failed")
            return f"Error saving file: {e}"

    def _get_extension(self, content_type: str) -> str:
        extensions = {'markdown': 'md', 'text': 'txt', 'html': 'html', 'json': 'json', 'report': 'md'}
        return extensions.get(content_type.lower(), 'txt')

    def _format_content(self, content: str, content_type: str, metadata: Dict) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if content_type.lower() == 'markdown':
            header = f"""---
title: {metadata.get('title', 'AI Generated Content')}
author: {metadata.get('agent', 'AI Agent')}
created: {timestamp}
type: {content_type}
tags: {', '.join(metadata.get('tags', []))}
---
"""
            return header + content
        else:
            header = f"""# Metadata
# Created: {timestamp}
# Author: {metadata.get('agent', 'AI Agent')}
# Type: {content_type}
# Title: {metadata.get('title', 'AI Generated Content')}
"""
            return header + content

    def _update_metadata(self, filename: str, filepath: str, content_type: str, metadata: Dict):
        try:
            if os.path.exists(self.metadata_file):
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    all_metadata = json.load(f)
            else:
                all_metadata = {}
            file_metadata = {
                'filepath': filepath,
                'content_type': content_type,
                'created': datetime.now().isoformat(),
                'size': os.path.getsize(filepath),
                **(metadata or {}),
            }
            all_metadata[filename] = file_metadata
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(all_metadata, f, indent=2)
        except Exception as e:
            logger.exception("Updating metadata failed")

#
# ---------- Agent factory and workflow ----------
#

class AgentFactory:
    """Factory class for creating specialized agents"""

    @staticmethod
    def create_research_agent(config: AgentConfig) -> Agent:
        """Create an advanced research agent"""
        llm = AgentFactory._get_llm(config)
        tools = [
            DuckDuckGoCrewTool(),         # CrewAI-compatible wrapper
            AdvancedWebScrapingTool(),    # CrewAI BaseTool subclass
        ]
        print(f"Tools being passed to Agent: {[type(tool) for tool in tools]}")

        return Agent(
            role="Senior Research Specialist",
            goal="""Conduct comprehensive, accurate research using advanced web search and scraping techniques.
Gather data from multiple credible sources, verify information accuracy, and provide detailed source citations.""",
            backstory="""You are an experienced research professional with advanced skills in information gathering and synthesis.""",
            tools=tools,
            llm=llm,
            verbose=config.verbose,
            allow_delegation=False,
            max_retry=config.max_retry,
            max_execution_time=300
        )

    @staticmethod
    def create_analysis_agent(config: AgentConfig) -> Agent:
        llm = AgentFactory._get_llm(config)
        tools = [AdvancedDataAnalysisTool()]
        return Agent(
            role="Data Analysis Expert",
            goal="Analyze research data and produce insights.",
            backstory="Skilled data analyst.",
            tools=tools,
            llm=llm,
            verbose=config.verbose,
            allow_delegation=False,
            max_retry=config.max_retry,
            max_execution_time=180
        )

    @staticmethod
    def create_content_agent(config: AgentConfig) -> Agent:
        llm = AgentFactory._get_llm(config)
        tools = [SmartFileManagerTool()]
        return Agent(
            role="Content Creation Specialist",
            goal="Create high-quality content based on research and analysis.",
            backstory="Professional writer.",
            tools=tools,
            llm=llm,
            verbose=config.verbose,
            allow_delegation=False,
            max_retry=config.max_retry,
            max_execution_time=240
        )

    @staticmethod
    def create_review_agent(config: AgentConfig) -> Agent:
        llm = AgentFactory._get_llm(config)
        tools = [SmartFileManagerTool(), AdvancedDataAnalysisTool()]
        return Agent(
            role="Quality Assurance Editor",
            goal="Review and improve content.",
            backstory="Experienced editor.",
            tools=tools,
            llm=llm,
            verbose=config.verbose,
            allow_delegation=False,
            max_retry=config.max_retry,
            max_execution_time=180
        )

    @staticmethod
    def create_coordinator_agent(config: AgentConfig) -> Agent:
        llm = AgentFactory._get_llm(config)
        tools = [SmartFileManagerTool()]
        return Agent(
            role="Project Coordination Manager",
            goal="Coordinate multi-agent workflow and deliver outputs.",
            backstory="Project manager.",
            tools=tools,
            llm=llm,
            verbose=config.verbose,
            allow_delegation=True,
            max_retry=config.max_retry,
            max_execution_time=120
        )

    @staticmethod
    def _get_llm(config: AgentConfig):
        """Get LLM instance based on configuration"""
        if config.model_type.lower() == "openai":
            return ChatOpenAI(
                model=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens
            )
        else:
            return ChatOllama(
                model=config.model_name,
                temperature=config.temperature,
                num_predict=config.max_tokens
            )

class AgenticWorkflow:
    """Main workflow orchestrator for agentic AI system"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.agents = self._initialize_agents()
        self.results = []

    def _initialize_agents(self) -> Dict[str, Agent]:
        """Initialize all agents"""
        return {
            'researcher': AgentFactory.create_research_agent(self.config),
            'analyst': AgentFactory.create_analysis_agent(self.config),
            'writer': AgentFactory.create_content_agent(self.config),
            'reviewer': AgentFactory.create_review_agent(self.config),
            'coordinator': AgentFactory.create_coordinator_agent(self.config)
        }

    def create_tasks(self, project_details: Dict[str, Any]) -> List[Task]:
        """Create tasks based on project requirements"""
        topic = project_details.get('topic', 'AI Technology')
        format_type = project_details.get('format', 'article')
        audience = project_details.get('audience', 'general')
        tasks = [
            Task(
                description=f"Conduct comprehensive research on the topic: \"{topic}\"",
                agent=self.agents['researcher'],
                expected_output="Comprehensive research report with organized findings and source citations"
            ),
            Task(
                description="Analyze the research data provided and generate actionable insights.",
                agent=self.agents['analyst'],
                expected_output="Detailed analysis report with insights, trends, and recommendations"
            ),
            Task(
                description=f"Create a high-quality {format_type} about \"{topic}\" for {audience}.",
                agent=self.agents['writer'],
                expected_output=f"Well-structured, engaging {format_type} saved to file with metadata"
            ),
            Task(
                description="Review and enhance the created content for maximum quality and impact.",
                agent=self.agents['reviewer'],
                expected_output="Enhanced final content with quality assessment report"
            ),
            Task(
                description="Coordinate the entire project and create a comprehensive delivery package.",
                agent=self.agents['coordinator'],
                expected_output="Complete project delivery package with all components organized"
            )
        ]
        return tasks

    def execute_workflow(self, project_details: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete agentic workflow"""
        try:
            logger.info("Starting agentic workflow execution")
            start_time = datetime.now()
            tasks = self.create_tasks(project_details)
            crew = Crew(
                agents=list(self.agents.values()),
                tasks=tasks,
                process=Process.sequential,
                verbose=self.config.verbose,
                max_rpm=10,
                memory=True
            )
            result = crew.kickoff()
            execution_time = (datetime.now() - start_time).total_seconds()
            workflow_result = {
                'success': True,
                'execution_time': execution_time,
                'final_output': str(result),
                'agents_used': len(self.agents),
                'tasks_completed': len(tasks),
                'project_details': project_details,
                'timestamp': datetime.now().isoformat()
            }
            logger.info(f"Workflow completed successfully in {execution_time:.1f} seconds")
            return workflow_result
        except Exception as e:
            logger.exception("Workflow execution failed")
            return {
                'success': False,
                'error': str(e),
                'execution_time': 0,
                'timestamp': datetime.now().isoformat()
            }


# Example usage and testing
if __name__ == "__main__":
    config = AgentConfig(
        model_name="llama3.2:3b",
        model_type="ollama",
        temperature=0.7,
        verbose=True
    )
    project = {
        'topic': 'Artificial Intelligence in Healthcare',
        'format': 'article',
        'audience': 'healthcare professionals'
    }
    workflow = AgenticWorkflow(config)
    print("Advanced Agent System initialized successfully!")
    print("Running workflow (synchronous kickoff).")

    # This will trigger the CrewAI workflow; LLMs and remote tools may need credentials.
    try:
        result = workflow.execute_workflow(project)
        print("Result:", result)
    except Exception as e:
        print("Workflow raised an exception at runtime:", e)
