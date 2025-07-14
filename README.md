# Research Assistant Agent

## Feedback

* [ ] Env vars (legacy & current) are confusing and very not clearly documented -- the whole `LANGSMITH_*` vs. `LANGCHAIN_*` env vars are not well documented. 
* [ ] ExperimentResults (actual returning results) documentation should be easier to find. Or perhaps I just didn't search hard enough... But I had to use python REPL to examine what has been returned. from `from langsmith_evaluation import run_evaluation_experiment`
* [ ] I haven’t tested with other LLM integration, but I do know the integration with Azure OpenAI, especially the documentation, can improve a lot…
  * [ ] For instance, when visiting https://python.langchain.com/docs/integrations/llms/azure_openai/, it says, “This page goes over how to use LangChain with Azure OpenAI.”. But that page leads to a Azure OpenAI sales page… It probably should lead to this page https://learn.microsoft.com/en-us/azure/ai-foundry/how-to/develop/langchain
  * [ ] Another small thing is stuff like this [Azure OpenAI | 🦜️🔗](https://python.langchain.com/docs/integrations/llms/azure_openai/) LangChain more specifically see screenshot. It really should be **BOLD, highlighted or CALLOUT**
![alt text](<docs/Screenshot 2025-07-14 at 11.47.46.png>)

## Wishful features for LangSmith

* [ ] LoRA (low-rank adaptation) or any other fine-tuning techniques in LangSmith

## How to run

1. create conda environment
   
```bash
conda create -f ls-academy.yaml
```

2. run streamlit app

```bash
streamlit run app.py
```

## Overview

This is a research assistant agent built using LangGraph and LangSmith that helps users conduct academic research, literature reviews, and paper searches. The system uses a multi-agent architecture with specialized tools for academic paper search, web search, and content analysis. The application features a Streamlit frontend for user interaction and comprehensive evaluation capabilities through LangSmith.

## System Architecture

The system follows a modular architecture with clear separation of concerns:

1. **Frontend Layer**: Streamlit web application (`app.py`) providing an intuitive chat interface
2. **Agent Layer**: LangGraph-based research agent (`research_agent.py`) with stateful conversation management
3. **Tools Layer**: Specialized research tools (`tools.py`) for academic paper search and analysis
4. **Evaluation Layer**: Comprehensive evaluation framework (`langsmith_evaluation.py`, `evaluators.py`) for agent performance assessment
5. **Utilities Layer**: Supporting utilities (`utils.py`) for environment setup and data formatting

## Key Components

### Research Agent (`research_agent.py`)
- **Purpose**: Core conversational agent using LangGraph for structured research workflows
- **Technologies**: LangGraph, OpenAI GPT-4o, LangChain
- **Features**: 
  - Stateful conversation management with memory persistence
  - Multi-step research workflows
  - Tool integration for academic research
  - Configurable threading for session management

### Research Tools (`tools.py`)
- **ArXiv Search**: Academic paper search with metadata extraction
- **Web Search**: General web search using TAVILY (upgraded from DuckDuckGo for better search quality)
- **Content Analysis**: Paper content extraction and analysis
- **Citation Extraction**: Academic citation parsing and formatting

### Evaluation System
- **LangSmith Integration**: Comprehensive evaluation framework with multiple evaluators
- **Evaluation Metrics**: Research accuracy, source credibility, conversation helpfulness, research depth, citation quality
- **Dataset Generation**: Synthetic dataset creation for systematic evaluation (`evaluation_dataset.py`)

### Frontend (`app.py`)
- **Streamlit Interface**: Clean, responsive web interface
- **Session Management**: Persistent conversation state
- **Configuration Panel**: API key status and agent configuration
- **Real-time Chat**: Interactive research assistance

## Data Flow

1. **User Input**: User submits research question through Streamlit interface
2. **Agent Processing**: Research agent processes query using LangGraph workflow
3. **Tool Execution**: Agent selects and executes appropriate research tools (ArXiv, web search, etc.)
4. **Response Generation**: Agent synthesizes findings into comprehensive response
5. **Memory Storage**: Conversation state persisted for multi-turn interactions
6. **Evaluation**: Optional evaluation of responses using LangSmith evaluators

## External Dependencies

### Required APIs
- **OpenAI API**: For GPT-4o language model access
- **LangSmith API**: For evaluation and monitoring
- **ArXiv API**: For academic paper search
- **TAVILY API**: For enhanced web search capabilities (upgraded from DuckDuckGo)

### Key Libraries
- **LangGraph**: For agent workflow orchestration
- **LangChain**: For LLM integration and tool management
- **Streamlit**: For web interface
- **LangSmith**: For evaluation and monitoring
- **Trafilatura**: For web content extraction
- **ArXiv Python Client**: For academic paper access

### Environment Setup
- Set `OPENAI_API_KEY` for language model access
- Set `LANGSMITH_API_KEY` for evaluation and monitoring
- Set `TAVILY_API_KEY` for enhanced web search capabilities
- Set `LANGSMITH_TRACING` to enable tracing
- Set `LANGSMITH_PROJECT` for organized evaluation tracking

The system prioritizes simplicity and effectiveness, using established patterns and well-documented libraries to ensure maintainability and extensibility.
