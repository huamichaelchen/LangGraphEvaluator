import os
from typing import Dict, List, Any, Annotated
from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from tools import arxiv_search, web_search, analyze_paper_content, extract_citations
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class ResearchAgent:
    def __init__(self):
        """Initialize the research agent with LangGraph workflow"""
        # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # do not change this unless explicitly requested by the user
        try:
            self.model = ChatOpenAI(
                model="gpt-4o",
                temperature=0.1,
                api_key=os.getenv("OPENAI_API_KEY")
            )
        except Exception as e:
            print(f"Warning: Could not initialize OpenAI model: {e}")
            self.model = None
        
        # Define tools
        self.tools = [
            arxiv_search,
            web_search,
            analyze_paper_content,
            extract_citations
        ]
        
        # Create the workflow
        if self.model is not None:
            self.workflow = self._create_workflow()
            
            # Add memory for conversation persistence
            self.memory = MemorySaver()
            
            # Compile the graph
            self.graph = self.workflow.compile(checkpointer=self.memory)
        else:
            self.workflow = None
            self.memory = None
            self.graph = None
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow for research agent"""
        
        # Define the research assistant prompt
        research_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a knowledgeable research assistant specializing in academic research and literature review.
            
Your responsibilities:
1. Search for relevant academic papers and research content
2. Analyze and synthesize research findings
3. Provide accurate citations and source attribution
4. Maintain conversation context across multiple turns
5. Guide users through iterative research refinement
6. Ensure research depth and accuracy

Available tools:
- arxiv_search: Search for academic papers on arXiv
- web_search: Search the web for research content
- analyze_paper_content: Analyze paper content and extract key insights
- extract_citations: Extract and format citations from papers

Guidelines:
- Always provide proper citations for your sources
- Be thorough in your research approach
- Ask clarifying questions when needed
- Build on previous conversation context
- Prioritize credible academic sources
- Explain your research methodology when helpful
- Use tools sparingly - typically 1-3 searches per response
- Stop tool usage when you have sufficient information
- Provide direct answers based on your knowledge when appropriate"""),
            ("placeholder", "{messages}")
        ])
        
        # Create the workflow graph
        workflow = StateGraph(MessagesState)
        
        # Add nodes
        workflow.add_node("agent", self._create_agent_node(research_prompt))
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Add edges - removed the circular edge that may cause issues
        workflow.add_edge("tools", "agent")
        
        # Set conditional edges
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {
                "tools": "tools",
                "__end__": "__end__"
            }
        )
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        return workflow
    
    def _create_agent_node(self, prompt: ChatPromptTemplate):
        """Create the agent node with model and tools"""
        def agent_node(state: MessagesState) -> Dict[str, Any]:
            # Bind tools to the model
            model_with_tools = self.model.bind_tools(self.tools)
            
            # Create the chain
            chain = prompt | model_with_tools
            
            # Invoke the chain
            response = chain.invoke(state)
            
            # Return the response in the expected format
            return {"messages": [response]}
        
        return agent_node
    
    def invoke(self, input_data: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Invoke the research agent with input data"""
        if self.graph is None:
            return {
                "messages": [AIMessage(content="Research agent is not properly initialized. Please check your API keys in the .env file.")]
            }
        
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        
        # Ensure messages are in the correct format
        if "messages" in input_data:
            messages = []
            for msg in input_data["messages"]:
                if isinstance(msg, dict):
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
                    elif msg["role"] == "system":
                        messages.append(SystemMessage(content=msg["content"]))
                else:
                    messages.append(msg)
            
            input_data["messages"] = messages
        
        # Add recursion limit and timeout to prevent infinite loops
        config["recursion_limit"] = 25  # Increased limit for proper functioning
        config["timeout"] = 60  # 60 second timeout
        
        try:
            # Invoke the graph
            result = self.graph.invoke(input_data, config=config)
            return result
        except Exception as e:
            error_message = str(e).lower()
            print(f"DEBUG: Error in research agent: {str(e)}")
            # Handle various error types gracefully
            if "recursion limit" in error_message:
                return {
                    "messages": [AIMessage(content="I've reached the complexity limit for this research query. Based on my knowledge and the information I've gathered, let me provide you with a comprehensive response. If you need more specific details, please ask a more focused question.")]
                }
            elif "timeout" in error_message:
                return {
                    "messages": [AIMessage(content="The research process is taking longer than expected. Let me provide you with a direct answer based on my current knowledge. For more detailed research, please try breaking your question into smaller parts.")]
                }
            else:
                return {
                    "messages": [AIMessage(content="I encountered an issue while processing your research request. Let me provide a direct response based on my knowledge. If you need specific research assistance, please rephrase your question.")]
                }
    
    def stream(self, input_data: Dict[str, Any], config: Dict[str, Any] = None):
        """Stream the research agent response"""
        if config is None:
            config = {"configurable": {"thread_id": "default"}}
        
        # Ensure messages are in the correct format
        if "messages" in input_data:
            messages = []
            for msg in input_data["messages"]:
                if isinstance(msg, dict):
                    if msg["role"] == "user":
                        messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        messages.append(AIMessage(content=msg["content"]))
                    elif msg["role"] == "system":
                        messages.append(SystemMessage(content=msg["content"]))
                else:
                    messages.append(msg)
            
            input_data["messages"] = messages
        
        # Add recursion limit and timeout to prevent infinite loops
        config["recursion_limit"] = 25  # Increased limit for proper functioning
        config["timeout"] = 60  # 60 second timeout
        
        try:
            # Stream the graph
            for chunk in self.graph.stream(input_data, config=config):
                yield chunk
        except Exception as e:
            error_message = str(e).lower()
            # Handle various error types gracefully
            if "recursion limit" in error_message:
                yield {
                    "messages": [AIMessage(content="I've reached the complexity limit for this research query. Based on my knowledge and the information I've gathered, let me provide you with a comprehensive response. If you need more specific details, please ask a more focused question.")]
                }
            elif "timeout" in error_message:
                yield {
                    "messages": [AIMessage(content="The research process is taking longer than expected. Let me provide you with a direct answer based on my current knowledge. For more detailed research, please try breaking your question into smaller parts.")]
                }
            else:
                yield {
                    "messages": [AIMessage(content="I encountered an issue while processing your research request. Let me provide a direct response based on my knowledge. If you need specific research assistance, please rephrase your question.")]
                }
    
    def get_conversation_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a specific thread"""
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Get the current state
            state = self.graph.get_state(config)
            
            if state and "messages" in state.values:
                return [
                    {
                        "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                        "content": msg.content
                    }
                    for msg in state.values["messages"]
                ]
            else:
                return []
        except Exception as e:
            print(f"Error retrieving conversation history: {e}")
            return []
