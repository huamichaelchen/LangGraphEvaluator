import os
from typing import Dict, Any, List, Optional, Annotated
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
import json

from tools import arxiv_search, web_search, analyze_paper_content, extract_citations

class State(TypedDict):
    messages: Annotated[list, add_messages]

class BasicToolNode:
    """A node that runs the tools requested in the last AIMessage."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            try:
                tool_result = self.tools_by_name[tool_call["name"]].invoke(
                    tool_call["args"]
                )
                outputs.append(
                    ToolMessage(
                        content=json.dumps(tool_result),
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
            except Exception as e:
                outputs.append(
                    ToolMessage(
                        content=f"Error using tool {tool_call['name']}: {str(e)}",
                        name=tool_call["name"],
                        tool_call_id=tool_call["id"],
                    )
                )
        return {"messages": outputs}

class ResearchAgent:
    def __init__(self):
        """Initialize the research agent with LangGraph workflow following official patterns"""
        self.model = ChatOpenAI(
            model="gpt-4o",  # the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
            temperature=0.1,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Set up tools
        self.tools = [
            arxiv_search,
            web_search,
            analyze_paper_content,
            extract_citations
        ]
        
        self.memory = MemorySaver()
        self.graph = self._create_workflow()
        
    def _create_workflow(self):
        """Create the LangGraph workflow for research agent following official patterns"""
        
        # Create state graph
        graph_builder = StateGraph(State)
        
        # Create model with tools
        llm_with_tools = self.model.bind_tools(self.tools)
        
        def chatbot(state: State):
            """Main chatbot node that processes messages and potentially calls tools"""
            # Add research context to system if needed
            messages = state["messages"]
            
            # Add research assistant system message if it's the first interaction
            if not messages or not any(
                isinstance(msg, AIMessage) and "research assistant" in msg.content.lower() 
                for msg in messages
            ):
                system_context = AIMessage(content="""I am a research assistant specializing in academic research and literature reviews. 
                
I help with:
- Finding relevant academic papers and research articles
- Analyzing research content and methodologies  
- Extracting key insights and citations
- Providing comprehensive literature reviews
- Guiding research methodology decisions

I have access to tools for searching ArXiv, web search, content analysis, and citation extraction. I always provide detailed, well-sourced responses with proper citations.""")
                messages = [system_context] + messages
            
            return {"messages": [llm_with_tools.invoke(messages)]}
        
        def route_tools(state: State):
            """Route to tools if the last message has tool calls, otherwise end"""
            if isinstance(state, list):
                ai_message = state[-1]
            elif messages := state.get("messages", []):
                ai_message = messages[-1]
            else:
                raise ValueError(f"No messages found in input state: {state}")
            
            if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
                return "tools"
            return END
        
        # Create tool node
        tool_node = BasicToolNode(tools=self.tools)
        
        # Add nodes to graph
        graph_builder.add_node("chatbot", chatbot)
        graph_builder.add_node("tools", tool_node)
        
        # Add edges
        graph_builder.add_edge(START, "chatbot")
        graph_builder.add_conditional_edges(
            "chatbot",
            route_tools,
            {"tools": "tools", END: END},
        )
        graph_builder.add_edge("tools", "chatbot")
        
        # Compile the graph with memory
        return graph_builder.compile(checkpointer=self.memory)
    
    def invoke(self, input_data: Dict[str, Any], config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Invoke the research agent with input data"""
        try:
            # Set default config if none provided
            if config is None:
                config = {"configurable": {"thread_id": "default"}}
            
            # Convert dict messages to proper message objects if needed
            if "messages" in input_data:
                messages = []
                for msg in input_data["messages"]:
                    if isinstance(msg, dict):
                        if msg["role"] == "user":
                            messages.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            messages.append(AIMessage(content=msg["content"]))
                    else:
                        messages.append(msg)
                input_data = {"messages": messages}
            
            # Add recursion limit to prevent infinite loops
            config["recursion_limit"] = 20
            
            # Invoke the graph
            result = self.graph.invoke(input_data, config=config)
            return result
            
        except Exception as e:
            print(f"Research agent error: {e}")
            return {
                "messages": [AIMessage(content=f"I encountered an error: {str(e)}. Let me provide a direct response based on my knowledge instead.")]
            }
    
    def stream(self, input_data: Dict[str, Any], config: Dict[str, Any] = None):
        """Stream the research agent response"""
        try:
            # Set default config if none provided
            if config is None:
                config = {"configurable": {"thread_id": "default"}}
            
            # Convert dict messages to proper message objects if needed
            if "messages" in input_data:
                messages = []
                for msg in input_data["messages"]:
                    if isinstance(msg, dict):
                        if msg["role"] == "user":
                            messages.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            messages.append(AIMessage(content=msg["content"]))
                    else:
                        messages.append(msg)
                input_data = {"messages": messages}
            
            # Add recursion limit to prevent infinite loops
            config["recursion_limit"] = 20
            
            # Stream the graph
            for chunk in self.graph.stream(input_data, config=config):
                yield chunk
                
        except Exception as e:
            print(f"Research agent streaming error: {e}")
            yield {
                "messages": [AIMessage(content=f"I encountered an error: {str(e)}. Let me provide a direct response based on my knowledge instead.")]
            }
    
    def get_conversation_history(self, thread_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a specific thread"""
        try:
            config = {"configurable": {"thread_id": thread_id}}
            state = self.graph.get_state(config)
            
            if state and "messages" in state.values:
                return [
                    {
                        "role": "user" if isinstance(msg, HumanMessage) else "assistant",
                        "content": msg.content
                    }
                    for msg in state.values["messages"]
                ]
            return []
        except Exception as e:
            print(f"Error retrieving conversation history: {e}")
            return []