import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

class SimpleResearchAgent:
    def __init__(self):
        """Initialize simple research agent"""
        try:
            self.model = ChatOpenAI(
                model="gpt-4o",
                temperature=0.1,
                api_key=os.getenv("OPENAI_API_KEY")
            )
            self.initialized = True
        except Exception as e:
            print(f"Failed to initialize simple agent: {e}")
            self.initialized = False
    
    def invoke(self, input_data, config=None):
        """Simple invoke method"""
        if not self.initialized:
            return {
                "messages": [AIMessage(content="Agent not initialized - check API keys")]
            }
        
        try:
            # Get the user message
            user_msg = input_data["messages"][-1]
            if isinstance(user_msg, dict):
                content = user_msg["content"]
            else:
                content = user_msg.content
            
            # Call OpenAI directly
            response = self.model.invoke([HumanMessage(content=f"You are a research assistant. Answer this question: {content}")])
            
            return {
                "messages": [response]
            }
        except Exception as e:
            return {
                "messages": [AIMessage(content=f"Error: {str(e)}")]
            }