#!/usr/bin/env python3
"""
Simple test to diagnose research agent issues
"""

import os
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from research_agent import ResearchAgent

def test_openai_connection():
    """Test basic OpenAI connection"""
    try:
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            print("❌ No OpenAI API key found")
            return False
        
        # Test basic OpenAI connection
        model = ChatOpenAI(model="gpt-4o", temperature=0.1)
        response = model.invoke([HumanMessage(content="Hello, reply with just 'OK'")])
        print(f"✅ OpenAI connection successful: {response.content}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
        return False

def test_research_agent():
    """Test research agent initialization"""
    try:
        agent = ResearchAgent()
        if agent.graph is None:
            print("❌ Research agent graph is None")
            return False
        
        print("✅ Research agent initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ Research agent initialization failed: {e}")
        return False

def test_simple_query():
    """Test simple query without tools"""
    try:
        agent = ResearchAgent()
        
        # Test simple query
        response = agent.invoke({
            "messages": [HumanMessage(content="Say hello and do not use any tools")]
        })
        
        if response and "messages" in response:
            print(f"✅ Simple query successful: {response['messages'][-1].content[:100]}...")
            return True
        else:
            print("❌ Simple query failed - no response")
            return False
        
    except Exception as e:
        print(f"❌ Simple query failed: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Testing Research Agent Components...")
    print("=" * 50)
    
    print("1. Testing OpenAI connection...")
    openai_ok = test_openai_connection()
    
    print("\n2. Testing research agent initialization...")
    agent_ok = test_research_agent()
    
    print("\n3. Testing simple query...")
    query_ok = test_simple_query()
    
    print("\n" + "=" * 50)
    if openai_ok and agent_ok and query_ok:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed - check the errors above")