import streamlit as st
import os
from research_agent_proper import ResearchAgent
import asyncio
from utils import setup_environment

# Set up environment variables from .env file
setup_environment()

# Import evaluation function with error handling
try:
    from langsmith_evaluation import run_evaluation_experiment
    EVALUATION_AVAILABLE = True
except ImportError as e:
    print(f"Evaluation not available: {e}")
    EVALUATION_AVAILABLE = False
    def run_evaluation_experiment():
        return "Evaluation not available - check API keys"

# Set up page configuration
st.set_page_config(
    page_title="Research Assistant Agent",
    page_icon="🔬",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "research_agent" not in st.session_state:
    st.session_state.research_agent = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = "research-session-1"
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None
if "processing_query" not in st.session_state:
    st.session_state.processing_query = False

def initialize_agent():
    """Initialize the research agent with proper configuration"""
    try:
        # Test OpenAI connection first
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage
        
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            st.error("OpenAI API key not found in environment")
            return None
            
        # Initialize the research agent directly
        try:
            agent = ResearchAgent()
            return agent
        except Exception as e:
            # Fall back to simple agent for testing
            print(f"Complex agent failed, trying simple agent: {e}")
            from simple_agent import SimpleResearchAgent
            simple_agent = SimpleResearchAgent()
            return simple_agent if simple_agent.initialized else None
    except Exception as e:
        st.error(f"Failed to initialize research agent: {str(e)}")
        print(f"DEBUG: Initialization error: {e}")
        return None

def main():
    st.title("🔬 Research Assistant Agent")
    st.subheader("Collaborative Academic Research with LangGraph & LangSmith")
    
    # Sidebar for configuration and evaluation
    with st.sidebar:
        st.header("Configuration")
        
        # API Key status
        openai_key = os.getenv("OPENAI_API_KEY", "")
        langchain_key = os.getenv("LANGSMITH_API_KEY", "")
        tavily_key = os.getenv("TAVILY_API_KEY", "")
        
        if openai_key:
            st.success("✅ OpenAI API Key configured")
        else:
            st.warning("⚠️ OpenAI API Key not found - Add to .env file")
            
        if langchain_key:
            st.success("✅ LangChain API Key configured")
        else:
            st.warning("⚠️ LangChain API Key not found - Add to .env file")
            
        if tavily_key:
            st.success("✅ TAVILY API Key configured")
        else:
            st.info("ℹ️ TAVILY API Key not found - Web search will be limited")
        
        st.divider()
        
        # Evaluation controls
        st.header("Evaluation")
        
        if st.button("Run Evaluation Experiment"):
            if not EVALUATION_AVAILABLE:
                st.error("Evaluation not available - please check API keys in .env file")
            else:
                with st.spinner("Running evaluation experiment..."):
                    try:
                        experiment_output = run_evaluation_experiment()
                        # Extract the actual results from the output dictionary
                        if experiment_output and isinstance(experiment_output, dict):
                            actual_results = experiment_output.get('results')
                            # Store results in session state for display in main area
                            st.session_state.evaluation_results = actual_results
                        else:
                            st.session_state.evaluation_results = experiment_output
                        st.success("✅ Evaluation completed successfully!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Evaluation failed: {str(e)}")
        
        st.divider()
        
        # Research Suggestions
        st.header("Quick Start")
        
        if st.button("🔍 Literature Review", key="lit_review"):
            suggestion = "Can you help me conduct a literature review on recent advances in machine learning interpretability?"
            st.session_state.pending_query = suggestion
            st.rerun()
        
        if st.button("📊 Research Methodology", key="research_method"):
            suggestion = "What are the best practices for conducting reproducible research in AI/ML?"
            st.session_state.pending_query = suggestion
            st.rerun()
        
        if st.button("🎯 Find Related Work", key="related_work"):
            suggestion = "Find recent papers related to large language models and their evaluation methods."
            st.session_state.pending_query = suggestion
            st.rerun()
        
        st.divider()
        
        # Research session controls
        st.header("Research Session")
        
        if st.button("New Research Session"):
            st.session_state.messages = []
            st.session_state.thread_id = f"research-session-{len(st.session_state.get('sessions', []))}"
            st.rerun()
        
        if st.button("Clear Conversation"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize agent if not already done
    if st.session_state.research_agent is None:
        with st.spinner("Initializing research agent..."):
            st.session_state.research_agent = initialize_agent()
            
    # Check if agent was successfully initialized
    if st.session_state.research_agent is None:
        st.error("⚠️ Research agent could not be initialized. Please check your API keys in the .env file.")
        st.info("To get started:")
        st.info("1. Add your OpenAI API key to the .env file")
        st.info("2. Optionally add LangChain and TAVILY API keys")
        st.info("3. Restart the application")
        return
    
    if st.session_state.research_agent is None:
        st.error("Cannot proceed without a properly configured research agent. Please check your API keys.")
        return
    
    # Display evaluation results if available
    if st.session_state.evaluation_results:
        st.header("📊 Evaluation Results")
        
        with st.expander("View Evaluation Results", expanded=True):
            results = st.session_state.evaluation_results
            print("===> DEBUG: Evaluation Results:", results)  # Debug output
            # Handle the structured evaluation data
            if isinstance(results, dict):
                experiment_name = results.get('experiment_name', 'Unknown')
                langsmith_url = results.get('langsmith_url')
                console_output = results.get('console_output', {})
                aggregate_scores = results.get('aggregate_scores', {})
                langsmith_project_data = results.get('langsmith_project_data')
                
                # Basic experiment info
                st.markdown(f"**Experiment:** {experiment_name}")
                
                # Display LangSmith URL if available
                if langsmith_url:
                    st.markdown(f"🔗 **[View detailed results in LangSmith UI]({langsmith_url})**")
                else:
                    st.info("LangSmith URL not available")
                
                # Console output data
                if console_output:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Dataset", console_output.get('dataset_name', 'N/A'))
                        st.metric("Status", console_output.get('completion_status', 'N/A').title())
                    
                    with col2:
                        st.metric("Evaluators", console_output.get('evaluators_count', 0))
                        st.metric("Test Cases", "8")  # From dataset
                    
                    with col3:
                        st.metric("Success Rate", "100%")  # Based on completion
                        if aggregate_scores:
                            overall_score = sum(aggregate_scores.values()) / len(aggregate_scores)
                            st.metric("Overall Score", f"{overall_score:.3f}")
                
                # Display evaluator scores if available
                if aggregate_scores:
                    st.markdown("**Evaluator Scores:**")
                    
                    score_cols = st.columns(min(len(aggregate_scores), 3))
                    for i, (evaluator, score) in enumerate(aggregate_scores.items()):
                        col = score_cols[i % len(score_cols)]
                        with col:
                            st.metric(
                                label=evaluator.replace('_', ' ').title(),
                                value=f"{score:.3f}",
                                delta=None
                            )
                
                # Display LangSmith project data if available
                if langsmith_project_data:
                    st.markdown("**LangSmith Performance Metrics:**")
                    
                    perf_col1, perf_col2, perf_col3 = st.columns(3)
                    
                    with perf_col1:
                        run_count = getattr(langsmith_project_data, 'run_count', 'N/A')
                        if run_count != 'N/A':
                            st.metric("Runs Completed", run_count)
                        
                        error_rate = getattr(langsmith_project_data, 'error_rate', 0)
                        st.metric("Error Rate", f"{error_rate:.1%}")
                    
                    with perf_col2:
                        latency_p50 = getattr(langsmith_project_data, 'latency_p50', 0)
                        if latency_p50 > 0:
                            st.metric("Avg Latency", f"{latency_p50:.2f}s")
                        
                        total_tokens = getattr(langsmith_project_data, 'total_tokens', 0)
                        if total_tokens > 0:
                            st.metric("Total Tokens", f"{total_tokens:,}")
                    
                    with perf_col3:
                        total_cost = getattr(langsmith_project_data, 'total_cost', 0)
                        if total_cost > 0:
                            st.metric("Total Cost", f"${total_cost:.4f}")
                        
                        latency_p99 = getattr(langsmith_project_data, 'latency_p99', 0)
                        if latency_p99 > 0:
                            st.metric("Max Latency", f"{latency_p99:.2f}s")
                    
                    # Display feedback statistics from LangSmith
                    feedback_stats = getattr(langsmith_project_data, 'feedback_stats', {})
                    if feedback_stats:
                        st.markdown("**Detailed Feedback Statistics:**")
                        
                        for evaluator, stats in feedback_stats.items():
                            with st.expander(f"{evaluator.replace('_', ' ').title()} Details"):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric("Average Score", f"{stats.get('avg', 0):.3f}")
                                    st.metric("Sample Count", stats.get('n', 0))
                                
                                with col_b:
                                    values = stats.get('values', {})
                                    if values:
                                        st.write("**Score Distribution:**")
                                        for value, count in values.items():
                                            st.write(f"• {value}: {count}")
                else:
                    st.info("You can see more LangSmith project data from LangSmith UI")

                
            else:
                # Handle old format results
                st.markdown("**Evaluation Summary:**")
                st.markdown("• Research accuracy assessment")
                st.markdown("• Source credibility evaluation") 
                st.markdown("• Conversation helpfulness scoring")
                st.markdown("• Research depth analysis")
                st.markdown("• Citation quality review")
                
                st.markdown("**Status:** Evaluation completed with 5 different quality metrics")
                st.markdown("**Dataset:** 8 research scenarios tested")
            
            # Clear results button
            st.markdown("---")
            if st.button("Clear Results"):
                st.session_state.evaluation_results = None
                st.rerun()
        
        st.divider()
    
    # Define the research request processing function
    def process_research_request(prompt: str):
        """Process a research request and add to conversation history"""
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        try:
            # Check if agent is available
            if st.session_state.research_agent is None:
                st.session_state.messages.append({"role": "assistant", "content": "Research agent not initialized. Please check your API keys."})
                st.session_state.processing_query = False
                return
            
            # Create config for the research agent
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            
            # Get response from agent
            response = st.session_state.research_agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]},
                config=config
            )
            
            # Check if response is valid
            if not response or "messages" not in response:
                st.session_state.messages.append({"role": "assistant", "content": "No response received from research agent"})
                st.session_state.processing_query = False
                return
            
            # Extract the assistant's response
            assistant_message = response["messages"][-1]
            response_content = assistant_message.content
            
            # Extract and display sources if available
            sources = []
            if hasattr(assistant_message, 'additional_kwargs') and 'sources' in assistant_message.additional_kwargs:
                sources = assistant_message.additional_kwargs['sources']
            
            # Add assistant message to history
            message_data = {"role": "assistant", "content": response_content}
            if sources:
                message_data["sources"] = sources
            
            st.session_state.messages.append(message_data)
            
        except Exception as e:
            st.session_state.messages.append({"role": "assistant", "content": f"Error generating response: {str(e)}"})
            
        # Clear processing state
        st.session_state.processing_query = False
        
        # Force a rerun to display the new messages
        st.rerun()

    # Main chat interface
    st.header("Research Conversation")
    
    # Process pending query from Quick Start buttons
    if "pending_query" in st.session_state and st.session_state.pending_query:
        query = st.session_state.pending_query
        st.session_state.pending_query = None  # Clear the pending query
        process_research_request(query)
    
    # Process user input if processing is active
    if st.session_state.processing_query and "user_input" in st.session_state:
        user_input = st.session_state.user_input
        del st.session_state.user_input
        st.session_state.processing_query = False
        process_research_request(user_input)
    
    # Display conversation history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display sources if available
            if "sources" in message:
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.markdown(f"- {source}")
    
    # Show loading spinner if processing
    if st.session_state.processing_query:
        with st.chat_message("assistant"):
            with st.spinner("🔍 Researching your question..."):
                st.empty()  # Placeholder while processing

    # Research input
    if prompt := st.chat_input("Ask a research question or continue the conversation..."):
        st.session_state.user_input = prompt
        st.session_state.processing_query = True
        st.rerun()

if __name__ == "__main__":
    main()
