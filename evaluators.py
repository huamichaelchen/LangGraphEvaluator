import json
import re
from typing import Dict, Any, List, Optional
from langchain_core.messages import HumanMessage, AIMessage
from langsmith.evaluation import RunEvaluator
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
def get_model():
    """Get ChatOpenAI model with proper error handling"""
    try:
        return ChatOpenAI(model="gpt-4o", temperature=0.1)
    except Exception as e:
        print(f"Warning: Could not initialize OpenAI model: {e}")
        return None

model = get_model()

def research_accuracy_evaluator(run, example):
    """
    Evaluate the accuracy of research information provided by the agent.
    
    Checks for:
    - Factual correctness of research claims
    - Proper use of academic terminology
    - Logical consistency in explanations
    """
    try:
        # Extract the assistant's response
        if run.outputs and "messages" in run.outputs:
            assistant_response = run.outputs["messages"][-1].content
        else:
            return {"key": "research_accuracy", "score": 0.0, "comment": "No response found"}
        
        # Get the expected response for comparison
        expected_response = example.outputs.get("answer", "")
        
        # Use GPT-4o to evaluate research accuracy
        evaluation_prompt = f"""
        You are an expert research evaluator. Evaluate the accuracy of the following research response.
        
        User Question: {example.inputs.get("question", "")}
        
        Assistant Response: {assistant_response}
        
        Expected/Reference Response: {expected_response}
        
        Evaluate the response on a scale of 0-1 based on:
        1. Factual accuracy of research claims
        2. Proper use of academic terminology
        3. Logical consistency
        4. Completeness of information
        
        Provide your evaluation as JSON with 'score' (0-1) and 'reasoning' fields.
        """
        
        if model is None:
            return {"key": "research_accuracy", "score": 0.5, "comment": "Model not available"}
        
        evaluation = model.invoke([{"role": "user", "content": evaluation_prompt}])
        
        try:
            result = json.loads(evaluation.content)
            score = max(0.0, min(1.0, float(result.get("score", 0.5))))
            reasoning = result.get("reasoning", "No reasoning provided")
        except:
            # Fallback scoring based on simple heuristics
            score = 0.5
            reasoning = "Could not parse evaluation result"
        
        return {
            "key": "research_accuracy",
            "score": score,
            "comment": reasoning
        }
        
    except Exception as e:
        return {
            "key": "research_accuracy",
            "score": 0.0,
            "comment": f"Error in evaluation: {str(e)}"
        }

def source_credibility_evaluator(run, example):
    """
    Evaluate the credibility and appropriateness of sources cited by the agent.
    
    Checks for:
    - Use of peer-reviewed sources
    - Appropriate citation format
    - Relevance of sources to the query
    """
    try:
        # Extract the assistant's response
        if run.outputs and "messages" in run.outputs:
            assistant_response = run.outputs["messages"][-1].content
        else:
            return {"key": "source_credibility", "score": 0.0, "comment": "No response found"}
        
        # Check for common academic source indicators
        credible_indicators = [
            r'arxiv\.org',
            r'doi:',
            r'https?://[^/]*\.edu',
            r'https?://[^/]*\.ac\.',
            r'ieee\.org',
            r'acm\.org',
            r'springer\.com',
            r'nature\.com',
            r'science\.org'
        ]
        
        source_count = 0
        credible_source_count = 0
        
        # Count total sources (URLs and citations)
        url_pattern = r'https?://[^\s)]+'
        citation_pattern = r'\[[\d,\s]+\]|\([^)]+\d{4}[^)]*\)'
        
        urls = re.findall(url_pattern, assistant_response)
        citations = re.findall(citation_pattern, assistant_response)
        
        source_count = len(urls) + len(citations)
        
        # Count credible sources
        for url in urls:
            if any(re.search(pattern, url, re.IGNORECASE) for pattern in credible_indicators):
                credible_source_count += 1
        
        # Score based on credible source ratio
        if source_count == 0:
            score = 0.0
            comment = "No sources cited"
        else:
            score = credible_source_count / source_count
            comment = f"Found {credible_source_count} credible sources out of {source_count} total sources"
        
        return {
            "key": "source_credibility",
            "score": score,
            "comment": comment
        }
        
    except Exception as e:
        return {
            "key": "source_credibility",
            "score": 0.0,
            "comment": f"Error in evaluation: {str(e)}"
        }

def conversation_helpfulness_evaluator(run, example):
    """
    Evaluate how helpful the agent's response is for research purposes.
    
    Checks for:
    - Relevance to the research question
    - Depth of analysis
    - Actionable insights
    - Follow-up guidance
    """
    try:
        # Extract the assistant's response
        if run.outputs and "messages" in run.outputs:
            assistant_response = run.outputs["messages"][-1].content
        else:
            return {"key": "conversation_helpfulness", "score": 0.0, "comment": "No response found"}
        
        user_question = example.inputs.get("question", "")
        
        # Use GPT-4o to evaluate helpfulness
        evaluation_prompt = f"""
        You are an expert research assistant evaluator. Evaluate how helpful the following response is for research purposes.
        
        User Question: {user_question}
        
        Assistant Response: {assistant_response}
        
        Evaluate the response on a scale of 0-1 based on:
        1. Relevance to the research question
        2. Depth of analysis and insights
        3. Actionable information provided
        4. Clear guidance for next steps
        5. Comprehensiveness of coverage
        
        Provide your evaluation as JSON with 'score' (0-1) and 'reasoning' fields.
        """
        
        if model is None:
            return {"key": "conversation_helpfulness", "score": 0.5, "comment": "Model not available"}
        
        evaluation = model.invoke([{"role": "user", "content": evaluation_prompt}])
        
        try:
            result = json.loads(evaluation.content)
            score = max(0.0, min(1.0, float(result.get("score", 0.5))))
            reasoning = result.get("reasoning", "No reasoning provided")
        except:
            # Fallback scoring based on response length and structure
            score = min(1.0, len(assistant_response) / 1000)  # Simple length-based scoring
            reasoning = "Fallback scoring based on response completeness"
        
        return {
            "key": "conversation_helpfulness",
            "score": score,
            "comment": reasoning
        }
        
    except Exception as e:
        return {
            "key": "conversation_helpfulness",
            "score": 0.0,
            "comment": f"Error in evaluation: {str(e)}"
        }

def research_depth_evaluator(run, example):
    """
    Evaluate the depth and thoroughness of the research provided.
    
    Checks for:
    - Multiple perspectives considered
    - Comprehensive coverage of the topic
    - Critical analysis of sources
    - Identification of research gaps
    """
    try:
        # Extract the assistant's response
        if run.outputs and "messages" in run.outputs:
            assistant_response = run.outputs["messages"][-1].content
        else:
            return {"key": "research_depth", "score": 0.0, "comment": "No response found"}
        
        # Analyze depth indicators
        depth_indicators = {
            "multiple_perspectives": [
                r"however,", r"on the other hand", r"alternatively", r"in contrast",
                r"different approach", r"another perspective", r"various viewpoints"
            ],
            "critical_analysis": [
                r"strengths", r"weaknesses", r"limitations", r"advantages", r"disadvantages",
                r"critique", r"evaluation", r"assessment", r"analysis"
            ],
            "comprehensive_coverage": [
                r"furthermore", r"additionally", r"moreover", r"in addition",
                r"also", r"besides", r"comprehensive", r"thorough"
            ],
            "research_gaps": [
                r"gap", r"limitation", r"future work", r"further research",
                r"unexplored", r"needs investigation", r"requires study"
            ]
        }
        
        depth_score = 0.0
        depth_comments = []
        
        for category, patterns in depth_indicators.items():
            matches = sum(1 for pattern in patterns 
                         if re.search(pattern, assistant_response, re.IGNORECASE))
            if matches > 0:
                category_score = min(1.0, matches / 3)  # Normalize to 0-1
                depth_score += category_score
                depth_comments.append(f"{category}: {matches} indicators found")
        
        # Normalize final score
        final_score = depth_score / len(depth_indicators)
        
        return {
            "key": "research_depth",
            "score": final_score,
            "comment": "; ".join(depth_comments) if depth_comments else "Limited depth indicators found"
        }
        
    except Exception as e:
        return {
            "key": "research_depth",
            "score": 0.0,
            "comment": f"Error in evaluation: {str(e)}"
        }

def citation_quality_evaluator(run, example):
    """
    Evaluate the quality and format of citations provided.
    
    Checks for:
    - Proper citation format
    - Complete citation information
    - Appropriate citation placement
    """
    try:
        # Extract the assistant's response
        if run.outputs and "messages" in run.outputs:
            assistant_response = run.outputs["messages"][-1].content
        else:
            return {"key": "citation_quality", "score": 0.0, "comment": "No response found"}
        
        # Citation format patterns
        citation_patterns = {
            "academic_style": r'\([^)]+\d{4}[^)]*\)',  # (Author, Year)
            "numbered": r'\[[\d,\s]+\]',  # [1, 2, 3]
            "url_citations": r'https?://[^\s)]+',  # URLs
            "doi": r'doi:\s*[^\s]+',  # DOI references
            "arxiv": r'arXiv:\s*[^\s]+'  # arXiv references
        }
        
        total_citations = 0
        quality_score = 0.0
        format_comments = []
        
        for format_name, pattern in citation_patterns.items():
            matches = re.findall(pattern, assistant_response, re.IGNORECASE)
            if matches:
                total_citations += len(matches)
                # Higher score for academic formats
                if format_name in ["academic_style", "doi", "arxiv"]:
                    quality_score += len(matches) * 1.0
                else:
                    quality_score += len(matches) * 0.5
                
                format_comments.append(f"{format_name}: {len(matches)} citations")
        
        if total_citations == 0:
            return {
                "key": "citation_quality",
                "score": 0.0,
                "comment": "No citations found"
            }
        
        # Normalize score
        normalized_score = min(1.0, quality_score / (total_citations * 1.0))
        
        return {
            "key": "citation_quality",
            "score": normalized_score,
            "comment": f"Total citations: {total_citations}; " + "; ".join(format_comments)
        }
        
    except Exception as e:
        return {
            "key": "citation_quality",
            "score": 0.0,
            "comment": f"Error in evaluation: {str(e)}"
        }

# Summary evaluator for experiment-level metrics
def research_collaboration_summary_evaluator(runs, examples):
    """
    Evaluate the overall quality of a research collaboration session.
    
    Provides experiment-level metrics for:
    - Average research quality across turns
    - Conversation coherence
    - Research progression
    """
    try:
        if not runs or not examples:
            return {"research_collaboration_quality": 0.0}
        
        # Calculate average scores from individual evaluations
        total_accuracy = 0.0
        total_helpfulness = 0.0
        total_depth = 0.0
        valid_runs = 0
        
        for run in runs:
            if hasattr(run, 'feedback') and run.feedback:
                accuracy = next((f.score for f in run.feedback if f.key == "research_accuracy"), 0.0)
                helpfulness = next((f.score for f in run.feedback if f.key == "conversation_helpfulness"), 0.0)
                depth = next((f.score for f in run.feedback if f.key == "research_depth"), 0.0)
                
                total_accuracy += accuracy
                total_helpfulness += helpfulness
                total_depth += depth
                valid_runs += 1
        
        if valid_runs == 0:
            return {"research_collaboration_quality": 0.0}
        
        # Calculate overall research collaboration quality
        avg_accuracy = total_accuracy / valid_runs
        avg_helpfulness = total_helpfulness / valid_runs
        avg_depth = total_depth / valid_runs
        
        # Weighted average (accuracy and depth are more important)
        collaboration_quality = (avg_accuracy * 0.4 + avg_helpfulness * 0.3 + avg_depth * 0.3)
        
        return {
            "research_collaboration_quality": collaboration_quality,
            "average_accuracy": avg_accuracy,
            "average_helpfulness": avg_helpfulness,
            "average_depth": avg_depth,
            "total_conversations": valid_runs
        }
        
    except Exception as e:
        return {"research_collaboration_quality": 0.0, "error": str(e)}
