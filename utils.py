import os
import json
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime
import re
from dotenv import load_dotenv

def setup_environment():
    """Set up environment variables for LangSmith and OpenAI"""
    # Load environment variables from .env file
    load_dotenv()
    
    return {
        "langsmith_enabled": bool(os.getenv("LANGSMITH_API_KEY")),
        "openai_enabled": bool(os.getenv("OPENAI_API_KEY")),
        "tavily_enabled": bool(os.getenv("TAVILY_API_KEY")),
        "tracing_enabled": True
    }

def validate_api_keys():
    """Validate that required API keys are present"""
    required_keys = ["LANGSMITH_API_KEY", "OPENAI_API_KEY"]
    optional_keys = ["TAVILY_API_KEY"]
    missing_keys = []
    missing_optional = []
    
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    for key in optional_keys:
        if not os.getenv(key):
            missing_optional.append(key)
    
    if missing_keys:
        print(f"❌ Missing required API keys: {', '.join(missing_keys)}")
        return False
    
    if missing_optional:
        print(f"⚠️  Missing optional API keys: {', '.join(missing_optional)} (web search will be limited)")
    
    print("✅ All required API keys are configured")
    return True

def format_conversation_history(messages: List[Dict[str, Any]]) -> str:
    """Format conversation history for display"""
    formatted = []
    for msg in messages:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        
        if role == "user":
            formatted.append(f"👤 User: {content}")
        elif role == "assistant":
            formatted.append(f"🤖 Assistant: {content}")
        else:
            formatted.append(f"{role}: {content}")
    
    return "\n\n".join(formatted)

def extract_sources_from_response(response: str) -> List[str]:
    """Extract source URLs and citations from response text"""
    sources = []
    
    # URL pattern
    url_pattern = r'https?://[^\s)\]]+(?:\.[^\s)\]]+)*'
    urls = re.findall(url_pattern, response)
    sources.extend(urls)
    
    # Citation patterns
    citation_patterns = [
        r'\[[\d,\s]+\]',  # [1, 2, 3]
        r'\([^)]+\d{4}[^)]*\)',  # (Author, 2024)
        r'doi:\s*[^\s]+',  # DOI references
        r'arXiv:\s*[^\s]+'  # arXiv references
    ]
    
    for pattern in citation_patterns:
        citations = re.findall(pattern, response, re.IGNORECASE)
        sources.extend(citations)
    
    return list(set(sources))  # Remove duplicates

def generate_thread_id(base_name: str = "research") -> str:
    """Generate a unique thread ID for conversations"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_suffix = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:6]
    return f"{base_name}_{timestamp}_{random_suffix}"

def clean_response_text(text: str) -> str:
    """Clean and format response text for better readability"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Normalize line breaks
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Clean up common formatting issues
    text = re.sub(r'^\s*[-*]\s*', '• ', text, flags=re.MULTILINE)
    
    return text

def calculate_response_metrics(response: str) -> Dict[str, Any]:
    """Calculate basic metrics for response quality"""
    metrics = {
        "word_count": len(response.split()),
        "character_count": len(response),
        "sentence_count": len(re.findall(r'[.!?]+', response)),
        "paragraph_count": len(response.split('\n\n')),
        "source_count": len(extract_sources_from_response(response)),
        "has_citations": bool(re.search(r'\[[^\]]+\]|\([^)]+\d{4}[^)]*\)', response)),
        "has_urls": bool(re.search(r'https?://', response)),
        "readability_score": min(100, max(0, 100 - (len(response.split()) / 10)))  # Simple readability heuristic
    }
    
    return metrics

def validate_research_response(response: str) -> Dict[str, Any]:
    """Validate research response quality"""
    validation = {
        "is_valid": True,
        "issues": [],
        "suggestions": []
    }
    
    # Check minimum length
    if len(response) < 100:
        validation["is_valid"] = False
        validation["issues"].append("Response too short for research quality")
        validation["suggestions"].append("Provide more comprehensive analysis")
    
    # Check for sources
    if not extract_sources_from_response(response):
        validation["issues"].append("No sources or citations found")
        validation["suggestions"].append("Include credible sources and citations")
    
    # Check for research indicators
    research_indicators = [
        r'research', r'study', r'analysis', r'evidence', r'findings',
        r'methodology', r'approach', r'results', r'conclusion'
    ]
    
    indicator_count = sum(1 for indicator in research_indicators 
                         if re.search(indicator, response, re.IGNORECASE))
    
    if indicator_count < 3:
        validation["issues"].append("Limited research terminology")
        validation["suggestions"].append("Use more academic and research-oriented language")
    
    return validation

def format_evaluation_results(results: Dict[str, Any]) -> str:
    """Format evaluation results for display"""
    if not results:
        return "No evaluation results available"
    
    formatted = ["📊 Evaluation Results Summary", "=" * 40]
    
    # Overall performance
    if "aggregate_scores" in results:
        scores = results["aggregate_scores"]
        overall_score = sum(scores.values()) / len(scores) if scores else 0
        
        formatted.append(f"🎯 Overall Performance: {overall_score:.3f}")
        formatted.append("")
        
        # Individual metrics
        formatted.append("📈 Individual Metrics:")
        for metric, score in scores.items():
            emoji = "🟢" if score >= 0.8 else "🟡" if score >= 0.6 else "🔴"
            formatted.append(f"  {emoji} {metric}: {score:.3f}")
    
    # Recommendations
    if "recommendations" in results:
        formatted.append("")
        formatted.append("💡 Recommendations:")
        for rec in results["recommendations"]:
            formatted.append(f"  • {rec}")
    
    return "\n".join(formatted)

def save_conversation_log(thread_id: str, messages: List[Dict[str, Any]], 
                         filename: Optional[str] = None) -> str:
    """Save conversation log to file"""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_log_{thread_id}_{timestamp}.json"
    
    log_data = {
        "thread_id": thread_id,
        "timestamp": datetime.now().isoformat(),
        "message_count": len(messages),
        "messages": messages
    }
    
    try:
        with open(filename, 'w') as f:
            json.dump(log_data, f, indent=2)
        return filename
    except Exception as e:
        print(f"Error saving conversation log: {str(e)}")
        return ""

def load_conversation_log(filename: str) -> Optional[Dict[str, Any]]:
    """Load conversation log from file"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading conversation log: {str(e)}")
        return None

def get_research_domain_keywords() -> Dict[str, List[str]]:
    """Get keyword lists for different research domains"""
    return {
        "machine_learning": [
            "neural networks", "deep learning", "training", "optimization",
            "gradient descent", "backpropagation", "overfitting", "regularization"
        ],
        "natural_language_processing": [
            "tokenization", "embeddings", "transformers", "attention",
            "language models", "seq2seq", "BERT", "GPT"
        ],
        "computer_vision": [
            "convolutional", "image classification", "object detection",
            "segmentation", "feature extraction", "CNN", "vision transformer"
        ],
        "reinforcement_learning": [
            "agent", "environment", "reward", "policy", "Q-learning",
            "exploration", "exploitation", "value function"
        ],
        "academic_research": [
            "methodology", "literature review", "hypothesis", "experiment",
            "statistical significance", "peer review", "citation", "publication"
        ]
    }

def identify_research_domain(text: str) -> str:
    """Identify the research domain based on text content"""
    keywords = get_research_domain_keywords()
    domain_scores = {}
    
    text_lower = text.lower()
    
    for domain, domain_keywords in keywords.items():
        score = sum(1 for keyword in domain_keywords if keyword in text_lower)
        domain_scores[domain] = score
    
    if domain_scores:
        return max(domain_scores, key=domain_scores.get)
    else:
        return "general"

def format_citation_apa(title: str, authors: List[str], year: str, 
                       journal: str = "", url: str = "") -> str:
    """Format citation in APA style"""
    # Format authors
    if len(authors) == 1:
        author_str = authors[0]
    elif len(authors) == 2:
        author_str = f"{authors[0]} & {authors[1]}"
    else:
        author_str = f"{', '.join(authors[:-1])}, & {authors[-1]}"
    
    # Build citation
    citation = f"{author_str} ({year}). {title}."
    
    if journal:
        citation += f" {journal}."
    
    if url:
        citation += f" Retrieved from {url}"
    
    return citation

def create_research_summary(responses: List[str]) -> str:
    """Create a summary of research conversation"""
    if not responses:
        return "No research content to summarize"
    
    # Combine all responses
    full_text = " ".join(responses)
    
    # Extract key information
    metrics = calculate_response_metrics(full_text)
    sources = extract_sources_from_response(full_text)
    domain = identify_research_domain(full_text)
    
    newline = chr(10)
    sources_list = newline.join(f"• {source}" for source in sources[:10])
    key_topics = ', '.join(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', full_text)[:20])
    
    summary = f"""
Research Conversation Summary
===========================

📊 Content Metrics:
- Total words: {metrics['word_count']}
- Sources cited: {metrics['source_count']}
- Research domain: {domain}

🔗 Sources Found:
{sources_list}

📝 Key Topics Covered:
{key_topics}

🎯 Research Quality:
- Citations present: {'✅' if metrics['has_citations'] else '❌'}
- URLs included: {'✅' if metrics['has_urls'] else '❌'}
- Comprehensive coverage: {'✅' if metrics['word_count'] > 500 else '❌'}
"""
    
    return summary

# Configuration constants
DEFAULT_CONFIG = {
    "max_sources_per_search": 5,
    "max_conversation_turns": 20,
    "response_timeout": 30,
    "evaluation_batch_size": 5,
    "default_temperature": 0.1,
    "max_tokens": 2000
}

def get_config(key: str, default: Any = None) -> Any:
    """Get configuration value"""
    return DEFAULT_CONFIG.get(key, default)

def update_config(key: str, value: Any) -> None:
    """Update configuration value"""
    DEFAULT_CONFIG[key] = value

# Initialize environment on import
if __name__ == "__main__":
    setup_result = setup_environment()
    validation_result = validate_api_keys()
    
    if validation_result:
        print("🚀 Research Assistant Utils initialized successfully")
    else:
        print("⚠️  Research Assistant Utils initialized with warnings")
