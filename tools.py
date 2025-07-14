import os
import json
from typing import List, Dict, Any, Optional
from langchain_core.tools import tool
import arxiv
import requests
from tavily import TavilyClient
import trafilatura
import re
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@tool
def arxiv_search(query: str, max_results: int = 5) -> str:
    """
    Search for academic papers on arXiv.
    
    Args:
        query: Search query for academic papers
        max_results: Maximum number of results to return (default 5)
    
    Returns:
        JSON string containing paper details including title, authors, abstract, and arXiv URL
    """
    try:
        # Create arXiv client
        client = arxiv.Client()
        
        # Search for papers
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate
        )
        
        results = []
        for result in client.results(search):
            paper_data = {
                "title": result.title,
                "authors": [str(author) for author in result.authors],
                "abstract": result.summary,
                "published": result.published.strftime("%Y-%m-%d"),
                "url": str(result.entry_id),
                "pdf_url": str(result.pdf_url),
                "categories": result.categories,
                "primary_category": result.primary_category
            }
            results.append(paper_data)
        
        return json.dumps(results, indent=2)
        
    except Exception as e:
        return f"Error searching arXiv: {str(e)}"

@tool
def web_search(query: str, max_results: int = 5) -> str:
    """
    Search the web for research-related content using TAVILY.
    
    Args:
        query: Search query for web content
        max_results: Maximum number of results to return (default 5)
    
    Returns:
        JSON string containing search results with title, URL, and snippet
    """
    try:
        # Initialize TAVILY client
        tavily_api_key = os.getenv("TAVILY_API_KEY")
        if not tavily_api_key:
            return json.dumps({"error": "TAVILY_API_KEY not found in environment variables"})
        
        tavily = TavilyClient(api_key=tavily_api_key)
        
        # Perform search with TAVILY
        response = tavily.search(
            query=query,
            search_depth="basic",
            max_results=max_results,
            include_answer=False,
            include_raw_content=False
        )
        
        results = []
        for result in response.get("results", []):
            results.append({
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "snippet": result.get("content", ""),
                "score": result.get("score", 0),
                "source": "tavily_search"
            })
        
        return json.dumps(results, indent=2)
        
    except Exception as e:
        return f"Error in TAVILY web search: {str(e)}"

@tool
def analyze_paper_content(url: str) -> str:
    """
    Analyze the content of a paper or research article from a URL.
    
    Args:
        url: URL of the paper or research article
    
    Returns:
        JSON string containing extracted content analysis
    """
    try:
        # For arXiv URLs, try to get the abstract page
        if "arxiv.org" in url:
            if "/pdf/" in url:
                # Convert PDF URL to abstract URL
                url = url.replace("/pdf/", "/abs/").replace(".pdf", "")
        
        # Fetch and extract content
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return json.dumps({"error": "Could not fetch content from URL"})
        
        text = trafilatura.extract(downloaded)
        if not text:
            return json.dumps({"error": "Could not extract text from content"})
        
        # Basic content analysis
        word_count = len(text.split())
        
        # Extract key sections (basic heuristics)
        sections = {}
        
        # Look for common academic paper sections
        section_patterns = {
            "abstract": r"abstract\s*:?\s*(.*?)(?=\n\n|\n[A-Z]|\n\d+\.|\nintroduction|$)",
            "introduction": r"introduction\s*:?\s*(.*?)(?=\n\n|\n[A-Z]|\n\d+\.|\nmethod|$)",
            "conclusion": r"conclusion\s*:?\s*(.*?)(?=\n\n|\n[A-Z]|\n\d+\.|\nreferences|$)",
            "methodology": r"method(?:ology)?\s*:?\s*(.*?)(?=\n\n|\n[A-Z]|\n\d+\.|\nresults|$)"
        }
        
        text_lower = text.lower()
        for section, pattern in section_patterns.items():
            match = re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL)
            if match:
                sections[section] = match.group(1).strip()[:500]  # Limit length
        
        # Extract potential citations (simple heuristic)
        citation_pattern = r'\[(\d+)\]|\(([^)]+,\s*\d{4})\)'
        citations = re.findall(citation_pattern, text)
        
        analysis = {
            "url": url,
            "word_count": word_count,
            "sections": sections,
            "potential_citations": len(citations),
            "content_preview": text[:1000] + "..." if len(text) > 1000 else text,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        return json.dumps(analysis, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Error analyzing content: {str(e)}"})

@tool
def extract_citations(text: str) -> str:
    """
    Extract and format citations from academic text.
    
    Args:
        text: Academic text containing citations
    
    Returns:
        JSON string containing extracted citations and references
    """
    try:
        # Pattern for different citation formats
        patterns = {
            "numbered": r'\[(\d+)\]',
            "author_year": r'\(([A-Za-z\s]+,\s*\d{4})\)',
            "author_year_alt": r'([A-Za-z\s]+\s*\(\d{4}\))',
            "doi": r'doi:\s*([^\s]+)',
            "url": r'https?://[^\s]+',
            "arxiv": r'arXiv:\s*([^\s]+)'
        }
        
        extracted = {}
        
        for pattern_name, pattern in patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                extracted[pattern_name] = list(set(matches))  # Remove duplicates
        
        # Try to extract reference section
        ref_patterns = [
            r'references\s*:?\s*(.*?)(?=\n\n|\nappendix|$)',
            r'bibliography\s*:?\s*(.*?)(?=\n\n|\nappendix|$)',
            r'works cited\s*:?\s*(.*?)(?=\n\n|\nappendix|$)'
        ]
        
        reference_section = None
        text_lower = text.lower()
        
        for pattern in ref_patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE | re.DOTALL)
            if match:
                reference_section = match.group(1).strip()
                break
        
        result = {
            "extracted_citations": extracted,
            "reference_section": reference_section[:2000] if reference_section else None,
            "total_citations": sum(len(v) for v in extracted.values()),
            "extraction_timestamp": datetime.now().isoformat()
        }
        
        return json.dumps(result, indent=2)
        
    except Exception as e:
        return json.dumps({"error": f"Error extracting citations: {str(e)}"})

# Additional utility functions for the tools
def clean_text(text: str) -> str:
    """Clean and normalize text content"""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove common academic formatting artifacts
    text = re.sub(r'\n+', '\n', text)
    text = re.sub(r'^\s*\d+\.\s*', '', text, flags=re.MULTILINE)
    
    return text

def is_academic_url(url: str) -> bool:
    """Check if a URL is likely to be an academic source"""
    academic_domains = [
        'arxiv.org', 'ieee.org', 'acm.org', 'springer.com',
        'nature.com', 'science.org', 'cell.com', 'pnas.org',
        'nih.gov', 'edu', 'researchgate.net', 'semanticscholar.org'
    ]
    
    return any(domain in url.lower() for domain in academic_domains)

def extract_paper_metadata(text: str) -> Dict[str, Any]:
    """Extract metadata from academic paper text"""
    metadata = {}
    
    # Extract title (usually first line or after specific patterns)
    title_patterns = [
        r'^([A-Z][^.!?]*(?:[.!?][A-Z][^.!?]*)*[.!?])\s*\n',
        r'title\s*:?\s*([^\n]+)',
        r'^([A-Z][^.!?]*(?:[.!?][A-Z][^.!?]*)*[.!?])\s*$'
    ]
    
    for pattern in title_patterns:
        match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        if match:
            metadata['title'] = match.group(1).strip()
            break
    
    # Extract authors (common patterns)
    author_patterns = [
        r'authors?\s*:?\s*([^\n]+)',
        r'by\s+([A-Z][^.!?]*(?:,\s*[A-Z][^.!?]*)*)',
        r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)*)\s*$'
    ]
    
    for pattern in author_patterns:
        match = re.search(pattern, text, re.MULTILINE | re.IGNORECASE)
        if match:
            metadata['authors'] = [author.strip() for author in match.group(1).split(',')]
            break
    
    return metadata
