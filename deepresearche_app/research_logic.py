"""
Deep Research Logic Module
Handles all Tavily API interactions and research processing
"""

import os
import re
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

from tavily import TavilyClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class ResearchResult:
    """Data class to store research results"""
    topic: str
    key_points: List[str] = field(default_factory=list)
    sources: List[Dict[str, str]] = field(default_factory=list)
    summary: str = ""
    raw_results: List[Dict] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    success: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class TavilyResearchError(Exception):
    """Custom exception for Tavily research errors"""
    pass


class DeepResearcher:
    """
    Deep Research class that performs comprehensive web research using Tavily API
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the DeepResearcher with Tavily API key
        
        Args:
            api_key: Tavily API key. If not provided, will try to get from environment
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise TavilyResearchError(
                "Tavily API key not found. Please set TAVILY_API_KEY in .env file or pass it directly."
            )
        
        try:
            self.client = TavilyClient(api_key=self.api_key)
        except Exception as e:
            raise TavilyResearchError(f"Failed to initialize Tavily client: {str(e)}")
    
    def _validate_topic(self, topic: str) -> bool:
        """
        Validate the research topic
        
        Args:
            topic: The topic to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not topic or not isinstance(topic, str):
            return False
        
        # Remove whitespace and check length
        cleaned_topic = topic.strip()
        if len(cleaned_topic) < 3:
            return False
        
        if len(cleaned_topic) > 500:
            return False
        
        return True
    
    def _generate_search_queries(self, topic: str) -> List[str]:
        """
        Generate multiple search queries for comprehensive research
        
        Args:
            topic: The main research topic
            
        Returns:
            List of search queries
        """
        queries = [
            topic.strip(),
            f"{topic} latest developments 2024 2025",
            f"{topic} key features benefits",
            f"{topic} tutorial guide how to use",
            f"{topic} comparison vs alternatives",
        ]
        
        # Add AI-specific queries if topic seems AI-related
        ai_keywords = ['ai', 'artificial intelligence', 'machine learning', 'ml', 'llm', 
                      'model', 'gpt', 'neural', 'deep learning', 'automation']
        topic_lower = topic.lower()
        
        if any(keyword in topic_lower for keyword in ai_keywords):
            queries.extend([
                f"{topic} best practices",
                f"{topic} use cases examples",
                f"{topic} limitations challenges",
            ])
        
        return queries[:6]  # Limit to 6 queries
    
    def _extract_key_points(self, results: List[Dict], topic: str) -> List[str]:
        """
        Extract key points from search results
        
        Args:
            results: List of search results from Tavily
            topic: The research topic
            
        Returns:
            List of key points
        """
        key_points = []
        seen_points = set()
        
        for result in results:
            content = result.get('content', '')
            if not content:
                continue
            
            # Split content into sentences
            sentences = re.split(r'[.!?]+', content)
            
            for sentence in sentences:
                sentence = sentence.strip()
                
                # Filter for meaningful sentences
                if len(sentence) < 30 or len(sentence) > 300:
                    continue
                
                # Check if sentence contains relevant information
                if self._is_informative_sentence(sentence, topic):
                    # Normalize for deduplication
                    normalized = sentence.lower().strip()
                    
                    if normalized not in seen_points:
                        seen_points.add(normalized)
                        key_points.append(sentence)
                        
                        # Limit key points
                        if len(key_points) >= 15:
                            break
            
            if len(key_points) >= 15:
                break
        
        return key_points
    
    def _is_informative_sentence(self, sentence: str, topic: str) -> bool:
        """
        Check if a sentence contains informative content
        
        Args:
            sentence: The sentence to check
            topic: The research topic
            
        Returns:
            bool: True if informative
        """
        # Keywords that indicate informative content
        informative_keywords = [
            'allows', 'enables', 'provides', 'offers', 'supports', 'features',
            'includes', 'designed', 'used for', 'helps', 'improves', 'enhances',
            'creates', 'generates', 'automates', 'streamlines', 'simplifies',
            'integrates', 'compatible', 'powered', 'built', 'develop', 'release',
            'version', 'update', 'launch', 'announce', 'introduce', 'capability'
        ]
        
        sentence_lower = sentence.lower()
        topic_words = topic.lower().split()
        
        # Check if sentence contains informative keywords
        has_informative_keyword = any(kw in sentence_lower for kw in informative_keywords)
        
        # Check if sentence relates to topic
        relates_to_topic = any(word in sentence_lower for word in topic_words if len(word) > 3)
        
        return has_informative_keyword and relates_to_topic
    
    def _extract_sources(self, results: List[Dict]) -> List[Dict[str, str]]:
        """
        Extract source information from results
        
        Args:
            results: List of search results
            
        Returns:
            List of source dictionaries
        """
        sources = []
        seen_urls = set()
        
        for result in results:
            url = result.get('url', '')
            title = result.get('title', 'Untitled')
            
            if url and url not in seen_urls:
                seen_urls.add(url)
                sources.append({
                    'title': title,
                    'url': url
                })
        
        return sources[:10]  # Limit to top 10 sources
    
    def _generate_summary(self, key_points: List[str], topic: str) -> str:
        """
        Generate a brief summary from key points
        
        Args:
            key_points: List of key points
            topic: The research topic
            
        Returns:
            Summary string
        """
        if not key_points:
            return f"No summary available for '{topic}'."
        
        # Combine first few key points for summary
        summary_points = key_points[:5]
        summary = f"Research on '{topic}' reveals several important aspects:\n\n"
        
        for i, point in enumerate(summary_points, 1):
            summary += f"{i}. {point}.\n"
        
        return summary
    
    def research(self, topic: str, search_depth: str = "advanced", 
                 max_results: int = 10) -> ResearchResult:
        """
        Perform deep research on a topic
        
        Args:
            topic: The topic to research
            search_depth: Search depth ('basic' or 'advanced')
            max_results: Maximum results per query
            
        Returns:
            ResearchResult object containing all research data
        """
        result = ResearchResult(topic=topic)
        
        # Validate topic
        if not self._validate_topic(topic):
            result.error_message = "Invalid topic. Please provide a topic with 3-500 characters."
            return result
        
        try:
            # Generate multiple search queries
            search_queries = self._generate_search_queries(topic)
            result.search_queries = search_queries
            
            all_results = []
            
            # Perform searches with rate limiting
            for i, query in enumerate(search_queries):
                try:
                    # Add delay between requests to avoid rate limiting
                    if i > 0:
                        time.sleep(0.5)
                    
                    response = self.client.search(
                        query=query,
                        search_depth=search_depth,
                        max_results=max_results,
                        include_answer=True,
                        include_raw_content=True
                    )
                    
                    if 'results' in response:
                        all_results.extend(response['results'])
                    
                except Exception as e:
                    # Continue with other queries if one fails
                    print(f"Warning: Search query '{query}' failed: {str(e)}")
                    continue
            
            # Check if we got any results
            if not all_results:
                result.error_message = "No results found for the given topic. Please try a different topic."
                return result
            
            # Process results
            result.raw_results = all_results
            result.key_points = self._extract_key_points(all_results, topic)
            result.sources = self._extract_sources(all_results)
            result.summary = self._generate_summary(result.key_points, topic)
            result.success = True
            
        except Exception as e:
            result.error_message = f"Research failed: {str(e)}"
        
        return result
    
    def quick_search(self, topic: str) -> ResearchResult:
        """
        Perform a quick basic search on a topic
        
        Args:
            topic: The topic to search
            
        Returns:
            ResearchResult object
        """
        return self.research(topic, search_depth="basic", max_results=5)


def format_research_output(result: ResearchResult) -> str:
    """
    Format research result as a readable string
    
    Args:
        result: ResearchResult object
        
    Returns:
        Formatted string
    """
    if not result.success:
        return f"❌ Error: {result.error_message}"
    
    output = []
    output.append("=" * 80)
    output.append(f"🔍 RESEARCH REPORT: {result.topic.upper()}")
    output.append("=" * 80)
    output.append("")
    
    # Summary section
    output.append("📋 SUMMARY")
    output.append("-" * 80)
    output.append(result.summary)
    output.append("")
    
    # Key Points section
    output.append("🎯 KEY POINTS")
    output.append("-" * 80)
    for i, point in enumerate(result.key_points, 1):
        output.append(f"{i}. {point}")
    output.append("")
    
    # Sources section
    output.append("📚 SOURCES")
    output.append("-" * 80)
    for i, source in enumerate(result.sources, 1):
        output.append(f"{i}. {source['title']}")
        output.append(f"   URL: {source['url']}")
    output.append("")
    
    # Search metadata
    output.append("🔧 SEARCH DETAILS")
    output.append("-" * 80)
    output.append(f"Search Queries Used: {len(result.search_queries)}")
    output.append(f"Total Sources Found: {len(result.sources)}")
    output.append(f"Key Points Extracted: {len(result.key_points)}")
    output.append(f"Research Time: {result.timestamp}")
    output.append("")
    output.append("=" * 80)
    
    return "\n".join(output)


# Example usage
if __name__ == "__main__":
    try:
        researcher = DeepResearcher()
        result = researcher.research("OpenAI GPT-5")
        print(format_research_output(result))
    except Exception as e:
        print(f"Error: {e}")
