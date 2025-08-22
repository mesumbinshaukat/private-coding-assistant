"""
Web search utilities for deep information gathering
Integrates multiple free APIs and scraping tools
"""

import asyncio
import aiohttp
import json
import re
from typing import Dict, List, Any, Optional
from urllib.parse import urlencode, urlparse, quote
from bs4 import BeautifulSoup
import logging

# DuckDuckGo search
try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

logger = logging.getLogger(__name__)

class WebSearcher:
    """
    Comprehensive web search engine integrating multiple sources
    
    Features:
    - DuckDuckGo search (free, no API key required)
    - Stack Overflow API integration
    - GitHub API integration  
    - Web scraping for documentation
    - Result synthesis and ranking
    """
    
    def __init__(self):
        self.session = None
        self.user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        
        # API endpoints
        self.stackoverflow_api = "https://api.stackexchange.com/2.3/search/advanced"
        self.github_api = "https://api.github.com/search/repositories"
        
        # Search result cache
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour
    
    async def initialize(self):
        """Initialize the web searcher"""
        self.session = aiohttp.ClientSession(
            headers={"User-Agent": self.user_agent},
            timeout=aiohttp.ClientTimeout(total=30)
        )
        logger.info("WebSearcher initialized")
    
    async def close(self):
        """Close the session"""
        if self.session:
            await self.session.close()
    
    async def search_multiple_sources(self, 
                                    query: str, 
                                    depth: int = 5, 
                                    include_code: bool = True,
                                    sources: List[str] = None) -> List[Dict[str, Any]]:
        """
        Search multiple sources and combine results
        
        Args:
            query: Search query
            depth: Number of results per source
            include_code: Whether to include code examples
            sources: List of sources to search
            
        Returns:
            Combined and ranked search results
        """
        if sources is None:
            sources = ["duckduckgo", "stackoverflow", "github"]
        
        # Check cache first
        cache_key = f"{query}:{':'.join(sources)}:{depth}:{include_code}"
        if cache_key in self.cache:
            logger.info(f"Returning cached results for: {query}")
            return self.cache[cache_key]
        
        all_results = []
        search_tasks = []
        
        # Create search tasks for each source
        if "duckduckgo" in sources:
            search_tasks.append(self._search_duckduckgo(query, depth))
        
        if "stackoverflow" in sources:
            search_tasks.append(self._search_stackoverflow(query, depth))
        
        if "github" in sources and include_code:
            search_tasks.append(self._search_github(query, depth))
        
        # Execute searches concurrently
        try:
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Search task {i} failed: {result}")
                    continue
                
                if isinstance(result, list):
                    all_results.extend(result)
        
        except Exception as e:
            logger.error(f"Multi-source search failed: {e}")
            return []
        
        # Rank and deduplicate results
        ranked_results = self._rank_and_deduplicate(all_results, query)
        
        # Cache results
        self.cache[cache_key] = ranked_results
        
        return ranked_results[:depth * len(sources)]
    
    async def _search_duckduckgo(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo"""
        try:
            if DDGS is None:
                logger.warning("DuckDuckGo search not available")
                return []
            
            # Add programming context to query
            programming_query = f"{query} programming code python"
            
            results = []
            
            # Use DuckDuckGo search
            with DDGS() as ddgs:
                search_results = list(ddgs.text(programming_query, max_results=num_results))
                
                for result in search_results:
                    processed_result = {
                        "title": result.get("title", ""),
                        "url": result.get("href", ""),
                        "snippet": result.get("body", ""),
                        "source": "duckduckgo",
                        "relevance_score": 0.7,  # Base score
                        "content_type": "web"
                    }
                    
                    # Try to extract additional content
                    try:
                        content = await self._scrape_page(result.get("href", ""))
                        if content:
                            processed_result["full_content"] = content[:1000]  # Limit content
                            processed_result["has_code"] = bool(re.search(r'<code>|```|def |class |function', content))
                    except:
                        pass
                    
                    results.append(processed_result)
            
            logger.info(f"DuckDuckGo search returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return []
    
    async def _search_stackoverflow(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search Stack Overflow using their API"""
        try:
            params = {
                "order": "desc",
                "sort": "relevance",
                "q": query,
                "site": "stackoverflow",
                "pagesize": num_results,
                "filter": "withbody"
            }
            
            url = f"{self.stackoverflow_api}?{urlencode(params)}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    items = data.get("items", [])
                    
                    results = []
                    for item in items:
                        result = {
                            "title": item.get("title", ""),
                            "url": item.get("link", ""),
                            "snippet": self._clean_html(item.get("body", ""))[:300],
                            "source": "stackoverflow",
                            "relevance_score": min(1.0, item.get("score", 0) / 10),
                            "content_type": "qa",
                            "tags": item.get("tags", []),
                            "is_answered": item.get("is_answered", False),
                            "answer_count": item.get("answer_count", 0),
                            "has_code": bool(re.search(r'<code>|```', item.get("body", "")))
                        }
                        results.append(result)
                    
                    logger.info(f"Stack Overflow search returned {len(results)} results")
                    return results
                else:
                    logger.warning(f"Stack Overflow API returned status {response.status}")
                    return []
        
        except Exception as e:
            logger.error(f"Stack Overflow search failed: {e}")
            return []
    
    async def _search_github(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search GitHub repositories"""
        try:
            # Add language filter for better results
            github_query = f"{query} language:python"
            
            params = {
                "q": github_query,
                "sort": "stars",
                "order": "desc",
                "per_page": num_results
            }
            
            url = f"{self.github_api}?{urlencode(params)}"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    items = data.get("items", [])
                    
                    results = []
                    for item in items:
                        result = {
                            "title": item.get("full_name", ""),
                            "url": item.get("html_url", ""),
                            "snippet": item.get("description", ""),
                            "source": "github",
                            "relevance_score": min(1.0, item.get("stargazers_count", 0) / 1000),
                            "content_type": "repository",
                            "language": item.get("language", ""),
                            "stars": item.get("stargazers_count", 0),
                            "forks": item.get("forks_count", 0),
                            "has_code": True
                        }
                        results.append(result)
                    
                    logger.info(f"GitHub search returned {len(results)} results")
                    return results
                else:
                    logger.warning(f"GitHub API returned status {response.status}")
                    return []
        
        except Exception as e:
            logger.error(f"GitHub search failed: {e}")
            return []
    
    async def _scrape_page(self, url: str) -> Optional[str]:
        """Scrape content from a web page"""
        try:
            # Basic URL validation
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return None
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Parse with BeautifulSoup
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Get text content
                    text = soup.get_text()
                    
                    # Clean up whitespace
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)
                    
                    return text
        
        except Exception as e:
            logger.debug(f"Failed to scrape {url}: {e}")
            return None
    
    def _clean_html(self, html_content: str) -> str:
        """Clean HTML content and extract text"""
        if not html_content:
            return ""
        
        # Remove HTML tags
        soup = BeautifulSoup(html_content, 'html.parser')
        return soup.get_text()
    
    def _rank_and_deduplicate(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Rank and deduplicate search results"""
        if not results:
            return []
        
        # Remove duplicates based on URL
        seen_urls = set()
        unique_results = []
        
        for result in results:
            url = result.get("url", "")
            if url not in seen_urls:
                seen_urls.add(url)
                unique_results.append(result)
        
        # Calculate relevance scores
        query_words = set(query.lower().split())
        
        for result in unique_results:
            title_words = set(result.get("title", "").lower().split())
            snippet_words = set(result.get("snippet", "").lower().split())
            
            # Calculate word overlap
            title_overlap = len(query_words.intersection(title_words)) / len(query_words) if query_words else 0
            snippet_overlap = len(query_words.intersection(snippet_words)) / len(query_words) if query_words else 0
            
            # Adjust relevance score
            base_score = result.get("relevance_score", 0.5)
            relevance_boost = (title_overlap * 0.3) + (snippet_overlap * 0.2)
            
            # Boost for code content
            if result.get("has_code", False):
                relevance_boost += 0.2
            
            # Boost for answered Stack Overflow questions
            if result.get("source") == "stackoverflow" and result.get("is_answered", False):
                relevance_boost += 0.1
            
            result["relevance_score"] = min(1.0, base_score + relevance_boost)
        
        # Sort by relevance score
        ranked_results = sorted(unique_results, key=lambda x: x.get("relevance_score", 0), reverse=True)
        
        return ranked_results
    
    async def search_documentation(self, query: str, domain: str = "python") -> List[Dict[str, Any]]:
        """Search official documentation"""
        try:
            doc_urls = {
                "python": "https://docs.python.org/3/search.html",
                "javascript": "https://developer.mozilla.org/en-US/search",
                "react": "https://reactjs.org/docs",
                "django": "https://docs.djangoproject.com/en/stable/search/"
            }
            
            if domain not in doc_urls:
                return []
            
            # For now, use DuckDuckGo with site-specific search
            site_query = f"site:{doc_urls[domain]} {query}"
            return await self._search_duckduckgo(site_query, 3)
        
        except Exception as e:
            logger.error(f"Documentation search failed: {e}")
            return []

# Singleton instance
web_searcher = WebSearcher()

# Utility functions
async def search_coding_solutions(query: str, language: str = "python") -> List[Dict[str, Any]]:
    """Search for coding solutions across multiple sources"""
    enhanced_query = f"{query} {language} programming"
    return await web_searcher.search_multiple_sources(
        enhanced_query,
        depth=10,
        include_code=True,
        sources=["stackoverflow", "github", "duckduckgo"]
    )

async def search_algorithms(algorithm_name: str) -> List[Dict[str, Any]]:
    """Search for algorithm implementations and explanations"""
    query = f"{algorithm_name} algorithm implementation explanation"
    return await web_searcher.search_multiple_sources(
        query,
        depth=8,
        include_code=True
    )

# Testing
if __name__ == "__main__":
    async def test_search():
        searcher = WebSearcher()
        await searcher.initialize()
        
        try:
            # Test search
            results = await searcher.search_multiple_sources(
                "python fibonacci algorithm",
                depth=3,
                include_code=True
            )
            
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results):
                print(f"{i+1}. {result.get('title', 'No title')}")
                print(f"   Source: {result.get('source', 'Unknown')}")
                print(f"   Relevance: {result.get('relevance_score', 0):.2f}")
                print(f"   URL: {result.get('url', 'No URL')}")
                print()
        
        finally:
            await searcher.close()
    
    asyncio.run(test_search())
