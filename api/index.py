"""
Autonomous AI Agent - Phase 1: Basic Web and Search
Progressive Enhancement System Implementation
"""

from http.server import BaseHTTPRequestHandler
import json
import logging
from datetime import datetime
import urllib.parse
import subprocess
import sys
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProgressiveDependencyManager:
    """Manages progressive installation of dependencies"""
    
    def __init__(self):
        self.available_features = set()
        self.installed_packages = set()
        self.feature_status = {}
        self._initialize_features()
    
    def _initialize_features(self):
        """Initialize available features"""
        self.available_features = {
            "web_search", "code_generation", "reasoning", 
            "memory", "training", "advanced_ml"
        }
        
        # Initialize all features as not available initially
        for feature in self.available_features:
            self.feature_status[feature] = "not_installed"
        
        # Don't check packages during initialization - do it on demand
        logger.info("Dependency manager initialized - packages will be checked on demand")
    
    def _check_package_available(self, package_name: str) -> bool:
        """Check if a specific package is available"""
        try:
            import importlib
            importlib.import_module(package_name)
            return True
        except ImportError:
            return False
    
    def install_package(self, package_name: str) -> bool:
        """Install a specific package"""
        try:
            logger.info(f"Installing package: {package_name}")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package_name
            ], capture_output=True, text=True)
            self.installed_packages.add(package_name)
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package_name}: {e}")
            return False
    
    def enable_feature(self, feature_name: str) -> bool:
        """Enable a specific feature by installing required packages"""
        if feature_name not in self.available_features:
            return False
        
        if feature_name == "web_search":
            required_packages = ["requests", "beautifulsoup4", "lxml"]
            for package in required_packages:
                if not self._check_package_available(package):
                    if not self.install_package(package):
                        return False
                else:
                    self.installed_packages.add(package)
            self.feature_status[feature_name] = "enabled"
            return True
        
        return False
    
    def get_enhanced_search(self, query: str, depth: int = 5) -> dict:
        """Enhanced web search using progressive dependency loading"""
        # Always try to enable web search first
        if self.feature_status.get("web_search") != "enabled":
            if not self.enable_feature("web_search"):
                return self._get_mock_search(query, depth)
        
        try:
            # Dynamic import after ensuring packages are available
            # Use importlib for safer dynamic imports
            import importlib
            
            try:
                requests = importlib.import_module("requests")
                bs4 = importlib.import_module("bs4")
                BeautifulSoup = getattr(bs4, "BeautifulSoup")
            except ImportError as e:
                logger.error(f"Failed to import required modules: {e}")
                return self._get_mock_search(query, depth)
            
            # Real web search implementation
            search_results = []
            
            # Use DuckDuckGo for search (no API key required)
            search_url = f"https://duckduckgo.com/html/?q={urllib.parse.quote(query)}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract search results
            for result in soup.find_all('a', class_='result__a')[:depth]:
                title = result.get_text(strip=True)
                url = result.get('href')
                if title and url:
                    search_results.append({
                        "title": title,
                        "url": url,
                        "snippet": "Search result from DuckDuckGo"
                    })
            
            return {
                "results": search_results,
                "synthesized_answer": f"Found {len(search_results)} real web search results for '{query}'",
                "confidence": 0.9,
                "method": "enhanced_web_search",
                "query": query,
                "result_count": len(search_results),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Enhanced search failed: {str(e)}")
            return self._get_mock_search(query, depth)
    
    def _get_mock_search(self, query: str, depth: int) -> dict:
        """Fallback mock search when dependencies aren't available"""
        return {
            "results": [
                {
                    "title": f"Mock result for: {query}",
                    "url": "https://example.com/mock",
                    "snippet": "This is a mock search result. Install dependencies to enable real web search."
                }
            ],
            "synthesized_answer": f"Mock search results for '{query}' (dependencies not installed)",
            "confidence": 0.1,
            "method": "mock_search",
            "query": query,
            "result_count": 1,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_status(self) -> dict:
        """Get current system status"""
        return {
            "available_features": list(self.available_features),
            "installed_packages": list(self.installed_packages),
            "feature_status": self.feature_status,
            "timestamp": datetime.now().isoformat()
        }

# Initialize dependency manager
try:
    dependency_manager = ProgressiveDependencyManager()
    logger.info("Dependency manager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize dependency manager: {e}")
    dependency_manager = None

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        try:
            if self.path == "/":
                response = {
                    "message": "Autonomous AI Agent API",
                    "status": "Phase 1: Basic Web and Search",
                    "version": "1.0.0",
                    "timestamp": datetime.now().isoformat(),
                    "features": ["web_search", "code_generation", "reasoning"],
                    "endpoints": {
                        "GET": ["/", "/status", "/dependencies"],
                        "POST": ["/search", "/generate", "/reason"]
                    }
                }
            elif self.path == "/status":
                response = {
                    "status": "operational",
                    "phase": "Phase 1: Basic Web and Search",
                    "timestamp": datetime.now().isoformat(),
                    "dependency_manager": dependency_manager.get_status() if dependency_manager else {"error": "Not initialized"}
                }
            elif self.path == "/dependencies":
                response = {
                    "dependencies": dependency_manager.get_status() if dependency_manager else {"error": "Not initialized"},
                    "installation_guide": "Use POST /dependencies to install packages",
                    "timestamp": datetime.now().isoformat()
                }
            elif self.path == "/test":
                response = {
                    "message": "Test GET endpoint working",
                    "timestamp": datetime.now().isoformat()
                }
            else:
                response = {
                    "error": "Endpoint not found",
                    "available_endpoints": ["/", "/status", "/dependencies"],
                    "timestamp": datetime.now().isoformat()
                }
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                return
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            error_response = {
                "error": "GET request failed",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode())
    
    def do_POST(self):
        """Handle POST requests"""
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length > 0:
                post_data = self.rfile.read(content_length)
                request_data = json.loads(post_data.decode('utf-8'))
            else:
                request_data = {}
            
            # Route to appropriate handler
            if self.path == "/search":
                response = self._handle_search(request_data)
            elif self.path == "/generate":
                response = self._handle_code_generation(request_data)
            elif self.path == "/reason":
                response = self._handle_reasoning(request_data)
            elif self.path == "/dependencies":
                response = self._handle_dependencies(request_data)
            elif self.path == "/test":
                response = self._handle_test(request_data)
            else:
                response = {
                    "error": "Endpoint not found",
                    "available_endpoints": ["/search", "/generate", "/reason", "/dependencies"],
                    "timestamp": datetime.now().isoformat()
                }
                self.send_response(404)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())
                return
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())
            
        except json.JSONDecodeError as e:
            error_response = {
                "error": "Invalid JSON in request body",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode())
        except Exception as e:
            error_response = {
                "error": "POST request failed",
                "details": str(e),
                "timestamp": datetime.now().isoformat()
            }
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(error_response).encode())
    
    def _handle_search(self, request_data: dict) -> dict:
        """Handle web search requests"""
        query = request_data.get('query', '')
        depth = request_data.get('depth', 5)
        
        if not query:
            return {
                "error": "Query parameter is required",
                "timestamp": datetime.now().isoformat()
            }
        
        # Use progressive enhancement for search
        if dependency_manager:
            return dependency_manager.get_enhanced_search(query, depth)
        else:
            return {
                "error": "Dependency manager not available",
                "timestamp": datetime.now().isoformat()
            }
    
    def _handle_code_generation(self, request_data: dict) -> dict:
        """Handle code generation requests"""
        prompt = request_data.get('prompt', '')
        language = request_data.get('language', 'python')
        
        if not prompt:
            return {
                "error": "Prompt parameter is required",
                "timestamp": datetime.now().isoformat()
            }
        
        # Phase 1: Basic code generation with templates
        if "function" in prompt.lower():
            code = f"""def {prompt.lower().replace(' ', '_')}():
    \"\"\"
    Generated function based on prompt: {prompt}
    \"\"\"
    # TODO: Implement function logic
    pass

# Example usage
if __name__ == "__main__":
    result = {prompt.lower().replace(' ', '_')}()
    print(result)"""
        else:
            code = f"""# Generated code for: {prompt}
# Language: {language}

# TODO: Implement the requested functionality
print("Code generation for: {prompt}")"""
        
        return {
            "code": code,
            "language": language,
            "prompt": prompt,
            "method": "template_generation",
            "confidence": 0.7,
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_reasoning(self, request_data: dict) -> dict:
        """Handle reasoning requests"""
        problem = request_data.get('problem', '')
        
        if not problem:
            return {
                "error": "Problem parameter is required",
                "timestamp": datetime.now().isoformat()
            }
        
        # Phase 1: Basic reasoning with step-by-step analysis
        reasoning_steps = [
            f"1. Analyzing problem: {problem}",
            "2. Breaking down into components",
            "3. Identifying key requirements",
            "4. Planning solution approach",
            "5. Considering edge cases"
        ]
        
        return {
            "problem": problem,
            "reasoning_steps": reasoning_steps,
            "conclusion": f"Problem '{problem}' analyzed with {len(reasoning_steps)} reasoning steps",
            "method": "basic_reasoning",
            "confidence": 0.8,
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_dependencies(self, request_data: dict) -> dict:
        """Handle dependency management requests"""
        if not dependency_manager:
            return {
                "error": "Dependency manager not available",
                "timestamp": datetime.now().isoformat()
            }
        
        action = request_data.get('action', 'status')
        
        if action == 'install_package':
            package = request_data.get('package')
            if package:
                success = dependency_manager.install_package(package)
                return {
                    "action": "install_package",
                    "package": package,
                    "success": success,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "error": "Package name required for install_package action",
                    "timestamp": datetime.now().isoformat()
                }
        
        elif action == 'enable_feature':
            feature = request_data.get('feature')
            if feature:
                success = dependency_manager.enable_feature(feature)
                return {
                    "action": "enable_feature",
                    "feature": feature,
                    "success": success,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "error": "Feature name required for enable_feature action",
                    "timestamp": datetime.now().isoformat()
                }
        
        else:  # Default to status
            return {
                "action": "status",
                "status": dependency_manager.get_status(),
                "timestamp": datetime.now().isoformat()
            }
    
    def _handle_test(self, request_data: dict) -> dict:
        """Handle test requests - simple endpoint to verify POST works"""
        return {
            "message": "Test POST endpoint working",
            "received_data": request_data,
            "timestamp": datetime.now().isoformat()
        }
