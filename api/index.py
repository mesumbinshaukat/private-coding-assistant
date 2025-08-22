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
            # Use stdout and stderr instead of capture_output for Python 3.6 compatibility
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package_name
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
        # Try to use real web search if available
        if ADVANCED_UTILS_AVAILABLE and web_searcher:
            try:
                # Use real web search
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    results = loop.run_until_complete(
                        web_searcher.search_multiple_sources(query, depth, include_code=True)
                    )
                    loop.close()
                    
                    return {
                        "results": results,
                        "synthesized_answer": f"Found {len(results)} real web search results for '{query}'",
                        "confidence": 0.95,
                        "method": "advanced_web_search",
                        "query": query,
                        "result_count": len(results),
                        "timestamp": datetime.now().isoformat()
                    }
                except Exception as e:
                    logger.error(f"Real web search failed: {e}")
                    loop.close()
            except Exception as e:
                logger.error(f"Web search initialization failed: {e}")
        
        # Fallback to basic web search or mock
        if "web_search" not in self.feature_status or self.feature_status["web_search"] != "enabled":
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
    
    def get_enhanced_code_generation(self, prompt: str, language: str = "python") -> dict:
        """Enhanced code generation using progressive dependency loading"""
        # Phase 2: Basic enhanced code generation with templates
        if "function" in prompt.lower():
            code = f"""def {prompt.lower().replace(' ', '_').replace('-', '_')}():
    \"\"\"
    Generated function based on prompt: {prompt}
    \"\"\"
    # TODO: Implement function logic
    pass

# Example usage
if __name__ == "__main__":
    result = {prompt.lower().replace(' ', '_').replace('-', '_')}()
    print(result)"""
        elif "class" in prompt.lower():
            code = f"""class {prompt.lower().replace(' ', '_').replace('-', '_').title()}:
    \"\"\"
    Generated class based on prompt: {prompt}
    \"\"\"
    
    def __init__(self):
        # TODO: Initialize class attributes
        pass
    
    def process(self):
        # TODO: Implement main processing logic
        pass

# Example usage
if __name__ == "__main__":
    instance = {prompt.lower().replace(' ', '_').replace('-', '_').title()}()
    instance.process()"""
        else:
            code = f"""# Generated code for: {prompt}
# Language: {language}

# TODO: Implement the requested functionality
print("Code generation for: {prompt}")"""
        
        return {
            "code": code,
            "language": language,
            "prompt": prompt,
            "method": "enhanced_template_generation",
            "confidence": 0.8,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_enhanced_reasoning(self, problem: str) -> dict:
        """Enhanced reasoning using progressive dependency loading"""
        # Phase 2: Enhanced reasoning with more detailed analysis
        reasoning_steps = [
            f"1. Problem Analysis: {problem}",
            "2. Context Understanding: Breaking down the problem domain",
            "3. Component Identification: Key elements and requirements",
            "4. Solution Strategy: Planning the approach",
            "5. Implementation Steps: Detailed execution plan",
            "6. Edge Case Consideration: Potential issues and solutions",
            "7. Validation: How to verify the solution works"
        ]
        
        # Add problem-specific reasoning
        if "algorithm" in problem.lower():
            reasoning_steps.append("8. Algorithm Complexity: Time and space analysis")
            reasoning_steps.append("9. Optimization: Potential improvements")
        elif "machine learning" in problem.lower():
            reasoning_steps.append("8. Data Requirements: What data is needed")
            reasoning_steps.append("9. Model Selection: Choosing appropriate algorithms")
        
        return {
            "problem": problem,
            "reasoning_steps": reasoning_steps,
            "conclusion": f"Problem '{problem}' analyzed with {len(reasoning_steps)} detailed reasoning steps",
            "method": "enhanced_reasoning",
            "confidence": 0.9,
            "timestamp": datetime.now().isoformat()
        }
    
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
    
    def get_memory_status(self) -> dict:
        """Get memory system status"""
        return {
            "status": "available" if "memory" in self.available_features else "not_available",
            "type": "vector_store",
            "storage": "in_memory",
            "timestamp": datetime.now().isoformat()
        }
    
    def get_models_status(self) -> dict:
        """Get models status"""
        return {
            "available_models": [],
            "trained_models": [],
            "status": "Phase 2: Models will be available after dependency installation",
            "timestamp": datetime.now().isoformat()
        }
    
    def store_in_memory(self, content: str) -> bool:
        """Store content in memory (real implementation when available)"""
        try:
            # Try to use real memory manager if available
            if ADVANCED_UTILS_AVAILABLE and memory_manager:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    success = loop.run_until_complete(
                        memory_manager.store_interaction("user_input", {"content": content, "timestamp": datetime.now().isoformat()})
                    )
                    loop.close()
                    return success
                except Exception as e:
                    logger.error(f"Real memory storage failed: {e}")
                    loop.close()
            
            # Fallback to mock memory storage
            logger.info(f"Storing content in memory: {content[:50]}...")
            return True
        except Exception as e:
            logger.error(f"Failed to store in memory: {e}")
            return False
    
    def retrieve_from_memory(self, query: str) -> list:
        """Retrieve content from memory (real implementation when available)"""
        try:
            # Try to use real memory manager if available
            if ADVANCED_UTILS_AVAILABLE and memory_manager:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    results = loop.run_until_complete(
                        memory_manager.search_memories(query, limit=5)
                    )
                    loop.close()
                    return results
                except Exception as e:
                    logger.error(f"Real memory retrieval failed: {e}")
                    loop.close()
            
            # Fallback to mock memory retrieval
            logger.info(f"Retrieving from memory with query: {query}")
            return [
                {
                    "content": f"Mock retrieved content for query: {query}",
                    "relevance": 0.8,
                    "timestamp": datetime.now().isoformat()
                }
            ]
        except Exception as e:
            logger.error(f"Failed to retrieve from memory: {e}")
            return []
    
    def start_training(self, config: dict) -> bool:
        """Start training process (real implementation when available)"""
        try:
            # Try to use real training manager if available
            if ADVANCED_UTILS_AVAILABLE and training_manager:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    success = loop.run_until_complete(
                        training_manager.start_training(config)
                    )
                    loop.close()
                    return success
                except Exception as e:
                    logger.error(f"Real training start failed: {e}")
                    loop.close()
            
            # Fallback to mock training start
            logger.info(f"Starting training with config: {config}")
            return True
        except Exception as e:
            logger.error(f"Failed to start training: {e}")
            return False
    
    def get_training_status(self) -> dict:
        """Get training status (real implementation when available)"""
        try:
            # Try to use real training manager if available
            if ADVANCED_UTILS_AVAILABLE and training_manager:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    status = loop.run_until_complete(
                        training_manager.get_training_status()
                    )
                    loop.close()
                    return status
                except Exception as e:
                    logger.error(f"Real training status failed: {e}")
                    loop.close()
            
            # Fallback to mock training status
            return {
                "status": "idle",
                "progress": 0,
                "current_epoch": 0,
                "total_epochs": 0,
                "loss": 0.0,
                "accuracy": 0.0,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to get training status: {e}")
            return {"error": str(e)}
    
    def analyze_code(self, code: str) -> dict:
        """Analyze code (real implementation when available)"""
        try:
            # Try to use real code executor if available
            if ADVANCED_UTILS_AVAILABLE and code_executor:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    analysis = loop.run_until_complete(
                        code_executor.analyze_code(code)
                    )
                    loop.close()
                    return analysis
                except Exception as e:
                    logger.error(f"Real code analysis failed: {e}")
                    loop.close()
            
            # Fallback to mock code analysis
            logger.info(f"Analyzing code: {code[:50]}...")
            return {
                "complexity": "medium",
                "quality_score": 0.7,
                "suggestions": [
                    "Consider adding type hints",
                    "Add error handling",
                    "Include docstrings"
                ],
                "language_detected": "python",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to analyze code: {e}")
            return {"error": str(e)}
    
    def analyze_text(self, text: str) -> dict:
        """Analyze text (real implementation when available)"""
        try:
            # Try to use real math engine if available
            if ADVANCED_UTILS_AVAILABLE and math_engine:
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    analysis = loop.run_until_complete(
                        math_engine.analyze_text(text)
                    )
                    loop.close()
                    return analysis
                except Exception as e:
                    logger.error(f"Real text analysis failed: {e}")
                    loop.close()
            
            # Fallback to mock text analysis
            logger.info(f"Analyzing text: {text[:50]}...")
            return {
                "sentiment": "neutral",
                "sentiment_score": 0.0,
                "language": "english",
                "word_count": len(text.split()),
                "key_topics": ["sample", "text", "analysis"],
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to analyze text: {e}")
            return {"error": str(e)}

# Initialize dependency manager
try:
    dependency_manager = ProgressiveDependencyManager()
    logger.info("Dependency manager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize dependency manager: {e}")
    dependency_manager = None

# Import existing advanced utilities
try:
    from api.utils.memory_manager import MemoryManager
    from api.utils.training_manager import TrainingManager
    from api.utils.web_search import WebSearcher
    from api.utils.math_engine import MathEngine
    from api.utils.code_executor import CodeExecutor
    ADVANCED_UTILS_AVAILABLE = True
    logger.info("Advanced utils imported successfully")
except ImportError as e:
    ADVANCED_UTILS_AVAILABLE = False
    logger.warning(f"Advanced utils not available: {e}")

# Initialize advanced components
memory_manager = None
training_manager = None
web_searcher = None
math_engine = None
code_executor = None

if ADVANCED_UTILS_AVAILABLE:
    try:
        # Initialize advanced components
        memory_manager = MemoryManager()
        training_manager = TrainingManager()
        web_searcher = WebSearcher()
        math_engine = MathEngine()
        code_executor = CodeExecutor()
        logger.info("Advanced components initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize advanced components: {e}")
        ADVANCED_UTILS_AVAILABLE = False

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        try:
            if self.path == "/":
                response = {
                    "message": "Autonomous AI Agent API",
                    "status": "Phase 3: Advanced ML & Real Dependencies" if ADVANCED_UTILS_AVAILABLE else "Phase 2: Machine Learning Core & Memory",
                    "version": "3.0.0" if ADVANCED_UTILS_AVAILABLE else "2.0.0",
                    "timestamp": datetime.now().isoformat(),
                    "features": ["web_search", "code_generation", "reasoning", "memory", "training", "analysis"],
                    "endpoints": {
                        "GET": ["/", "/status", "/dependencies", "/memory", "/models", "/advanced"],
                        "POST": ["/search", "/generate", "/reason", "/dependencies", "/memory", "/train", "/analyze"]
                    }
                }
            elif self.path == "/status":
                response = {
                    "status": "operational",
                    "phase": "Phase 3: Advanced ML & Real Dependencies" if ADVANCED_UTILS_AVAILABLE else "Phase 2: Machine Learning Core & Memory",
                    "timestamp": datetime.now().isoformat(),
                    "dependency_manager": dependency_manager.get_status() if dependency_manager else {"error": "Not initialized"}
                }
            elif self.path == "/dependencies":
                response = {
                    "dependencies": dependency_manager.get_status() if dependency_manager else {"error": "Not initialized"},
                    "installation_guide": "Use POST /dependencies to install packages",
                    "timestamp": datetime.now().isoformat()
                }
            elif self.path == "/memory":
                response = {
                    "memory": dependency_manager.get_memory_status() if dependency_manager else {"error": "Not initialized"},
                    "timestamp": datetime.now().isoformat()
                }
            elif self.path == "/models":
                response = {
                    "models": dependency_manager.get_models_status() if dependency_manager else {"error": "Not initialized"},
                    "timestamp": datetime.now().isoformat()
                }
            elif self.path == "/advanced":
                response = {
                    "advanced_utils": {
                        "available": ADVANCED_UTILS_AVAILABLE,
                        "components": {
                            "memory_manager": memory_manager is not None,
                            "training_manager": training_manager is not None,
                            "web_searcher": web_searcher is not None,
                            "math_engine": math_engine is not None,
                            "code_executor": code_executor is not None
                        },
                        "status": "Phase 3: Advanced ML & Real Dependencies" if ADVANCED_UTILS_AVAILABLE else "Phase 2: Mock implementations"
                    },
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
            elif self.path == "/memory":
                response = self._handle_memory(request_data)
            elif self.path == "/train":
                response = self._handle_training(request_data)
            elif self.path == "/analyze":
                response = self._handle_analysis(request_data)
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
    
    def _handle_memory(self, request_data: dict) -> dict:
        """Handle memory operations"""
        if not dependency_manager:
            return {
                "error": "Dependency manager not available",
                "timestamp": datetime.now().isoformat()
            }
        
        action = request_data.get('action', 'status')
        
        if action == 'store':
            content = request_data.get('content')
            if content:
                success = dependency_manager.store_in_memory(content)
                return {
                    "action": "store",
                    "success": success,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "error": "Content required for store action",
                    "timestamp": datetime.now().isoformat()
                }
        
        elif action == 'retrieve':
            query = request_data.get('query')
            if query:
                results = dependency_manager.retrieve_from_memory(query)
                return {
                    "action": "retrieve",
                    "results": results,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "error": "Query required for retrieve action",
                    "timestamp": datetime.now().isoformat()
                }
        
        else:  # Default to status
            return {
                "action": "status",
                "memory": dependency_manager.get_memory_status(),
                "timestamp": datetime.now().isoformat()
            }
    
    def _handle_training(self, request_data: dict) -> dict:
        """Handle training operations"""
        if not dependency_manager:
            return {
                "error": "Dependency manager not available",
                "timestamp": datetime.now().isoformat()
            }
        
        action = request_data.get('action', 'status')
        
        if action == 'start_training':
            training_config = request_data.get('config', {})
            success = dependency_manager.start_training(training_config)
            return {
                "action": "start_training",
                "success": success,
                "timestamp": datetime.now().isoformat()
            }
        
        elif action == 'training_status':
            status = dependency_manager.get_training_status()
            return {
                "action": "training_status",
                "status": status,
                "timestamp": datetime.now().isoformat()
            }
        
        else:  # Default to status
            return {
                "action": "status",
                "training": dependency_manager.get_training_status(),
                "timestamp": datetime.now().isoformat()
            }
    
    def _handle_analysis(self, request_data: dict) -> dict:
        """Handle analysis operations"""
        if not dependency_manager:
            return {
                "error": "Dependency manager not available",
                "timestamp": datetime.now().isoformat()
            }
        
        action = request_data.get('action', 'status')
        
        if action == 'analyze_code':
            code = request_data.get('code')
            if code:
                analysis = dependency_manager.analyze_code(code)
                return {
                    "action": "analyze_code",
                    "analysis": analysis,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "error": "Code required for analyze_code action",
                    "timestamp": datetime.now().isoformat()
                }
        
        elif action == 'analyze_text':
            text = request_data.get('text')
            if text:
                analysis = dependency_manager.analyze_text(text)
                return {
                    "action": "analyze_text",
                    "analysis": analysis,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "error": "Text required for analyze_text action",
                    "timestamp": datetime.now().isoformat()
                }
        
        else:  # Default to status
            return {
                "action": "status",
                "analysis": "Available analysis methods: analyze_code, analyze_text",
                "timestamp": datetime.now().isoformat()
            }
    
    def _handle_test(self, request_data: dict) -> dict:
        """Handle test requests - simple endpoint to verify POST works"""
        return {
            "message": "Test POST endpoint working",
            "received_data": request_data,
            "timestamp": datetime.now().isoformat()
        }
