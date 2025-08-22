"""
Progressive Dependency Manager for Vercel Python Serverless Functions
Allows dynamic loading of heavy dependencies based on feature requirements
"""

import importlib
import subprocess
import sys
import os
import json
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProgressiveDependencyManager:
    """Manages progressive installation and loading of Python dependencies"""
    
    def __init__(self):
        self.installed_packages = set()
        self.available_features = set()
        self.feature_dependencies = {
            "web_search": ["requests", "beautifulsoup4"],
            "ml_models": ["torch", "transformers", "sentence-transformers"],
            "vector_store": ["faiss-cpu", "numpy"],
            "code_analysis": ["ast", "black", "flake8"],
            "advanced_auth": ["pyjwt", "cryptography"],
            "data_processing": ["pandas", "numpy", "scipy"],
            "graph_analysis": ["networkx", "matplotlib"],
            "nlp_enhanced": ["nltk", "spacy", "textblob"]
        }
        
        # Initialize with basic features
        self.available_features.update([
            "basic_api", "authentication", "code_generation", "reasoning"
        ])
        
        # Check what's already available
        self._check_installed_packages()
    
    def _check_installed_packages(self):
        """Check which packages are already available"""
        for package in ["requests", "torch", "transformers", "numpy", "pandas"]:
            try:
                importlib.import_module(package)
                self.installed_packages.add(package)
                logger.info(f"Package {package} is already available")
            except ImportError:
                logger.info(f"Package {package} is not available")
    
    def install_package(self, package_name: str) -> bool:
        """Install a single package using pip"""
        try:
            logger.info(f"Installing package: {package_name}")
            
            # Use subprocess to install package
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", 
                package_name, "--quiet", "--no-cache-dir"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                self.installed_packages.add(package_name)
                logger.info(f"Successfully installed {package_name}")
                return True
            else:
                logger.error(f"Failed to install {package_name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout installing {package_name}")
            return False
        except Exception as e:
            logger.error(f"Error installing {package_name}: {str(e)}")
            return False
    
    def enable_feature(self, feature_name: str) -> bool:
        """Enable a specific feature by installing its dependencies"""
        if feature_name in self.available_features:
            logger.info(f"Feature {feature_name} is already available")
            return True
        
        if feature_name not in self.feature_dependencies:
            logger.error(f"Unknown feature: {feature_name}")
            return False
        
        dependencies = self.feature_dependencies[feature_name]
        logger.info(f"Enabling feature {feature_name} with dependencies: {dependencies}")
        
        success_count = 0
        for dep in dependencies:
            if dep not in self.installed_packages:
                if self.install_package(dep):
                    success_count += 1
                else:
                    logger.warning(f"Failed to install dependency {dep} for feature {feature_name}")
            else:
                success_count += 1
        
        if success_count == len(dependencies):
            self.available_features.add(feature_name)
            logger.info(f"Successfully enabled feature: {feature_name}")
            return True
        else:
            logger.error(f"Failed to enable feature {feature_name}: {success_count}/{len(dependencies)} dependencies installed")
            return False
    
    def get_feature_status(self) -> Dict[str, bool]:
        """Get status of all features"""
        return {feature: feature in self.available_features for feature in self.feature_dependencies.keys()}
    
    def load_module_safely(self, module_name: str) -> Optional[object]:
        """Safely load a module, installing it if necessary"""
        try:
            return importlib.import_module(module_name)
        except ImportError:
            logger.info(f"Module {module_name} not found, attempting to install...")
            if self.install_package(module_name):
                try:
                    return importlib.import_module(module_name)
                except ImportError:
                    logger.error(f"Failed to load {module_name} even after installation")
                    return None
            else:
                logger.error(f"Failed to install {module_name}")
                return None
    
    def get_enhanced_search(self, query: str, depth: int = 5) -> Dict:
        """Enhanced web search using progressive dependency loading"""
        if "web_search" not in self.available_features:
            if not self.enable_feature("web_search"):
                return self._get_mock_search(query, depth)
        
        try:
            import requests
            from bs4 import BeautifulSoup
            
            # Real web search implementation
            search_results = []
            
            # DuckDuckGo search (no API key required)
            search_url = f"https://duckduckgo.com/html/?q={query}"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(search_url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                results = soup.find_all('div', class_='result')[:depth]
                
                for result in results:
                    title_elem = result.find('a', class_='result__a')
                    snippet_elem = result.find('div', class_='result__snippet')
                    
                    if title_elem and snippet_elem:
                        search_results.append({
                            "title": title_elem.get_text(strip=True),
                            "snippet": snippet_elem.get_text(strip=True),
                            "url": title_elem.get('href', ''),
                            "type": "web_search"
                        })
            
            return {
                "results": search_results,
                "synthesized_answer": f"Found {len(search_results)} real web search results for '{query}'",
                "confidence": 0.9,
                "method": "enhanced_web_search",
                "query": query,
                "result_count": len(search_results),
                "timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            logger.error(f"Enhanced search failed: {str(e)}")
            return self._get_mock_search(query, depth)
    
    def get_enhanced_code_generation(self, prompt: str, language: str = "python") -> Dict:
        """Enhanced code generation with ML models if available"""
        if "ml_models" not in self.available_features:
            if not self.enable_feature("ml_models"):
                return self._get_template_code_generation(prompt, language)
        
        try:
            # Try to use transformers for enhanced code generation
            transformers = self.load_module_safely("transformers")
            if transformers:
                return self._get_ml_enhanced_code(prompt, language)
            else:
                return self._get_template_code_generation(prompt, language)
                
        except Exception as e:
            logger.error(f"Enhanced code generation failed: {str(e)}")
            return self._get_template_code_generation(prompt, language)
    
    def get_enhanced_reasoning(self, problem: str) -> Dict:
        """Enhanced reasoning with ML models if available"""
        if "nlp_enhanced" not in self.available_features:
            if not self.enable_feature("nlp_enhanced"):
                return self._get_basic_reasoning(problem)
        
        try:
            # Try to use NLTK for enhanced reasoning
            nltk = self.load_module_safely("nltk")
            if nltk:
                return self._get_nlp_enhanced_reasoning(problem)
            else:
                return self._get_basic_reasoning(problem)
                
        except Exception as e:
            logger.error(f"Enhanced reasoning failed: {str(e)}")
            return self._get_basic_reasoning(problem)
    
    def _get_mock_search(self, query: str, depth: int) -> Dict:
        """Fallback mock search"""
        mock_results = [
            {
                "title": f"Search Result for: {query}",
                "snippet": f"This is a comprehensive result about {query} with detailed information and examples.",
                "url": f"https://example.com/search?q={query}",
                "type": "mock_result"
            }
        ]
        
        return {
            "results": mock_results[:depth],
            "synthesized_answer": f"Mock search results for '{query}' (enhanced search not available)",
            "confidence": 0.5,
            "method": "mock_search",
            "query": query,
            "result_count": len(mock_results[:depth]),
            "timestamp": self._get_timestamp()
        }
    
    def _get_template_code_generation(self, prompt: str, language: str) -> Dict:
        """Template-based code generation (fallback)"""
        # This would contain the existing template logic
        return {
            "code": f"# {prompt}\n# Template-based implementation\n# TODO: Implement solution",
            "explanation": f"Template-based {language} code for: {prompt}",
            "confidence": 0.7,
            "method": "template_generation",
            "timestamp": self._get_timestamp()
        }
    
    def _get_ml_enhanced_code(self, prompt: str, language: str) -> Dict:
        """ML-enhanced code generation using transformers"""
        try:
            from transformers import pipeline
            
            # Use code generation pipeline
            generator = pipeline("text-generation", model="microsoft/DialoGPT-medium")
            enhanced_prompt = f"Generate {language} code for: {prompt}"
            
            result = generator(enhanced_prompt, max_length=200, num_return_sequences=1)
            generated_code = result[0]['generated_text']
            
            return {
                "code": generated_code,
                "explanation": f"ML-enhanced {language} code generation for: {prompt}",
                "confidence": 0.85,
                "method": "ml_enhanced",
                "timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            logger.error(f"ML-enhanced code generation failed: {str(e)}")
            return self._get_template_code_generation(prompt, language)
    
    def _get_basic_reasoning(self, problem: str) -> Dict:
        """Basic reasoning (fallback)"""
        return {
            "analysis": f"Problem analysis: {problem}",
            "solution": "Basic reasoning approach",
            "confidence": 0.6,
            "method": "basic_reasoning",
            "timestamp": self._get_timestamp()
        }
    
    def _get_nlp_enhanced_reasoning(self, problem: str) -> Dict:
        """NLP-enhanced reasoning using NLTK"""
        try:
            import nltk
            
            # Download required NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            
            # Tokenize and analyze the problem
            tokens = nltk.word_tokenize(problem)
            pos_tags = nltk.pos_tag(tokens)
            
            # Extract key concepts
            nouns = [word for word, pos in pos_tags if pos.startswith('NN')]
            verbs = [word for word, pos in pos_tags if pos.startswith('VB')]
            
            analysis = {
                "problem": problem,
                "key_concepts": nouns,
                "actions": verbs,
                "token_count": len(tokens),
                "complexity": "high" if len(tokens) > 20 else "medium" if len(tokens) > 10 else "low"
            }
            
            return {
                "analysis": analysis,
                "solution": f"NLP-enhanced analysis with {len(nouns)} key concepts and {len(verbs)} actions",
                "confidence": 0.8,
                "method": "nlp_enhanced_reasoning",
                "timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            logger.error(f"NLP-enhanced reasoning failed: {str(e)}")
            return self._get_basic_reasoning(problem)
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_system_status(self) -> Dict:
        """Get comprehensive system status"""
        return {
            "available_features": list(self.available_features),
            "installed_packages": list(self.installed_packages),
            "feature_status": self.get_feature_status(),
            "total_features": len(self.feature_dependencies),
            "enabled_features": len(self.available_features),
            "timestamp": self._get_timestamp()
        }

# Global instance
dependency_manager = ProgressiveDependencyManager()
