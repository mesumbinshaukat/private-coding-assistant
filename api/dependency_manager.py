"""
Dynamic Dependency Manager for Autonomous AI Agent

This module handles progressive installation of heavy dependencies
after successful deployment to avoid OOM errors during build.
"""

import os
import sys
import subprocess
import importlib
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import asyncio
from datetime import datetime

logger = logging.getLogger(__name__)

class DependencyManager:
    """Manages dynamic installation of dependencies post-deployment"""
    
    def __init__(self):
        self.dependency_phases = {
            "phase_1_core": [
                "fastapi==0.104.1",
                "uvicorn==0.24.0", 
                "pydantic==2.5.2",
                "pyjwt==2.8.0",
                "requests==2.31.0",
                "python-dotenv==1.0.0"
            ],
            "phase_2_ai_lightweight": [
                "numpy==1.24.4",
                "beautifulsoup4==4.12.2",
                "aiohttp==3.9.1",
                "duckduckgo-search==3.9.6"
            ],
            "phase_3_ai_core": [
                "torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu",
                "transformers==4.35.2",
                "tokenizers==0.15.0"
            ],
            "phase_4_ai_advanced": [
                "langchain==0.0.340",
                "langchain-community==0.0.6",
                "sentence-transformers==2.2.2",
                "faiss-cpu==1.7.4"
            ],
            "phase_5_training": [
                "datasets==2.14.7",
                "peft==0.6.2", 
                "accelerate==0.24.1"
            ],
            "phase_6_math_science": [
                "scipy==1.11.4",
                "sympy==1.12",
                "networkx==3.2.1",
                "scikit-learn==1.3.2",
                "pandas==2.1.4",
                "nltk==3.8.1"
            ]
        }
        
        self.status_file = Path("dependency_status.json")
        self.load_status()
    
    def load_status(self):
        """Load dependency installation status"""
        try:
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    self.status = json.load(f)
            else:
                self.status = {
                    "phases_completed": ["phase_1_core"],  # Core is pre-installed
                    "current_phase": "phase_2_ai_lightweight",
                    "failed_packages": [],
                    "last_updated": datetime.now().isoformat(),
                    "installation_mode": "auto"  # auto, manual, disabled
                }
                self.save_status()
        except Exception as e:
            logger.error(f"Failed to load dependency status: {e}")
            self.status = {
                "phases_completed": ["phase_1_core"],
                "current_phase": "phase_2_ai_lightweight", 
                "failed_packages": [],
                "last_updated": datetime.now().isoformat(),
                "installation_mode": "auto"
            }
    
    def save_status(self):
        """Save dependency installation status"""
        try:
            self.status["last_updated"] = datetime.now().isoformat()
            with open(self.status_file, 'w') as f:
                json.dump(self.status, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save dependency status: {e}")
    
    def check_package_installed(self, package_name: str) -> bool:
        """Check if a package is installed"""
        try:
            # Extract package name from requirement string
            pkg_name = package_name.split("==")[0].split(">=")[0].split("[")[0]
            importlib.import_module(pkg_name.replace("-", "_"))
            return True
        except ImportError:
            return False
        except Exception:
            return False
    
    async def install_package(self, package: str) -> Tuple[bool, str]:
        """Install a single package"""
        try:
            logger.info(f"Installing package: {package}")
            
            # Run pip install in subprocess
            cmd = [sys.executable, "-m", "pip", "install", package, "--no-cache-dir", "--timeout", "300"]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=600)
            
            if process.returncode == 0:
                logger.info(f"Successfully installed: {package}")
                return True, f"Installed {package}"
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"Failed to install {package}: {error_msg}")
                return False, f"Failed to install {package}: {error_msg}"
                
        except asyncio.TimeoutError:
            return False, f"Timeout installing {package}"
        except Exception as e:
            logger.error(f"Exception installing {package}: {e}")
            return False, f"Exception: {str(e)}"
    
    async def install_phase(self, phase_name: str) -> Dict[str, Any]:
        """Install all packages in a phase"""
        if phase_name not in self.dependency_phases:
            return {"success": False, "error": f"Unknown phase: {phase_name}"}
        
        if phase_name in self.status["phases_completed"]:
            return {"success": True, "message": f"Phase {phase_name} already completed"}
        
        packages = self.dependency_phases[phase_name]
        results = []
        success_count = 0
        
        logger.info(f"Starting installation of phase: {phase_name}")
        
        for package in packages:
            # Check if already installed
            pkg_name = package.split("==")[0].split(">=")[0]
            if self.check_package_installed(pkg_name):
                results.append({"package": package, "status": "already_installed"})
                success_count += 1
                continue
            
            # Install package
            success, message = await self.install_package(package)
            
            result = {
                "package": package,
                "status": "success" if success else "failed",
                "message": message
            }
            results.append(result)
            
            if success:
                success_count += 1
            else:
                self.status["failed_packages"].append(package)
        
        # Update status
        if success_count == len(packages):
            self.status["phases_completed"].append(phase_name)
            logger.info(f"Phase {phase_name} completed successfully")
        
        # Set next phase
        phase_order = list(self.dependency_phases.keys())
        current_index = phase_order.index(phase_name)
        if current_index + 1 < len(phase_order):
            next_phase = phase_order[current_index + 1]
            if next_phase not in self.status["phases_completed"]:
                self.status["current_phase"] = next_phase
        
        self.save_status()
        
        return {
            "success": success_count == len(packages),
            "phase": phase_name,
            "total_packages": len(packages),
            "successful_packages": success_count,
            "failed_packages": len(packages) - success_count,
            "results": results
        }
    
    async def install_next_phase(self) -> Dict[str, Any]:
        """Install the next phase automatically"""
        if self.status["installation_mode"] == "disabled":
            return {"success": False, "message": "Auto-installation is disabled"}
        
        current_phase = self.status["current_phase"]
        
        if current_phase not in self.dependency_phases:
            return {"success": False, "message": "All phases completed or invalid phase"}
        
        return await self.install_phase(current_phase)
    
    def get_status(self) -> Dict[str, Any]:
        """Get current dependency status"""
        return {
            "dependency_status": self.status,
            "available_phases": list(self.dependency_phases.keys()),
            "phase_descriptions": {
                "phase_1_core": "Essential FastAPI and authentication",
                "phase_2_ai_lightweight": "Basic utilities and web scraping", 
                "phase_3_ai_core": "PyTorch and Transformers",
                "phase_4_ai_advanced": "LangChain and vector search",
                "phase_5_training": "Model training capabilities",
                "phase_6_math_science": "Scientific computing libraries"
            }
        }
    
    def get_available_features(self) -> Dict[str, Any]:
        """Get features available based on installed dependencies"""
        features = {
            "core_api": True,  # Always available
            "authentication": True,  # Always available
            "web_search_basic": "phase_2_ai_lightweight" in self.status["phases_completed"],
            "ai_code_generation": "phase_3_ai_core" in self.status["phases_completed"],
            "advanced_reasoning": "phase_4_ai_advanced" in self.status["phases_completed"],
            "self_training": "phase_5_training" in self.status["phases_completed"],
            "mathematical_analysis": "phase_6_math_science" in self.status["phases_completed"]
        }
        
        return features
    
    def set_installation_mode(self, mode: str) -> bool:
        """Set installation mode: auto, manual, disabled"""
        if mode in ["auto", "manual", "disabled"]:
            self.status["installation_mode"] = mode
            self.save_status()
            return True
        return False
    
    async def install_specific_package(self, package: str) -> Dict[str, Any]:
        """Install a specific package manually"""
        success, message = await self.install_package(package)
        return {
            "success": success,
            "package": package,
            "message": message
        }
    
    def retry_failed_packages(self) -> List[str]:
        """Get list of failed packages for retry"""
        return self.status["failed_packages"].copy()
    
    async def retry_failed(self) -> Dict[str, Any]:
        """Retry installing failed packages"""
        failed_packages = self.retry_failed_packages()
        if not failed_packages:
            return {"success": True, "message": "No failed packages to retry"}
        
        results = []
        success_count = 0
        
        for package in failed_packages:
            success, message = await self.install_package(package)
            results.append({
                "package": package,
                "status": "success" if success else "failed",
                "message": message
            })
            
            if success:
                success_count += 1
                # Remove from failed list
                if package in self.status["failed_packages"]:
                    self.status["failed_packages"].remove(package)
        
        self.save_status()
        
        return {
            "success": success_count > 0,
            "total_retried": len(failed_packages),
            "successful_retries": success_count,
            "results": results
        }

# Global dependency manager instance
dependency_manager = DependencyManager()
