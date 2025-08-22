#!/usr/bin/env python3
"""
Progressive Dependency Installer for Autonomous AI Agent
Installs dependencies in phases to avoid Vercel build issues
"""

import subprocess
import sys
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

class ProgressiveInstaller:
    """Installs dependencies progressively in phases"""
    
    def __init__(self):
        self.phases = {
            "phase_1": {
                "name": "Basic Web and Search",
                "description": "Lightweight web scraping and HTTP requests",
                "packages": ["requests==2.31.0", "beautifulsoup4==4.12.2", "lxml==4.9.3"],
                "estimated_size": "15MB"
            },
            "phase_2": {
                "name": "Machine Learning Core",
                "description": "PyTorch and Transformers for AI capabilities",
                "packages": ["torch==2.1.0+cpu", "transformers==4.35.2", "tokenizers==0.15.0", "numpy==1.24.4"],
                "estimated_size": "800MB"
            },
            "phase_3": {
                "name": "Advanced AI Features",
                "description": "Vector search and LangChain integration",
                "packages": ["sentence-transformers==2.2.2", "faiss-cpu==1.7.4", "langchain==0.0.340", "langchain-community==0.0.6"],
                "estimated_size": "500MB"
            },
            "phase_4": {
                "name": "Data Processing",
                "description": "Pandas, SciPy, and scikit-learn",
                "packages": ["pandas==2.1.4", "scipy==1.11.4", "scikit-learn==1.3.2", "networkx==3.2.1"],
                "estimated_size": "200MB"
            },
            "phase_5": {
                "name": "Natural Language Processing",
                "description": "NLTK, spaCy, and TextBlob",
                "packages": ["nltk==3.8.1", "spacy==3.7.2", "textblob==0.17.1"],
                "estimated_size": "300MB"
            }
        }
        
        self.status_file = Path("installation_status.json")
        self.load_status()
    
    def load_status(self):
        """Load installation status from file"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    self.status = json.load(f)
            except:
                self.status = self._get_default_status()
        else:
            self.status = self._get_default_status()
        self.save_status()
    
    def _get_default_status(self):
        """Get default installation status"""
        return {
            "phases_completed": [],
            "current_phase": "phase_1",
            "failed_packages": [],
            "total_installed": 0,
            "last_updated": self._get_timestamp(),
            "installation_mode": "manual"
        }
    
    def save_status(self):
        """Save installation status to file"""
        self.status["last_updated"] = self._get_timestamp()
        try:
            with open(self.status_file, 'w') as f:
                json.dump(self.status, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save status: {e}")
    
    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def install_package(self, package: str) -> Tuple[bool, str]:
        """Install a single package"""
        try:
            print(f"Installing: {package}")
            
            # Handle PyTorch CPU-only installation
            if "torch==2.1.0+cpu" in package:
                cmd = [sys.executable, "-m", "pip", "install", "torch==2.1.0+cpu", 
                       "--index-url", "https://download.pytorch.org/whl/cpu", 
                       "--no-cache-dir", "--quiet"]
            else:
                cmd = [sys.executable, "-m", "pip", "install", package, "--no-cache-dir", "--quiet"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print(f"✅ Successfully installed: {package}")
                return True, f"Installed {package}"
            else:
                error_msg = result.stderr if result.stderr else "Unknown error"
                print(f"❌ Failed to install {package}: {error_msg}")
                return False, f"Failed to install {package}: {error_msg}"
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Timeout installing {package}")
            return False, f"Timeout installing {package}"
        except Exception as e:
            print(f"💥 Exception installing {package}: {e}")
            return False, f"Exception: {str(e)}"
    
    def install_phase(self, phase_name: str) -> Dict:
        """Install all packages in a specific phase"""
        if phase_name not in self.phases:
            return {"success": False, "error": f"Unknown phase: {phase_name}"}
        
        if phase_name in self.status["phases_completed"]:
            return {"success": True, "message": f"Phase {phase_name} already completed"}
        
        phase_info = self.phases[phase_name]
        packages = phase_info["packages"]
        
        print(f"\n🚀 Starting Phase: {phase_info['name']}")
        print(f"📝 Description: {phase_info['description']}")
        print(f"📦 Packages: {len(packages)}")
        print(f"💾 Estimated Size: {phase_info['estimated_size']}")
        print("=" * 60)
        
        results = []
        success_count = 0
        
        for package in packages:
            success, message = self.install_package(package)
            results.append({
                "package": package,
                "status": "success" if success else "failed",
                "message": message
            })
            
            if success:
                success_count += 1
                self.status["total_installed"] += 1
            else:
                self.status["failed_packages"].append(package)
        
        # Update status
        if success_count == len(packages):
            self.status["phases_completed"].append(phase_name)
            print(f"\n🎉 Phase {phase_name} completed successfully!")
        else:
            print(f"\n⚠️  Phase {phase_name} partially completed: {success_count}/{len(packages)}")
        
        # Set next phase
        phase_order = list(self.phases.keys())
        current_index = phase_order.index(phase_name)
        if current_index + 1 < len(phase_order):
            next_phase = phase_order[current_index + 1]
            if next_phase not in self.status["phases_completed"]:
                self.status["current_phase"] = next_phase
        
        self.save_status()
        
        return {
            "success": success_count == len(packages),
            "phase": phase_name,
            "phase_info": phase_info,
            "total_packages": len(packages),
            "successful_packages": success_count,
            "failed_packages": len(packages) - success_count,
            "results": results
        }
    
    def install_next_phase(self) -> Dict:
        """Install the next available phase"""
        current_phase = self.status["current_phase"]
        
        if current_phase not in self.phases:
            return {"success": False, "error": "All phases completed or invalid phase"}
        
        return self.install_phase(current_phase)
    
    def install_all_phases(self) -> Dict:
        """Install all remaining phases"""
        print("🚀 Installing all remaining phases...")
        
        results = []
        total_success = 0
        
        for phase_name in self.phases:
            if phase_name not in self.status["phases_completed"]:
                result = self.install_phase(phase_name)
                results.append(result)
                if result["success"]:
                    total_success += 1
        
        return {
            "success": total_success == len(results),
            "total_phases": len(results),
            "successful_phases": total_success,
            "results": results
        }
    
    def get_status(self) -> Dict:
        """Get current installation status"""
        return {
            "installation_status": self.status,
            "available_phases": list(self.phases.keys()),
            "phase_descriptions": {k: v["description"] for k, v in self.phases.items()},
            "next_phase": self.status["current_phase"],
            "total_phases": len(self.phases),
            "completed_phases": len(self.status["phases_completed"])
        }
    
    def retry_failed(self) -> Dict:
        """Retry installing failed packages"""
        failed_packages = self.status["failed_packages"].copy()
        if not failed_packages:
            return {"success": True, "message": "No failed packages to retry"}
        
        print(f"🔄 Retrying {len(failed_packages)} failed packages...")
        
        results = []
        success_count = 0
        
        for package in failed_packages:
            success, message = self.install_package(package)
            results.append({
                "package": package,
                "status": "success" if success else "failed",
                "message": message
            })
            
            if success:
                success_count += 1
                self.status["failed_packages"].remove(package)
                self.status["total_installed"] += 1
        
        self.save_status()
        
        return {
            "success": success_count > 0,
            "total_retried": len(failed_packages),
            "successful_retries": success_count,
            "results": results
        }
    
    def reset_status(self):
        """Reset installation status"""
        self.status = self._get_default_status()
        self.save_status()
        print("🔄 Installation status reset")

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(description="Progressive Dependency Installer")
    parser.add_argument("--phase", help="Install specific phase (e.g., phase_1)")
    parser.add_argument("--next", action="store_true", help="Install next available phase")
    parser.add_argument("--all", action="store_true", help="Install all remaining phases")
    parser.add_argument("--status", action="store_true", help="Show installation status")
    parser.add_argument("--retry", action="store_true", help="Retry failed packages")
    parser.add_argument("--reset", action="store_true", help="Reset installation status")
    parser.add_argument("--auto", action="store_true", help="Auto-install next phase")
    
    args = parser.parse_args()
    
    installer = ProgressiveInstaller()
    
    if args.status:
        status = installer.get_status()
        print(json.dumps(status, indent=2))
    
    elif args.reset:
        installer.reset_status()
    
    elif args.retry:
        result = installer.retry_failed()
        print(json.dumps(result, indent=2))
    
    elif args.phase:
        result = installer.install_phase(args.phase)
        print(json.dumps(result, indent=2))
    
    elif args.next:
        result = installer.install_next_phase()
        print(json.dumps(result, indent=2))
    
    elif args.all:
        result = installer.install_all_phases()
        print(json.dumps(result, indent=2))
    
    elif args.auto:
        result = installer.install_next_phase()
        print(json.dumps(result, indent=2))
    
    else:
        # Show help
        print("Progressive Dependency Installer")
        print("Available commands:")
        print("  --status     Show installation status")
        print("  --next       Install next available phase")
        print("  --phase X    Install specific phase")
        print("  --all        Install all remaining phases")
        print("  --retry      Retry failed packages")
        print("  --reset      Reset installation status")
        print("  --auto       Auto-install next phase")
        
        # Show current status
        status = installer.get_status()
        print(f"\nCurrent Status:")
        print(f"  Completed: {status['completed_phases']}/{status['total_phases']} phases")
        print(f"  Next Phase: {status['next_phase']}")

if __name__ == "__main__":
    main()
