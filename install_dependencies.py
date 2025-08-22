#!/usr/bin/env python3
"""
Post-Deployment Dependency Installation Script

This script helps install heavy dependencies after successful Vercel deployment
to avoid Out of Memory (OOM) errors during the build process.

Usage:
    python install_dependencies.py --phase phase_2_ai_lightweight
    python install_dependencies.py --all
    python install_dependencies.py --auto
"""

import requests
import json
import time
import argparse
from typing import Dict, Any

class DependencyInstaller:
    """Client for installing dependencies via API"""
    
    def __init__(self, api_base: str, token: str):
        self.api_base = api_base.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get current dependency status"""
        try:
            response = requests.get(f"{self.api_base}/dependencies/status", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def install_phase(self, phase: str) -> Dict[str, Any]:
        """Install a specific phase"""
        try:
            data = {"phase": phase}
            response = requests.post(f"{self.api_base}/dependencies/install", 
                                   json=data, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def install_next(self) -> Dict[str, Any]:
        """Install next phase automatically"""
        try:
            response = requests.post(f"{self.api_base}/dependencies/install", 
                                   json={}, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def set_mode(self, mode: str) -> Dict[str, Any]:
        """Set installation mode"""
        try:
            data = {"mode": mode}
            response = requests.post(f"{self.api_base}/dependencies/install",
                                   json=data, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def retry_failed(self) -> Dict[str, Any]:
        """Retry failed installations"""
        try:
            response = requests.post(f"{self.api_base}/dependencies/retry", headers=self.headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}

def print_status(status: Dict[str, Any]):
    """Print formatted status"""
    print("\n" + "="*60)
    print("DEPENDENCY STATUS")
    print("="*60)
    
    if "error" in status:
        print(f"❌ Error: {status['error']}")
        return
    
    dep_status = status.get("dependency_status", {})
    
    print(f"Installation Mode: {dep_status.get('installation_mode', 'unknown')}")
    print(f"Current Phase: {dep_status.get('current_phase', 'unknown')}")
    print(f"Completed Phases: {', '.join(dep_status.get('phases_completed', []))}")
    
    failed = dep_status.get('failed_packages', [])
    if failed:
        print(f"Failed Packages: {len(failed)}")
    
    print("\nAvailable Phases:")
    descriptions = status.get("phase_descriptions", {})
    for phase, desc in descriptions.items():
        completed = "✅" if phase in dep_status.get('phases_completed', []) else "⏳"
        print(f"  {completed} {phase}: {desc}")

def install_all_phases(installer: DependencyInstaller):
    """Install all phases sequentially"""
    phases = [
        "phase_2_ai_lightweight",
        "phase_3_ai_core", 
        "phase_4_ai_advanced",
        "phase_5_training",
        "phase_6_math_science"
    ]
    
    print("🚀 Installing all phases sequentially...")
    
    for phase in phases:
        print(f"\n📦 Installing {phase}...")
        result = installer.install_phase(phase)
        
        if "error" in result:
            print(f"❌ Failed to install {phase}: {result['error']}")
            break
        
        if result.get("success"):
            success_count = result.get("successful_packages", 0)
            total_count = result.get("total_packages", 0)
            print(f"✅ {phase} completed: {success_count}/{total_count} packages installed")
        else:
            failed_count = result.get("failed_packages", 0)
            print(f"⚠️ {phase} partially completed: {failed_count} packages failed")
        
        # Wait between phases
        print("⏳ Waiting 30 seconds before next phase...")
        time.sleep(30)
    
    print("\n🎉 All phases installation completed!")

def auto_install(installer: DependencyInstaller):
    """Enable auto-installation mode"""
    print("🔄 Enabling auto-installation mode...")
    
    result = installer.set_mode("auto")
    if "error" in result:
        print(f"❌ Failed to set auto mode: {result['error']}")
        return
    
    print("✅ Auto-installation mode enabled")
    print("🚀 Dependencies will install automatically in the background")
    
    # Check status periodically
    for i in range(12):  # Check for 2 minutes
        time.sleep(10)
        status = installer.get_status()
        
        if "error" not in status:
            current_phase = status.get("dependency_status", {}).get("current_phase")
            completed = status.get("dependency_status", {}).get("phases_completed", [])
            print(f"📊 Status check {i+1}: Current phase: {current_phase}, Completed: {len(completed)}")

def main():
    parser = argparse.ArgumentParser(description="Install dependencies for Autonomous AI Agent")
    parser.add_argument("--api-base", default="https://your-app.vercel.app", 
                       help="API base URL")
    parser.add_argument("--token", default="autonomous-ai-agent-secret-key-2024",
                       help="API authentication token")
    parser.add_argument("--phase", help="Specific phase to install")
    parser.add_argument("--all", action="store_true", help="Install all phases")
    parser.add_argument("--auto", action="store_true", help="Enable auto-installation")
    parser.add_argument("--status", action="store_true", help="Show status only")
    parser.add_argument("--retry", action="store_true", help="Retry failed installations")
    
    args = parser.parse_args()
    
    installer = DependencyInstaller(args.api_base, args.token)
    
    # Always show status first
    print("📊 Checking current status...")
    status = installer.get_status()
    print_status(status)
    
    if args.status:
        return
    
    if args.retry:
        print("\n🔄 Retrying failed installations...")
        result = installer.retry_failed()
        print(f"Result: {json.dumps(result, indent=2)}")
        return
    
    if args.auto:
        auto_install(installer)
        return
    
    if args.all:
        install_all_phases(installer)
        return
    
    if args.phase:
        print(f"\n📦 Installing phase: {args.phase}")
        result = installer.install_phase(args.phase)
        print(f"Result: {json.dumps(result, indent=2)}")
        return
    
    # Default: install next phase
    print("\n📦 Installing next phase...")
    result = installer.install_next()
    print(f"Result: {json.dumps(result, indent=2)}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Installation cancelled by user")
    except Exception as e:
        print(f"\n💥 Installation failed: {e}")
        print("\nTroubleshooting:")
        print("1. Check that your API is deployed and accessible")
        print("2. Verify the API token is correct")
        print("3. Ensure the API base URL is correct")
        print("4. Check API logs for detailed error information")
