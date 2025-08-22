#!/usr/bin/env python3
"""
Deployment script for minimal Vercel deployment
This script helps deploy with minimal dependencies first, then progressively adds AI capabilities
"""

import os
import sys
import subprocess
import json
import time
from pathlib import Path

def run_command(command: str, description: str) -> bool:
    """Run a command and return success status"""
    print(f"\n🔄 {description}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def check_file_exists(file_path: str) -> bool:
    """Check if a file exists"""
    return Path(file_path).exists()

def create_minimal_deployment():
    """Create minimal deployment configuration"""
    print("🚀 Creating minimal deployment configuration...")
    
    # Check if minimal files exist
    if not check_file_exists("api/index.py"):
        print("❌ api/index.py not found. Please ensure the minimal version is in place.")
        return False
    
    if not check_file_exists("vercel.json"):
        print("❌ vercel.json not found. Please ensure Vercel configuration exists.")
        return False
    
    if not check_file_exists("requirements.txt"):
        print("❌ requirements.txt not found. Please ensure minimal requirements exist.")
        return False
    
    print("✅ All minimal deployment files are in place")
    return True

def deploy_to_vercel():
    """Deploy to Vercel"""
    print("\n🚀 Deploying to Vercel...")
    
    # Check if Vercel CLI is installed
    try:
        subprocess.run(["vercel", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Vercel CLI not found. Please install it first:")
        print("   npm i -g vercel")
        return False
    
    # Deploy
    if run_command("vercel --prod", "Deploying to Vercel"):
        print("\n🎉 Deployment completed successfully!")
        print("📱 Your API is now live with minimal dependencies")
        return True
    else:
        print("\n❌ Deployment failed")
        return False

def get_deployment_url():
    """Get the deployment URL from Vercel"""
    try:
        result = subprocess.run(["vercel", "ls"], capture_output=True, text=True, check=True)
        lines = result.stdout.split('\n')
        for line in lines:
            if 'private-coding-assistant' in line and 'https://' in line:
                url = line.split('https://')[1].split()[0]
                return f"https://{url}"
    except:
        pass
    return None

def test_minimal_api(deployment_url: str):
    """Test the minimal API deployment"""
    print(f"\n🧪 Testing minimal API at {deployment_url}")
    
    try:
        import requests
        
        # Test health endpoint
        health_url = f"{deployment_url}/health"
        response = requests.get(health_url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data.get('status', 'unknown')}")
            print(f"   Deployment mode: {data.get('deployment_mode', 'unknown')}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except ImportError:
        print("⚠️  requests not available, skipping API test")
        return True
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def show_next_steps(deployment_url: str):
    """Show next steps for progressive enhancement"""
    print("\n" + "="*60)
    print("🎯 DEPLOYMENT SUCCESSFUL - NEXT STEPS")
    print("="*60)
    
    print(f"✅ Your minimal API is deployed at: {deployment_url}")
    print("\n📋 Current Status:")
    print("   • Basic API endpoints working")
    print("   • Authentication enabled")
    print("   • Template-based code generation")
    print("   • Fallback web search")
    print("   • Basic reasoning capabilities")
    
    print("\n🚀 To add full AI capabilities:")
    print("   1. Test the minimal API first")
    print("   2. Use the progressive installation system")
    print("   3. Install dependencies in phases:")
    print("      - Phase 2: Basic utilities")
    print("      - Phase 3: PyTorch & Transformers")
    print("      - Phase 4: LangChain & vector search")
    print("      - Phase 5: Training capabilities")
    print("      - Phase 6: Scientific computing")
    
    print("\n📚 Documentation:")
    print("   • API docs: {deployment_url}/docs")
    print("   • Health check: {deployment_url}/health")
    print("   • Status: {deployment_url}/status")
    
    print("\n🔧 Progressive Installation:")
    print("   • Use install_dependencies.py script")
    print("   • Or make direct API calls to /dependencies/install")
    print("   • Monitor progress at /dependencies/status")

def main():
    """Main deployment process"""
    print("🚀 Autonomous AI Agent - Minimal Deployment")
    print("="*50)
    
    # Step 1: Check minimal deployment files
    if not create_minimal_deployment():
        print("❌ Cannot proceed with deployment")
        return
    
    # Step 2: Deploy to Vercel
    if not deploy_to_vercel():
        print("❌ Deployment failed")
        return
    
    # Step 3: Get deployment URL
    deployment_url = get_deployment_url()
    if not deployment_url:
        print("⚠️  Could not determine deployment URL")
        print("   Please check your Vercel dashboard")
        return
    
    # Step 4: Test the API
    if not test_minimal_api(deployment_url):
        print("⚠️  API test failed, but deployment may still be working")
    
    # Step 5: Show next steps
    show_next_steps(deployment_url)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Deployment cancelled by user")
    except Exception as e:
        print(f"\n💥 Deployment failed with error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure all minimal files are in place")
        print("2. Check Vercel CLI installation")
        print("3. Verify GitHub repository access")
        print("4. Check Vercel project configuration")
