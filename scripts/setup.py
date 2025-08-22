#!/usr/bin/env python3
"""
Setup Script for Autonomous AI Agent

This script handles the complete setup process for the AI agent project,
including dependency installation, model setup, configuration, and deployment.
"""

import os
import sys
import json
import subprocess
import platform
import shutil
import urllib.request
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse

class ProjectSetup:
    """Complete project setup and configuration"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.system = platform.system().lower()
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        
        print(f"🚀 Autonomous AI Agent Setup")
        print(f"📁 Project root: {self.project_root}")
        print(f"💻 System: {self.system}")
        print(f"🐍 Python: {self.python_version}")
        print("="*50)
    
    def check_prerequisites(self) -> Dict[str, bool]:
        """Check system prerequisites"""
        print("🔍 Checking prerequisites...")
        
        checks = {
            "python": False,
            "pip": False,
            "git": False,
            "node": False,
            "npm": False,
            "docker": False
        }
        
        # Check Python version
        if sys.version_info >= (3, 8):
            checks["python"] = True
            print("✅ Python 3.8+ found")
        else:
            print("❌ Python 3.8+ required")
        
        # Check pip
        try:
            subprocess.run(["pip", "--version"], capture_output=True, check=True)
            checks["pip"] = True
            print("✅ pip found")
        except:
            print("❌ pip not found")
        
        # Check git
        try:
            subprocess.run(["git", "--version"], capture_output=True, check=True)
            checks["git"] = True
            print("✅ git found")
        except:
            print("❌ git not found")
        
        # Check Node.js (for desktop app)
        try:
            subprocess.run(["node", "--version"], capture_output=True, check=True)
            checks["node"] = True
            print("✅ Node.js found")
        except:
            print("⚠️ Node.js not found (needed for desktop app)")
        
        # Check npm
        try:
            subprocess.run(["npm", "--version"], capture_output=True, check=True)
            checks["npm"] = True
            print("✅ npm found")
        except:
            print("⚠️ npm not found (needed for desktop app)")
        
        # Check Docker (optional)
        try:
            subprocess.run(["docker", "--version"], capture_output=True, check=True)
            checks["docker"] = True
            print("✅ Docker found")
        except:
            print("⚠️ Docker not found (optional)")
        
        return checks
    
    def setup_python_environment(self) -> bool:
        """Set up Python virtual environment"""
        print("\n🐍 Setting up Python environment...")
        
        venv_path = self.project_root / "venv"
        
        try:
            # Create virtual environment
            if not venv_path.exists():
                print("Creating virtual environment...")
                subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
                print("✅ Virtual environment created")
            else:
                print("✅ Virtual environment already exists")
            
            # Determine activation script
            if self.system == "windows":
                activate_script = venv_path / "Scripts" / "activate.bat"
                pip_executable = venv_path / "Scripts" / "pip.exe"
            else:
                activate_script = venv_path / "bin" / "activate"
                pip_executable = venv_path / "bin" / "pip"
            
            # Install/upgrade pip
            print("Upgrading pip...")
            subprocess.run([str(pip_executable), "install", "--upgrade", "pip"], check=True)
            
            # Install requirements
            requirements_file = self.project_root / "requirements.txt"
            if requirements_file.exists():
                print("Installing Python dependencies...")
                subprocess.run([
                    str(pip_executable), "install", "-r", str(requirements_file)
                ], check=True)
                print("✅ Python dependencies installed")
            else:
                print("⚠️ requirements.txt not found")
            
            # Create activation instructions
            activation_file = self.project_root / "activate_env.txt"
            with open(activation_file, 'w') as f:
                if self.system == "windows":
                    f.write(f"venv\\Scripts\\activate.bat\n")
                else:
                    f.write(f"source venv/bin/activate\n")
            
            print(f"✅ Environment setup complete")
            print(f"📝 Activation command saved to: {activation_file}")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to set up Python environment: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
    
    def setup_configuration(self) -> bool:
        """Set up configuration files"""
        print("\n⚙️ Setting up configuration...")
        
        try:
            # Create .env file if it doesn't exist
            env_file = self.project_root / ".env"
            if not env_file.exists():
                env_content = """# Autonomous AI Agent Configuration

# API Configuration
SECRET_KEY=autonomous-ai-agent-2024-change-in-production
LOG_LEVEL=INFO

# Model Configuration
MODEL_NAME=distilgpt2
MAX_TOKENS=512
TEMPERATURE=0.7

# Web Search Configuration
ENABLE_WEB_SEARCH=true
SEARCH_RATE_LIMIT=10

# Training Configuration
ENABLE_SELF_TRAINING=true
TRAINING_ITERATIONS=10
LEARNING_RATE=0.0001

# Memory Configuration
MEMORY_VECTOR_DIM=384
MEMORY_INDEX_PATH=memory_index.faiss

# Rate Limiting
RATE_LIMIT_REQUESTS=10
RATE_LIMIT_WINDOW=60

# Development
DEBUG=false
"""
                with open(env_file, 'w') as f:
                    f.write(env_content)
                print("✅ .env configuration file created")
            else:
                print("✅ .env file already exists")
            
            # Create .gitignore if it doesn't exist
            gitignore_file = self.project_root / ".gitignore"
            if not gitignore_file.exists():
                gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# Environment Variables
.env
.env.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
*.log
logs/

# Model files
models/
*.bin
*.pt
*.pth

# Memory files
*.faiss
*.faiss.metadata

# OS
.DS_Store
Thumbs.db

# Node.js (for desktop app)
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Electron
dist/
*.exe
*.dmg
*.pkg

# Temporary files
tmp/
temp/
*.tmp

# Testing
.pytest_cache/
.coverage
htmlcov/

# Docker
.dockerignore
"""
                with open(gitignore_file, 'w') as f:
                    f.write(gitignore_content)
                print("✅ .gitignore file created")
            else:
                print("✅ .gitignore file already exists")
            
            return True
            
        except Exception as e:
            print(f"❌ Configuration setup failed: {e}")
            return False
    
    def setup_directories(self) -> bool:
        """Create necessary directories"""
        print("\n📁 Setting up directories...")
        
        directories = [
            "models",
            "logs",
            "temp",
            "data",
            "tests/fixtures",
            "docs/images",
            "scripts"
        ]
        
        try:
            for directory in directories:
                dir_path = self.project_root / directory
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✅ Directory created: {directory}")
            
            # Create __init__.py files for Python packages
            python_packages = [
                "api",
                "api/utils",
                "tests"
            ]
            
            for package in python_packages:
                init_file = self.project_root / package / "__init__.py"
                if not init_file.exists():
                    init_file.touch()
                    print(f"✅ Created __init__.py in {package}")
            
            return True
            
        except Exception as e:
            print(f"❌ Directory setup failed: {e}")
            return False
    
    def download_initial_model(self) -> bool:
        """Download and setup initial model"""
        print("\n🤖 Setting up initial model...")
        
        try:
            # This would normally download the model, but for this example,
            # we'll just create a placeholder and let the application download it
            models_dir = self.project_root / "models"
            models_dir.mkdir(exist_ok=True)
            
            # Create a model info file
            model_info = {
                "model_name": "distilgpt2",
                "model_type": "causal_lm",
                "source": "huggingface",
                "status": "not_downloaded",
                "download_url": "https://huggingface.co/distilgpt2",
                "setup_date": "will_be_set_on_first_run"
            }
            
            with open(models_dir / "model_info.json", 'w') as f:
                json.dump(model_info, f, indent=2)
            
            print("✅ Model configuration created")
            print("ℹ️ Model will be downloaded automatically on first run")
            
            return True
            
        except Exception as e:
            print(f"❌ Model setup failed: {e}")
            return False
    
    def setup_desktop_app(self) -> bool:
        """Set up the desktop application"""
        print("\n🖥️ Setting up desktop application...")
        
        desktop_app_dir = self.project_root / "desktop-app" / "electron-app"
        
        if not desktop_app_dir.exists():
            print("❌ Desktop app directory not found")
            return False
        
        try:
            # Check if package.json exists
            package_json = desktop_app_dir / "package.json"
            if not package_json.exists():
                # Create package.json
                package_content = {
                    "name": "autonomous-ai-agent-desktop",
                    "version": "1.0.0",
                    "description": "Desktop application for Autonomous AI Agent",
                    "main": "main.js",
                    "scripts": {
                        "start": "electron .",
                        "build": "electron-builder",
                        "dev": "electron . --dev",
                        "pack": "electron-builder --dir",
                        "dist": "electron-builder"
                    },
                    "keywords": ["ai", "agent", "coding", "automation"],
                    "author": "Your Name",
                    "license": "MIT",
                    "devDependencies": {
                        "electron": "^22.0.0",
                        "electron-builder": "^23.0.0"
                    },
                    "dependencies": {
                        "axios": "^1.0.0"
                    },
                    "build": {
                        "appId": "com.yourcompany.autonomous-ai-agent",
                        "productName": "Autonomous AI Agent",
                        "directories": {
                            "output": "dist"
                        },
                        "files": [
                            "**/*",
                            "!node_modules",
                            "!src",
                            "!.git"
                        ],
                        "win": {
                            "target": "nsis",
                            "icon": "assets/icon.ico"
                        },
                        "mac": {
                            "target": "dmg",
                            "icon": "assets/icon.icns"
                        },
                        "linux": {
                            "target": "AppImage",
                            "icon": "assets/icon.png"
                        }
                    }
                }
                
                with open(package_json, 'w') as f:
                    json.dump(package_content, f, indent=2)
                print("✅ package.json created")
            
            # Install dependencies if npm is available
            try:
                print("Installing Node.js dependencies...")
                subprocess.run(["npm", "install"], cwd=desktop_app_dir, check=True, 
                             capture_output=True, text=True)
                print("✅ Node.js dependencies installed")
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Failed to install Node.js dependencies: {e}")
                print("You can install them later with: cd desktop-app/electron-app && npm install")
            except FileNotFoundError:
                print("⚠️ npm not found, skipping Node.js dependency installation")
            
            return True
            
        except Exception as e:
            print(f"❌ Desktop app setup failed: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Run the test suite to verify setup"""
        print("\n🧪 Running test suite...")
        
        try:
            # Determine Python executable in venv
            if self.system == "windows":
                python_executable = self.project_root / "venv" / "Scripts" / "python.exe"
            else:
                python_executable = self.project_root / "venv" / "bin" / "python"
            
            if not python_executable.exists():
                python_executable = sys.executable
            
            # Run basic tests
            test_command = [str(python_executable), "-m", "pytest", "tests/", "-v", "--tb=short"]
            
            result = subprocess.run(test_command, cwd=self.project_root, 
                                  capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print("✅ All tests passed")
                print(f"Test output:\n{result.stdout}")
                return True
            else:
                print("⚠️ Some tests failed")
                print(f"Test output:\n{result.stdout}")
                print(f"Errors:\n{result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⚠️ Tests timed out")
            return False
        except FileNotFoundError:
            print("⚠️ pytest not found, skipping tests")
            return False
        except Exception as e:
            print(f"⚠️ Test execution failed: {e}")
            return False
    
    def create_deployment_guide(self) -> bool:
        """Create deployment guide"""
        print("\n📋 Creating deployment guide...")
        
        try:
            deployment_guide = """# Deployment Guide - Autonomous AI Agent

## 🚀 Vercel Deployment

### Prerequisites
- GitHub/GitLab account
- Vercel account (free tier)
- Repository pushed to GitHub/GitLab

### Steps

1. **Prepare Repository**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy to Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your repository
   - Configure settings:
     - Framework Preset: Other
     - Build Command: `pip install -r requirements.txt`
     - Output Directory: `api`
     - Install Command: (leave empty)

3. **Set Environment Variables**
   In Vercel dashboard → Settings → Environment Variables:
   ```
   SECRET_KEY=your-secret-key-here
   LOG_LEVEL=INFO
   MODEL_NAME=distilgpt2
   ```

4. **Verify Deployment**
   - Check deployment logs
   - Test API endpoints
   - Monitor performance

### 🔧 Local Development

1. **Activate Environment**
   ```bash
   # Windows
   venv\\Scripts\\activate

   # Linux/Mac
   source venv/bin/activate
   ```

2. **Start Development Server**
   ```bash
   cd api
   python -m uvicorn index:app --reload --host 0.0.0.0 --port 8000
   ```

3. **Test API**
   ```bash
   curl -X GET http://localhost:8000/status \\
        -H "Authorization: Bearer autonomous-ai-agent-2024"
   ```

### 🖥️ Desktop Application

1. **Setup**
   ```bash
   cd desktop-app/electron-app
   npm install
   ```

2. **Development**
   ```bash
   npm run dev
   ```

3. **Build**
   ```bash
   npm run build
   ```

### 🔒 Security Considerations

- Change default API key before deployment
- Use environment variables for secrets
- Enable HTTPS in production
- Implement proper rate limiting
- Regular security updates

### 📊 Monitoring

- Check Vercel deployment logs
- Monitor API response times
- Track error rates
- Set up alerts for failures

### 🐛 Troubleshooting

**Common Issues:**

1. **Import Errors**
   - Check requirements.txt
   - Verify Python version compatibility

2. **Memory Limits**
   - Optimize model size
   - Use quantization
   - Consider upgrading Vercel plan

3. **Cold Start Issues**
   - Implement health check endpoint
   - Use Vercel's edge functions if needed

4. **Rate Limiting**
   - Check API usage
   - Implement backoff strategies
   - Cache responses when possible

### 📞 Support

- Check logs in Vercel dashboard
- Review GitHub issues
- Refer to documentation in /docs folder
"""
            
            with open(self.project_root / "DEPLOYMENT.md", 'w') as f:
                f.write(deployment_guide)
            
            print("✅ Deployment guide created: DEPLOYMENT.md")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create deployment guide: {e}")
            return False
    
    def create_startup_script(self) -> bool:
        """Create platform-specific startup scripts"""
        print("\n📝 Creating startup scripts...")
        
        try:
            # Windows batch script
            if self.system == "windows":
                startup_script = """@echo off
echo Starting Autonomous AI Agent...

REM Activate virtual environment
call venv\\Scripts\\activate.bat

REM Start the API server
cd api
python -m uvicorn index:app --host 0.0.0.0 --port 8000

pause
"""
                with open(self.project_root / "start_agent.bat", 'w') as f:
                    f.write(startup_script)
                print("✅ Windows startup script created: start_agent.bat")
            
            # Unix shell script
            unix_script = """#!/bin/bash
echo "Starting Autonomous AI Agent..."

# Activate virtual environment
source venv/bin/activate

# Start the API server
cd api
python -m uvicorn index:app --host 0.0.0.0 --port 8000
"""
            with open(self.project_root / "start_agent.sh", 'w') as f:
                f.write(unix_script)
            
            # Make shell script executable
            os.chmod(self.project_root / "start_agent.sh", 0o755)
            print("✅ Unix startup script created: start_agent.sh")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to create startup scripts: {e}")
            return False
    
    def run_complete_setup(self, skip_tests: bool = False, skip_desktop: bool = False) -> bool:
        """Run the complete setup process"""
        print("🎯 Running complete setup...\n")
        
        setup_steps = [
            ("Prerequisites", self.check_prerequisites),
            ("Directories", self.setup_directories),
            ("Configuration", self.setup_configuration),
            ("Python Environment", self.setup_python_environment),
            ("Model Setup", self.download_initial_model),
            ("Deployment Guide", self.create_deployment_guide),
            ("Startup Scripts", self.create_startup_script)
        ]
        
        if not skip_desktop:
            setup_steps.append(("Desktop App", self.setup_desktop_app))
        
        if not skip_tests:
            setup_steps.append(("Tests", self.run_tests))
        
        success_count = 0
        total_steps = len(setup_steps)
        
        for step_name, step_function in setup_steps:
            print(f"\n{'='*20} {step_name} {'='*20}")
            try:
                if step_function():
                    success_count += 1
                    print(f"✅ {step_name} completed successfully")
                else:
                    print(f"⚠️ {step_name} completed with warnings")
            except Exception as e:
                print(f"❌ {step_name} failed: {e}")
        
        print(f"\n{'='*60}")
        print(f"🎉 Setup Complete!")
        print(f"✅ {success_count}/{total_steps} steps completed successfully")
        
        if success_count == total_steps:
            print("\n🚀 Your Autonomous AI Agent is ready!")
            print("Next steps:")
            print("1. Review the .env file and update configuration")
            print("2. Deploy to Vercel using DEPLOYMENT.md guide")
            print("3. Test the API endpoints")
            print("4. Build the desktop app if needed")
            return True
        else:
            print(f"\n⚠️ Some setup steps had issues. Please check the output above.")
            return False

def main():
    """Main setup function"""
    parser = argparse.ArgumentParser(description="Setup Autonomous AI Agent Project")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--skip-desktop", action="store_true", help="Skip desktop app setup")
    parser.add_argument("--check-only", action="store_true", help="Only check prerequisites")
    
    args = parser.parse_args()
    
    setup = ProjectSetup(args.project_root)
    
    if args.check_only:
        setup.check_prerequisites()
        return
    
    success = setup.run_complete_setup(
        skip_tests=args.skip_tests,
        skip_desktop=args.skip_desktop
    )
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Setup failed with unexpected error: {e}")
        sys.exit(1)
