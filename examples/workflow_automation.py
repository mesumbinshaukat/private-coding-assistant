#!/usr/bin/env python3
"""
Workflow Automation Examples for Autonomous AI Agent

This script demonstrates automated workflows for common development tasks,
including CI/CD integration, automated code reviews, and development pipelines.
"""

import os
import json
import time
import requests
import subprocess
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import git
from datetime import datetime

# Configuration
API_BASE = "https://your-deployment.vercel.app"
API_TOKEN = "autonomous-ai-agent-2024"

class WorkflowAutomation:
    """Automated workflow management for development tasks"""
    
    def __init__(self, base_url: str = API_BASE, token: str = API_TOKEN):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def _make_request(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict[str, Any]:
        """Make HTTP request to the API"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "GET":
                response = self.session.get(url, timeout=60)
            elif method == "POST":
                response = self.session.post(url, json=data, timeout=60)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def automated_code_review(self, repo_path: str, target_branch: str = "main") -> Dict[str, Any]:
        """Perform automated code review on git changes"""
        try:
            repo = git.Repo(repo_path)
            
            # Get current branch
            current_branch = repo.active_branch.name
            
            # Get diff between current branch and target
            diff = repo.git.diff(f"{target_branch}..{current_branch}")
            
            if not diff:
                return {"message": "No changes found for review", "files_reviewed": 0}
            
            # Split diff by files
            files_changed = []
            current_file = None
            current_content = []
            
            for line in diff.split('\n'):
                if line.startswith('diff --git'):
                    if current_file:
                        files_changed.append({
                            "file": current_file,
                            "diff": '\n'.join(current_content)
                        })
                    
                    # Extract filename
                    parts = line.split(' ')
                    current_file = parts[3][2:] if len(parts) > 3 else "unknown"
                    current_content = []
                else:
                    current_content.append(line)
            
            # Add last file
            if current_file:
                files_changed.append({
                    "file": current_file,
                    "diff": '\n'.join(current_content)
                })
            
            # Review each changed file
            review_results = []
            total_issues = 0
            
            for file_change in files_changed:
                print(f"🔍 Reviewing {file_change['file']}...")
                
                # Get AI review
                review_prompt = f"""
Perform a thorough code review of this change:

File: {file_change['file']}
Diff:
{file_change['diff']}

Provide:
1. Code quality assessment
2. Potential bugs or issues
3. Security concerns
4. Performance implications
5. Best practices violations
6. Suggestions for improvement
7. Overall rating (1-10)
"""
                
                review_result = self._make_request("/reason", "POST", {
                    "problem": review_prompt,
                    "domain": "code_review",
                    "include_math": False
                })
                
                if "error" not in review_result:
                    # Extract issues count (simplified)
                    issues_found = 0
                    if "solution" in review_result:
                        solution_text = review_result["solution"].lower()
                        if any(word in solution_text for word in ["bug", "issue", "problem", "concern", "violation"]):
                            issues_found = solution_text.count("issue") + solution_text.count("bug") + solution_text.count("problem")
                    
                    total_issues += issues_found
                    
                    review_results.append({
                        "file": file_change['file'],
                        "review": review_result,
                        "issues_found": issues_found,
                        "lines_changed": len([l for l in file_change['diff'].split('\n') if l.startswith('+') or l.startswith('-')])
                    })
                
                time.sleep(2)  # Rate limiting
            
            return {
                "branch": current_branch,
                "target_branch": target_branch,
                "files_reviewed": len(files_changed),
                "total_issues": total_issues,
                "review_results": review_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Code review failed: {str(e)}"}
    
    def create_ci_cd_pipeline(self, repo_path: str, language: str = "python") -> Dict[str, Any]:
        """Generate CI/CD pipeline configuration"""
        try:
            pipeline_prompt = f"""
Create a comprehensive CI/CD pipeline configuration for a {language} project.

Include:
1. Automated testing (unit tests, integration tests)
2. Code quality checks (linting, formatting)
3. Security scanning
4. Build process
5. Deployment configuration
6. Environment management
7. Notifications
8. Performance monitoring

Provide configurations for:
- GitHub Actions
- GitLab CI
- Jenkins (Jenkinsfile)
- Docker containerization

Make it production-ready with proper error handling and rollback mechanisms.
"""
            
            pipeline_result = self._make_request("/generate", "POST", {
                "prompt": pipeline_prompt,
                "language": "yaml",
                "context": f"CI/CD pipeline for {language} project"
            })
            
            if "error" in pipeline_result:
                return pipeline_result
            
            # Save pipeline files to repo
            repo_path = Path(repo_path)
            
            # Create .github/workflows directory
            workflows_dir = repo_path / ".github" / "workflows"
            workflows_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate GitHub Actions workflow
            github_workflow = self._generate_github_workflow(language)
            
            with open(workflows_dir / "ci-cd.yml", 'w') as f:
                f.write(github_workflow)
            
            # Create Dockerfile if not exists
            dockerfile_path = repo_path / "Dockerfile"
            if not dockerfile_path.exists():
                dockerfile_content = self._generate_dockerfile(language)
                with open(dockerfile_path, 'w') as f:
                    f.write(dockerfile_content)
            
            # Create docker-compose.yml
            docker_compose_path = repo_path / "docker-compose.yml"
            if not docker_compose_path.exists():
                compose_content = self._generate_docker_compose(language)
                with open(docker_compose_path, 'w') as f:
                    f.write(compose_content)
            
            return {
                "status": "success",
                "files_created": [
                    ".github/workflows/ci-cd.yml",
                    "Dockerfile",
                    "docker-compose.yml"
                ],
                "pipeline_config": pipeline_result,
                "recommendations": [
                    "Configure repository secrets for deployment",
                    "Set up staging and production environments",
                    "Configure monitoring and alerting",
                    "Set up automated dependency updates"
                ]
            }
            
        except Exception as e:
            return {"error": f"Pipeline creation failed: {str(e)}"}
    
    def _generate_github_workflow(self, language: str) -> str:
        """Generate GitHub Actions workflow"""
        if language.lower() == "python":
            return """name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10', '3.11']

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
        restore-keys: |
          ${{ runner.os }}-pip-
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov black flake8 safety bandit
    
    - name: Code formatting check
      run: black --check .
    
    - name: Linting
      run: flake8 .
    
    - name: Security check
      run: |
        safety check
        bandit -r . -f json -o bandit-report.json
    
    - name: Run tests
      run: |
        pytest --cov=. --cov-report=xml --cov-report=html
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker image
      run: |
        docker build -t my-app:${{ github.sha }} .
        docker tag my-app:${{ github.sha }} my-app:latest
    
    - name: Run container tests
      run: |
        docker run --rm my-app:${{ github.sha }} python -m pytest
    
    - name: Deploy to staging
      if: github.ref == 'refs/heads/main'
      run: |
        echo "Deploying to staging environment"
        # Add your deployment commands here
    
    - name: Deploy to production
      if: github.ref == 'refs/heads/main' && github.event_name == 'push'
      run: |
        echo "Deploying to production environment"
        # Add your deployment commands here
"""
        else:
            return f"""name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Setup {language} environment
      run: |
        echo "Setting up {language} environment"
        # Add language-specific setup here
    
    - name: Install dependencies
      run: |
        echo "Installing dependencies"
        # Add dependency installation here
    
    - name: Run tests
      run: |
        echo "Running tests"
        # Add test commands here
    
    - name: Build application
      run: |
        echo "Building application"
        # Add build commands here
"""
    
    def _generate_dockerfile(self, language: str) -> str:
        """Generate Dockerfile for the project"""
        if language.lower() == "python":
            return """FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && \\
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        else:
            return f"""FROM ubuntu:20.04

WORKDIR /app

# Install system dependencies for {language}
RUN apt-get update && apt-get install -y \\
    curl \\
    wget \\
    && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY . .

# Build application
RUN echo "Build commands for {language}"

EXPOSE 8000

CMD ["echo", "Run command for {language}"]
"""
    
    def _generate_docker_compose(self, language: str) -> str:
        """Generate docker-compose.yml"""
        return f"""version: '3.8'

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENV=development
      - LOG_LEVEL=INFO
    volumes:
      - .:/app
    depends_on:
      - redis
      - postgres

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: myuser
      POSTGRES_PASSWORD: mypassword
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - app

volumes:
  redis_data:
  postgres_data:
"""
    
    def automated_testing_suite(self, repo_path: str, test_type: str = "comprehensive") -> Dict[str, Any]:
        """Generate and run automated testing suite"""
        try:
            repo_path = Path(repo_path)
            
            # Analyze existing code to generate tests
            code_files = []
            for pattern in ["*.py", "*.js", "*.ts", "*.java"]:
                code_files.extend(list(repo_path.rglob(pattern)))
            
            if not code_files:
                return {"error": "No code files found for testing"}
            
            test_results = []
            
            for code_file in code_files[:10]:  # Limit to first 10 files
                print(f"🧪 Generating tests for {code_file.name}...")
                
                try:
                    with open(code_file, 'r', encoding='utf-8') as f:
                        code_content = f.read()
                except:
                    continue
                
                # Generate tests for this file
                test_prompt = f"""
Generate comprehensive {test_type} tests for this code:

File: {code_file.name}
Code:
{code_content}

Include:
1. Unit tests for all functions/methods
2. Edge case testing
3. Error condition testing
4. Integration tests where applicable
5. Performance tests for critical functions
6. Mocking for external dependencies

Use appropriate testing framework for the language.
"""
                
                test_result = self._make_request("/generate", "POST", {
                    "prompt": test_prompt,
                    "language": code_file.suffix[1:],  # Remove the dot
                    "context": f"Generating {test_type} tests"
                })
                
                if "error" not in test_result and "code" in test_result:
                    # Save generated test
                    test_dir = repo_path / "tests"
                    test_dir.mkdir(exist_ok=True)
                    
                    test_filename = f"test_{code_file.stem}{code_file.suffix}"
                    test_path = test_dir / test_filename
                    
                    with open(test_path, 'w', encoding='utf-8') as f:
                        f.write(test_result["code"])
                    
                    test_results.append({
                        "source_file": str(code_file),
                        "test_file": str(test_path),
                        "success": True,
                        "test_content": test_result["code"]
                    })
                else:
                    test_results.append({
                        "source_file": str(code_file),
                        "success": False,
                        "error": test_result.get("error", "Unknown error")
                    })
                
                time.sleep(2)  # Rate limiting
            
            # Run the generated tests
            test_execution_results = self._run_tests(repo_path)
            
            return {
                "test_generation": {
                    "files_processed": len(code_files),
                    "tests_generated": len([r for r in test_results if r["success"]]),
                    "generation_results": test_results
                },
                "test_execution": test_execution_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Testing suite generation failed: {str(e)}"}
    
    def _run_tests(self, repo_path: Path) -> Dict[str, Any]:
        """Run the generated tests"""
        try:
            test_dir = repo_path / "tests"
            if not test_dir.exists():
                return {"error": "No tests directory found"}
            
            # Try to run pytest
            result = subprocess.run(
                ["python", "-m", "pytest", str(test_dir), "-v", "--tb=short"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=300
            )
            
            return {
                "exit_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "success": result.returncode == 0
            }
            
        except subprocess.TimeoutExpired:
            return {"error": "Test execution timed out"}
        except Exception as e:
            return {"error": f"Test execution failed: {str(e)}"}
    
    def performance_optimization_workflow(self, repo_path: str) -> Dict[str, Any]:
        """Analyze and optimize code performance"""
        try:
            repo_path = Path(repo_path)
            
            # Find performance-critical files
            critical_files = []
            for pattern in ["*.py"]:  # Focus on Python for now
                for file_path in repo_path.rglob(pattern):
                    if file_path.is_file():
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                
                            # Simple heuristics for performance-critical code
                            if any(keyword in content.lower() for keyword in [
                                "for", "while", "recursion", "loop", "sort", "search", 
                                "algorithm", "process", "compute", "calculate"
                            ]):
                                critical_files.append(file_path)
                        except:
                            continue
            
            optimization_results = []
            
            for file_path in critical_files[:5]:  # Limit to 5 files
                print(f"⚡ Optimizing {file_path.name}...")
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    code_content = f.read()
                
                # Analyze performance
                analysis_prompt = f"""
Analyze this code for performance optimization opportunities:

File: {file_path.name}
Code:
{code_content}

Provide:
1. Time complexity analysis (Big O notation)
2. Space complexity analysis
3. Performance bottlenecks identification
4. Optimization recommendations
5. Optimized code implementation
6. Expected performance improvements
7. Memory usage optimization

Focus on:
- Algorithm efficiency
- Data structure optimization
- Loop optimization
- Memory management
- Caching opportunities
"""
                
                analysis_result = self._make_request("/reason", "POST", {
                    "problem": analysis_prompt,
                    "domain": "performance_optimization",
                    "include_math": True
                })
                
                # Generate optimized code
                optimization_prompt = f"""
Optimize this code for better performance:

{code_content}

Apply optimizations based on the analysis:
1. Improve algorithm efficiency
2. Optimize data structures
3. Reduce memory usage
4. Add caching where appropriate
5. Implement parallel processing if beneficial

Maintain the same functionality while improving performance.
"""
                
                optimization_result = self._make_request("/generate", "POST", {
                    "prompt": optimization_prompt,
                    "language": "python",
                    "context": f"Performance optimization for {file_path.name}"
                })
                
                optimization_results.append({
                    "file": str(file_path),
                    "analysis": analysis_result,
                    "optimization": optimization_result,
                    "timestamp": datetime.now().isoformat()
                })
                
                time.sleep(3)  # Rate limiting
            
            return {
                "files_analyzed": len(critical_files),
                "optimizations_generated": len(optimization_results),
                "optimization_results": optimization_results,
                "recommendations": [
                    "Profile your code to identify actual bottlenecks",
                    "Measure performance before and after optimizations",
                    "Consider using profiling tools like cProfile",
                    "Implement monitoring for production performance",
                    "Set up automated performance regression tests"
                ]
            }
            
        except Exception as e:
            return {"error": f"Performance optimization failed: {str(e)}"}
    
    def security_audit_workflow(self, repo_path: str) -> Dict[str, Any]:
        """Perform automated security audit"""
        try:
            repo_path = Path(repo_path)
            
            # Security analysis for code files
            security_results = []
            
            for pattern in ["*.py", "*.js", "*.ts"]:
                for file_path in repo_path.rglob(pattern):
                    if file_path.is_file():
                        print(f"🔒 Security audit for {file_path.name}...")
                        
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                code_content = f.read()
                        except:
                            continue
                        
                        security_prompt = f"""
Perform a comprehensive security audit of this code:

File: {file_path.name}
Code:
{code_content}

Analyze for:
1. SQL injection vulnerabilities
2. Cross-site scripting (XSS) risks
3. Authentication and authorization issues
4. Input validation problems
5. Sensitive data exposure
6. Insecure dependencies
7. Cryptographic issues
8. API security concerns
9. Error handling that leaks information
10. OWASP Top 10 vulnerabilities

Provide:
- Vulnerability assessment
- Risk rating (High/Medium/Low)
- Remediation recommendations
- Secure code alternatives
"""
                        
                        security_result = self._make_request("/reason", "POST", {
                            "problem": security_prompt,
                            "domain": "security_audit",
                            "include_math": False
                        })
                        
                        if "error" not in security_result:
                            # Extract vulnerability count (simplified)
                            vulnerabilities = 0
                            if "solution" in security_result:
                                solution_text = security_result["solution"].lower()
                                vulnerabilities = (
                                    solution_text.count("vulnerability") + 
                                    solution_text.count("risk") + 
                                    solution_text.count("security issue")
                                )
                            
                            security_results.append({
                                "file": str(file_path),
                                "vulnerabilities_found": vulnerabilities,
                                "audit_result": security_result,
                                "risk_level": "high" if vulnerabilities > 3 else "medium" if vulnerabilities > 1 else "low"
                            })
                        
                        time.sleep(2)
                        
                        # Limit to prevent overuse
                        if len(security_results) >= 10:
                            break
            
            # Generate security report
            total_vulnerabilities = sum(r["vulnerabilities_found"] for r in security_results)
            high_risk_files = len([r for r in security_results if r["risk_level"] == "high"])
            
            return {
                "files_audited": len(security_results),
                "total_vulnerabilities": total_vulnerabilities,
                "high_risk_files": high_risk_files,
                "audit_results": security_results,
                "recommendations": [
                    "Implement static code analysis tools (bandit, eslint-security)",
                    "Set up dependency vulnerability scanning",
                    "Use secure coding guidelines",
                    "Implement automated security testing in CI/CD",
                    "Regular security training for developers",
                    "Penetration testing for production systems"
                ],
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Security audit failed: {str(e)}"}
    
    def documentation_generation_workflow(self, repo_path: str) -> Dict[str, Any]:
        """Generate comprehensive project documentation"""
        try:
            repo_path = Path(repo_path)
            
            # Generate different types of documentation
            docs_generated = []
            
            # 1. API Documentation
            api_files = list(repo_path.rglob("*api*")) + list(repo_path.rglob("*routes*")) + list(repo_path.rglob("*endpoints*"))
            
            if api_files:
                print("📚 Generating API documentation...")
                
                api_content = ""
                for api_file in api_files[:3]:  # Limit to 3 files
                    try:
                        with open(api_file, 'r', encoding='utf-8') as f:
                            api_content += f"\n\n--- {api_file.name} ---\n{f.read()}"
                    except:
                        continue
                
                api_doc_prompt = f"""
Generate comprehensive API documentation for this code:

{api_content}

Include:
1. API overview and purpose
2. Authentication requirements
3. Endpoint documentation with examples
4. Request/response schemas
5. Error codes and handling
6. Rate limiting information
7. Usage examples in multiple languages
8. OpenAPI/Swagger specification

Format as professional API documentation.
"""
                
                api_doc_result = self._make_request("/generate", "POST", {
                    "prompt": api_doc_prompt,
                    "language": "markdown",
                    "context": "API documentation generation"
                })
                
                if "error" not in api_doc_result and "code" in api_doc_result:
                    docs_dir = repo_path / "docs"
                    docs_dir.mkdir(exist_ok=True)
                    
                    with open(docs_dir / "api_documentation.md", 'w', encoding='utf-8') as f:
                        f.write(api_doc_result["code"])
                    
                    docs_generated.append("API Documentation")
            
            # 2. Architecture Documentation
            print("🏗️ Generating architecture documentation...")
            
            # Analyze project structure
            structure_info = self._analyze_project_structure(repo_path)
            
            arch_doc_prompt = f"""
Generate comprehensive architecture documentation for this project:

Project Structure:
{json.dumps(structure_info, indent=2)}

Include:
1. System architecture overview
2. Component diagrams
3. Data flow diagrams
4. Technology stack description
5. Design patterns used
6. Scalability considerations
7. Security architecture
8. Deployment architecture
9. Database schema (if applicable)
10. Integration points

Format as professional architecture documentation with diagrams described in text.
"""
            
            arch_doc_result = self._make_request("/generate", "POST", {
                "prompt": arch_doc_prompt,
                "language": "markdown",
                "context": "Architecture documentation generation"
            })
            
            if "error" not in arch_doc_result and "code" in arch_doc_result:
                docs_dir = repo_path / "docs"
                docs_dir.mkdir(exist_ok=True)
                
                with open(docs_dir / "architecture.md", 'w', encoding='utf-8') as f:
                    f.write(arch_doc_result["code"])
                
                docs_generated.append("Architecture Documentation")
            
            # 3. User Guide
            print("👥 Generating user guide...")
            
            user_guide_prompt = f"""
Generate a comprehensive user guide for this project:

Project Structure: {structure_info.get('summary', 'Unknown project')}

Include:
1. Getting started guide
2. Installation instructions
3. Configuration guide
4. Usage examples
5. Troubleshooting section
6. FAQ
7. Best practices
8. Tips and tricks
9. Common workflows
10. Support information

Make it user-friendly for both technical and non-technical users.
"""
            
            user_guide_result = self._make_request("/generate", "POST", {
                "prompt": user_guide_prompt,
                "language": "markdown",
                "context": "User guide generation"
            })
            
            if "error" not in user_guide_result and "code" in user_guide_result:
                with open(repo_path / "USER_GUIDE.md", 'w', encoding='utf-8') as f:
                    f.write(user_guide_result["code"])
                
                docs_generated.append("User Guide")
            
            return {
                "documentation_generated": docs_generated,
                "files_created": [
                    "docs/api_documentation.md",
                    "docs/architecture.md", 
                    "USER_GUIDE.md"
                ],
                "project_structure": structure_info,
                "recommendations": [
                    "Keep documentation up to date with code changes",
                    "Add inline code comments for complex logic",
                    "Create video tutorials for complex workflows",
                    "Set up automated documentation generation",
                    "Implement documentation linting and validation"
                ]
            }
            
        except Exception as e:
            return {"error": f"Documentation generation failed: {str(e)}"}
    
    def _analyze_project_structure(self, repo_path: Path) -> Dict[str, Any]:
        """Analyze project structure for documentation"""
        structure = {
            "directories": [],
            "file_types": {},
            "total_files": 0,
            "total_lines": 0
        }
        
        for item in repo_path.rglob("*"):
            if item.is_file():
                structure["total_files"] += 1
                ext = item.suffix.lower()
                
                if ext not in structure["file_types"]:
                    structure["file_types"][ext] = 0
                structure["file_types"][ext] += 1
                
                try:
                    with open(item, 'r', encoding='utf-8') as f:
                        structure["total_lines"] += len(f.readlines())
                except:
                    pass
            elif item.is_dir():
                structure["directories"].append(str(item.relative_to(repo_path)))
        
        # Generate summary
        main_language = max(structure["file_types"].items(), key=lambda x: x[1], default=("unknown", 0))[0]
        structure["summary"] = f"Project with {structure['total_files']} files, primarily {main_language} codebase"
        
        return structure

def main():
    """Main function for workflow automation examples"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Workflow Automation for Development Tasks")
    parser.add_argument("--repo", required=True, help="Repository path")
    parser.add_argument("--workflow", 
                       choices=["review", "pipeline", "testing", "performance", "security", "docs", "all"],
                       default="review", help="Workflow to execute")
    parser.add_argument("--language", default="python", help="Primary project language")
    parser.add_argument("--target-branch", default="main", help="Target branch for comparison")
    
    args = parser.parse_args()
    
    automation = WorkflowAutomation()
    
    print(f"🚀 Starting {args.workflow} workflow for {args.repo}")
    
    if args.workflow == "review":
        result = automation.automated_code_review(args.repo, args.target_branch)
    elif args.workflow == "pipeline":
        result = automation.create_ci_cd_pipeline(args.repo, args.language)
    elif args.workflow == "testing":
        result = automation.automated_testing_suite(args.repo)
    elif args.workflow == "performance":
        result = automation.performance_optimization_workflow(args.repo)
    elif args.workflow == "security":
        result = automation.security_audit_workflow(args.repo)
    elif args.workflow == "docs":
        result = automation.documentation_generation_workflow(args.repo)
    elif args.workflow == "all":
        # Run all workflows
        workflows = ["review", "testing", "security", "performance", "docs"]
        results = {}
        
        for workflow in workflows:
            print(f"\n{'='*50}")
            print(f"Running {workflow} workflow...")
            print('='*50)
            
            if workflow == "review":
                results[workflow] = automation.automated_code_review(args.repo, args.target_branch)
            elif workflow == "testing":
                results[workflow] = automation.automated_testing_suite(args.repo)
            elif workflow == "security":
                results[workflow] = automation.security_audit_workflow(args.repo)
            elif workflow == "performance":
                results[workflow] = automation.performance_optimization_workflow(args.repo)
            elif workflow == "docs":
                results[workflow] = automation.documentation_generation_workflow(args.repo)
        
        result = {"all_workflows": results}
    
    print(f"\n✅ {args.workflow.title()} workflow completed!")
    
    if "error" in result:
        print(f"❌ Error: {result['error']}")
    else:
        print(json.dumps(result, indent=2, default=str))

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Workflow cancelled. Goodbye!")
    except Exception as e:
        print(f"\n💥 An unexpected error occurred: {e}")
