"""
Minimal API entry point for Vercel deployment
Optimized to stay under 250MB limit while preserving functionality
"""

import os
import json
import logging
from typing import Dict, Any
from datetime import datetime

# Only import essential FastAPI components
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Basic imports
import jwt
import requests

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "autonomous-ai-agent-secret-key-2024")

# Initialize FastAPI app
app = FastAPI(
    title="Autonomous AI Agent API",
    description="Production-ready autonomous AI agent with progressive loading",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return True
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

# Request/Response models
class CodeRequest(BaseModel):
    prompt: str = Field(..., description="Code generation prompt")
    language: str = Field("python", description="Programming language")
    context: str = Field(None, description="Additional context")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    depth: int = Field(5, description="Search depth")
    include_code: bool = Field(True, description="Include code examples")

class ReasonRequest(BaseModel):
    problem: str = Field(..., description="Problem to reason about")
    domain: str = Field("coding", description="Problem domain")
    include_math: bool = Field(True, description="Include mathematical reasoning")

# Fallback implementations (lightweight)
def fallback_code_generation(prompt: str, language: str) -> Dict[str, Any]:
    """Template-based code generation when AI libraries aren't loaded"""
    
    templates = {
        "fibonacci": {
            "python": """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Example usage
print(fibonacci(10))""",
            "javascript": """function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n-1) + fibonacci(n-2);
}

console.log(fibonacci(10));"""
        },
        "binary_search": {
            "python": """def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# Example usage
arr = [1, 3, 5, 7, 9, 11]
print(binary_search(arr, 7))"""
        }
    }
    
    prompt_lower = prompt.lower()
    
    # Match templates
    for template_name, template_code in templates.items():
        if template_name in prompt_lower:
            code = template_code.get(language, template_code.get("python", ""))
            return {
                "code": code,
                "explanation": f"Template-based {language} implementation for {template_name}",
                "confidence": 0.8,
                "method": "template"
            }
    
    # Default template
    if language == "python":
        code = f"""# {prompt}

def solve_problem():
    '''
    Solution for: {prompt}
    '''
    # TODO: Implement your solution here
    pass

if __name__ == "__main__":
    solve_problem()"""
    else:
        code = f"// {prompt}\n// TODO: Implement solution in {language}"
    
    return {
        "code": code,
        "explanation": f"Basic template for {language}",
        "confidence": 0.6,
        "method": "basic_template"
    }

def fallback_web_search(query: str) -> Dict[str, Any]:
    """Basic web search when advanced search isn't available"""
    try:
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            results = []
            
            if data.get("AbstractText"):
                results.append({
                    "title": data.get("AbstractSource", "DuckDuckGo"),
                    "snippet": data.get("AbstractText"),
                    "url": data.get("AbstractURL", "")
                })
            
            return {
                "results": results,
                "synthesized_answer": data.get("AbstractText", "No direct answer found"),
                "confidence": 0.7,
                "method": "duckduckgo_api"
            }
        
        return {
            "results": [],
            "synthesized_answer": "Search unavailable",
            "confidence": 0.0,
            "method": "fallback"
        }
    except Exception as e:
        logger.error(f"Fallback search error: {e}")
        return {
            "results": [],
            "synthesized_answer": f"Search error: {str(e)}",
            "confidence": 0.0,
            "method": "error"
        }

def fallback_reasoning(problem: str, domain: str) -> Dict[str, Any]:
    """Basic reasoning when advanced reasoning isn't available"""
    reasoning_steps = [
        {
            "step": 1,
            "description": "Problem Analysis",
            "content": f"Analyzing the {domain} problem: {problem}"
        },
        {
            "step": 2,
            "description": "Solution Planning", 
            "content": "Breaking down the problem into manageable components"
        },
        {
            "step": 3,
            "description": "Implementation Strategy",
            "content": f"Determining approach for this {domain} challenge"
        }
    ]
    
    solution = f"""
Basic approach for: {problem}

1. Understand the requirements
2. Design the solution
3. Implement step by step
4. Test and validate
5. Optimize if needed

This {domain} problem requires systematic analysis.
"""
    
    return {
        "reasoning_steps": reasoning_steps,
        "solution": solution.strip(),
        "confidence": 0.6,
        "method": "basic_reasoning"
    }

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Autonomous AI Agent API - Minimal Deployment",
        "status": "active",
        "version": "2.0.0",
        "deployment_mode": "minimal",
        "features": ["basic_api", "authentication", "fallback_templates"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "deployment_mode": "minimal",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/generate")
async def generate_code(request: CodeRequest, token: bool = Depends(verify_token)):
    """Generate code with fallback template system"""
    try:
        result = fallback_code_generation(request.prompt, request.language)
        result["timestamp"] = datetime.now().isoformat()
        result["deployment_mode"] = "minimal"
        return result
        
    except Exception as e:
        logger.error(f"Code generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/search")
async def deep_search(request: SearchRequest, token: bool = Depends(verify_token)):
    """Perform web search with fallback capabilities"""
    try:
        result = fallback_web_search(request.query)
        result["timestamp"] = datetime.now().isoformat()
        result["deployment_mode"] = "minimal"
        return result
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/reason")
async def reason_step_by_step(request: ReasonRequest, token: bool = Depends(verify_token)):
    """Perform reasoning with fallback capabilities"""
    try:
        result = fallback_reasoning(request.problem, request.domain)
        result["timestamp"] = datetime.now().isoformat()
        result["deployment_mode"] = "minimal"
        return result
        
    except Exception as e:
        logger.error(f"Reasoning error: {e}")
        raise HTTPException(status_code=500, detail=f"Reasoning failed: {str(e)}")

@app.get("/status")
async def get_status(token: bool = Depends(verify_token)):
    """Get API status"""
    return {
        "api_status": "active",
        "deployment_mode": "minimal",
        "available_features": ["basic_api", "authentication", "fallback_templates"],
        "message": "This is a minimal deployment. Use /dependencies/install to add full AI capabilities.",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/dependencies/status")
async def get_dependency_status():
    """Get dependency status"""
    return {
        "deployment_mode": "minimal",
        "message": "Full dependencies not loaded. Use progressive installation to add AI capabilities.",
        "available_features": ["basic_api", "authentication", "fallback_templates"],
        "next_steps": "Deploy with minimal dependencies first, then install AI libraries progressively"
    }

# For Vercel
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
