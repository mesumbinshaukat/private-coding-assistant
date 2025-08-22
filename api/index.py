"""
Minimal API entry point for Vercel deployment - Debug Version
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any
import jwt
import requests
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="Autonomous AI Agent API",
    description="Production-ready autonomous AI agent",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
SECRET_KEY = "autonomous-ai-agent-secret-key-2024"

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

# Fallback implementations
def fallback_code_generation(prompt: str, language: str) -> Dict[str, Any]:
    """Template-based code generation"""
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
        }
    }
    
    prompt_lower = prompt.lower()
    
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
    """Basic web search"""
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
        return {
            "results": [],
            "synthesized_answer": f"Search error: {str(e)}",
            "confidence": 0.0,
            "method": "error"
        }

def fallback_reasoning(problem: str, domain: str) -> Dict[str, Any]:
    """Basic reasoning"""
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
        "message": "Autonomous AI Agent API - Working!",
        "status": "active",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/generate")
async def generate_code(request: CodeRequest, token: bool = Depends(verify_token)):
    """Generate code with fallback template system"""
    try:
        result = fallback_code_generation(request.prompt, request.language)
        result["timestamp"] = datetime.now().isoformat()
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/search")
async def deep_search(request: SearchRequest, token: bool = Depends(verify_token)):
    """Perform web search with fallback capabilities"""
    try:
        result = fallback_web_search(request.query)
        result["timestamp"] = datetime.now().isoformat()
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/reason")
async def reason_step_by_step(request: ReasonRequest, token: bool = Depends(verify_token)):
    """Perform reasoning with fallback capabilities"""
    try:
        result = fallback_reasoning(request.problem, request.domain)
        result["timestamp"] = datetime.now().isoformat()
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reasoning failed: {str(e)}")

@app.get("/status")
async def get_status(token: bool = Depends(verify_token)):
    """Get API status"""
    return {
        "api_status": "active",
        "available_features": ["code_generation", "web_search", "reasoning"],
        "timestamp": datetime.now().isoformat()
    }

# Vercel handler - This is required for Vercel to work
from http.server import BaseHTTPRequestHandler
import json

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        
        if self.path == '/':
            response = {"Hello": "World"}
        elif self.path == '/health':
            response = {"status": "ok"}
        else:
            response = {"error": "Not found"}
        
        self.wfile.write(json.dumps(response).encode())
        return