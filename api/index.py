"""
Autonomous AI Agent API with Dynamic Dependency Management

This API starts with minimal dependencies and progressively loads
heavy AI/ML libraries after successful deployment to avoid OOM errors.
"""

import os
import json
import asyncio
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import importlib

# Core imports (always available)
from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import jwt
import requests

# Dynamic dependency management
from api.dependency_manager import dependency_manager

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "autonomous-ai-agent-secret-key-2024")
MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")

# Initialize FastAPI app
app = FastAPI(
    title="Autonomous AI Agent API",
    description="Production-ready autonomous AI agent with progressive dependency loading",
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

# Global agent instance (loaded dynamically)
_agent = None

async def get_agent():
    """Get or create the autonomous agent (lazy loading)"""
    global _agent
    
    if _agent is None:
        features = dependency_manager.get_available_features()
        
        if features["ai_code_generation"]:
            try:
                from api.agent_core import AutonomousAgent
                _agent = AutonomousAgent()
                await _agent.initialize()
                logger.info("Full AutonomousAgent loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load AutonomousAgent: {e}")
                _agent = "fallback"
        else:
            _agent = "fallback"
    
    return _agent

# Request/Response models
class CodeRequest(BaseModel):
    prompt: str = Field(..., description="Code generation prompt")
    language: str = Field("python", description="Programming language")
    context: Optional[str] = Field(None, description="Additional context")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    depth: int = Field(5, description="Search depth")
    include_code: bool = Field(True, description="Include code examples")

class ReasonRequest(BaseModel):
    problem: str = Field(..., description="Problem to reason about")
    domain: str = Field("coding", description="Problem domain")
    include_math: bool = Field(True, description="Include mathematical reasoning")

class DependencyRequest(BaseModel):
    phase: Optional[str] = Field(None, description="Specific phase to install")
    package: Optional[str] = Field(None, description="Specific package to install")
    mode: Optional[str] = Field(None, description="Installation mode: auto, manual, disabled")

# Fallback implementations for when dependencies aren't loaded
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
    features = dependency_manager.get_available_features()
    return {
        "message": "Autonomous AI Agent API with Progressive Loading",
        "status": "active",
        "version": "2.0.0",
        "available_features": features,
        "dependency_status": dependency_manager.status["phases_completed"],
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    features = dependency_manager.get_available_features()
    return {
        "status": "healthy",
        "features": features,
        "dependency_phases": dependency_manager.status["phases_completed"],
        "installation_mode": dependency_manager.status["installation_mode"],
        "timestamp": datetime.now().isoformat()
    }

@app.post("/generate")
async def generate_code(request: CodeRequest, token: bool = Depends(verify_token)):
    """Generate code with progressive AI capabilities"""
    try:
        agent = await get_agent()
        
        if agent != "fallback":
            # Use full AI agent
            result = await agent.generate_code(request.prompt, request.language, request.context)
        else:
            # Use fallback template-based generation
            result = fallback_code_generation(request.prompt, request.language)
        
        result["timestamp"] = datetime.now().isoformat()
        return result
        
    except Exception as e:
        logger.error(f"Code generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/search")
async def deep_search(request: SearchRequest, token: bool = Depends(verify_token)):
    """Perform web search with progressive capabilities"""
    try:
        agent = await get_agent()
        
        if agent != "fallback":
            # Use full AI agent
            result = await agent.deep_search(request.query, request.depth, request.include_code)
        else:
            # Use fallback search
            result = fallback_web_search(request.query)
        
        result["timestamp"] = datetime.now().isoformat()
        return result
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/reason")
async def reason_step_by_step(request: ReasonRequest, token: bool = Depends(verify_token)):
    """Perform reasoning with progressive capabilities"""
    try:
        agent = await get_agent()
        
        if agent != "fallback":
            # Use full AI agent
            result = await agent.reason_step_by_step(request.problem, request.domain, request.include_math)
        else:
            # Use fallback reasoning
            result = fallback_reasoning(request.problem, request.domain)
        
        result["timestamp"] = datetime.now().isoformat()
        return result
        
    except Exception as e:
        logger.error(f"Reasoning error: {e}")
        raise HTTPException(status_code=500, detail=f"Reasoning failed: {str(e)}")

@app.post("/train")
async def trigger_training(request: dict, token: bool = Depends(verify_token)):
    """Trigger model training (requires training dependencies)"""
    try:
        features = dependency_manager.get_available_features()
        
        if not features["self_training"]:
            return {
                "status": "unavailable",
                "message": "Training features not available. Install phase_5_training dependencies.",
                "required_phase": "phase_5_training"
            }
        
        agent = await get_agent()
        if agent != "fallback":
            await agent.self_train(
                request.get("dataset_name"),
                request.get("training_type", "rlhf"),
                request.get("iterations", 10)
            )
            return {"status": "training_started", "message": "Training initiated successfully"}
        else:
            return {"status": "unavailable", "message": "Full agent not loaded"}
            
    except Exception as e:
        logger.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.get("/status")
async def get_status(token: bool = Depends(verify_token)):
    """Get comprehensive agent status"""
    features = dependency_manager.get_available_features()
    dependency_status = dependency_manager.get_status()
    
    agent = await get_agent()
    
    if agent != "fallback":
        try:
            agent_status = await agent.get_status()
        except:
            agent_status = {"agent_loaded": True, "status": "basic"}
    else:
        agent_status = {"agent_loaded": False, "status": "fallback_mode"}
    
    return {
        "api_status": "active",
        "features": features,
        "dependency_status": dependency_status,
        "agent_status": agent_status,
        "timestamp": datetime.now().isoformat()
    }

# Dependency Management Endpoints
@app.get("/dependencies/status")
async def get_dependency_status():
    """Get detailed dependency status"""
    return dependency_manager.get_status()

@app.post("/dependencies/install")
async def install_dependencies(
    request: DependencyRequest, 
    background_tasks: BackgroundTasks,
    token: bool = Depends(verify_token)
):
    """Install dependencies progressively"""
    try:
        if request.mode:
            dependency_manager.set_installation_mode(request.mode)
            return {"status": "mode_updated", "mode": request.mode}
        
        if request.package:
            # Install specific package
            result = await dependency_manager.install_specific_package(request.package)
            return result
        
        if request.phase:
            # Install specific phase
            result = await dependency_manager.install_phase(request.phase)
            return result
        
        # Install next phase
        result = await dependency_manager.install_next_phase()
        return result
        
    except Exception as e:
        logger.error(f"Dependency installation error: {e}")
        raise HTTPException(status_code=500, detail=f"Installation failed: {str(e)}")

@app.post("/dependencies/retry")
async def retry_failed_dependencies(token: bool = Depends(verify_token)):
    """Retry failed dependency installations"""
    try:
        result = await dependency_manager.retry_failed()
        return result
    except Exception as e:
        logger.error(f"Retry failed: {e}")
        raise HTTPException(status_code=500, detail=f"Retry failed: {str(e)}")

@app.get("/features")
async def list_features():
    """List all available features and their status"""
    features = dependency_manager.get_available_features()
    dependency_status = dependency_manager.get_status()
    
    return {
        "features": features,
        "phase_descriptions": dependency_status["phase_descriptions"],
        "installation_guide": {
            "phase_1_core": "Pre-installed (FastAPI, authentication)",
            "phase_2_ai_lightweight": "Basic utilities and web scraping",
            "phase_3_ai_core": "PyTorch and Transformers for AI generation",
            "phase_4_ai_advanced": "LangChain and vector search capabilities",
            "phase_5_training": "Model training and fine-tuning",
            "phase_6_math_science": "Scientific computing and analysis"
        },
        "next_steps": {
            "current_phase": dependency_manager.status["current_phase"],
            "auto_install": dependency_manager.status["installation_mode"] == "auto",
            "manual_install_command": f"POST /dependencies/install with phase: {dependency_manager.status['current_phase']}"
        }
    }

# Auto-installation on startup (if enabled)
@app.on_event("startup")
async def startup_event():
    """Initialize and optionally auto-install next phase"""
    logger.info("Autonomous AI Agent API starting up")
    
    if dependency_manager.status["installation_mode"] == "auto":
        try:
            # Install next phase in background
            asyncio.create_task(dependency_manager.install_next_phase())
            logger.info("Auto-installation initiated in background")
        except Exception as e:
            logger.error(f"Auto-installation failed: {e}")

# For Vercel
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)