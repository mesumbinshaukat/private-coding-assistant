"""
Lightweight FastAPI application for Vercel deployment
Optimized for serverless environments with minimal dependencies
"""

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import logging
import os
import time
from collections import defaultdict
import asyncio

# Import the lightweight agent
from agent_core_lite import get_agent

# Simple logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app with metadata
app = FastAPI(
    title="Autonomous AI Agent API",
    description="Lightweight autonomous AI agent optimized for serverless deployment",
    version="1.0.0-lite",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple rate limiting
rate_limit_store = defaultdict(list)
RATE_LIMIT = 10  # requests per minute
RATE_WINDOW = 60  # seconds

def check_rate_limit(client_ip: str):
    """Simple in-memory rate limiting"""
    current_time = time.time()
    
    # Clean old requests
    rate_limit_store[client_ip] = [
        req_time for req_time in rate_limit_store[client_ip]
        if current_time - req_time < RATE_WINDOW
    ]
    
    # Check limit
    if len(rate_limit_store[client_ip]) >= RATE_LIMIT:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait before making more requests."
        )
    
    # Record this request
    rate_limit_store[client_ip].append(current_time)

def verify_token(authorization: Optional[str] = Header(None)):
    """Simple token verification"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authorization scheme")
        
        # Simple token check (in production, use proper JWT validation)
        valid_tokens = [
            "autonomous-ai-agent-2024",
            "autonomous-ai-agent-secret-key-2024",
            os.getenv("SECRET_KEY", "default-secret-key")
        ]
        
        if token not in valid_tokens:
            raise HTTPException(status_code=401, detail="Invalid token")
            
        return token
        
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization format")

# Request models
class CodeRequest(BaseModel):
    prompt: str = Field(..., description="Code generation prompt")
    language: str = Field(default="python", description="Programming language")
    context: Optional[str] = Field(None, description="Additional context")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    depth: int = Field(default=5, description="Search depth")
    include_code: bool = Field(default=True, description="Include code examples")

class ReasonRequest(BaseModel):
    problem: str = Field(..., description="Problem to reason about")
    domain: str = Field(default="coding", description="Problem domain")
    include_math: bool = Field(default=True, description="Include mathematical analysis")

class FeedbackRequest(BaseModel):
    feedback_type: str = Field(..., description="Type of feedback")
    content: str = Field(..., description="Feedback content")
    rating: Optional[int] = Field(None, description="Rating 1-5")

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Autonomous AI Agent API",
        "version": "1.0.0-lite",
        "status": "operational",
        "deployment": "serverless",
        "docs": "/docs",
        "endpoints": ["/generate", "/search", "/reason", "/feedback", "/status"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": time.time()}

@app.post("/generate")
async def generate_code(
    request: CodeRequest,
    req: Request,
    authorization: str = Header(None)
):
    """Generate code based on prompt"""
    # Rate limiting and auth
    client_ip = req.client.host
    check_rate_limit(client_ip)
    verify_token(authorization)
    
    try:
        agent = await get_agent()
        result = await agent.generate_code(
            request.prompt, 
            request.language, 
            request.context
        )
        
        return {
            "success": True,
            "result": result,
            "metadata": {
                "client_ip": client_ip,
                "timestamp": time.time()
            }
        }
        
    except Exception as e:
        logger.error(f"Code generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def deep_search(
    request: SearchRequest,
    req: Request,
    authorization: str = Header(None)
):
    """Perform deep web search"""
    # Rate limiting and auth
    client_ip = req.client.host
    check_rate_limit(client_ip)
    verify_token(authorization)
    
    try:
        agent = await get_agent()
        result = await agent.deep_search(
            request.query, 
            request.depth, 
            request.include_code
        )
        
        return {
            "success": True,
            "result": result,
            "metadata": {
                "client_ip": client_ip,
                "timestamp": time.time()
            }
        }
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reason")
async def reason_step_by_step(
    request: ReasonRequest,
    req: Request,
    authorization: str = Header(None)
):
    """Perform step-by-step reasoning"""
    # Rate limiting and auth
    client_ip = req.client.host
    check_rate_limit(client_ip)
    verify_token(authorization)
    
    try:
        agent = await get_agent()
        result = await agent.reason_step_by_step(
            request.problem, 
            request.domain, 
            request.include_math
        )
        
        return {
            "success": True,
            "result": result,
            "metadata": {
                "client_ip": client_ip,
                "timestamp": time.time()
            }
        }
        
    except Exception as e:
        logger.error(f"Reasoning error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    req: Request,
    authorization: str = Header(None)
):
    """Submit feedback"""
    # Rate limiting and auth
    client_ip = req.client.host
    check_rate_limit(client_ip)
    verify_token(authorization)
    
    try:
        agent = await get_agent()
        result = await agent.process_feedback({
            "feedback_type": request.feedback_type,
            "content": request.content,
            "rating": request.rating,
            "timestamp": time.time()
        })
        
        return {
            "success": True,
            "result": result,
            "message": "Feedback received successfully"
        }
        
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status(
    req: Request,
    authorization: str = Header(None)
):
    """Get agent status"""
    # Rate limiting and auth
    client_ip = req.client.host
    check_rate_limit(client_ip)
    verify_token(authorization)
    
    try:
        agent = await get_agent()
        result = await agent.get_status()
        
        return {
            "success": True,
            "result": result,
            "api_info": {
                "version": "1.0.0-lite",
                "deployment": "serverless",
                "rate_limit": f"{RATE_LIMIT} requests per {RATE_WINDOW} seconds"
            }
        }
        
    except Exception as e:
        logger.error(f"Status error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def train_model(
    req: Request,
    authorization: str = Header(None)
):
    """Training endpoint (mock for serverless)"""
    # Rate limiting and auth
    client_ip = req.client.host
    check_rate_limit(client_ip)
    verify_token(authorization)
    
    return {
        "success": True,
        "message": "Training is not available in serverless deployment",
        "suggestion": "Use the Google Colab training scripts for model fine-tuning",
        "colab_script": "scripts/train_colab.py"
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler"""
    return {
        "success": False,
        "error": exc.detail,
        "status_code": exc.status_code,
        "timestamp": time.time()
    }

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler"""
    logger.error(f"Unexpected error: {str(exc)}")
    return {
        "success": False,
        "error": "Internal server error",
        "message": "An unexpected error occurred",
        "timestamp": time.time()
    }

# For Vercel serverless deployment
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
