"""
Autonomous AI Agent API - Main Entry Point
Production-ready FastAPI server for Vercel deployment
"""

import os
import json
import jwt
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from api.agent_core import AutonomousAgent
from api.utils.auth import verify_token
from api.utils.rate_limiter import RateLimiter
from api.utils.logger import setup_logger

# Setup logging
logger = setup_logger()

# Global agent instance
agent = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the AI agent on startup"""
    global agent
    logger.info("Initializing Autonomous AI Agent...")
    agent = AutonomousAgent()
    await agent.initialize()
    logger.info("Agent initialized successfully")
    yield
    logger.info("Shutting down agent...")

# Initialize FastAPI app
app = FastAPI(
    title="Autonomous AI Agent API",
    description="Production-ready autonomous AI agent for coding tasks",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for local desktop app access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()
rate_limiter = RateLimiter()

# Pydantic models
class CodeRequest(BaseModel):
    prompt: str = Field(..., description="Coding task or problem description")
    language: str = Field(default="python", description="Programming language")
    context: Optional[str] = Field(None, description="Additional context or existing code")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    depth: int = Field(default=5, description="Search depth (1-10)")
    include_code: bool = Field(default=True, description="Include code examples in search")

class TrainingRequest(BaseModel):
    dataset_name: Optional[str] = Field(None, description="Hugging Face dataset name")
    training_type: str = Field(default="rlhf", description="Training type: rlhf, fine_tune, or continuous")
    iterations: int = Field(default=10, description="Number of training iterations")

class ReasoningRequest(BaseModel):
    problem: str = Field(..., description="Problem to solve step-by-step")
    domain: str = Field(default="coding", description="Problem domain")
    include_math: bool = Field(default=True, description="Include mathematical reasoning")

# Dependency for authentication
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and rate limiting"""
    await rate_limiter.check_rate_limit(credentials.credentials)
    return verify_token(credentials.credentials)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "agent_status": "initialized" if agent else "not_initialized",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.post("/generate")
async def generate_code(
    request: CodeRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    """
    Generate code based on the prompt using the autonomous agent
    
    The agent will:
    1. Analyze the prompt using ReAct framework
    2. Generate code with explanations
    3. Test and validate the code
    4. Provide optimization suggestions
    """
    try:
        logger.info(f"Code generation request: {request.prompt[:100]}...")
        
        # Use the autonomous agent to generate code
        result = await agent.generate_code(
            prompt=request.prompt,
            language=request.language,
            context=request.context
        )
        
        # Background task for continuous learning
        background_tasks.add_task(agent.learn_from_interaction, "code_generation", request.dict(), result)
        
        return {
            "code": result["code"],
            "explanation": result["explanation"],
            "test_cases": result["test_cases"],
            "complexity_analysis": result["complexity_analysis"],
            "optimizations": result["optimizations"],
            "reasoning_steps": result["reasoning_steps"],
            "confidence": result["confidence"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Code generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Code generation failed: {str(e)}")

@app.post("/search")
async def deep_search(
    request: SearchRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    """
    Perform deep web search for coding-related information
    
    The agent will:
    1. Search multiple sources (Stack Overflow, GitHub, documentation)
    2. Analyze and synthesize results
    3. Extract relevant code examples
    4. Provide confidence scoring
    """
    try:
        logger.info(f"Deep search request: {request.query[:100]}...")
        
        result = await agent.deep_search(
            query=request.query,
            depth=request.depth,
            include_code=request.include_code
        )
        
        # Background learning
        background_tasks.add_task(agent.learn_from_interaction, "search", request.dict(), result)
        
        return {
            "results": result["results"],
            "synthesized_answer": result["synthesized_answer"],
            "code_examples": result["code_examples"],
            "sources": result["sources"],
            "confidence": result["confidence"],
            "related_queries": result["related_queries"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Search error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/train")
async def trigger_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    """
    Trigger self-training of the agent
    
    The agent will:
    1. Load specified dataset or use accumulated interaction data
    2. Perform LoRA fine-tuning
    3. Evaluate improvements
    4. Update model weights if improvements are detected
    """
    try:
        logger.info(f"Training request: {request.training_type} for {request.iterations} iterations")
        
        # Start training in background
        background_tasks.add_task(
            agent.self_train,
            dataset_name=request.dataset_name,
            training_type=request.training_type,
            iterations=request.iterations
        )
        
        return {
            "status": "training_started",
            "training_type": request.training_type,
            "iterations": request.iterations,
            "dataset": request.dataset_name,
            "estimated_duration": f"{request.iterations * 2} minutes",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")

@app.post("/reason")
async def step_by_step_reasoning(
    request: ReasoningRequest,
    background_tasks: BackgroundTasks,
    user: dict = Depends(get_current_user)
):
    """
    Perform step-by-step reasoning on complex problems
    
    The agent will:
    1. Break down the problem into sub-problems
    2. Apply domain-specific reasoning (mathematical, algorithmic, etc.)
    3. Provide detailed explanations for each step
    4. Validate the solution
    """
    try:
        logger.info(f"Reasoning request: {request.problem[:100]}...")
        
        result = await agent.reason_step_by_step(
            problem=request.problem,
            domain=request.domain,
            include_math=request.include_math
        )
        
        # Background learning
        background_tasks.add_task(agent.learn_from_interaction, "reasoning", request.dict(), result)
        
        return {
            "reasoning_steps": result["reasoning_steps"],
            "solution": result["solution"],
            "mathematical_analysis": result["mathematical_analysis"],
            "alternative_approaches": result["alternative_approaches"],
            "confidence": result["confidence"],
            "complexity_analysis": result["complexity_analysis"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Reasoning error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reasoning failed: {str(e)}")

@app.get("/status")
async def get_agent_status(user: dict = Depends(get_current_user)):
    """Get detailed agent status and capabilities"""
    try:
        status = await agent.get_status()
        return {
            "agent_status": status,
            "memory_usage": agent.get_memory_usage(),
            "model_info": agent.get_model_info(),
            "training_history": agent.get_training_history(),
            "capabilities": agent.get_capabilities(),
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Status error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

@app.post("/feedback")
async def provide_feedback(
    feedback: Dict[str, Any],
    user: dict = Depends(get_current_user)
):
    """Provide feedback to improve the agent's performance"""
    try:
        await agent.process_feedback(feedback)
        return {
            "status": "feedback_received",
            "message": "Thank you for the feedback. The agent will use this to improve.",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Feedback error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feedback processing failed: {str(e)}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    logger.error(f"HTTP error: {exc.detail}")
    return {"error": exc.detail, "status_code": exc.status_code}

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {str(exc)}")
    return {"error": "Internal server error", "status_code": 500}

# For local development
if __name__ == "__main__":
    uvicorn.run(
        "index:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
