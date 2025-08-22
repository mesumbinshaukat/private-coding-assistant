#!/usr/bin/env python3
"""
Vercel-Optimized Entry Point for Autonomous AI Agent API

Lightweight version optimized for Vercel's serverless environment
with reduced memory footprint and faster cold start times.
"""

import os
import json
import asyncio
from typing import Dict, Any, Optional
import logging

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Basic imports for minimal functionality
import jwt
from datetime import datetime
import requests
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "autonomous-ai-agent-secret-key-2024")
MODEL_NAME = os.getenv("MODEL_NAME", "distilgpt2")

# Initialize FastAPI app
app = FastAPI(
    title="Autonomous AI Agent API",
    description="Production-ready AI agent for coding tasks",
    version="1.0.0"
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
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

# Global variables for model (lazy loading)
_model = None
_tokenizer = None

def get_model():
    global _model, _tokenizer
    if _model is None:
        logger.info(f"Loading model: {MODEL_NAME}")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        logger.info("Model loaded successfully")
    return _model, _tokenizer

# Request/Response models
class CodeRequest(BaseModel):
    prompt: str = Field(..., description="Code generation prompt")
    language: str = Field("python", description="Programming language")
    max_tokens: int = Field(150, description="Maximum tokens to generate")

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(5, description="Maximum results")

class CodeResponse(BaseModel):
    code: str
    explanation: str
    confidence: float

class SearchResponse(BaseModel):
    results: list
    summary: str

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint - health check"""
    return {
        "message": "Autonomous AI Agent API",
        "status": "active",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/generate", response_model=CodeResponse)
async def generate_code(
    request: CodeRequest,
    token: dict = Depends(verify_token)
) -> CodeResponse:
    """Generate code based on prompt"""
    try:
        model, tokenizer = get_model()
        
        # Create a simple prompt
        prompt = f"# {request.language.title()} code for: {request.prompt}\n"
        
        # Generate text
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=request.max_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        
        result = generator(prompt, max_new_tokens=request.max_tokens)[0]['generated_text']
        
        # Extract the generated part
        generated = result[len(prompt):].strip()
        
        return CodeResponse(
            code=generated,
            explanation=f"Generated {request.language} code for: {request.prompt}",
            confidence=0.8
        )
        
    except Exception as e:
        logger.error(f"Code generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

@app.post("/search", response_model=SearchResponse)
async def search_web(
    request: SearchRequest,
    token: dict = Depends(verify_token)
) -> SearchResponse:
    """Simple web search using DuckDuckGo"""
    try:
        # Simple DuckDuckGo instant answer API
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={
                "q": request.query,
                "format": "json",
                "no_html": "1",
                "skip_disambig": "1"
            },
            timeout=10
        )
        
        data = response.json()
        
        results = []
        if data.get("AbstractText"):
            results.append({
                "title": data.get("AbstractSource", "DuckDuckGo"),
                "snippet": data.get("AbstractText"),
                "url": data.get("AbstractURL", "")
            })
        
        # Add related topics
        for topic in data.get("RelatedTopics", [])[:request.max_results]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append({
                    "title": topic.get("FirstURL", "").split("/")[-1],
                    "snippet": topic.get("Text"),
                    "url": topic.get("FirstURL", "")
                })
        
        return SearchResponse(
            results=results,
            summary=f"Found {len(results)} results for: {request.query}"
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.post("/reason")
async def reason_problem(
    request: dict,
    token: dict = Depends(verify_token)
):
    """Simple reasoning endpoint"""
    try:
        problem = request.get("problem", "")
        
        # Simple reasoning response
        reasoning_steps = [
            {
                "step": 1,
                "description": "Analyze the problem",
                "content": f"Understanding: {problem}"
            },
            {
                "step": 2,
                "description": "Break down the solution",
                "content": "Identifying key components and approach"
            },
            {
                "step": 3,
                "description": "Provide solution",
                "content": "Implementing the solution step by step"
            }
        ]
        
        return {
            "reasoning_steps": reasoning_steps,
            "solution": f"Solution approach for: {problem}",
            "confidence": 0.7
        }
        
    except Exception as e:
        logger.error(f"Reasoning error: {e}")
        raise HTTPException(status_code=500, detail=f"Reasoning failed: {str(e)}")

@app.get("/status")
async def get_status(token: dict = Depends(verify_token)):
    """Get agent status"""
    return {
        "model_loaded": _model is not None,
        "model_name": MODEL_NAME,
        "status": "active",
        "memory_usage": "optimized",
        "features": ["code_generation", "web_search", "reasoning"],
        "timestamp": datetime.now().isoformat()
    }

# For Vercel deployment
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
