from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import jwt
import os

# Configuration
SECRET_KEY = os.getenv("SECRET_KEY", "autonomous-ai-agent-secret-key-2024")

# Initialize FastAPI app
app = FastAPI(title="Autonomous AI Agent API", version="1.0.0")

# Authentication
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return True
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

# Request models
class CodeRequest(BaseModel):
    prompt: str
    language: str = "python"

class SearchRequest(BaseModel):
    query: str

# Endpoints
@app.get("/")
async def root():
    return {"message": "Autonomous AI Agent API", "status": "active"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/generate")
async def generate_code(request: CodeRequest, token: bool = Depends(verify_token)):
    try:
        return {
            "code": f"# Generated {request.language} code for: {request.prompt}\nprint('Hello, World!')",
            "explanation": f"Basic {request.language} implementation",
            "confidence": 0.8
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/search")
async def search_web(request: SearchRequest, token: bool = Depends(verify_token)):
    try:
        return {
            "results": [{"title": "Example", "snippet": f"Search results for: {request.query}"}],
            "summary": f"Found results for {request.query}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reason")
async def reason_problem(request: dict, token: bool = Depends(verify_token)):
    try:
        problem = request.get("problem", "")
        return {
            "reasoning_steps": [
                {"step": 1, "description": "Analyze", "content": f"Understanding: {problem}"}
            ],
            "solution": f"Solution for: {problem}",
            "confidence": 0.7
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status(token: bool = Depends(verify_token)):
    return {
        "status": "active",
        "features": ["code_generation", "web_search", "reasoning"]
    }