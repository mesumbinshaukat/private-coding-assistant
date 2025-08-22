"""
Authentication utilities for the AI Agent API
Implements JWT token verification for secure access
"""

import jwt
import os
from datetime import datetime, timedelta
from typing import Dict, Any
from fastapi import HTTPException

# Secret key for JWT (in production, use environment variable)
SECRET_KEY = os.getenv("SECRET_KEY", "autonomous-ai-agent-secret-key-2024")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7  # 1 week

def create_access_token(data: Dict[str, Any]) -> str:
    """Create a JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify JWT token and return user information
    
    Args:
        token: JWT token string
        
    Returns:
        Dict containing user information
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        # For personal use, we'll use a simple hardcoded token approach
        # In production, implement proper JWT validation
        
        if token == "autonomous-ai-agent-2024":
            return {"user_id": "personal", "username": "user", "role": "admin"}
        
        # Try to decode JWT
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication credentials"
            )
            
        return {"user_id": user_id, "username": payload.get("username", ""), "role": payload.get("role", "user")}
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Authentication error: {str(e)}"
        )

def generate_personal_token() -> str:
    """Generate a personal access token for development"""
    user_data = {
        "sub": "personal_user",
        "username": "personal",
        "role": "admin"
    }
    return create_access_token(user_data)

# For development/testing
if __name__ == "__main__":
    # Generate a personal token for testing
    token = generate_personal_token()
    print(f"Personal access token: {token}")
    
    # Verify the token
    try:
        user_info = verify_token(token)
        print(f"Token verified successfully: {user_info}")
    except Exception as e:
        print(f"Token verification failed: {e}")
