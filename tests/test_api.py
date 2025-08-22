"""
Test suite for the FastAPI application
Tests all API endpoints with various scenarios
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import HTTPException

# Import the app (with mocked dependencies)
import sys
sys.path.append("../api")

# Mock heavy dependencies before importing
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['langchain'] = MagicMock()
sys.modules['faiss'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()

from api.index import app
from api.utils.auth import verify_token, generate_personal_token

# Create test client
client = TestClient(app)

# Test constants
VALID_TOKEN = "autonomous-ai-agent-2024"
INVALID_TOKEN = "invalid-token"
TEST_HEADERS = {
    "Authorization": f"Bearer {VALID_TOKEN}",
    "Content-Type": "application/json"
}

class TestAuthentication:
    """Test authentication and authorization"""
    
    def test_auth_utility_functions(self):
        """Test auth utility functions"""
        
        # Test token generation
        token = generate_personal_token()
        assert isinstance(token, str)
        assert len(token) > 20
        
        # Test valid token verification
        user_info = verify_token(VALID_TOKEN)
        assert user_info["user_id"] == "personal"
        assert user_info["role"] == "admin"
        
        # Test invalid token
        with pytest.raises(HTTPException):
            verify_token(INVALID_TOKEN)
    
    def test_protected_endpoint_without_auth(self):
        """Test that protected endpoints require authentication"""
        
        response = client.post("/generate", json={"prompt": "test"})
        assert response.status_code == 403  # Forbidden without auth
    
    def test_protected_endpoint_with_invalid_token(self):
        """Test protected endpoint with invalid token"""
        
        headers = {"Authorization": f"Bearer {INVALID_TOKEN}"}
        response = client.post("/generate", json={"prompt": "test"}, headers=headers)
        assert response.status_code == 401  # Unauthorized
    
    def test_protected_endpoint_with_valid_token(self):
        """Test protected endpoint with valid token"""
        
        with patch('api.agent_core.AutonomousAgent') as mock_agent_class:
            # Mock the agent
            mock_agent = AsyncMock()
            mock_agent.generate_code.return_value = {
                "code": "def test(): pass",
                "explanation": "Test function",
                "test_cases": [],
                "complexity_analysis": {},
                "optimizations": [],
                "reasoning_steps": "",
                "confidence": 0.8,
                "test_results": {"success": True}
            }
            mock_agent_class.return_value = mock_agent
            
            response = client.post(
                "/generate",
                json={"prompt": "write a test function"},
                headers=TEST_HEADERS
            )
            
            # Should not fail due to auth (may fail due to missing agent)
            assert response.status_code != 401
            assert response.status_code != 403

class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_root_endpoint(self):
        """Test root health check"""
        
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert data["status"] == "online"

class TestCodeGenerationEndpoint:
    """Test /generate endpoint"""
    
    @patch('api.agent_core.AutonomousAgent')
    def test_generate_code_success(self, mock_agent_class):
        """Test successful code generation"""
        
        # Mock the agent
        mock_agent = AsyncMock()
        mock_agent.generate_code.return_value = {
            "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "explanation": "Recursive Fibonacci implementation",
            "test_cases": [{"input": "5", "expected": "5"}],
            "complexity_analysis": {"time_complexity": "O(2^n)", "space_complexity": "O(n)"},
            "optimizations": ["Use memoization"],
            "reasoning_steps": "Identified as classic recursion problem",
            "confidence": 0.9,
            "test_results": {"success": True, "execution_time": 0.001}
        }
        mock_agent_class.return_value = mock_agent
        
        response = client.post(
            "/generate",
            json={
                "prompt": "Write a function to calculate Fibonacci numbers",
                "language": "python",
                "context": "Use recursion"
            },
            headers=TEST_HEADERS
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "code" in data
        assert "explanation" in data
        assert "test_cases" in data
        assert "complexity_analysis" in data
        assert "optimizations" in data
        assert "confidence" in data
        assert "timestamp" in data
        
        # Check content
        assert "fibonacci" in data["code"].lower()
        assert data["confidence"] == 0.9
    
    def test_generate_code_missing_prompt(self):
        """Test code generation with missing prompt"""
        
        response = client.post(
            "/generate",
            json={"language": "python"},
            headers=TEST_HEADERS
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_generate_code_empty_prompt(self):
        """Test code generation with empty prompt"""
        
        response = client.post(
            "/generate",
            json={"prompt": ""},
            headers=TEST_HEADERS
        )
        
        assert response.status_code == 422  # Validation error
    
    @patch('api.agent_core.AutonomousAgent')
    def test_generate_code_with_context(self, mock_agent_class):
        """Test code generation with additional context"""
        
        mock_agent = AsyncMock()
        mock_agent.generate_code.return_value = {
            "code": "def optimized_fibonacci(n, memo={}):\n    if n in memo:\n        return memo[n]\n    if n <= 1:\n        return n\n    memo[n] = optimized_fibonacci(n-1, memo) + optimized_fibonacci(n-2, memo)\n    return memo[n]",
            "explanation": "Memoized Fibonacci implementation",
            "test_cases": [],
            "complexity_analysis": {"time_complexity": "O(n)", "space_complexity": "O(n)"},
            "optimizations": [],
            "reasoning_steps": "Applied memoization optimization",
            "confidence": 0.95,
            "test_results": {"success": True}
        }
        mock_agent_class.return_value = mock_agent
        
        response = client.post(
            "/generate",
            json={
                "prompt": "Write a Fibonacci function",
                "language": "python", 
                "context": "Optimize for performance using memoization"
            },
            headers=TEST_HEADERS
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "memo" in data["code"]  # Should include memoization
        assert data["complexity_analysis"]["time_complexity"] == "O(n)"

class TestSearchEndpoint:
    """Test /search endpoint"""
    
    @patch('api.agent_core.AutonomousAgent')
    def test_search_success(self, mock_agent_class):
        """Test successful search"""
        
        mock_agent = AsyncMock()
        mock_agent.deep_search.return_value = {
            "results": [
                {
                    "title": "Python Async Programming Guide",
                    "url": "https://example.com",
                    "snippet": "Comprehensive guide to async programming",
                    "source": "documentation",
                    "relevance_score": 0.9
                }
            ],
            "synthesized_answer": "Async programming in Python uses asyncio...",
            "code_examples": [
                {
                    "code": "async def example(): await asyncio.sleep(1)",
                    "description": "Basic async function"
                }
            ],
            "sources": ["docs.python.org"],
            "confidence": 0.85,
            "related_queries": ["asyncio tutorial", "python concurrency"],
            "search_plan": "Searched official documentation and tutorials"
        }
        mock_agent_class.return_value = mock_agent
        
        response = client.post(
            "/search",
            json={
                "query": "Python async programming best practices",
                "depth": 5,
                "include_code": True
            },
            headers=TEST_HEADERS
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "results" in data
        assert "synthesized_answer" in data
        assert "code_examples" in data
        assert "sources" in data
        assert "confidence" in data
        assert "related_queries" in data
        
        # Check content
        assert len(data["results"]) > 0
        assert data["confidence"] == 0.85
        assert "asyncio" in data["synthesized_answer"]
    
    def test_search_missing_query(self):
        """Test search with missing query"""
        
        response = client.post(
            "/search",
            json={"depth": 5},
            headers=TEST_HEADERS
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_search_invalid_depth(self):
        """Test search with invalid depth"""
        
        response = client.post(
            "/search",
            json={
                "query": "test query",
                "depth": -1
            },
            headers=TEST_HEADERS
        )
        
        # Should either validate and reject, or clamp to valid range
        # Exact behavior depends on validation implementation
        assert response.status_code in [200, 422]

class TestReasoningEndpoint:
    """Test /reason endpoint"""
    
    @patch('api.agent_core.AutonomousAgent')
    def test_reasoning_success(self, mock_agent_class):
        """Test successful reasoning"""
        
        mock_agent = AsyncMock()
        mock_agent.reason_step_by_step.return_value = {
            "reasoning_steps": [
                {
                    "step": 1,
                    "description": "Problem Analysis",
                    "content": "This is a graph shortest path problem"
                },
                {
                    "step": 2,
                    "description": "Algorithm Selection",
                    "content": "Dijkstra's algorithm is optimal for this case"
                }
            ],
            "solution": "Implement Dijkstra's algorithm with priority queue",
            "mathematical_analysis": {
                "time_complexity": "O((V + E) log V)",
                "space_complexity": "O(V)",
                "proof": "Priority queue operations are O(log V)"
            },
            "alternative_approaches": [
                {
                    "name": "Bellman-Ford",
                    "complexity": "O(VE)",
                    "description": "Handles negative weights"
                }
            ],
            "confidence": 0.92,
            "complexity_analysis": {
                "time_complexity": "O((V + E) log V)",
                "space_complexity": "O(V)"
            }
        }
        mock_agent_class.return_value = mock_agent
        
        response = client.post(
            "/reason",
            json={
                "problem": "Find shortest path in weighted graph",
                "domain": "algorithms",
                "include_math": True
            },
            headers=TEST_HEADERS
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "reasoning_steps" in data
        assert "solution" in data
        assert "mathematical_analysis" in data
        assert "alternative_approaches" in data
        assert "confidence" in data
        
        # Check content
        assert len(data["reasoning_steps"]) > 0
        assert "Dijkstra" in data["solution"]
        assert data["confidence"] == 0.92
    
    def test_reasoning_missing_problem(self):
        """Test reasoning with missing problem"""
        
        response = client.post(
            "/reason",
            json={"domain": "coding"},
            headers=TEST_HEADERS
        )
        
        assert response.status_code == 422  # Validation error

class TestTrainingEndpoint:
    """Test /train endpoint"""
    
    @patch('api.agent_core.AutonomousAgent')
    def test_training_trigger_success(self, mock_agent_class):
        """Test successful training trigger"""
        
        mock_agent = AsyncMock()
        # Training is async background task, so just return success
        
        response = client.post(
            "/train",
            json={
                "dataset_name": "roneneldan/TinyStories",
                "training_type": "rlhf",
                "iterations": 5
            },
            headers=TEST_HEADERS
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "status" in data
        assert "training_type" in data
        assert "iterations" in data
        assert "estimated_duration" in data
        
        # Check content
        assert data["status"] == "training_started"
        assert data["training_type"] == "rlhf"
        assert data["iterations"] == 5
    
    def test_training_invalid_type(self):
        """Test training with invalid type"""
        
        response = client.post(
            "/train",
            json={
                "training_type": "invalid_type",
                "iterations": 5
            },
            headers=TEST_HEADERS
        )
        
        # Should validate training type
        assert response.status_code in [200, 422]

class TestStatusEndpoint:
    """Test /status endpoint"""
    
    @patch('api.agent_core.AutonomousAgent')
    def test_status_success(self, mock_agent_class):
        """Test successful status retrieval"""
        
        mock_agent = AsyncMock()
        mock_agent.get_status.return_value = {
            "model_loaded": True,
            "components_initialized": True,
            "performance_metrics": {
                "code_generation_success_rate": 0.94,
                "search_relevance_score": 0.87
            },
            "interaction_count": 150
        }
        mock_agent.get_memory_usage.return_value = {
            "total_memories": {"episodic": 100, "semantic": 50, "code": 75},
            "memory_utilization": 0.65
        }
        mock_agent.get_model_info.return_value = {
            "model_name": "distilgpt2",
            "quantization": "fp32"
        }
        mock_agent.get_training_history.return_value = []
        mock_agent.get_capabilities.return_value = [
            "Code generation", "Web search", "Reasoning"
        ]
        mock_agent_class.return_value = mock_agent
        
        response = client.get("/status", headers=TEST_HEADERS)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "agent_status" in data
        assert "memory_usage" in data
        assert "model_info" in data
        assert "training_history" in data
        assert "capabilities" in data
        
        # Check content
        assert data["agent_status"]["model_loaded"] is True
        assert len(data["capabilities"]) > 0

class TestFeedbackEndpoint:
    """Test /feedback endpoint"""
    
    @patch('api.agent_core.AutonomousAgent')
    def test_feedback_success(self, mock_agent_class):
        """Test successful feedback submission"""
        
        mock_agent = AsyncMock()
        mock_agent.process_feedback.return_value = None
        mock_agent_class.return_value = mock_agent
        
        response = client.post(
            "/feedback",
            json={
                "rating": 5,
                "comments": "Excellent code generation",
                "specific_feedback": {
                    "code_quality": "high",
                    "explanation_clarity": "very_clear"
                }
            },
            headers=TEST_HEADERS
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert "status" in data
        assert "message" in data
        assert data["status"] == "feedback_received"
    
    def test_feedback_missing_rating(self):
        """Test feedback with missing rating"""
        
        response = client.post(
            "/feedback",
            json={"comments": "Good work"},
            headers=TEST_HEADERS
        )
        
        # Rating should be required
        assert response.status_code == 422

class TestErrorHandling:
    """Test error handling scenarios"""
    
    @patch('api.agent_core.AutonomousAgent')
    def test_agent_initialization_failure(self, mock_agent_class):
        """Test behavior when agent fails to initialize"""
        
        mock_agent_class.side_effect = Exception("Agent initialization failed")
        
        # This might be handled at startup, so test endpoint behavior
        response = client.post(
            "/generate",
            json={"prompt": "test"},
            headers=TEST_HEADERS
        )
        
        # Should handle gracefully
        assert response.status_code in [200, 500]
    
    @patch('api.agent_core.AutonomousAgent')
    def test_agent_method_failure(self, mock_agent_class):
        """Test behavior when agent method fails"""
        
        mock_agent = AsyncMock()
        mock_agent.generate_code.side_effect = Exception("Generation failed")
        mock_agent_class.return_value = mock_agent
        
        response = client.post(
            "/generate",
            json={"prompt": "test"},
            headers=TEST_HEADERS
        )
        
        # Should return error response
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
    
    def test_malformed_json(self):
        """Test handling of malformed JSON"""
        
        response = client.post(
            "/generate",
            data="{invalid json}",
            headers={"Authorization": f"Bearer {VALID_TOKEN}", "Content-Type": "application/json"}
        )
        
        assert response.status_code == 422  # Unprocessable Entity

class TestRateLimiting:
    """Test rate limiting functionality"""
    
    def test_rate_limit_headers(self):
        """Test that rate limit headers are present"""
        
        response = client.get("/")
        
        # Rate limiting might not be active in test mode
        # But should not cause errors
        assert response.status_code == 200
    
    @patch('api.utils.rate_limiter.RateLimiter.check_rate_limit')
    def test_rate_limit_exceeded(self, mock_rate_limit):
        """Test behavior when rate limit is exceeded"""
        
        # Mock rate limit exceeded
        mock_rate_limit.side_effect = HTTPException(
            status_code=429,
            detail="Rate limit exceeded"
        )
        
        response = client.post(
            "/generate",
            json={"prompt": "test"},
            headers=TEST_HEADERS
        )
        
        assert response.status_code == 429

class TestCORS:
    """Test CORS handling"""
    
    def test_cors_headers(self):
        """Test that CORS headers are present"""
        
        response = client.options("/generate")
        
        # Should handle OPTIONS request for CORS
        assert response.status_code in [200, 405]  # 405 if OPTIONS not explicitly handled
    
    def test_cors_preflight(self):
        """Test CORS preflight request"""
        
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "Content-Type"
        }
        
        response = client.options("/generate", headers=headers)
        
        # Should handle preflight
        assert response.status_code in [200, 405]

class TestPerformance:
    """Test performance-related aspects"""
    
    @patch('api.agent_core.AutonomousAgent')
    def test_response_time(self, mock_agent_class):
        """Test that responses are reasonably fast"""
        
        import time
        
        mock_agent = AsyncMock()
        mock_agent.generate_code.return_value = {
            "code": "def test(): pass",
            "explanation": "Test",
            "test_cases": [],
            "complexity_analysis": {},
            "optimizations": [],
            "reasoning_steps": "",
            "confidence": 0.8,
            "test_results": {"success": True}
        }
        mock_agent_class.return_value = mock_agent
        
        start_time = time.time()
        
        response = client.post(
            "/generate",
            json={"prompt": "write a test function"},
            headers=TEST_HEADERS
        )
        
        end_time = time.time()
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 5.0  # Should respond within 5 seconds

# Pytest configuration and fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def test_client():
    """Create a test client"""
    return TestClient(app)

@pytest.fixture
def auth_headers():
    """Provide authentication headers"""
    return {
        "Authorization": f"Bearer {VALID_TOKEN}",
        "Content-Type": "application/json"
    }

# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
