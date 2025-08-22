"""
Test suite for utility modules
Tests authentication, rate limiting, logging, web search, code execution, etc.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import AsyncMock, patch, MagicMock
import numpy as np

# Mock heavy dependencies
import sys
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['faiss'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['scipy'] = MagicMock()
sys.modules['sympy'] = MagicMock()
sys.modules['beautifulsoup4'] = MagicMock()
sys.modules['aiohttp'] = MagicMock()

from api.utils.auth import verify_token, generate_personal_token, create_access_token
from api.utils.rate_limiter import RateLimiter, AdvancedRateLimiter
from api.utils.logger import setup_logger, AgentLogger
from fastapi import HTTPException

class TestAuthentication:
    """Test authentication utilities"""
    
    def test_create_access_token(self):
        """Test JWT token creation"""
        
        test_data = {
            "sub": "test_user",
            "username": "testuser",
            "role": "user"
        }
        
        token = create_access_token(test_data)
        
        assert isinstance(token, str)
        assert len(token) > 20  # JWT tokens are long
        assert "." in token  # JWT has dots separating sections
    
    def test_verify_valid_token(self):
        """Test verification of valid token"""
        
        # Test hardcoded token
        user_info = verify_token("autonomous-ai-agent-2024")
        
        assert user_info["user_id"] == "personal"
        assert user_info["username"] == "user"
        assert user_info["role"] == "admin"
    
    def test_verify_invalid_token(self):
        """Test verification of invalid token"""
        
        with pytest.raises(HTTPException) as exc_info:
            verify_token("invalid-token")
        
        assert exc_info.value.status_code == 401
        assert "Invalid token" in str(exc_info.value.detail)
    
    def test_verify_jwt_token(self):
        """Test verification of actual JWT token"""
        
        # Create a token and verify it
        test_data = {"sub": "test", "username": "test", "role": "user"}
        token = create_access_token(test_data)
        
        user_info = verify_token(token)
        
        assert user_info["user_id"] == "test"
        assert user_info["username"] == "test"
        assert user_info["role"] == "user"
    
    def test_generate_personal_token(self):
        """Test personal token generation"""
        
        token = generate_personal_token()
        
        assert isinstance(token, str)
        assert len(token) > 20
        
        # Should be verifiable
        user_info = verify_token(token)
        assert user_info["user_id"] == "personal_user"

class TestRateLimiter:
    """Test rate limiting functionality"""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_basic_functionality(self):
        """Test basic rate limiting"""
        
        # Create a rate limiter with low limits for testing
        limiter = RateLimiter(requests_per_minute=2, burst_size=2)
        
        # First request should succeed
        result1 = await limiter.check_rate_limit("test_user")
        assert result1 is True
        
        # Second request should succeed (within burst)
        result2 = await limiter.check_rate_limit("test_user")
        assert result2 is True
        
        # Third request should fail (burst exceeded)
        with pytest.raises(HTTPException) as exc_info:
            await limiter.check_rate_limit("test_user")
        
        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in str(exc_info.value.detail)
    
    @pytest.mark.asyncio
    async def test_rate_limiter_different_users(self):
        """Test rate limiting for different users"""
        
        limiter = RateLimiter(requests_per_minute=1, burst_size=1)
        
        # Different users should have separate limits
        result1 = await limiter.check_rate_limit("user1")
        assert result1 is True
        
        result2 = await limiter.check_rate_limit("user2")
        assert result2 is True
        
        # Each user should be limited independently
        with pytest.raises(HTTPException):
            await limiter.check_rate_limit("user1")
        
        with pytest.raises(HTTPException):
            await limiter.check_rate_limit("user2")
    
    def test_rate_limiter_bucket_status(self):
        """Test rate limiter bucket status"""
        
        limiter = RateLimiter(requests_per_minute=60, burst_size=10)
        
        status = limiter.get_bucket_status("test_user")
        
        assert "current_tokens" in status
        assert "max_tokens" in status
        assert "refill_rate" in status
        assert "time_to_full" in status
        
        assert status["max_tokens"] == 10
        assert status["current_tokens"] == 10  # Should start full
    
    def test_reset_bucket(self):
        """Test bucket reset functionality"""
        
        limiter = RateLimiter(requests_per_minute=1, burst_size=1)
        
        # Use up the bucket
        asyncio.run(limiter.check_rate_limit("test_user"))
        
        # Reset the bucket
        limiter.reset_bucket("test_user")
        
        # Should be able to make request again
        result = asyncio.run(limiter.check_rate_limit("test_user"))
        assert result is True
    
    @pytest.mark.asyncio
    async def test_advanced_rate_limiter(self):
        """Test advanced rate limiter with roles"""
        
        limiter = AdvancedRateLimiter()
        
        # Admin should have higher limits
        result = await limiter.check_rate_limit("admin_user", "admin")
        assert result is True
        
        # User should have standard limits
        result = await limiter.check_rate_limit("regular_user", "user")
        assert result is True
        
        # Anonymous should have lowest limits
        result = await limiter.check_rate_limit("anon_user", "anonymous")
        assert result is True
    
    def test_get_limits_for_role(self):
        """Test getting limits for different roles"""
        
        limiter = AdvancedRateLimiter()
        
        admin_limits = limiter.get_limits_for_role("admin")
        user_limits = limiter.get_limits_for_role("user")
        anon_limits = limiter.get_limits_for_role("anonymous")
        
        # Admin should have highest limits
        assert admin_limits["requests_per_minute"] > user_limits["requests_per_minute"]
        assert user_limits["requests_per_minute"] > anon_limits["requests_per_minute"]

class TestLogger:
    """Test logging utilities"""
    
    def test_logger_setup(self):
        """Test logger setup"""
        
        logger = setup_logger("test_logger")
        
        assert logger.name == "test_logger"
        assert len(logger.handlers) > 0
    
    def test_agent_logger_creation(self):
        """Test AgentLogger creation"""
        
        agent_logger = AgentLogger("test_agent")
        
        assert agent_logger.name == "test_agent"
        assert agent_logger.logger is not None
    
    def test_agent_logger_methods(self):
        """Test AgentLogger methods"""
        
        agent_logger = AgentLogger("test_agent")
        
        # Test different log levels
        agent_logger.info("Test info message", extra_data="test")
        agent_logger.debug("Test debug message")
        agent_logger.warning("Test warning message")
        agent_logger.error("Test error message")
        agent_logger.critical("Test critical message")
        
        # Should not raise exceptions
        assert True
    
    def test_performance_logging(self):
        """Test performance logging"""
        
        agent_logger = AgentLogger("test_agent")
        
        agent_logger.log_performance("test_operation", 1.5, metric1=100, metric2=200)
        
        # Should not raise exceptions
        assert True
    
    def test_request_logging(self):
        """Test API request logging"""
        
        agent_logger = AgentLogger("test_agent")
        
        agent_logger.log_request("/generate", "POST", "user123", 2.0)
        
        # Should not raise exceptions
        assert True

class TestWebSearch:
    """Test web search utilities"""
    
    @pytest.fixture
    def web_searcher(self):
        """Create a web searcher for testing"""
        
        # Mock the actual search implementation
        sys.modules['duckduckgo_search'] = MagicMock()
        
        from api.utils.web_search import WebSearcher
        searcher = WebSearcher()
        return searcher
    
    @pytest.mark.asyncio
    async def test_web_searcher_initialization(self, web_searcher):
        """Test web searcher initialization"""
        
        await web_searcher.initialize()
        
        assert web_searcher.session is not None
        assert hasattr(web_searcher, 'cache')
    
    @pytest.mark.asyncio
    async def test_duckduckgo_search(self, web_searcher):
        """Test DuckDuckGo search functionality"""
        
        with patch('api.utils.web_search.DDGS') as mock_ddgs:
            # Mock DuckDuckGo search results
            mock_ddgs.return_value.__enter__.return_value.text.return_value = [
                {
                    "title": "Python Programming Guide",
                    "href": "https://example.com/python-guide",
                    "body": "Comprehensive Python programming tutorial"
                }
            ]
            
            results = await web_searcher._search_duckduckgo("python programming", 1)
            
            assert len(results) > 0
            assert results[0]["title"] == "Python Programming Guide"
            assert results[0]["source"] == "duckduckgo"
    
    @pytest.mark.asyncio
    async def test_stackoverflow_search(self, web_searcher):
        """Test Stack Overflow search functionality"""
        
        # Mock aiohttp session
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "items": [
                {
                    "title": "How to use async in Python?",
                    "link": "https://stackoverflow.com/questions/123",
                    "body": "Async programming in Python using asyncio",
                    "score": 42,
                    "tags": ["python", "asyncio"],
                    "is_answered": True,
                    "answer_count": 3
                }
            ]
        }
        
        web_searcher.session = AsyncMock()
        web_searcher.session.get.return_value.__aenter__.return_value = mock_response
        
        results = await web_searcher._search_stackoverflow("python async", 1)
        
        assert len(results) > 0
        assert results[0]["title"] == "How to use async in Python?"
        assert results[0]["source"] == "stackoverflow"
        assert results[0]["is_answered"] is True
    
    @pytest.mark.asyncio
    async def test_github_search(self, web_searcher):
        """Test GitHub search functionality"""
        
        # Mock GitHub API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = {
            "items": [
                {
                    "full_name": "example/python-examples",
                    "html_url": "https://github.com/example/python-examples",
                    "description": "Python code examples and tutorials",
                    "stargazers_count": 1500,
                    "forks_count": 300,
                    "language": "Python"
                }
            ]
        }
        
        web_searcher.session = AsyncMock()
        web_searcher.session.get.return_value.__aenter__.return_value = mock_response
        
        results = await web_searcher._search_github("python examples", 1)
        
        assert len(results) > 0
        assert results[0]["title"] == "example/python-examples"
        assert results[0]["source"] == "github"
        assert results[0]["has_code"] is True
    
    def test_rank_and_deduplicate(self, web_searcher):
        """Test result ranking and deduplication"""
        
        test_results = [
            {
                "title": "Python Guide",
                "url": "https://example.com/guide",
                "snippet": "Python programming tutorial",
                "relevance_score": 0.5
            },
            {
                "title": "Python Guide",  # Duplicate
                "url": "https://example.com/guide",
                "snippet": "Python programming tutorial",
                "relevance_score": 0.5
            },
            {
                "title": "Advanced Python",
                "url": "https://example.com/advanced",
                "snippet": "Advanced Python concepts",
                "relevance_score": 0.7
            }
        ]
        
        ranked_results = web_searcher._rank_and_deduplicate(test_results, "python programming")
        
        # Should remove duplicates
        assert len(ranked_results) == 2
        
        # Should be ranked by relevance
        assert ranked_results[0]["relevance_score"] >= ranked_results[1]["relevance_score"]
    
    def test_clean_html(self, web_searcher):
        """Test HTML cleaning functionality"""
        
        html_content = "<p>This is <b>HTML</b> content with <a href='#'>links</a></p>"
        
        cleaned = web_searcher._clean_html(html_content)
        
        assert "<p>" not in cleaned
        assert "<b>" not in cleaned
        assert "This is HTML content with links" in cleaned

class TestCodeExecutor:
    """Test code execution utilities"""
    
    @pytest.fixture
    def code_executor(self):
        """Create a code executor for testing"""
        from api.utils.code_executor import CodeExecutor
        return CodeExecutor()
    
    def test_syntax_validation_valid_code(self, code_executor):
        """Test syntax validation with valid code"""
        
        valid_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
        
        is_valid = code_executor._validate_python_syntax(valid_code)
        assert is_valid is True
    
    def test_syntax_validation_invalid_code(self, code_executor):
        """Test syntax validation with invalid code"""
        
        invalid_code = """
def fibonacci(n)
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2
"""
        
        is_valid = code_executor._validate_python_syntax(invalid_code)
        assert is_valid is False
    
    def test_security_analysis_safe_code(self, code_executor):
        """Test security analysis with safe code"""
        
        safe_code = """
def add_numbers(a, b):
    return a + b

result = add_numbers(5, 3)
print(result)
"""
        
        security_analysis = code_executor._analyze_security(safe_code)
        
        assert security_analysis["has_security_issues"] is False
        assert security_analysis["risk_level"] == "low"
    
    def test_security_analysis_dangerous_code(self, code_executor):
        """Test security analysis with dangerous code"""
        
        dangerous_code = """
import os
import subprocess

os.system("rm -rf /")
subprocess.call(["ls", "/etc/passwd"])
"""
        
        security_analysis = code_executor._analyze_security(dangerous_code)
        
        assert security_analysis["has_security_issues"] is True
        assert len(security_analysis["issues"]) > 0
        assert security_analysis["risk_level"] in ["medium", "high"]
    
    def test_complexity_analysis(self, code_executor):
        """Test code complexity analysis"""
        
        test_code = """
def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr
"""
        
        complexity = code_executor._analyze_complexity(test_code)
        
        assert "lines_of_code" in complexity
        assert "cyclomatic_complexity" in complexity
        assert "function_count" in complexity
        assert "loop_count" in complexity
        assert "complexity_score" in complexity
        
        assert complexity["function_count"] == 1
        assert complexity["loop_count"] == 2  # Two nested loops
    
    def test_time_complexity_estimation(self, code_executor):
        """Test time complexity estimation"""
        
        # Linear complexity
        linear_code = "for i in range(n): print(i)"
        linear_complexity = code_executor._estimate_time_complexity(linear_code)
        assert linear_complexity == "O(n)"
        
        # Quadratic complexity
        quadratic_code = "for i in range(n):\n    for j in range(n): print(i, j)"
        quadratic_complexity = code_executor._estimate_time_complexity(quadratic_code)
        assert quadratic_complexity == "O(n²)"
        
        # Constant complexity
        constant_code = "x = 5\ny = 10\nprint(x + y)"
        constant_complexity = code_executor._estimate_time_complexity(constant_code)
        assert constant_complexity == "O(1)"
    
    def test_space_complexity_estimation(self, code_executor):
        """Test space complexity estimation"""
        
        # Linear space complexity
        linear_space_code = "arr = [i for i in range(n)]"
        linear_space = code_executor._estimate_space_complexity(linear_space_code)
        assert linear_space == "O(n)"
        
        # Constant space complexity
        constant_space_code = "x = 5\ny = 10"
        constant_space = code_executor._estimate_space_complexity(constant_space_code)
        assert constant_space == "O(1)"
    
    @pytest.mark.asyncio
    async def test_code_execution_success(self, code_executor):
        """Test successful code execution"""
        
        simple_code = """
def add(a, b):
    return a + b

result = add(2, 3)
print(f"Result: {result}")
"""
        
        execution_result = await code_executor._execute_python_code(simple_code)
        
        assert execution_result["success"] is True
        assert execution_result["return_code"] == 0
        assert "Result: 5" in execution_result["output"]
        assert execution_result["execution_time"] > 0
    
    @pytest.mark.asyncio
    async def test_code_execution_error(self, code_executor):
        """Test code execution with error"""
        
        error_code = """
def divide_by_zero():
    return 10 / 0

result = divide_by_zero()
"""
        
        execution_result = await code_executor._execute_python_code(error_code)
        
        assert execution_result["success"] is False
        assert "ZeroDivisionError" in execution_result["error"]
    
    def test_test_case_generation(self, code_executor):
        """Test automatic test case generation"""
        
        fibonacci_func = MagicMock()
        fibonacci_func.name = "fibonacci"
        fibonacci_func.args.args = [MagicMock()]  # One argument
        
        test_cases = code_executor._generate_test_cases_for_function(fibonacci_func)
        
        assert len(test_cases) > 0
        # Should include typical Fibonacci test cases
        inputs = [case["input"] for case in test_cases]
        assert "0" in inputs
        assert "1" in inputs
        assert "5" in inputs

class TestMathEngine:
    """Test mathematical reasoning engine"""
    
    @pytest.fixture
    def math_engine(self):
        """Create a math engine for testing"""
        
        # Mock scipy and sympy
        sys.modules['scipy.stats'] = MagicMock()
        sys.modules['scipy.optimize'] = MagicMock()
        sys.modules['scipy.integrate'] = MagicMock()
        sys.modules['sympy'] = MagicMock()
        
        from api.utils.math_engine import MathEngine
        return MathEngine()
    
    def test_math_engine_initialization(self, math_engine):
        """Test math engine initialization"""
        
        assert hasattr(math_engine, 'constants')
        assert 'pi' in math_engine.constants
        assert 'e' in math_engine.constants
        assert math_engine.constants['pi'] == 3.141592653589793
    
    @pytest.mark.asyncio
    async def test_algorithm_analysis(self, math_engine):
        """Test algorithm analysis functionality"""
        
        test_code = """
def binary_search(arr, target):
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
"""
        
        analysis = await math_engine.analyze_algorithm(test_code)
        
        assert "time_complexity" in analysis
        assert "space_complexity" in analysis
        assert "mathematical_operations" in analysis
        assert "optimization_opportunities" in analysis
        assert "complexity_score" in analysis
    
    def test_time_complexity_analysis(self, math_engine):
        """Test time complexity analysis"""
        
        # Test different complexity patterns
        linear_code = "for i in range(n): pass"
        linear_analysis = math_engine._analyze_time_complexity(linear_code)
        assert "O(n)" in linear_analysis["estimated_complexity"]
        
        nested_code = "for i in range(n):\n    for j in range(n): pass"
        nested_analysis = math_engine._analyze_time_complexity(nested_code)
        assert "O(n²)" in nested_analysis["estimated_complexity"]
    
    def test_space_complexity_analysis(self, math_engine):
        """Test space complexity analysis"""
        
        linear_space_code = "[i for i in range(n)]"
        space_analysis = math_engine._analyze_space_complexity(linear_space_code)
        assert "O(n)" in space_analysis["estimated_complexity"]
    
    def test_math_operations_identification(self, math_engine):
        """Test identification of mathematical operations"""
        
        code_with_ops = """
import math
result = math.sqrt(x) + y**2 - z/2
factorial = math.factorial(n)
"""
        
        operations = math_engine._identify_math_operations(code_with_ops)
        
        assert len(operations) > 0
        operation_names = [op["operation"] for op in operations]
        assert "addition" in operation_names
        assert "exponentiation" in operation_names
        assert "division" in operation_names
    
    def test_statistical_analysis(self, math_engine):
        """Test statistical data analysis"""
        
        test_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        with patch('numpy.array'), patch('numpy.mean'), patch('numpy.median'), \
             patch('numpy.std'), patch('numpy.var'):
            
            # Mock numpy functions
            import numpy as np
            np.array.return_value = test_data
            np.mean.return_value = 5.5
            np.median.return_value = 5.5
            np.std.return_value = 2.87
            np.var.return_value = 8.25
            
            analysis = math_engine.analyze_statistical_data(test_data)
            
            assert "descriptive_stats" in analysis
            assert "distribution_tests" in analysis
            assert "confidence_intervals" in analysis
    
    def test_derivative_calculation(self, math_engine):
        """Test symbolic derivative calculation"""
        
        with patch('sympy.Symbol'), patch('sympy.sympify'), patch('sympy.diff'):
            import sympy as sp
            
            # Mock sympy operations
            sp.Symbol.return_value = "x"
            sp.sympify.return_value = "x**2 + 3*x + 2"
            sp.diff.return_value = "2*x + 3"
            
            derivative = math_engine.calculate_derivative("x**2 + 3*x + 2", "x")
            
            assert isinstance(derivative, str)
            sp.diff.assert_called_once()
    
    def test_integral_calculation(self, math_engine):
        """Test symbolic integration"""
        
        with patch('sympy.Symbol'), patch('sympy.sympify'), patch('sympy.integrate'):
            import sympy as sp
            
            # Mock sympy operations
            sp.Symbol.return_value = "x"
            sp.sympify.return_value = "x**2"
            sp.integrate.return_value = "x**3/3"
            
            integral = math_engine.calculate_integral("x**2", "x")
            
            assert isinstance(integral, str)
            sp.integrate.assert_called_once()

class TestMemoryManager:
    """Test memory management utilities"""
    
    @pytest.fixture
    def memory_manager(self):
        """Create a memory manager for testing"""
        
        # Mock dependencies
        sys.modules['faiss'] = MagicMock()
        sys.modules['sentence_transformers'] = MagicMock()
        
        from api.utils.memory_manager import MemoryManager
        manager = MemoryManager()
        
        # Mock the embedding model
        manager.embedding_model = MagicMock()
        manager.embedding_model.encode.return_value = [np.random.rand(384)]
        
        # Mock FAISS indices
        manager.episodic_index = MagicMock()
        manager.semantic_index = MagicMock()
        manager.code_index = MagicMock()
        
        return manager
    
    @pytest.mark.asyncio
    async def test_memory_manager_initialization(self, memory_manager):
        """Test memory manager initialization"""
        
        await memory_manager.initialize()
        
        assert memory_manager.embedding_model is not None
        assert hasattr(memory_manager, 'episodic_memories')
        assert hasattr(memory_manager, 'semantic_memories')
        assert hasattr(memory_manager, 'code_memories')
    
    @pytest.mark.asyncio
    async def test_store_interaction(self, memory_manager):
        """Test storing interactions in memory"""
        
        interaction_data = {
            "prompt": "Write a sorting function",
            "code": "def sort(arr): return sorted(arr)",
            "success": True
        }
        
        await memory_manager.store_interaction("code_generation", interaction_data)
        
        # Should add to code memories
        assert len(memory_manager.code_memories) > 0
        
        # Should have proper structure
        stored_memory = memory_manager.code_memories[-1]
        assert "id" in stored_memory
        assert "type" in stored_memory
        assert "timestamp" in stored_memory
        assert "data" in stored_memory
        assert "importance_score" in stored_memory
    
    @pytest.mark.asyncio
    async def test_store_knowledge(self, memory_manager):
        """Test storing general knowledge"""
        
        knowledge = "Dynamic programming is an optimization technique that solves problems by breaking them into subproblems."
        
        await memory_manager.store_knowledge(knowledge, "algorithms")
        
        # Should add to semantic memories
        assert len(memory_manager.semantic_memories) > 0
        
        stored_knowledge = memory_manager.semantic_memories[-1]
        assert stored_knowledge["category"] == "algorithms"
        assert stored_knowledge["content"] == knowledge
    
    @pytest.mark.asyncio
    async def test_retrieve_similar_memories(self, memory_manager):
        """Test retrieving similar memories"""
        
        # Add some test memories
        await memory_manager.store_interaction("code_generation", {
            "prompt": "sorting algorithm",
            "code": "def sort(arr): return sorted(arr)"
        })
        
        # Mock search results
        memory_manager._search_memory_index = AsyncMock(return_value=[
            {
                "memory": memory_manager.code_memories[0],
                "similarity_score": 0.9,
                "relevance": "high"
            }
        ])
        
        similar_memories = await memory_manager.retrieve_similar_memories(
            "sorting algorithm", "code", 5
        )
        
        assert len(similar_memories) > 0
        assert similar_memories[0]["similarity_score"] == 0.9
        assert similar_memories[0]["relevance"] == "high"
    
    def test_generate_memory_id(self, memory_manager):
        """Test memory ID generation"""
        
        test_data = {"key": "value", "number": 42}
        
        id1 = memory_manager._generate_memory_id(test_data)
        id2 = memory_manager._generate_memory_id(test_data)
        
        # Same data should generate same ID
        assert id1 == id2
        
        # Different data should generate different ID
        different_data = {"key": "different", "number": 43}
        id3 = memory_manager._generate_memory_id(different_data)
        assert id1 != id3
    
    def test_calculate_importance_score(self, memory_manager):
        """Test importance score calculation"""
        
        # High-quality interaction
        good_data = {
            "success": True,
            "confidence": 0.9,
            "complexity_analysis": {"complexity_score": 20}
        }
        
        good_score = memory_manager._calculate_importance_score("code_generation", good_data)
        
        # Low-quality interaction
        bad_data = {
            "success": False,
            "confidence": 0.3,
            "complexity_analysis": {"complexity_score": 80}
        }
        
        bad_score = memory_manager._calculate_importance_score("code_generation", bad_data)
        
        # Good interaction should have higher importance
        assert good_score > bad_score
    
    def test_extract_text_for_embedding(self, memory_manager):
        """Test text extraction for embedding generation"""
        
        interaction_data = {
            "prompt": "Write a function",
            "code": "def hello(): print('hello')",
            "explanation": "Simple greeting function",
            "other_data": {"nested": "value"}
        }
        
        text = memory_manager._extract_text_for_embedding(interaction_data)
        
        assert "Write a function" in text
        assert "def hello" in text
        assert "Simple greeting function" in text
    
    @pytest.mark.asyncio
    async def test_memory_stats(self, memory_manager):
        """Test memory statistics"""
        
        # Add some test data
        await memory_manager.store_interaction("code_generation", {"test": "data"})
        await memory_manager.store_knowledge("test knowledge", "test")
        
        stats = await memory_manager.get_memory_stats()
        
        assert "total_memories" in stats
        assert "memory_usage" in stats
        assert "metadata" in stats
        assert "recent_activity" in stats
        
        assert stats["total_memories"]["code"] > 0
        assert stats["total_memories"]["semantic"] > 0

class TestTrainingManager:
    """Test training management utilities"""
    
    @pytest.fixture
    def training_manager(self):
        """Create a training manager for testing"""
        
        # Mock heavy dependencies
        sys.modules['transformers'] = MagicMock()
        sys.modules['peft'] = MagicMock()
        sys.modules['datasets'] = MagicMock()
        
        from api.utils.training_manager import TrainingManager
        manager = TrainingManager()
        
        # Mock tokenizer
        manager.tokenizer = MagicMock()
        
        return manager
    
    @pytest.mark.asyncio
    async def test_training_manager_initialization(self, training_manager):
        """Test training manager initialization"""
        
        await training_manager.initialize()
        
        assert training_manager.tokenizer is not None
        assert hasattr(training_manager, 'training_history')
        assert hasattr(training_manager, 'current_model_version')
    
    @pytest.mark.asyncio
    async def test_prepare_interaction_data(self, training_manager):
        """Test preparing interaction data for training"""
        
        interactions = [
            {
                "type": "code_generation",
                "request": {"prompt": "Write a function"},
                "response": {
                    "code": "def test(): pass",
                    "explanation": "Simple test function"
                }
            },
            {
                "type": "reasoning",
                "request": {"problem": "Solve this problem"},
                "response": {"solution": "Use dynamic programming"}
            }
        ]
        
        with patch('datasets.Dataset.from_list') as mock_dataset:
            mock_dataset.return_value = MagicMock()
            
            dataset = await training_manager.prepare_interaction_data(interactions)
            
            # Should create dataset from interactions
            mock_dataset.assert_called()
            
            # Check that interactions were processed
            call_args = mock_dataset.call_args[0][0]
            assert len(call_args) > 0
            assert any("Write a function" in str(item) for item in call_args)
    
    @pytest.mark.asyncio
    async def test_create_synthetic_dataset(self, training_manager):
        """Test synthetic dataset creation"""
        
        with patch('datasets.Dataset.from_list') as mock_dataset:
            mock_dataset.return_value = MagicMock()
            
            dataset = await training_manager._create_synthetic_dataset()
            
            # Should create dataset with synthetic examples
            mock_dataset.assert_called()
            call_args = mock_dataset.call_args[0][0]
            assert len(call_args) > 0
            
            # Should contain common algorithms
            examples_text = str(call_args)
            assert "fibonacci" in examples_text.lower()
            assert "factorial" in examples_text.lower()
    
    def test_get_training_history(self, training_manager):
        """Test getting training history"""
        
        history = training_manager.get_training_history()
        
        assert isinstance(history, list)
    
    def test_get_current_version(self, training_manager):
        """Test getting current model version"""
        
        version = training_manager.get_current_version()
        
        assert isinstance(version, int)
        assert version >= 0
    
    def test_get_available_models(self, training_manager):
        """Test getting available models"""
        
        models = training_manager.get_available_models()
        
        assert isinstance(models, list)

# Integration tests
class TestUtilsIntegration:
    """Integration tests for utility modules"""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_with_auth(self):
        """Test rate limiter integration with authentication"""
        
        # Test that different user roles get different rate limits
        limiter = AdvancedRateLimiter()
        
        # Verify admin token
        admin_user = verify_token("autonomous-ai-agent-2024")
        assert admin_user["role"] == "admin"
        
        # Should allow admin requests
        result = await limiter.check_rate_limit("admin_user", "admin")
        assert result is True
    
    def test_logger_with_performance_data(self):
        """Test logger integration with performance monitoring"""
        
        logger = AgentLogger("integration_test")
        
        # Log various types of data
        logger.log_request("/generate", "POST", "user123", 1.5)
        logger.log_performance("code_generation", 2.0, lines_generated=50)
        logger.log_model_operation("inference", "distilgpt2", tokens=100)
        
        # Should not raise exceptions
        assert True

# Performance tests
class TestUtilsPerformance:
    """Performance tests for utility modules"""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_performance(self):
        """Test rate limiter performance under load"""
        
        limiter = RateLimiter(requests_per_minute=1000, burst_size=100)
        
        start_time = time.time()
        
        # Simulate many rapid requests
        tasks = []
        for i in range(50):
            task = limiter.check_rate_limit(f"user_{i}")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        
        # Should handle many requests quickly
        assert end_time - start_time < 1.0
        assert all(result is True for result in results)
    
    def test_memory_id_generation_performance(self):
        """Test memory ID generation performance"""
        
        from api.utils.memory_manager import MemoryManager
        manager = MemoryManager()
        
        start_time = time.time()
        
        # Generate many IDs
        for i in range(1000):
            data = {"iteration": i, "data": f"test_data_{i}"}
            manager._generate_memory_id(data)
        
        end_time = time.time()
        
        # Should generate IDs quickly
        assert end_time - start_time < 1.0

# Test configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
