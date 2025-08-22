"""
Pytest configuration and shared fixtures
"""

import pytest
import asyncio
import sys
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

# Add the api directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "api"))

# Mock heavy dependencies globally to speed up tests
@pytest.fixture(scope="session", autouse=True)
def mock_heavy_dependencies():
    """Mock heavy ML/AI dependencies for faster test execution"""
    
    # Store original modules
    original_modules = {}
    
    # List of modules to mock
    modules_to_mock = [
        'torch',
        'transformers',
        'langchain',
        'langchain.llms',
        'langchain.chains', 
        'langchain.memory',
        'langchain.prompts',
        'langchain.schema',
        'faiss',
        'sentence_transformers',
        'peft',
        'datasets',
        'scipy',
        'scipy.stats',
        'scipy.optimize',
        'scipy.integrate',
        'sympy',
        'beautifulsoup4',
        'aiohttp',
        'duckduckgo_search',
        'nltk',
        'sklearn',
        'pandas',
        'networkx'
    ]
    
    # Mock each module
    for module_name in modules_to_mock:
        if module_name in sys.modules:
            original_modules[module_name] = sys.modules[module_name]
        
        # Create mock module
        mock_module = MagicMock()
        sys.modules[module_name] = mock_module
        
        # Special handling for specific modules
        if module_name == 'torch':
            mock_module.float32 = "float32"
            mock_module.tensor = MagicMock(return_value=MagicMock())
            mock_module.no_grad = MagicMock()
            mock_module.cuda.is_available.return_value = False
        
        elif module_name == 'transformers':
            mock_module.AutoTokenizer.from_pretrained.return_value = MagicMock()
            mock_module.AutoModelForCausalLM.from_pretrained.return_value = MagicMock()
            mock_module.pipeline.return_value = MagicMock()
            mock_module.TrainingArguments = MagicMock
            mock_module.Trainer = MagicMock
            mock_module.DataCollatorForLanguageModeling = MagicMock
        
        elif module_name == 'langchain':
            mock_module.llms.HuggingFacePipeline = MagicMock
            mock_module.chains.ConversationChain = MagicMock
            mock_module.memory.ConversationBufferWindowMemory = MagicMock
            mock_module.prompts.PromptTemplate = MagicMock
            mock_module.schema.BaseOutputParser = MagicMock
        
        elif module_name == 'faiss':
            mock_module.IndexFlatIP.return_value = MagicMock()
            mock_module.read_index.return_value = MagicMock()
            mock_module.write_index = MagicMock()
        
        elif module_name == 'sentence_transformers':
            mock_transformer = MagicMock()
            mock_transformer.encode.return_value = [[0.1] * 384]  # Mock embedding
            mock_module.SentenceTransformer.return_value = mock_transformer
        
        elif module_name == 'datasets':
            mock_dataset = MagicMock()
            mock_dataset.map.return_value = mock_dataset
            mock_dataset.filter.return_value = mock_dataset
            mock_dataset.select.return_value = mock_dataset
            mock_dataset.train_test_split.return_value = {
                "train": mock_dataset,
                "test": mock_dataset
            }
            mock_module.Dataset.from_list.return_value = mock_dataset
            mock_module.Dataset.from_dict.return_value = mock_dataset
            mock_module.load_dataset.return_value = mock_dataset
    
    yield
    
    # Restore original modules (cleanup)
    for module_name, original_module in original_modules.items():
        sys.modules[module_name] = original_module

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture
def mock_agent():
    """Create a mock autonomous agent for testing"""
    
    agent = AsyncMock()
    
    # Mock common methods
    agent.generate_code.return_value = {
        "code": "def test_function():\n    return 'Hello, World!'",
        "explanation": "A simple test function that returns a greeting",
        "test_cases": [{"input": "", "expected": "Hello, World!"}],
        "complexity_analysis": {
            "time_complexity": "O(1)",
            "space_complexity": "O(1)"
        },
        "optimizations": [],
        "reasoning_steps": "Generated a simple function as requested",
        "confidence": 0.95,
        "test_results": {"success": True, "execution_time": 0.001}
    }
    
    agent.deep_search.return_value = {
        "results": [
            {
                "title": "Test Search Result",
                "url": "https://example.com/test",
                "snippet": "Test search result snippet",
                "source": "test_source",
                "relevance_score": 0.9
            }
        ],
        "synthesized_answer": "This is a synthesized answer from test search",
        "code_examples": [
            {
                "code": "print('Hello from search')",
                "description": "Example code from search"
            }
        ],
        "sources": ["example.com"],
        "confidence": 0.85,
        "related_queries": ["related query 1", "related query 2"],
        "search_plan": "Executed comprehensive search strategy"
    }
    
    agent.reason_step_by_step.return_value = {
        "reasoning_steps": [
            {
                "step": 1,
                "description": "Problem Analysis",
                "content": "Analyzing the given problem systematically"
            },
            {
                "step": 2,
                "description": "Solution Development",
                "content": "Developing a comprehensive solution approach"
            }
        ],
        "solution": "The optimal solution is to use a well-established algorithm",
        "mathematical_analysis": {
            "time_complexity": "O(n log n)",
            "space_complexity": "O(n)",
            "proof": "Mathematical proof of complexity"
        },
        "alternative_approaches": [
            {
                "name": "Alternative Method",
                "complexity": "O(n²)",
                "description": "Alternative but less efficient approach"
            }
        ],
        "confidence": 0.92,
        "complexity_analysis": {
            "time_complexity": "O(n log n)",
            "space_complexity": "O(n)",
            "optimality": "Near optimal for this problem class"
        }
    }
    
    agent.get_status.return_value = {
        "model_loaded": True,
        "components_initialized": True,
        "performance_metrics": {
            "code_generation_success_rate": 0.94,
            "search_relevance_score": 0.87,
            "reasoning_accuracy": 0.91
        },
        "interaction_count": 150
    }
    
    agent.get_memory_usage.return_value = {
        "total_memories": {
            "episodic": 100,
            "semantic": 50,
            "code": 75
        },
        "memory_utilization": 0.65,
        "recent_activity": {
            "last_24h": 25,
            "last_7d": 150
        }
    }
    
    agent.get_model_info.return_value = {
        "model_name": "test-model",
        "current_version": 1,
        "quantization": "fp32",
        "parameter_count": "82M"
    }
    
    agent.get_training_history.return_value = [
        {
            "version": 1,
            "timestamp": "2024-01-01T00:00:00Z",
            "training_type": "rlhf",
            "performance_improvement": 0.05
        }
    ]
    
    agent.get_capabilities.return_value = [
        "Code generation",
        "Deep web search",
        "Mathematical reasoning",
        "Self-training",
        "Step-by-step reasoning"
    ]
    
    agent.process_feedback = AsyncMock()
    agent.learn_from_interaction = AsyncMock()
    
    return agent

@pytest.fixture
def mock_web_searcher():
    """Create a mock web searcher"""
    
    searcher = AsyncMock()
    
    searcher.search_multiple_sources.return_value = [
        {
            "title": "Python Programming Guide",
            "url": "https://example.com/python-guide",
            "snippet": "Comprehensive guide to Python programming",
            "source": "documentation",
            "relevance_score": 0.9,
            "has_code": True
        },
        {
            "title": "Python Examples Repository",
            "url": "https://github.com/example/python-examples",
            "snippet": "Collection of Python code examples",
            "source": "github",
            "relevance_score": 0.85,
            "has_code": True
        }
    ]
    
    return searcher

@pytest.fixture
def mock_code_executor():
    """Create a mock code executor"""
    
    executor = AsyncMock()
    
    executor.test_code.return_value = {
        "success": True,
        "syntax_valid": True,
        "execution_result": {
            "success": True,
            "output": "Test output",
            "error": "",
            "execution_time": 0.001
        },
        "security_analysis": {
            "has_security_issues": False,
            "risk_level": "low"
        },
        "complexity_analysis": {
            "lines_of_code": 10,
            "cyclomatic_complexity": 2,
            "complexity_level": "low"
        },
        "test_cases_passed": 3,
        "test_cases_total": 3,
        "performance_metrics": {
            "time_complexity": "O(n)",
            "space_complexity": "O(1)"
        },
        "suggestions": ["Consider adding error handling"]
    }
    
    return executor

@pytest.fixture
def mock_math_engine():
    """Create a mock math engine"""
    
    engine = AsyncMock()
    
    engine.analyze_algorithm.return_value = {
        "time_complexity": {
            "estimated_complexity": "O(n)",
            "justification": "Single loop iteration",
            "mathematical_proof": "Linear iteration over input"
        },
        "space_complexity": {
            "estimated_complexity": "O(1)",
            "justification": "Constant space usage"
        },
        "mathematical_operations": [
            {
                "operation": "addition",
                "complexity": "O(1)"
            }
        ],
        "optimization_opportunities": [
            {
                "type": "algorithmic",
                "suggestion": "Consider using more efficient data structures"
            }
        ],
        "complexity_score": 25.0
    }
    
    engine.analyze_problem.return_value = {
        "problem_type": "algorithmic",
        "mathematical_concepts": ["iteration", "comparison"],
        "complexity_considerations": {
            "time": "O(n)",
            "space": "O(1)"
        }
    }
    
    return engine

@pytest.fixture
def mock_memory_manager():
    """Create a mock memory manager"""
    
    manager = AsyncMock()
    
    manager.store_interaction = AsyncMock()
    manager.store_knowledge = AsyncMock()
    manager.store_feedback = AsyncMock()
    
    manager.retrieve_similar_memories.return_value = [
        {
            "memory": {
                "id": "mem_123",
                "type": "code_generation",
                "data": {"prompt": "test prompt", "code": "test code"},
                "importance_score": 0.8
            },
            "similarity_score": 0.9,
            "relevance": "high"
        }
    ]
    
    manager.get_memory_stats.return_value = {
        "total_memories": {
            "episodic": 100,
            "semantic": 50,
            "code": 75
        },
        "memory_usage": {
            "current_size": 225,
            "max_size": 1000,
            "utilization": 0.225
        },
        "recent_activity": {
            "last_24h": 15,
            "last_7d": 80
        }
    }
    
    return manager

@pytest.fixture
def mock_training_manager():
    """Create a mock training manager"""
    
    manager = AsyncMock()
    
    manager.load_dataset.return_value = {
        "train": MagicMock(),
        "eval": MagicMock()
    }
    
    manager.prepare_interaction_data.return_value = {
        "train": MagicMock(),
        "eval": MagicMock()
    }
    
    manager.fine_tune_with_lora.return_value = {
        "train_result": {
            "training_loss": 1.5,
            "training_runtime": 300,
            "train_samples_per_second": 10.0
        },
        "eval_result": {
            "eval_loss": 1.2,
            "eval_runtime": 60
        },
        "model_path": "/path/to/model"
    }
    
    manager.get_training_history.return_value = [
        {
            "version": 1,
            "timestamp": "2024-01-01T00:00:00Z",
            "training_type": "rlhf",
            "performance_improvement": 0.08
        }
    ]
    
    manager.get_current_version.return_value = 1
    manager.get_available_models.return_value = ["model_v1", "model_v2"]
    
    return manager

@pytest.fixture
def sample_interaction_data():
    """Sample interaction data for testing"""
    
    return {
        "type": "code_generation",
        "timestamp": "2024-01-01T00:00:00Z",
        "request": {
            "prompt": "Write a function to calculate factorial",
            "language": "python",
            "context": "Use recursion"
        },
        "response": {
            "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)",
            "explanation": "Recursive factorial implementation",
            "confidence": 0.9,
            "test_results": {"success": True}
        }
    }

@pytest.fixture
def sample_search_results():
    """Sample search results for testing"""
    
    return [
        {
            "title": "Python Factorial Implementation",
            "url": "https://example.com/factorial",
            "snippet": "Learn how to implement factorial in Python",
            "source": "documentation",
            "relevance_score": 0.95,
            "has_code": True
        },
        {
            "title": "Recursive Algorithms in Python",
            "url": "https://github.com/example/recursive-algorithms",
            "snippet": "Collection of recursive algorithm implementations",
            "source": "github",
            "relevance_score": 0.88,
            "has_code": True
        }
    ]

@pytest.fixture
def test_code_samples():
    """Sample code snippets for testing"""
    
    return {
        "valid_python": """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
""",
        "invalid_python": """
def fibonacci(n)
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2
""",
        "dangerous_code": """
import os
import subprocess

os.system("rm -rf /")
subprocess.call(["cat", "/etc/passwd"])
""",
        "simple_function": """
def add_numbers(a, b):
    return a + b
""",
        "complex_algorithm": """
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    
    return quicksort(left) + middle + quicksort(right)
"""
    }

@pytest.fixture
def test_mathematical_data():
    """Sample mathematical data for testing"""
    
    return {
        "simple_dataset": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "normal_distribution": [
            1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
            3.2, 2.8, 3.1, 2.9, 3.3, 2.7, 3.4, 2.6, 3.5
        ],
        "skewed_distribution": [1, 1, 1, 2, 2, 3, 4, 5, 10, 15, 20],
        "outlier_dataset": [1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
    }

@pytest.fixture
def auth_headers():
    """Authentication headers for testing"""
    
    return {
        "Authorization": "Bearer autonomous-ai-agent-2024",
        "Content-Type": "application/json"
    }

@pytest.fixture
def invalid_auth_headers():
    """Invalid authentication headers for testing"""
    
    return {
        "Authorization": "Bearer invalid-token",
        "Content-Type": "application/json"
    }

# Utility functions for tests
def assert_valid_response_structure(response_data, required_fields):
    """Assert that response data has required structure"""
    
    assert isinstance(response_data, dict)
    
    for field in required_fields:
        assert field in response_data, f"Missing required field: {field}"
    
    # Common fields that should always be present
    if "timestamp" in response_data:
        assert isinstance(response_data["timestamp"], str)
    
    if "confidence" in response_data:
        assert isinstance(response_data["confidence"], (int, float))
        assert 0 <= response_data["confidence"] <= 1

def assert_valid_code_response(response_data):
    """Assert that code generation response is valid"""
    
    required_fields = [
        "code", "explanation", "test_cases", "complexity_analysis",
        "optimizations", "reasoning_steps", "confidence", "test_results"
    ]
    
    assert_valid_response_structure(response_data, required_fields)
    
    # Code-specific assertions
    assert isinstance(response_data["code"], str)
    assert len(response_data["code"]) > 0
    assert isinstance(response_data["test_cases"], list)
    assert isinstance(response_data["complexity_analysis"], dict)
    assert isinstance(response_data["optimizations"], list)

def assert_valid_search_response(response_data):
    """Assert that search response is valid"""
    
    required_fields = [
        "results", "synthesized_answer", "code_examples",
        "sources", "confidence", "related_queries"
    ]
    
    assert_valid_response_structure(response_data, required_fields)
    
    # Search-specific assertions
    assert isinstance(response_data["results"], list)
    assert isinstance(response_data["sources"], list)
    assert isinstance(response_data["related_queries"], list)

def assert_valid_reasoning_response(response_data):
    """Assert that reasoning response is valid"""
    
    required_fields = [
        "reasoning_steps", "solution", "mathematical_analysis",
        "alternative_approaches", "confidence", "complexity_analysis"
    ]
    
    assert_valid_response_structure(response_data, required_fields)
    
    # Reasoning-specific assertions
    assert isinstance(response_data["reasoning_steps"], list)
    assert len(response_data["reasoning_steps"]) > 0
    
    for step in response_data["reasoning_steps"]:
        assert "step" in step
        assert "description" in step
        assert "content" in step

# Async test utilities
async def run_async_test(coro):
    """Helper to run async tests"""
    return await coro

# Markers for different test categories
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.performance = pytest.mark.performance
pytest.mark.slow = pytest.mark.slow
