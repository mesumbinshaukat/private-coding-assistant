"""
Test suite for the Autonomous Agent Core
Tests the ReAct framework, reasoning, and learning capabilities
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import numpy as np
import json

# Mock heavy dependencies
import sys
sys.modules['torch'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['langchain'] = MagicMock()
sys.modules['faiss'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['peft'] = MagicMock()
sys.modules['datasets'] = MagicMock()

from api.agent_core import AutonomousAgent, ReActOutputParser

class TestReActOutputParser:
    """Test the ReAct framework output parser"""
    
    def test_parse_complete_react_output(self):
        """Test parsing complete ReAct output"""
        
        parser = ReActOutputParser()
        
        test_output = """
Thought: I need to analyze this coding problem step by step
Action: Generate a solution using dynamic programming
Observation: The solution works correctly with O(n) complexity
"""
        
        result = parser.parse(test_output)
        
        assert "thought" in result
        assert "action" in result
        assert "observation" in result
        
        assert "analyze this coding problem" in result["thought"]
        assert "dynamic programming" in result["action"]
        assert "O(n) complexity" in result["observation"]
    
    def test_parse_incomplete_react_output(self):
        """Test parsing incomplete ReAct output"""
        
        parser = ReActOutputParser()
        
        test_output = """
Thought: I need to solve this problem
Action: Use recursion
"""
        
        result = parser.parse(test_output)
        
        assert "thought" in result
        assert "action" in result
        assert result.get("observation", "") == ""
    
    def test_parse_empty_output(self):
        """Test parsing empty output"""
        
        parser = ReActOutputParser()
        result = parser.parse("")
        
        # Should return empty dict or handle gracefully
        assert isinstance(result, dict)

class TestAutonomousAgent:
    """Test the main Autonomous Agent class"""
    
    @pytest.fixture
    async def agent(self):
        """Create a test agent instance"""
        
        with patch.multiple(
            'api.agent_core',
            AutoTokenizer=MagicMock(),
            AutoModelForCausalLM=MagicMock(),
            HuggingFacePipeline=MagicMock(),
            ConversationChain=MagicMock()
        ):
            agent = AutonomousAgent(model_name="test-model")
            
            # Mock the components
            agent.web_searcher = AsyncMock()
            agent.code_executor = AsyncMock()
            agent.math_engine = AsyncMock()
            agent.memory_manager = AsyncMock()
            agent.training_manager = AsyncMock()
            
            # Mock model components
            agent.tokenizer = MagicMock()
            agent.model = MagicMock()
            agent.llm = MagicMock()
            agent.conversation_chain = MagicMock()
            
            return agent
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, agent):
        """Test agent initialization"""
        
        # Test that agent is created with proper attributes
        assert agent.model_name == "test-model"
        assert hasattr(agent, 'web_searcher')
        assert hasattr(agent, 'code_executor')
        assert hasattr(agent, 'math_engine')
        assert hasattr(agent, 'memory_manager')
        assert hasattr(agent, 'training_manager')
        
        # Test performance metrics initialization
        assert isinstance(agent.performance_metrics, dict)
        assert "code_generation_success_rate" in agent.performance_metrics
        assert "search_relevance_score" in agent.performance_metrics
        assert "reasoning_accuracy" in agent.performance_metrics
    
    @pytest.mark.asyncio
    async def test_generate_code_success(self, agent):
        """Test successful code generation"""
        
        # Mock ReAct response
        agent._react_generate = AsyncMock(return_value={
            "thought": "I need to implement a sorting algorithm",
            "action": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr",
            "observation": "The bubble sort algorithm is implemented correctly"
        })
        
        # Mock code executor
        agent.code_executor.test_code.return_value = {
            "success": True,
            "execution_time": 0.005,
            "syntax_valid": True,
            "output": "Test passed"
        }
        
        # Mock math engine
        agent.math_engine.analyze_algorithm.return_value = {
            "time_complexity": "O(n²)",
            "space_complexity": "O(1)",
            "optimization_opportunities": ["Use more efficient sorting algorithm"]
        }
        
        # Mock memory manager
        agent.memory_manager.store_interaction = AsyncMock()
        
        result = await agent.generate_code(
            prompt="Write a function to sort an array",
            language="python",
            context="Use simple algorithm"
        )
        
        # Verify result structure
        assert "code" in result
        assert "explanation" in result
        assert "test_cases" in result
        assert "complexity_analysis" in result
        assert "optimizations" in result
        assert "reasoning_steps" in result
        assert "confidence" in result
        assert "test_results" in result
        
        # Verify content
        assert "bubble_sort" in result["code"]
        assert result["test_results"]["success"] is True
        assert isinstance(result["confidence"], float)
        assert 0 <= result["confidence"] <= 1
        
        # Verify memory storage was called
        agent.memory_manager.store_interaction.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_code_failure(self, agent):
        """Test code generation failure handling"""
        
        # Mock failure
        agent._react_generate = AsyncMock(side_effect=Exception("Generation failed"))
        
        result = await agent.generate_code(
            prompt="Write invalid code",
            language="python"
        )
        
        # Should handle failure gracefully
        assert "code" in result
        assert "error" in result["code"] or "Error generating code" in result["code"]
        assert result["confidence"] == 0.0
        assert result["test_results"]["success"] is False
    
    @pytest.mark.asyncio
    async def test_deep_search_success(self, agent):
        """Test successful deep search"""
        
        # Mock ReAct planning
        agent._react_generate = AsyncMock(return_value={
            "thought": "I need to search for information about machine learning",
            "action": "Search multiple sources for comprehensive information",
            "observation": "Found relevant information from various sources"
        })
        
        # Mock web searcher
        agent.web_searcher.search_multiple_sources.return_value = [
            {
                "title": "Machine Learning Guide",
                "url": "https://example.com/ml-guide",
                "snippet": "Comprehensive guide to ML algorithms",
                "source": "documentation",
                "relevance_score": 0.9
            },
            {
                "title": "ML Implementation Examples",
                "url": "https://github.com/example/ml-examples", 
                "snippet": "Python implementations of ML algorithms",
                "source": "github",
                "relevance_score": 0.85
            }
        ]
        
        # Mock memory manager
        agent.memory_manager.store_interaction = AsyncMock()
        
        result = await agent.deep_search(
            query="machine learning algorithms",
            depth=5,
            include_code=True
        )
        
        # Verify result structure
        assert "results" in result
        assert "synthesized_answer" in result
        assert "code_examples" in result
        assert "sources" in result
        assert "confidence" in result
        assert "related_queries" in result
        
        # Verify content
        assert len(result["results"]) > 0
        assert isinstance(result["confidence"], float)
        assert len(result["sources"]) > 0
        
        # Verify search was called with correct parameters
        agent.web_searcher.search_multiple_sources.assert_called_once()
        call_args = agent.web_searcher.search_multiple_sources.call_args
        assert call_args[1]["query"] == "machine learning algorithms"
        assert call_args[1]["depth"] == 5
        assert call_args[1]["include_code"] is True
    
    @pytest.mark.asyncio
    async def test_deep_search_failure(self, agent):
        """Test deep search failure handling"""
        
        # Mock search failure
        agent.web_searcher.search_multiple_sources.side_effect = Exception("Search failed")
        
        result = await agent.deep_search(
            query="test query",
            depth=5,
            include_code=True
        )
        
        # Should handle failure gracefully
        assert "results" in result
        assert result["results"] == []
        assert "Search failed" in result["synthesized_answer"]
        assert result["confidence"] == 0.0
    
    @pytest.mark.asyncio
    async def test_step_by_step_reasoning_success(self, agent):
        """Test successful step-by-step reasoning"""
        
        # Mock ReAct responses
        agent._react_generate = AsyncMock(side_effect=[
            {  # Initial analysis
                "thought": "This is an optimization problem that requires careful analysis",
                "action": "Break down into subproblems",
                "observation": "Identified key components"
            },
            {  # Solution development
                "thought": "Based on analysis, the best approach is dynamic programming",
                "action": "Implement DP solution with memoization",
                "observation": "Solution provides optimal results"
            }
        ])
        
        # Mock math engine
        agent.math_engine.analyze_problem.return_value = {
            "problem_type": "optimization",
            "complexity": "O(n²)",
            "mathematical_formulation": "min f(x) subject to constraints"
        }
        
        # Mock memory manager
        agent.memory_manager.store_interaction = AsyncMock()
        
        result = await agent.reason_step_by_step(
            problem="Optimize this algorithm for better performance",
            domain="coding",
            include_math=True
        )
        
        # Verify result structure
        assert "reasoning_steps" in result
        assert "solution" in result
        assert "mathematical_analysis" in result
        assert "alternative_approaches" in result
        assert "confidence" in result
        assert "complexity_analysis" in result
        
        # Verify reasoning steps
        assert len(result["reasoning_steps"]) > 0
        for step in result["reasoning_steps"]:
            assert "step" in step
            assert "description" in step
            assert "content" in step
        
        # Verify math analysis was included
        assert result["mathematical_analysis"] is not None
        assert isinstance(result["confidence"], float)
    
    @pytest.mark.asyncio
    async def test_self_training_success(self, agent):
        """Test successful self-training"""
        
        # Mock training manager
        agent.training_manager.load_dataset.return_value = {
            "train": MagicMock(),
            "eval": MagicMock()
        }
        
        # Mock evaluation methods
        agent._evaluate_current_performance = AsyncMock(return_value={
            "accuracy": 0.8,
            "loss": 2.5
        })
        
        # Mock improvement calculation
        agent._calculate_improvement = MagicMock(return_value=0.1)  # 10% improvement
        
        # Mock training components
        with patch('api.agent_core.get_peft_model') as mock_peft, \
             patch('api.agent_core.Trainer') as mock_trainer:
            
            mock_trainer_instance = MagicMock()
            mock_trainer_instance.train.return_value = MagicMock()
            mock_trainer_instance.evaluate.return_value = {"eval_loss": 2.0}
            mock_trainer.return_value = mock_trainer_instance
            
            await agent.self_train(
                dataset_name="test_dataset",
                training_type="rlhf",
                iterations=2
            )
            
            # Verify training was initiated
            mock_trainer.assert_called()
            mock_trainer_instance.train.assert_called()
    
    @pytest.mark.asyncio
    async def test_learn_from_interaction(self, agent):
        """Test learning from user interactions"""
        
        # Mock memory manager
        agent.memory_manager.store_interaction = AsyncMock()
        agent._update_performance_metrics = AsyncMock()
        
        interaction_data = {
            "prompt": "test prompt",
            "language": "python"
        }
        
        response_data = {
            "code": "def test(): pass",
            "confidence": 0.9,
            "success": True
        }
        
        await agent.learn_from_interaction(
            interaction_type="code_generation",
            request_data=interaction_data,
            response_data=response_data
        )
        
        # Verify interaction was stored
        agent.memory_manager.store_interaction.assert_called_once()
        
        # Verify performance metrics were updated
        agent._update_performance_metrics.assert_called_once()
        
        # Verify interaction was added to history
        assert len(agent.interaction_history) > 0
        latest_interaction = agent.interaction_history[-1]
        assert latest_interaction["type"] == "code_generation"
        assert latest_interaction["request"] == interaction_data
        assert latest_interaction["response"] == response_data
    
    @pytest.mark.asyncio
    async def test_process_feedback(self, agent):
        """Test processing user feedback"""
        
        # Mock memory manager and feedback analysis
        agent.memory_manager.store_feedback = AsyncMock()
        agent._analyze_feedback = AsyncMock(return_value={
            "should_retrain": True,
            "sentiment": "positive"
        })
        agent.self_train = AsyncMock()
        
        feedback = {
            "rating": 5,
            "comments": "Excellent code generation!",
            "specific_feedback": {
                "code_quality": "high",
                "explanation_clarity": "very_clear"
            }
        }
        
        await agent.process_feedback(feedback)
        
        # Verify feedback was stored
        agent.memory_manager.store_feedback.assert_called_once()
        
        # Verify feedback was analyzed
        agent._analyze_feedback.assert_called_once()
        
        # Verify retraining was triggered (due to should_retrain=True)
        agent.self_train.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_get_status(self, agent):
        """Test getting agent status"""
        
        status = await agent.get_status()
        
        # Verify status structure
        assert "model_loaded" in status
        assert "components_initialized" in status
        assert "performance_metrics" in status
        assert "interaction_count" in status
        
        # Verify content
        assert isinstance(status["interaction_count"], int)
        assert isinstance(status["performance_metrics"], dict)
    
    def test_get_memory_usage(self, agent):
        """Test getting memory usage information"""
        
        memory_usage = agent.get_memory_usage()
        
        # Should return memory usage info
        assert isinstance(memory_usage, dict)
    
    def test_get_model_info(self, agent):
        """Test getting model information"""
        
        model_info = agent.get_model_info()
        
        # Verify model info structure
        assert "model_name" in model_info
        assert model_info["model_name"] == "test-model"
    
    def test_get_capabilities(self, agent):
        """Test getting agent capabilities"""
        
        capabilities = agent.get_capabilities()
        
        # Verify capabilities
        assert isinstance(capabilities, list)
        assert len(capabilities) > 0
        assert "Code generation" in capabilities
        assert "Deep web search" in capabilities
        assert "Mathematical reasoning" in capabilities

class TestAgentHelperMethods:
    """Test helper methods of the agent"""
    
    @pytest.fixture
    def agent(self):
        """Create a basic agent for testing helper methods"""
        return AutonomousAgent()
    
    def test_extract_code_from_response(self, agent):
        """Test code extraction from ReAct response"""
        
        response = {
            "thought": "I need to write a function",
            "action": "def hello():\n    print('Hello, World!')",
            "observation": "Function created successfully"
        }
        
        code = agent._extract_code_from_response(response)
        
        # Should extract the action as code
        assert "def hello" in code
        assert "print" in code
    
    def test_calculate_confidence(self, agent):
        """Test confidence calculation"""
        
        test_results = {
            "success": True,
            "execution_time": 0.001,
            "syntax_valid": True
        }
        
        math_analysis = {
            "complexity_score": 30,  # Good complexity
            "optimization_opportunities": []
        }
        
        confidence = agent._calculate_confidence(test_results, math_analysis)
        
        # Should return a reasonable confidence score
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
    
    def test_calculate_improvement(self, agent):
        """Test improvement calculation between metrics"""
        
        baseline_metrics = {
            "accuracy": 0.8,
            "loss": 2.5,
            "performance_score": 0.7
        }
        
        final_metrics = {
            "accuracy": 0.9,
            "loss": 2.0,
            "performance_score": 0.8
        }
        
        improvement = agent._calculate_improvement(baseline_metrics, final_metrics)
        
        # Should calculate positive improvement
        assert isinstance(improvement, float)
        assert improvement > 0  # Should detect improvement

class TestAgentIntegration:
    """Integration tests for agent components"""
    
    @pytest.mark.asyncio
    async def test_full_code_generation_pipeline(self):
        """Test the complete code generation pipeline"""
        
        with patch.multiple(
            'api.agent_core',
            AutoTokenizer=MagicMock(),
            AutoModelForCausalLM=MagicMock(),
            HuggingFacePipeline=MagicMock(),
            ConversationChain=MagicMock()
        ):
            agent = AutonomousAgent()
            
            # Mock all components
            agent.web_searcher = AsyncMock()
            agent.code_executor = AsyncMock()
            agent.math_engine = AsyncMock()
            agent.memory_manager = AsyncMock()
            agent.training_manager = AsyncMock()
            
            # Setup component responses
            agent._react_generate = AsyncMock(return_value={
                "thought": "Need to implement Fibonacci",
                "action": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
                "observation": "Recursive solution implemented"
            })
            
            agent.code_executor.test_code.return_value = {
                "success": True,
                "syntax_valid": True,
                "execution_time": 0.002
            }
            
            agent.math_engine.analyze_algorithm.return_value = {
                "time_complexity": "O(2^n)",
                "space_complexity": "O(n)"
            }
            
            agent._generate_test_cases = AsyncMock(return_value=[
                {"input": "5", "expected": "5"}
            ])
            
            agent._suggest_optimizations = AsyncMock(return_value=[
                "Use memoization to improve time complexity"
            ])
            
            agent.memory_manager.store_interaction = AsyncMock()
            
            # Test the full pipeline
            result = await agent.generate_code(
                prompt="Write a Fibonacci function",
                language="python"
            )
            
            # Verify all components were called
            agent._react_generate.assert_called()
            agent.code_executor.test_code.assert_called()
            agent.math_engine.analyze_algorithm.assert_called()
            agent.memory_manager.store_interaction.assert_called()
            
            # Verify result completeness
            assert all(key in result for key in [
                "code", "explanation", "test_cases", "complexity_analysis",
                "optimizations", "reasoning_steps", "confidence", "test_results"
            ])
    
    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """Test that errors are properly handled and propagated"""
        
        with patch.multiple(
            'api.agent_core',
            AutoTokenizer=MagicMock(),
            AutoModelForCausalLM=MagicMock(),
            HuggingFacePipeline=MagicMock(),
            ConversationChain=MagicMock()
        ):
            agent = AutonomousAgent()
            
            # Mock component failure
            agent.code_executor = AsyncMock()
            agent.code_executor.test_code.side_effect = Exception("Execution failed")
            
            agent._react_generate = AsyncMock(return_value={
                "action": "def test(): pass"
            })
            
            agent.math_engine = AsyncMock()
            agent.memory_manager = AsyncMock()
            
            # Should handle component failure gracefully
            result = await agent.generate_code("test prompt")
            
            # Should still return a result structure
            assert "code" in result
            assert "confidence" in result
            # Confidence should be low due to failure
            assert result["confidence"] < 0.5

# Performance tests
class TestAgentPerformance:
    """Test performance aspects of the agent"""
    
    @pytest.mark.asyncio
    async def test_response_time_under_load(self):
        """Test agent response time under simulated load"""
        
        with patch.multiple(
            'api.agent_core',
            AutoTokenizer=MagicMock(),
            AutoModelForCausalLM=MagicMock(),
            HuggingFacePipeline=MagicMock(),
            ConversationChain=MagicMock()
        ):
            agent = AutonomousAgent()
            
            # Mock fast responses
            agent._react_generate = AsyncMock(return_value={"action": "test"})
            agent.code_executor = AsyncMock()
            agent.code_executor.test_code.return_value = {"success": True}
            agent.math_engine = AsyncMock()
            agent.math_engine.analyze_algorithm.return_value = {}
            agent.memory_manager = AsyncMock()
            
            # Simulate concurrent requests
            import time
            start_time = time.time()
            
            tasks = []
            for i in range(5):
                task = agent.generate_code(f"test prompt {i}")
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Should handle concurrent requests efficiently
            assert len(results) == 5
            assert total_time < 5.0  # Should complete within reasonable time
            
            # All requests should succeed
            for result in results:
                assert "code" in result

# Test configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
