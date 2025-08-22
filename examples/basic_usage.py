#!/usr/bin/env python3
"""
Basic Usage Examples for Autonomous AI Agent API

This script demonstrates the fundamental features of the AI agent,
including code generation, web search, reasoning, and training.
"""

import requests
import json
import time
import asyncio
from typing import Dict, Any

# Configuration
API_BASE = "https://your-deployment.vercel.app"
API_TOKEN = "autonomous-ai-agent-2024"

# Headers for all requests
HEADERS = {
    "Authorization": f"Bearer {API_TOKEN}",
    "Content-Type": "application/json"
}

class AutonomousAIClient:
    """Simple client for interacting with the Autonomous AI Agent API"""
    
    def __init__(self, base_url: str = API_BASE, token: str = API_TOKEN):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    
    def _make_request(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict[str, Any]:
        """Make HTTP request to the API"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url, headers=self.headers, timeout=30)
            elif method == "POST":
                response = requests.post(url, headers=self.headers, json=data, timeout=30)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the API is online"""
        return self._make_request("/")
    
    def generate_code(self, prompt: str, language: str = "python", context: str = None) -> Dict[str, Any]:
        """Generate code with AI assistance"""
        return self._make_request("/generate", "POST", {
            "prompt": prompt,
            "language": language,
            "context": context
        })
    
    def search_web(self, query: str, depth: int = 5, include_code: bool = True) -> Dict[str, Any]:
        """Search web for coding information"""
        return self._make_request("/search", "POST", {
            "query": query,
            "depth": depth,
            "include_code": include_code
        })
    
    def reason_step_by_step(self, problem: str, domain: str = "coding", include_math: bool = True) -> Dict[str, Any]:
        """Get step-by-step reasoning for complex problems"""
        return self._make_request("/reason", "POST", {
            "problem": problem,
            "domain": domain,
            "include_math": include_math
        })
    
    def trigger_training(self, dataset_name: str = None, training_type: str = "rlhf", iterations: int = 10) -> Dict[str, Any]:
        """Trigger self-training of the agent"""
        return self._make_request("/train", "POST", {
            "dataset_name": dataset_name,
            "training_type": training_type,
            "iterations": iterations
        })
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status and metrics"""
        return self._make_request("/status")
    
    def provide_feedback(self, rating: int, comments: str = None, specific_feedback: Dict = None) -> Dict[str, Any]:
        """Provide feedback to improve the agent"""
        return self._make_request("/feedback", "POST", {
            "rating": rating,
            "comments": comments,
            "specific_feedback": specific_feedback
        })

def print_section(title: str):
    """Print a formatted section header"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)

def print_code_result(result: Dict[str, Any]):
    """Print code generation result in a formatted way"""
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"📝 Generated Code (Confidence: {result.get('confidence', 0):.2%}):")
    print("-" * 40)
    print(result.get('code', 'No code generated'))
    print("-" * 40)
    
    if result.get('explanation'):
        print(f"\n💡 Explanation:")
        print(result['explanation'])
    
    if result.get('complexity_analysis'):
        complexity = result['complexity_analysis']
        print(f"\n📊 Complexity Analysis:")
        print(f"  Time: {complexity.get('time_complexity', 'Unknown')}")
        print(f"  Space: {complexity.get('space_complexity', 'Unknown')}")
    
    if result.get('optimizations'):
        print(f"\n⚡ Optimization Suggestions:")
        for i, opt in enumerate(result['optimizations'], 1):
            print(f"  {i}. {opt}")

def print_search_result(result: Dict[str, Any]):
    """Print search result in a formatted way"""
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"🔍 Search Results (Confidence: {result.get('confidence', 0):.2%}):")
    print("-" * 40)
    
    if result.get('synthesized_answer'):
        print("📋 Synthesized Answer:")
        print(result['synthesized_answer'])
        print()
    
    if result.get('results'):
        print(f"📚 Found {len(result['results'])} sources:")
        for i, res in enumerate(result['results'][:3], 1):  # Show top 3
            print(f"  {i}. {res.get('title', 'Untitled')}")
            print(f"     Source: {res.get('source', 'Unknown')}")
            print(f"     URL: {res.get('url', 'No URL')}")
            print()
    
    if result.get('code_examples'):
        print(f"💻 Code Examples Found: {len(result['code_examples'])}")

def print_reasoning_result(result: Dict[str, Any]):
    """Print reasoning result in a formatted way"""
    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return
    
    print(f"🧠 Step-by-Step Reasoning (Confidence: {result.get('confidence', 0):.2%}):")
    print("-" * 40)
    
    if result.get('reasoning_steps'):
        for step in result['reasoning_steps']:
            print(f"Step {step.get('step', '?')}: {step.get('description', 'Unknown')}")
            print(f"  {step.get('content', 'No content')}")
            print()
    
    if result.get('solution'):
        print("🎯 Final Solution:")
        print(result['solution'])
    
    if result.get('mathematical_analysis'):
        math_analysis = result['mathematical_analysis']
        print(f"\n📐 Mathematical Analysis:")
        for key, value in math_analysis.items():
            print(f"  {key}: {value}")

def main():
    """Main demonstration function"""
    print("🤖 Autonomous AI Agent - Basic Usage Examples")
    print("This script demonstrates the core capabilities of the AI agent.")
    
    # Initialize client
    client = AutonomousAIClient()
    
    # 1. Health Check
    print_section("1. Health Check")
    health = client.health_check()
    if "error" not in health:
        print(f"✅ API is online!")
        print(f"   Status: {health.get('status', 'Unknown')}")
        print(f"   Version: {health.get('version', 'Unknown')}")
    else:
        print(f"❌ API is offline: {health.get('error')}")
        return
    
    # 2. Code Generation Examples
    print_section("2. Code Generation Examples")
    
    # Example 1: Simple algorithm
    print("🔄 Generating a sorting algorithm...")
    code_result = client.generate_code(
        prompt="Write a Python function to implement quicksort algorithm with comments explaining each step",
        language="python",
        context="Focus on readability and include time complexity analysis"
    )
    print_code_result(code_result)
    
    time.sleep(2)  # Rate limiting
    
    # Example 2: Data structure
    print("\n🔄 Generating a data structure...")
    code_result = client.generate_code(
        prompt="Create a Python class for a binary search tree with insert, search, and delete methods",
        language="python",
        context="Include docstrings and error handling"
    )
    print_code_result(code_result)
    
    # 3. Web Search Examples
    print_section("3. Web Search Examples")
    
    print("🔄 Searching for algorithm information...")
    search_result = client.search_web(
        query="Python machine learning optimization algorithms comparison",
        depth=5,
        include_code=True
    )
    print_search_result(search_result)
    
    time.sleep(2)  # Rate limiting
    
    # 4. Step-by-Step Reasoning
    print_section("4. Step-by-Step Reasoning")
    
    print("🔄 Reasoning about algorithm optimization...")
    reasoning_result = client.reason_step_by_step(
        problem="How to optimize a recursive Fibonacci function for better performance?",
        domain="algorithms",
        include_math=True
    )
    print_reasoning_result(reasoning_result)
    
    # 5. Agent Status
    print_section("5. Agent Status")
    
    print("🔄 Getting agent status...")
    status = client.get_status()
    if "error" not in status:
        print("📊 Agent Status:")
        agent_status = status.get('agent_status', {})
        print(f"  Model loaded: {agent_status.get('model_loaded', 'Unknown')}")
        print(f"  Interactions: {agent_status.get('interaction_count', 0)}")
        
        memory_usage = status.get('memory_usage', {})
        total_memories = memory_usage.get('total_memories', {})
        print(f"  Memory: {sum(total_memories.values())} total memories")
        
        model_info = status.get('model_info', {})
        print(f"  Model: {model_info.get('model_name', 'Unknown')}")
        print(f"  Version: {model_info.get('current_version', 'Unknown')}")
    else:
        print(f"❌ Status error: {status.get('error')}")
    
    # 6. Feedback Example
    print_section("6. Providing Feedback")
    
    print("🔄 Providing feedback to the agent...")
    feedback_result = client.provide_feedback(
        rating=5,
        comments="Excellent code generation and clear explanations!",
        specific_feedback={
            "code_quality": "high",
            "explanation_clarity": "very_clear",
            "usefulness": "very_useful",
            "accuracy": "accurate"
        }
    )
    
    if "error" not in feedback_result:
        print("✅ Feedback submitted successfully!")
        print(f"   Status: {feedback_result.get('status')}")
    else:
        print(f"❌ Feedback error: {feedback_result.get('error')}")
    
    # 7. Training Example (optional)
    print_section("7. Self-Training (Optional)")
    
    response = input("🤔 Would you like to trigger agent training? (y/N): ")
    if response.lower() == 'y':
        print("🔄 Triggering self-training...")
        training_result = client.trigger_training(
            dataset_name=None,  # Use interaction data
            training_type="continuous",
            iterations=3
        )
        
        if "error" not in training_result:
            print("✅ Training started!")
            print(f"   Status: {training_result.get('status')}")
            print(f"   Type: {training_result.get('training_type')}")
            print(f"   Duration: {training_result.get('estimated_duration')}")
        else:
            print(f"❌ Training error: {training_result.get('error')}")
    else:
        print("⏭️  Skipping training example")
    
    # Summary
    print_section("Summary")
    print("🎉 Basic usage examples completed!")
    print("\n📖 What you've learned:")
    print("  ✓ How to generate code with AI assistance")
    print("  ✓ How to search for coding information")
    print("  ✓ How to get step-by-step problem solving")
    print("  ✓ How to check agent status")
    print("  ✓ How to provide feedback for improvement")
    print("  ✓ How to trigger self-training")
    
    print("\n🚀 Next steps:")
    print("  • Explore advanced_training.py for complex training scenarios")
    print("  • Try desktop_integration.py for GUI application usage")
    print("  • Check the API documentation for more endpoints")
    print("  • Build your own applications using the agent!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Thanks for trying the Autonomous AI Agent!")
    except Exception as e:
        print(f"\n💥 An unexpected error occurred: {e}")
        print("Please check your configuration and try again.")
