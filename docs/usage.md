# Usage Guide

## Getting Started

This guide provides comprehensive instructions for using the Autonomous AI Agent for coding tasks. The agent can generate code, perform deep web searches, solve problems step-by-step, and continuously improve through self-training.

## Authentication

### Personal Access Token

For personal use, the agent supports a hardcoded token for quick access:

```python
TOKEN = "autonomous-ai-agent-2024"
```

### JWT Token Generation

For production or custom deployments, generate a JWT token:

```python
from api.utils.auth import generate_personal_token, verify_token

# Generate a new token
token = generate_personal_token()
print(f"Your access token: {token}")

# Verify token
try:
    user_info = verify_token(token)
    print(f"Token valid for user: {user_info}")
except Exception as e:
    print(f"Token invalid: {e}")
```

## API Endpoints Usage

### 1. Code Generation (`/generate`)

Generate high-quality code with comprehensive analysis.

#### Basic Usage

```python
import requests
import json

API_BASE = "https://your-deployment.vercel.app"
HEADERS = {
    "Authorization": "Bearer autonomous-ai-agent-2024",
    "Content-Type": "application/json"
}

def generate_code(prompt, language="python", context=None):
    response = requests.post(
        f"{API_BASE}/generate",
        headers=HEADERS,
        json={
            "prompt": prompt,
            "language": language,
            "context": context
        }
    )
    return response.json()

# Example: Generate a sorting algorithm
result = generate_code(
    prompt="Write a function to implement merge sort with detailed comments",
    language="python",
    context="Focus on time complexity analysis and optimization"
)

print("Generated Code:")
print(result["code"])
print("\nExplanation:")
print(result["explanation"])
print(f"\nComplexity: {result['complexity_analysis']}")
print(f"Confidence: {result['confidence']}")
```

#### Advanced Code Generation

```python
# Generate algorithm with mathematical analysis
result = generate_code(
    prompt="""
    Create a function to find the k-th largest element in an unsorted array.
    Include multiple solution approaches and compare their complexities.
    """,
    language="python",
    context="Consider both time and space efficiency. Include heap-based solution."
)

# Access detailed results
print("Code:", result["code"])
print("Test Cases:", result["test_cases"])
print("Optimizations:", result["optimizations"])
print("Reasoning Steps:", result["reasoning_steps"])
```

#### Response Format

```json
{
  "code": "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    ...",
  "explanation": "Merge sort is a divide-and-conquer algorithm...",
  "test_cases": [
    {
      "input": "[3, 1, 4, 1, 5]",
      "expected_output": "[1, 1, 3, 4, 5]"
    }
  ],
  "complexity_analysis": {
    "time_complexity": "O(n log n)",
    "space_complexity": "O(n)",
    "mathematical_proof": "T(n) = 2T(n/2) + O(n), solving gives O(n log n)"
  },
  "optimizations": [
    "Consider in-place sorting for memory efficiency",
    "Use insertion sort for small subarrays"
  ],
  "reasoning_steps": "First, I analyzed the problem requirements...",
  "confidence": 0.95,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### 2. Deep Web Search (`/search`)

Perform comprehensive searches across multiple coding resources.

#### Basic Search

```python
def deep_search(query, depth=5, include_code=True):
    response = requests.post(
        f"{API_BASE}/search",
        headers=HEADERS,
        json={
            "query": query,
            "depth": depth,
            "include_code": include_code
        }
    )
    return response.json()

# Search for algorithm information
result = deep_search(
    query="Python asyncio best practices and common pitfalls",
    depth=10,
    include_code=True
)

print("Synthesized Answer:")
print(result["synthesized_answer"])
print("\nCode Examples:")
for example in result["code_examples"]:
    print(f"- {example}")
print(f"\nSources: {len(result['sources'])} sources found")
print(f"Confidence: {result['confidence']}")
```

#### Advanced Search with Filtering

```python
# Search for specific programming concepts
result = deep_search(
    query="machine learning optimization algorithms gradient descent variants",
    depth=15,
    include_code=True
)

# Process results
sources_by_type = {}
for i, result_item in enumerate(result["results"]):
    source_type = result_item.get("source", "unknown")
    if source_type not in sources_by_type:
        sources_by_type[source_type] = []
    sources_by_type[source_type].append(result_item)

print("Results by Source:")
for source, items in sources_by_type.items():
    print(f"{source}: {len(items)} results")
```

#### Response Format

```json
{
  "results": [
    {
      "title": "How to use asyncio properly",
      "url": "https://stackoverflow.com/questions/...",
      "snippet": "Asyncio is Python's library for...",
      "source": "stackoverflow",
      "relevance_score": 0.92,
      "has_code": true
    }
  ],
  "synthesized_answer": "Based on multiple sources, asyncio best practices include...",
  "code_examples": [
    {
      "code": "async def example():\n    await asyncio.sleep(1)",
      "description": "Basic async function example",
      "source": "stack_overflow"
    }
  ],
  "sources": ["stackoverflow.com", "github.com"],
  "confidence": 0.87,
  "related_queries": ["asyncio vs threading", "async await patterns"],
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### 3. Step-by-Step Reasoning (`/reason`)

Solve complex problems with detailed mathematical and logical analysis.

#### Basic Reasoning

```python
def step_by_step_reasoning(problem, domain="coding", include_math=True):
    response = requests.post(
        f"{API_BASE}/reason",
        headers=HEADERS,
        json={
            "problem": problem,
            "domain": domain,
            "include_math": include_math
        }
    )
    return response.json()

# Analyze an algorithmic problem
result = step_by_step_reasoning(
    problem="""
    Design an efficient algorithm to find the longest palindromic substring
    in a given string. Analyze different approaches and their trade-offs.
    """,
    domain="coding",
    include_math=True
)

print("Reasoning Steps:")
for step in result["reasoning_steps"]:
    print(f"Step {step['step']}: {step['description']}")
    print(f"Content: {step['content']}\n")

print("Final Solution:")
print(result["solution"])
```

#### Mathematical Problem Solving

```python
# Solve optimization problem
result = step_by_step_reasoning(
    problem="""
    Optimize the following: Given n tasks with processing times t_i and deadlines d_i,
    schedule them to minimize total weighted tardiness. Provide mathematical formulation.
    """,
    domain="optimization",
    include_math=True
)

print("Mathematical Analysis:")
print(result["mathematical_analysis"])
print("\nAlternative Approaches:")
for approach in result["alternative_approaches"]:
    print(f"- {approach}")
```

#### Response Format

```json
{
  "reasoning_steps": [
    {
      "step": 1,
      "description": "Problem Analysis",
      "content": "The problem asks for finding the longest palindromic substring..."
    },
    {
      "step": 2,
      "description": "Mathematical Formulation",
      "content": "Let P(i,j) be true if substring s[i:j+1] is palindromic..."
    }
  ],
  "solution": "The optimal approach uses Manacher's algorithm with O(n) complexity...",
  "mathematical_analysis": {
    "complexity": "O(n)",
    "space_complexity": "O(n)",
    "mathematical_proof": "Manacher's algorithm processes each character at most twice..."
  },
  "alternative_approaches": [
    {
      "name": "Dynamic Programming",
      "complexity": "O(n²)",
      "description": "Build a 2D table to track palindromes"
    },
    {
      "name": "Expand Around Centers",
      "complexity": "O(n²)",
      "description": "Check all possible centers"
    }
  ],
  "confidence": 0.91,
  "complexity_analysis": {
    "time_complexity": "O(n)",
    "space_complexity": "O(n)",
    "optimality": "Optimal for this problem class"
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### 4. Self-Training (`/train`)

Trigger autonomous improvement of the AI agent.

#### Basic Training

```python
def trigger_training(dataset_name=None, training_type="rlhf", iterations=10):
    response = requests.post(
        f"{API_BASE}/train",
        headers=HEADERS,
        json={
            "dataset_name": dataset_name,
            "training_type": training_type,
            "iterations": iterations
        }
    )
    return response.json()

# Trigger training with external dataset
result = trigger_training(
    dataset_name="codeparrot/github-code",
    training_type="rlhf",
    iterations=5
)

print(f"Training Status: {result['status']}")
print(f"Estimated Duration: {result['estimated_duration']}")
```

#### Continuous Learning from Interactions

```python
# Trigger training based on recent interactions
result = trigger_training(
    dataset_name=None,  # Use interaction history
    training_type="continuous",
    iterations=3
)

print("Training initiated using interaction data")
print(f"Training type: {result['training_type']}")
```

#### Feedback-Based Training

```python
# First, provide feedback on a previous interaction
feedback_response = requests.post(
    f"{API_BASE}/feedback",
    headers=HEADERS,
    json={
        "interaction_id": "some-interaction-id",
        "rating": 5,
        "comments": "The generated code was excellent and well-optimized",
        "specific_feedback": {
            "code_quality": "high",
            "explanation_clarity": "very_clear",
            "usefulness": "very_useful"
        }
    }
)

# Then trigger feedback-based training
result = trigger_training(
    training_type="feedback",
    iterations=5
)
```

## System Status and Monitoring

### Check Agent Status

```python
def get_agent_status():
    response = requests.get(
        f"{API_BASE}/status",
        headers=HEADERS
    )
    return response.json()

status = get_agent_status()
print("Agent Status:")
print(f"Model Loaded: {status['agent_status']['model_loaded']}")
print(f"Memory Usage: {status['memory_usage']}")
print(f"Training History: {len(status['training_history'])} sessions")
print(f"Capabilities: {', '.join(status['capabilities'])}")
```

### Provide Feedback

```python
def provide_feedback(interaction_id, rating, comments):
    response = requests.post(
        f"{API_BASE}/feedback",
        headers=HEADERS,
        json={
            "interaction_id": interaction_id,
            "rating": rating,
            "comments": comments,
            "timestamp": "2024-01-01T00:00:00Z"
        }
    )
    return response.json()

# Provide feedback on agent performance
feedback_result = provide_feedback(
    interaction_id="abc123",
    rating=4,
    comments="Good code generation but could use better error handling"
)
```

## Advanced Usage Patterns

### 1. Iterative Problem Solving

```python
def iterative_solve(initial_problem):
    """Solve a problem iteratively by refining the solution."""
    
    # Step 1: Initial analysis
    analysis = step_by_step_reasoning(
        problem=initial_problem,
        domain="coding",
        include_math=True
    )
    
    print("Initial Analysis:")
    print(analysis["solution"])
    
    # Step 2: Generate code based on analysis
    code_result = generate_code(
        prompt=f"Implement the solution: {analysis['solution']}",
        context="Based on the mathematical analysis provided"
    )
    
    print("\nGenerated Code:")
    print(code_result["code"])
    
    # Step 3: Search for optimizations
    search_result = deep_search(
        query=f"optimization techniques for {initial_problem}",
        include_code=True
    )
    
    print("\nOptimization Ideas:")
    print(search_result["synthesized_answer"])
    
    # Step 4: Refine the solution
    refined_result = generate_code(
        prompt=f"Optimize this code: {code_result['code']}",
        context=f"Using these optimization ideas: {search_result['synthesized_answer']}"
    )
    
    return {
        "analysis": analysis,
        "initial_code": code_result,
        "optimization_research": search_result,
        "final_code": refined_result
    }

# Example usage
result = iterative_solve("Find the shortest path in a weighted graph")
```

### 2. Batch Processing

```python
def batch_code_generation(prompts, language="python"):
    """Generate code for multiple prompts efficiently."""
    
    results = []
    for i, prompt in enumerate(prompts):
        print(f"Processing {i+1}/{len(prompts)}: {prompt[:50]}...")
        
        result = generate_code(
            prompt=prompt,
            language=language,
            context="Part of a batch processing request"
        )
        
        results.append({
            "prompt": prompt,
            "result": result,
            "success": result.get("confidence", 0) > 0.7
        })
    
    return results

# Example: Generate multiple algorithms
algorithm_prompts = [
    "Implement binary search",
    "Create a hash table with collision handling",
    "Write a depth-first search for graphs",
    "Implement quicksort with random pivot"
]

batch_results = batch_code_generation(algorithm_prompts)

# Analyze results
successful = [r for r in batch_results if r["success"]]
print(f"Successfully generated {len(successful)}/{len(batch_results)} algorithms")
```

### 3. Learning from Feedback Loop

```python
def feedback_learning_cycle(problem, max_iterations=3):
    """Implement a feedback learning cycle for continuous improvement."""
    
    iteration = 0
    current_solution = None
    
    while iteration < max_iterations:
        print(f"\n--- Iteration {iteration + 1} ---")
        
        if iteration == 0:
            # Initial solution
            result = generate_code(
                prompt=problem,
                context="First attempt at solving the problem"
            )
        else:
            # Improve based on feedback
            result = generate_code(
                prompt=f"Improve this solution: {current_solution}",
                context="Based on previous feedback and analysis"
            )
        
        current_solution = result["code"]
        
        print(f"Generated solution with confidence: {result['confidence']}")
        
        # Simulate feedback (in real usage, this would be user input)
        if result["confidence"] > 0.9:
            print("High confidence reached, stopping iteration")
            break
        
        # Search for improvement ideas
        search_result = deep_search(
            query=f"how to improve {problem}",
            include_code=True
        )
        
        # Provide feedback to the system
        provide_feedback(
            interaction_id=f"iteration_{iteration}",
            rating=3 + int(result["confidence"] * 2),  # Convert confidence to rating
            comments=f"Iteration {iteration + 1} results"
        )
        
        iteration += 1
    
    return current_solution

# Example usage
final_solution = feedback_learning_cycle(
    "Write an efficient algorithm for finding prime numbers"
)
```

## Error Handling and Best Practices

### 1. Robust Error Handling

```python
import time
from typing import Optional, Dict, Any

def safe_api_call(endpoint: str, data: Dict[str, Any], max_retries: int = 3) -> Optional[Dict]:
    """Make API call with retry logic and error handling."""
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{API_BASE}/{endpoint}",
                headers=HEADERS,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limited, wait and retry
                wait_time = 2 ** attempt
                print(f"Rate limited, waiting {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            elif response.status_code == 401:
                print("Authentication failed, check your token")
                return None
            else:
                print(f"API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.Timeout:
            print(f"Request timeout on attempt {attempt + 1}")
        except requests.exceptions.ConnectionError:
            print(f"Connection error on attempt {attempt + 1}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        
        if attempt < max_retries - 1:
            time.sleep(1)
    
    print("Max retries exceeded")
    return None

# Usage with error handling
result = safe_api_call("generate", {
    "prompt": "Write a function to reverse a string",
    "language": "python"
})

if result:
    print("Success:", result["code"])
else:
    print("Failed to generate code")
```

### 2. Input Validation

```python
def validate_code_request(prompt: str, language: str = "python") -> bool:
    """Validate code generation request parameters."""
    
    if not prompt or len(prompt.strip()) < 10:
        print("Error: Prompt too short or empty")
        return False
    
    if language not in ["python", "javascript", "java", "cpp", "c"]:
        print(f"Error: Unsupported language: {language}")
        return False
    
    # Check for potentially dangerous requests
    dangerous_keywords = ["delete", "remove", "os.system", "subprocess", "exec"]
    if any(keyword in prompt.lower() for keyword in dangerous_keywords):
        print("Warning: Prompt contains potentially dangerous keywords")
        return False
    
    return True

# Safe code generation
def safe_generate_code(prompt: str, language: str = "python") -> Optional[Dict]:
    """Generate code with input validation."""
    
    if not validate_code_request(prompt, language):
        return None
    
    return safe_api_call("generate", {
        "prompt": prompt,
        "language": language
    })
```

### 3. Performance Optimization

```python
import asyncio
import aiohttp

async def async_batch_request(endpoints_data: list) -> list:
    """Make multiple API requests concurrently."""
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        for endpoint, data in endpoints_data:
            task = make_async_request(session, endpoint, data)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

async def make_async_request(session, endpoint: str, data: dict):
    """Make individual async API request."""
    
    try:
        async with session.post(
            f"{API_BASE}/{endpoint}",
            headers=HEADERS,
            json=data,
            timeout=aiohttp.ClientTimeout(total=30)
        ) as response:
            if response.status == 200:
                return await response.json()
            else:
                return {"error": f"HTTP {response.status}"}
    except Exception as e:
        return {"error": str(e)}

# Example: Concurrent requests
async def concurrent_example():
    requests_data = [
        ("generate", {"prompt": "Implement bubble sort", "language": "python"}),
        ("generate", {"prompt": "Implement merge sort", "language": "python"}),
        ("search", {"query": "sorting algorithms comparison", "depth": 5})
    ]
    
    results = await async_batch_request(requests_data)
    
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Request {i} failed: {result}")
        elif "error" in result:
            print(f"Request {i} error: {result['error']}")
        else:
            print(f"Request {i} succeeded")

# Run concurrent example
# asyncio.run(concurrent_example())
```

## Integration Examples

### 1. Jupyter Notebook Integration

```python
# In Jupyter Notebook
from IPython.display import display, Markdown, Code
import json

def display_code_result(result):
    """Display code generation result in Jupyter notebook."""
    
    # Display explanation
    display(Markdown(f"**Explanation:**\n{result['explanation']}"))
    
    # Display code with syntax highlighting
    display(Code(result['code'], language='python'))
    
    # Display complexity analysis
    complexity = result.get('complexity_analysis', {})
    display(Markdown(f"""
    **Complexity Analysis:**
    - Time Complexity: {complexity.get('time_complexity', 'N/A')}
    - Space Complexity: {complexity.get('space_complexity', 'N/A')}
    - Confidence: {result.get('confidence', 0):.2%}
    """))

# Generate and display code
result = generate_code("Implement a binary search tree with insert and search methods")
display_code_result(result)
```

### 2. Command Line Interface

```python
#!/usr/bin/env python3
"""Command line interface for the Autonomous AI Agent."""

import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Autonomous AI Agent CLI")
    parser.add_argument("command", choices=["generate", "search", "reason", "train", "status"])
    parser.add_argument("--prompt", help="Input prompt")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--language", default="python", help="Programming language")
    parser.add_argument("--depth", type=int, default=5, help="Search depth")
    parser.add_argument("--output", help="Output file")
    
    args = parser.parse_args()
    
    if args.command == "generate":
        if not args.prompt:
            print("Error: --prompt required for generate command")
            sys.exit(1)
        
        result = generate_code(args.prompt, args.language)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(result["code"])
            print(f"Code saved to {args.output}")
        else:
            print(result["code"])
    
    elif args.command == "search":
        if not args.query:
            print("Error: --query required for search command")
            sys.exit(1)
        
        result = deep_search(args.query, args.depth)
        print(result["synthesized_answer"])
    
    elif args.command == "status":
        status = get_agent_status()
        print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
```

## Troubleshooting

### Common Issues

1. **Authentication Errors (401)**
   ```python
   # Check token validity
   try:
       from api.utils.auth import verify_token
       user_info = verify_token("your-token-here")
       print("Token valid:", user_info)
   except Exception as e:
       print("Token invalid:", e)
   ```

2. **Rate Limiting (429)**
   ```python
   # Implement exponential backoff
   import time
   
   def handle_rate_limit(response):
       if response.status_code == 429:
           retry_after = response.headers.get('Retry-After', 1)
           print(f"Rate limited, waiting {retry_after} seconds")
           time.sleep(int(retry_after))
           return True
       return False
   ```

3. **Large Response Times**
   ```python
   # Use streaming for large requests
   def stream_generate_code(prompt):
       response = requests.post(
           f"{API_BASE}/generate",
           headers=HEADERS,
           json={"prompt": prompt},
           stream=True
       )
       
       for chunk in response.iter_content(chunk_size=1024):
           if chunk:
               yield chunk.decode('utf-8')
   ```

4. **Memory Issues**
   ```python
   # Clear model cache if needed
   def clear_cache():
       requests.post(
           f"{API_BASE}/admin/clear-cache",
           headers=HEADERS
       )
   ```

This comprehensive usage guide should help you effectively utilize all features of the Autonomous AI Agent. Remember to always validate inputs, handle errors gracefully, and provide feedback to help the agent improve over time.
