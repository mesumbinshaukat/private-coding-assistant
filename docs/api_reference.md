# API Reference

## Overview

The Autonomous AI Agent API provides RESTful endpoints for code generation, web search, reasoning, and self-training. All endpoints require authentication and return JSON responses with comprehensive data.

## Base URL

```
https://your-deployment.vercel.app
```

## Authentication

All endpoints require JWT token authentication:

```http
Authorization: Bearer <token>
```

### Personal Access Token
For development and personal use:
```
Token: autonomous-ai-agent-2024
```

### Custom JWT Token
Generate using the auth utility:
```python
from api.utils.auth import generate_personal_token
token = generate_personal_token()
```

## Rate Limiting

| User Role | Requests/Minute | Burst Size |
|-----------|-----------------|------------|
| Admin     | 1000           | 50         |
| User      | 60             | 10         |
| Anonymous | 20             | 5          |

Rate limit headers:
- `X-RateLimit-Limit`: Request limit
- `X-RateLimit-Remaining`: Remaining requests
- `X-RateLimit-Reset`: Reset timestamp

## Error Responses

Standard error format:
```json
{
  "error": "Error description",
  "status_code": 400,
  "timestamp": "2024-01-01T00:00:00Z",
  "request_id": "req_123456"
}
```

### HTTP Status Codes

| Code | Description |
|------|-------------|
| 200  | Success |
| 400  | Bad Request |
| 401  | Unauthorized |
| 403  | Forbidden |
| 404  | Not Found |
| 429  | Too Many Requests |
| 500  | Internal Server Error |

---

## Endpoints

### GET `/`

Health check endpoint.

**Response:**
```json
{
  "status": "online",
  "agent_status": "initialized",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0"
}
```

---

### POST `/generate`

Generate code with comprehensive analysis.

**Request Body:**
```json
{
  "prompt": "string",
  "language": "string", 
  "context": "string"
}
```

**Parameters:**
- `prompt` (required): Description of the code to generate
- `language` (optional): Programming language (default: "python")
- `context` (optional): Additional context or constraints

**Response:**
```json
{
  "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
  "explanation": "This function implements the Fibonacci sequence using recursion...",
  "test_cases": [
    {
      "input": "5",
      "expected_output": "5",
      "description": "Test with n=5"
    }
  ],
  "complexity_analysis": {
    "time_complexity": "O(2^n)",
    "space_complexity": "O(n)",
    "mathematical_proof": "T(n) = T(n-1) + T(n-2) + O(1)..."
  },
  "optimizations": [
    "Use memoization to reduce time complexity to O(n)",
    "Consider iterative approach for O(1) space complexity"
  ],
  "reasoning_steps": "First, I identified this as a classic recursion problem...",
  "confidence": 0.95,
  "test_results": {
    "success": true,
    "execution_time": 0.001,
    "all_tests_passed": true
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**cURL Example:**
```bash
curl -X POST "https://your-deployment.vercel.app/generate" \
  -H "Authorization: Bearer autonomous-ai-agent-2024" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a function to implement binary search",
    "language": "python",
    "context": "Include error handling and edge cases"
  }'
```

**Error Responses:**
- `400`: Invalid prompt or parameters
- `500`: Code generation failed

---

### POST `/search`

Perform deep web search across multiple sources.

**Request Body:**
```json
{
  "query": "string",
  "depth": "integer",
  "include_code": "boolean"
}
```

**Parameters:**
- `query` (required): Search query
- `depth` (optional): Number of results per source (default: 5, max: 20)
- `include_code` (optional): Include code examples (default: true)

**Response:**
```json
{
  "results": [
    {
      "title": "How to implement async programming in Python",
      "url": "https://stackoverflow.com/questions/...",
      "snippet": "Asyncio is Python's library for writing concurrent code...",
      "source": "stackoverflow",
      "relevance_score": 0.92,
      "content_type": "qa",
      "has_code": true,
      "tags": ["python", "asyncio", "concurrency"],
      "is_answered": true,
      "answer_count": 3
    }
  ],
  "synthesized_answer": "Based on multiple authoritative sources, async programming in Python...",
  "code_examples": [
    {
      "code": "import asyncio\n\nasync def main():\n    await asyncio.sleep(1)",
      "description": "Basic async function example",
      "source": "stackoverflow",
      "language": "python"
    }
  ],
  "sources": [
    "stackoverflow.com",
    "github.com", 
    "docs.python.org"
  ],
  "confidence": 0.87,
  "related_queries": [
    "asyncio vs threading python",
    "async await best practices",
    "python concurrency patterns"
  ],
  "search_plan": "Searched for async programming concepts across technical sources...",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**cURL Example:**
```bash
curl -X POST "https://your-deployment.vercel.app/search" \
  -H "Authorization: Bearer autonomous-ai-agent-2024" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning optimization algorithms",
    "depth": 10,
    "include_code": true
  }'
```

**Error Responses:**
- `400`: Invalid search parameters
- `500`: Search service unavailable

---

### POST `/reason`

Perform step-by-step reasoning on complex problems.

**Request Body:**
```json
{
  "problem": "string",
  "domain": "string",
  "include_math": "boolean"
}
```

**Parameters:**
- `problem` (required): Problem description
- `domain` (optional): Problem domain (default: "coding")
- `include_math` (optional): Include mathematical analysis (default: true)

**Response:**
```json
{
  "reasoning_steps": [
    {
      "step": 1,
      "description": "Problem Analysis",
      "content": "The problem asks for an efficient algorithm to find..."
    },
    {
      "step": 2,
      "description": "Mathematical Formulation", 
      "content": "We can model this as a graph traversal problem where..."
    },
    {
      "step": 3,
      "description": "Solution Development",
      "content": "The optimal approach uses Dijkstra's algorithm..."
    },
    {
      "step": 4,
      "description": "Validation",
      "content": "This solution handles all edge cases including..."
    }
  ],
  "solution": "Implement Dijkstra's algorithm with a priority queue...",
  "mathematical_analysis": {
    "complexity": {
      "time": "O((V + E) log V)",
      "space": "O(V)",
      "proof": "Priority queue operations are O(log V)..."
    },
    "correctness_proof": "The algorithm maintains the invariant that...",
    "optimality": "This is optimal for single-source shortest path problems"
  },
  "alternative_approaches": [
    {
      "name": "Bellman-Ford Algorithm",
      "complexity": "O(VE)",
      "description": "Handles negative edge weights",
      "trade_offs": "Slower but more general"
    },
    {
      "name": "A* Search",
      "complexity": "O(b^d)",
      "description": "Heuristic-guided search",
      "trade_offs": "Faster with good heuristic"
    }
  ],
  "confidence": 0.93,
  "complexity_analysis": {
    "time_complexity": "O((V + E) log V)",
    "space_complexity": "O(V)",
    "optimality": "Optimal for non-negative weights"
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**cURL Example:**
```bash
curl -X POST "https://your-deployment.vercel.app/reason" \
  -H "Authorization: Bearer autonomous-ai-agent-2024" \
  -H "Content-Type: application/json" \
  -d '{
    "problem": "Design an algorithm to find the shortest path in a weighted graph",
    "domain": "algorithms",
    "include_math": true
  }'
```

**Error Responses:**
- `400`: Invalid problem description
- `500`: Reasoning engine failed

---

### POST `/train`

Trigger self-training of the agent.

**Request Body:**
```json
{
  "dataset_name": "string",
  "training_type": "string",
  "iterations": "integer"
}
```

**Parameters:**
- `dataset_name` (optional): Hugging Face dataset name or null for interaction data
- `training_type` (optional): Training method ("rlhf", "fine_tune", "continuous") (default: "rlhf")
- `iterations` (optional): Number of training iterations (default: 10)

**Response:**
```json
{
  "status": "training_started",
  "training_id": "train_20240101_123456",
  "training_type": "rlhf",
  "iterations": 10,
  "dataset": "codeparrot/github-code",
  "estimated_duration": "20 minutes",
  "current_model_version": 5,
  "target_model_version": 6,
  "training_config": {
    "learning_rate": 5e-5,
    "batch_size": 2,
    "lora_rank": 8,
    "lora_alpha": 32
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Available Datasets:**
- `codeparrot/github-code`: Python code from GitHub
- `roneneldan/TinyStories`: Simple text for language modeling
- `null`: Use recent interaction data

**Training Types:**
- `rlhf`: Reinforcement Learning from Human Feedback
- `fine_tune`: Standard supervised fine-tuning
- `continuous`: Continuous learning from interactions

**cURL Example:**
```bash
curl -X POST "https://your-deployment.vercel.app/train" \
  -H "Authorization: Bearer autonomous-ai-agent-2024" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_name": "codeparrot/github-code",
    "training_type": "rlhf",
    "iterations": 5
  }'
```

**Error Responses:**
- `400`: Invalid training parameters
- `500`: Training initialization failed

---

### GET `/status`

Get comprehensive agent status and metrics.

**Response:**
```json
{
  "agent_status": {
    "model_loaded": true,
    "components_initialized": true,
    "performance_metrics": {
      "code_generation_success_rate": 0.94,
      "search_relevance_score": 0.87,
      "reasoning_accuracy": 0.91,
      "training_iterations": 15
    },
    "interaction_count": 1250
  },
  "memory_usage": {
    "total_memories": {
      "episodic": 856,
      "semantic": 234,
      "code": 412
    },
    "memory_utilization": 0.67,
    "recent_activity": {
      "last_24h": 45,
      "last_7d": 203
    }
  },
  "model_info": {
    "model_name": "distilgpt2",
    "current_version": 6,
    "quantization": "fp32",
    "parameter_count": "82M",
    "lora_parameters": "294K"
  },
  "training_history": [
    {
      "version": 6,
      "timestamp": "2024-01-01T10:00:00Z",
      "training_type": "rlhf",
      "performance_improvement": 0.08,
      "training_duration": 1200
    }
  ],
  "capabilities": [
    "Code generation",
    "Deep web search", 
    "Mathematical reasoning",
    "Self-training",
    "Step-by-step reasoning"
  ],
  "health_metrics": {
    "uptime": 7200,
    "avg_response_time": 2.3,
    "error_rate": 0.02,
    "cache_hit_rate": 0.76
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**cURL Example:**
```bash
curl -X GET "https://your-deployment.vercel.app/status" \
  -H "Authorization: Bearer autonomous-ai-agent-2024"
```

---

### POST `/feedback`

Provide feedback for agent improvement.

**Request Body:**
```json
{
  "interaction_id": "string",
  "rating": "integer",
  "comments": "string",
  "specific_feedback": "object"
}
```

**Parameters:**
- `interaction_id` (optional): ID of the interaction being rated
- `rating` (required): Rating from 1-5
- `comments` (optional): Text feedback
- `specific_feedback` (optional): Structured feedback object

**Response:**
```json
{
  "status": "feedback_received",
  "feedback_id": "fb_20240101_123456",
  "message": "Thank you for the feedback. The agent will use this to improve.",
  "processing_status": "queued_for_analysis",
  "estimated_impact": "medium",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Specific Feedback Structure:**
```json
{
  "code_quality": "high|medium|low",
  "explanation_clarity": "very_clear|clear|unclear|confusing",
  "usefulness": "very_useful|useful|somewhat_useful|not_useful",
  "accuracy": "accurate|mostly_accurate|somewhat_accurate|inaccurate",
  "completeness": "complete|mostly_complete|incomplete|very_incomplete",
  "categories": ["correctness", "performance", "readability", "documentation"]
}
```

**cURL Example:**
```bash
curl -X POST "https://your-deployment.vercel.app/feedback" \
  -H "Authorization: Bearer autonomous-ai-agent-2024" \
  -H "Content-Type: application/json" \
  -d '{
    "rating": 5,
    "comments": "Excellent code generation with clear explanations",
    "specific_feedback": {
      "code_quality": "high",
      "explanation_clarity": "very_clear",
      "usefulness": "very_useful",
      "accuracy": "accurate"
    }
  }'
```

---

## WebSocket Endpoints

### WS `/ws/training`

Real-time training progress updates.

**Connection:**
```javascript
const ws = new WebSocket('wss://your-deployment.vercel.app/ws/training?token=your-token');
```

**Message Format:**
```json
{
  "type": "training_update",
  "training_id": "train_20240101_123456",
  "progress": 0.65,
  "current_step": 650,
  "total_steps": 1000,
  "current_loss": 2.34,
  "estimated_time_remaining": 300,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

**Message Types:**
- `training_started`: Training initiated
- `training_update`: Progress update
- `training_completed`: Training finished
- `training_failed`: Training error
- `model_updated`: New model version deployed

---

## SDK Examples

### Python SDK

```python
import requests
import json

class AutonomousAgentClient:
    def __init__(self, base_url: str, token: str):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
    
    def generate_code(self, prompt: str, language: str = "python", context: str = None):
        """Generate code with analysis"""
        response = requests.post(
            f"{self.base_url}/generate",
            headers=self.headers,
            json={
                "prompt": prompt,
                "language": language,
                "context": context
            }
        )
        response.raise_for_status()
        return response.json()
    
    def search(self, query: str, depth: int = 5, include_code: bool = True):
        """Perform deep web search"""
        response = requests.post(
            f"{self.base_url}/search",
            headers=self.headers,
            json={
                "query": query,
                "depth": depth,
                "include_code": include_code
            }
        )
        response.raise_for_status()
        return response.json()
    
    def reason(self, problem: str, domain: str = "coding", include_math: bool = True):
        """Step-by-step reasoning"""
        response = requests.post(
            f"{self.base_url}/reason",
            headers=self.headers,
            json={
                "problem": problem,
                "domain": domain,
                "include_math": include_math
            }
        )
        response.raise_for_status()
        return response.json()
    
    def train(self, dataset_name: str = None, training_type: str = "rlhf", iterations: int = 10):
        """Trigger training"""
        response = requests.post(
            f"{self.base_url}/train",
            headers=self.headers,
            json={
                "dataset_name": dataset_name,
                "training_type": training_type,
                "iterations": iterations
            }
        )
        response.raise_for_status()
        return response.json()
    
    def get_status(self):
        """Get agent status"""
        response = requests.get(
            f"{self.base_url}/status",
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def provide_feedback(self, rating: int, comments: str = None, specific_feedback: dict = None):
        """Provide feedback"""
        response = requests.post(
            f"{self.base_url}/feedback",
            headers=self.headers,
            json={
                "rating": rating,
                "comments": comments,
                "specific_feedback": specific_feedback
            }
        )
        response.raise_for_status()
        return response.json()

# Usage example
client = AutonomousAgentClient(
    base_url="https://your-deployment.vercel.app",
    token="autonomous-ai-agent-2024"
)

# Generate code
result = client.generate_code("Write a function to implement quicksort")
print(f"Generated code:\n{result['code']}")

# Search for information
search_result = client.search("Python async programming best practices")
print(f"Search summary: {search_result['synthesized_answer']}")

# Reasoning
reasoning_result = client.reason("Optimize database query performance")
print(f"Solution: {reasoning_result['solution']}")
```

### JavaScript SDK

```javascript
class AutonomousAgentClient {
    constructor(baseUrl, token) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.headers = {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
        };
    }

    async generateCode(prompt, language = 'python', context = null) {
        const response = await fetch(`${this.baseUrl}/generate`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({
                prompt,
                language,
                context
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        }
        
        return await response.json();
    }

    async search(query, depth = 5, includeCode = true) {
        const response = await fetch(`${this.baseUrl}/search`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({
                query,
                depth,
                include_code: includeCode
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        }
        
        return await response.json();
    }

    async reason(problem, domain = 'coding', includeMath = true) {
        const response = await fetch(`${this.baseUrl}/reason`, {
            method: 'POST',
            headers: this.headers,
            body: JSON.stringify({
                problem,
                domain,
                include_math: includeMath
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        }
        
        return await response.json();
    }

    async getStatus() {
        const response = await fetch(`${this.baseUrl}/status`, {
            method: 'GET',
            headers: this.headers
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${await response.text()}`);
        }
        
        return await response.json();
    }
}

// Usage example
const client = new AutonomousAgentClient(
    'https://your-deployment.vercel.app',
    'autonomous-ai-agent-2024'
);

// Generate code
client.generateCode('Write a function to reverse a string')
    .then(result => {
        console.log('Generated code:', result.code);
        console.log('Explanation:', result.explanation);
    })
    .catch(error => console.error('Error:', error));
```

---

## Webhooks

### Training Completion Webhook

When training completes, the agent can send a webhook to a specified URL.

**Configuration:**
```json
{
  "webhook_url": "https://your-app.com/webhook/training-complete",
  "webhook_secret": "your-secret-key",
  "events": ["training_completed", "model_deployed"]
}
```

**Webhook Payload:**
```json
{
  "event": "training_completed",
  "training_id": "train_20240101_123456",
  "model_version": 6,
  "training_metrics": {
    "final_loss": 1.23,
    "training_duration": 1200,
    "performance_improvement": 0.08
  },
  "timestamp": "2024-01-01T00:00:00Z",
  "signature": "sha256=abcdef123456..."
}
```

---

## OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:
```
https://your-deployment.vercel.app/docs
```

Interactive documentation is available at:
```
https://your-deployment.vercel.app/redoc
```

---

This API reference provides comprehensive documentation for integrating with the Autonomous AI Agent. For additional examples and advanced usage patterns, refer to the [Usage Guide](usage.md) and [Architecture Documentation](architecture.md).
