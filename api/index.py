"""
Minimal API entry point for Vercel deployment - Debug Version
"""

from http.server import BaseHTTPRequestHandler
import json
from datetime import datetime
from urllib.parse import urlparse

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        """Handle GET requests"""
        try:
            parsed_path = urlparse(self.path)
            path = parsed_path.path
            
            if path == '/':
                response = {
                    "message": "Autonomous AI Agent API - Working!",
                    "status": "active",
                    "version": "2.0.0",
                    "timestamp": datetime.now().isoformat()
                }
            elif path == '/health':
                response = {
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat()
                }
            elif path == '/status':
                response = {
                    "api_status": "active",
                    "available_features": ["basic_api"],
                    "timestamp": datetime.now().isoformat()
                }
            else:
                response = {"error": "Endpoint not found"}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            self.end_headers()
            
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def do_POST(self):
        """Handle POST requests"""
        try:
            parsed_path = urlparse(self.path)
            path = parsed_path.path
            
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data.decode('utf-8'))
            
            if path == '/generate':
                response = self._handle_code_generation(request_data)
            elif path == '/search':
                response = self._handle_search(request_data)
            elif path == '/reason':
                response = self._handle_reasoning(request_data)
            else:
                response = {"error": "Endpoint not found"}
            
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
            self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
            self.end_headers()
            
            self.wfile.write(json.dumps(response).encode())
            
        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')
        self.end_headers()
    
    def _handle_code_generation(self, request_data):
        """Handle code generation requests"""
        prompt = request_data.get('prompt', '')
        language = request_data.get('language', 'python')
        
        # Template-based code generation
        templates = {
            "fibonacci": {
                "python": """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Example usage
print(fibonacci(10))""",
                "javascript": """function fibonacci(n) {
    if (n <= 1) return n;
    return fibonacci(n-1) + fibonacci(n-2);
}

console.log(fibonacci(10));"""
            }
        }
        
        prompt_lower = prompt.lower()
        
        for template_name, template_code in templates.items():
            if template_name in prompt_lower:
                code = template_code.get(language, template_code.get("python", ""))
                return {
                    "code": code,
                    "explanation": f"Template-based {language} implementation for {template_name}",
                    "confidence": 0.8,
                    "method": "template",
                    "timestamp": datetime.now().isoformat()
                }
        
        # Default template
        if language == "python":
            code = f"""# {prompt}

def solve_problem():
    '''
    Solution for: {prompt}
    '''
    # TODO: Implement your solution here
    pass

if __name__ == "__main__":
    solve_problem()"""
        else:
            code = f"// {prompt}\n// TODO: Implement solution in {language}"
        
        return {
            "code": code,
            "explanation": f"Basic template for {language}",
            "confidence": 0.6,
            "method": "basic_template",
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_search(self, request_data):
        """Handle search requests"""
        query = request_data.get('query', '')
        
        # Simple mock search response for now
        return {
            "results": [
                {
                    "title": "Search Result",
                    "snippet": f"Results for: {query}",
                    "url": "https://example.com"
                }
            ],
            "synthesized_answer": f"Search completed for: {query}",
            "confidence": 0.7,
            "method": "mock_search",
            "timestamp": datetime.now().isoformat()
        }
    
    def _handle_reasoning(self, request_data):
        """Handle reasoning requests"""
        problem = request_data.get('problem', '')
        domain = request_data.get('domain', 'coding')
        
        reasoning_steps = [
            {
                "step": 1,
                "description": "Problem Analysis",
                "content": f"Analyzing the {domain} problem: {problem}"
            },
            {
                "step": 2,
                "description": "Solution Planning", 
                "content": "Breaking down the problem into manageable components"
            },
            {
                "step": 3,
                "description": "Implementation Strategy",
                "content": f"Determining approach for this {domain} challenge"
            }
        ]
        
        solution = f"""
Basic approach for: {problem}

1. Understand the requirements
2. Design the solution
3. Implement step by step
4. Test and validate
5. Optimize if needed

This {domain} problem requires systematic analysis.
"""
        
        return {
            "reasoning_steps": reasoning_steps,
            "solution": solution.strip(),
            "confidence": 0.6,
            "method": "basic_reasoning",
            "timestamp": datetime.now().isoformat()
        }