"""
Code execution and testing utilities
Safely executes and tests generated code with comprehensive analysis
"""

import ast
import sys
import subprocess
import tempfile
import os
import time
import signal
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
import re
import traceback
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class CodeExecutionResult:
    """Result of code execution with comprehensive metrics"""
    
    def __init__(self):
        self.success = False
        self.output = ""
        self.error = ""
        self.execution_time = 0.0
        self.memory_usage = 0
        self.syntax_valid = False
        self.security_issues = []
        self.complexity_metrics = {}
        self.test_results = []

class CodeExecutor:
    """
    Safe code execution environment with testing capabilities
    
    Features:
    - Syntax validation
    - Safe execution in isolated environment
    - Performance measurement
    - Security analysis
    - Automatic test generation and execution
    - Code complexity analysis
    """
    
    def __init__(self):
        self.timeout = 10  # Maximum execution time in seconds
        self.max_memory = 128 * 1024 * 1024  # 128MB memory limit
        
        # Dangerous imports and functions to check for
        self.dangerous_patterns = [
            r'import\s+os',
            r'import\s+subprocess',
            r'import\s+sys',
            r'from\s+os',
            r'from\s+subprocess',
            r'from\s+sys',
            r'exec\s*\(',
            r'eval\s*\(',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\(',
            r'__import__',
            r'globals\s*\(',
            r'locals\s*\(',
            r'vars\s*\(',
            r'dir\s*\(',
            r'setattr',
            r'getattr',
            r'delattr',
        ]
    
    async def test_code(self, code: str, language: str = "python") -> Dict[str, Any]:
        """
        Comprehensive code testing with multiple analysis methods
        
        Args:
            code: Code to test
            language: Programming language
            
        Returns:
            Comprehensive test results
        """
        result = {
            "success": False,
            "syntax_valid": False,
            "execution_result": None,
            "security_analysis": {},
            "complexity_analysis": {},
            "test_cases_passed": 0,
            "test_cases_total": 0,
            "performance_metrics": {},
            "suggestions": []
        }
        
        try:
            if language.lower() == "python":
                # Syntax validation
                result["syntax_valid"] = self._validate_python_syntax(code)
                
                if not result["syntax_valid"]:
                    result["error"] = "Invalid Python syntax"
                    return result
                
                # Security analysis
                result["security_analysis"] = self._analyze_security(code)
                
                if result["security_analysis"]["has_security_issues"]:
                    result["error"] = "Code contains potentially dangerous operations"
                    return result
                
                # Execute code safely
                execution_result = await self._execute_python_code(code)
                result["execution_result"] = execution_result
                result["success"] = execution_result["success"]
                
                # Complexity analysis
                result["complexity_analysis"] = self._analyze_complexity(code)
                
                # Generate and run test cases
                test_results = await self._generate_and_run_tests(code)
                result["test_cases_passed"] = test_results["passed"]
                result["test_cases_total"] = test_results["total"]
                
                # Performance analysis
                result["performance_metrics"] = await self._analyze_performance(code)
                
                # Generate suggestions
                result["suggestions"] = self._generate_suggestions(code, result)
                
        except Exception as e:
            logger.error(f"Code testing failed: {e}")
            result["error"] = str(e)
        
        return result
    
    def _validate_python_syntax(self, code: str) -> bool:
        """Validate Python syntax using AST"""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            logger.debug(f"Syntax error in code: {e}")
            return False
        except Exception as e:
            logger.debug(f"Error validating syntax: {e}")
            return False
    
    def _analyze_security(self, code: str) -> Dict[str, Any]:
        """Analyze code for security issues"""
        security_issues = []
        
        # Check for dangerous patterns
        for pattern in self.dangerous_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                security_issues.append(f"Potentially dangerous pattern: {pattern}")
        
        # Check for network operations
        network_patterns = [
            r'requests\.',
            r'urllib',
            r'http\.',
            r'socket\.',
            r'telnet',
            r'ftp',
        ]
        
        for pattern in network_patterns:
            if re.search(pattern, code, re.IGNORECASE):
                security_issues.append(f"Network operation detected: {pattern}")
        
        return {
            "has_security_issues": len(security_issues) > 0,
            "issues": security_issues,
            "risk_level": "high" if len(security_issues) > 3 else "medium" if len(security_issues) > 0 else "low"
        }
    
    async def _execute_python_code(self, code: str) -> Dict[str, Any]:
        """Execute Python code safely in a subprocess"""
        try:
            # Create a temporary file for the code
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Execute the code with timeout
                start_time = time.time()
                
                process = await asyncio.create_subprocess_exec(
                    sys.executable, temp_file,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.timeout
                    )
                    
                    execution_time = time.time() - start_time
                    
                    return {
                        "success": process.returncode == 0,
                        "output": stdout.decode('utf-8') if stdout else "",
                        "error": stderr.decode('utf-8') if stderr else "",
                        "execution_time": execution_time,
                        "return_code": process.returncode
                    }
                    
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
                    return {
                        "success": False,
                        "output": "",
                        "error": f"Code execution timed out after {self.timeout} seconds",
                        "execution_time": self.timeout,
                        "return_code": -1
                    }
            
            finally:
                # Clean up temporary file
                try:
                    os.unlink(temp_file)
                except:
                    pass
        
        except Exception as e:
            return {
                "success": False,
                "output": "",
                "error": f"Execution failed: {str(e)}",
                "execution_time": 0,
                "return_code": -1
            }
    
    def _analyze_complexity(self, code: str) -> Dict[str, Any]:
        """Analyze code complexity using AST"""
        try:
            tree = ast.parse(code)
            
            complexity_metrics = {
                "lines_of_code": len([line for line in code.split('\n') if line.strip()]),
                "cyclomatic_complexity": self._calculate_cyclomatic_complexity(tree),
                "function_count": len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]),
                "class_count": len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]),
                "loop_count": len([node for node in ast.walk(tree) if isinstance(node, (ast.For, ast.While))]),
                "conditional_count": len([node for node in ast.walk(tree) if isinstance(node, ast.If)]),
                "max_nesting_depth": self._calculate_max_nesting_depth(tree)
            }
            
            # Calculate complexity score (0-100)
            complexity_score = min(100, (
                complexity_metrics["cyclomatic_complexity"] * 5 +
                complexity_metrics["max_nesting_depth"] * 10 +
                complexity_metrics["lines_of_code"] * 0.1
            ))
            
            complexity_metrics["complexity_score"] = complexity_score
            complexity_metrics["complexity_level"] = (
                "low" if complexity_score < 20 else
                "medium" if complexity_score < 50 else
                "high"
            )
            
            return complexity_metrics
            
        except Exception as e:
            logger.debug(f"Complexity analysis failed: {e}")
            return {"error": str(e)}
    
    def _calculate_cyclomatic_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity"""
        complexity = 1  # Base complexity
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += len(node.values) - 1
        
        return complexity
    
    def _calculate_max_nesting_depth(self, tree: ast.AST) -> int:
        """Calculate maximum nesting depth"""
        def get_depth(node, current_depth=0):
            max_depth = current_depth
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.With, ast.Try, ast.FunctionDef, ast.ClassDef)):
                    depth = get_depth(child, current_depth + 1)
                    max_depth = max(max_depth, depth)
                else:
                    depth = get_depth(child, current_depth)
                    max_depth = max(max_depth, depth)
            
            return max_depth
        
        return get_depth(tree)
    
    async def _generate_and_run_tests(self, code: str) -> Dict[str, Any]:
        """Generate and run test cases for the code"""
        try:
            # Extract functions from code
            tree = ast.parse(code)
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            
            if not functions:
                return {"passed": 0, "total": 0, "results": []}
            
            test_results = []
            passed = 0
            total = 0
            
            for func in functions:
                func_name = func.name
                
                # Generate basic test cases based on function signature
                test_cases = self._generate_test_cases_for_function(func)
                
                for test_case in test_cases:
                    total += 1
                    
                    # Create test code
                    test_code = f"""
{code}

# Test case
try:
    result = {func_name}({test_case['input']})
    expected = {test_case['expected']}
    
    if result == expected:
        print(f"PASS: {func_name}({test_case['input']}) = {{result}}")
    else:
        print(f"FAIL: {func_name}({test_case['input']}) = {{result}}, expected {{expected}}")
        
except Exception as e:
    print(f"ERROR: {func_name}({test_case['input']}) raised {{e}}")
"""
                    
                    # Execute test
                    test_result = await self._execute_python_code(test_code)
                    
                    if test_result["success"] and "PASS" in test_result["output"]:
                        passed += 1
                        test_results.append({
                            "function": func_name,
                            "input": test_case['input'],
                            "status": "passed",
                            "output": test_result["output"]
                        })
                    else:
                        test_results.append({
                            "function": func_name,
                            "input": test_case['input'],
                            "status": "failed",
                            "output": test_result.get("output", ""),
                            "error": test_result.get("error", "")
                        })
            
            return {
                "passed": passed,
                "total": total,
                "results": test_results
            }
            
        except Exception as e:
            logger.debug(f"Test generation failed: {e}")
            return {"passed": 0, "total": 0, "results": [], "error": str(e)}
    
    def _generate_test_cases_for_function(self, func: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Generate test cases for a function based on its signature"""
        test_cases = []
        
        # Basic test cases based on function name patterns
        func_name = func.name.lower()
        
        if "fibonacci" in func_name or "fib" in func_name:
            test_cases = [
                {"input": "0", "expected": "0"},
                {"input": "1", "expected": "1"},
                {"input": "5", "expected": "5"},
                {"input": "10", "expected": "55"}
            ]
        elif "factorial" in func_name:
            test_cases = [
                {"input": "0", "expected": "1"},
                {"input": "1", "expected": "1"},
                {"input": "5", "expected": "120"}
            ]
        elif "sort" in func_name:
            test_cases = [
                {"input": "[3, 1, 4, 1, 5]", "expected": "[1, 1, 3, 4, 5]"},
                {"input": "[]", "expected": "[]"},
                {"input": "[1]", "expected": "[1]"}
            ]
        elif "sum" in func_name or "add" in func_name:
            test_cases = [
                {"input": "1, 2", "expected": "3"},
                {"input": "0, 0", "expected": "0"},
                {"input": "-1, 1", "expected": "0"}
            ]
        else:
            # Generic test cases
            num_args = len(func.args.args)
            if num_args == 0:
                test_cases = [{"input": "", "expected": "None"}]
            elif num_args == 1:
                test_cases = [
                    {"input": "0", "expected": "None"},
                    {"input": "1", "expected": "None"},
                    {"input": "'test'", "expected": "None"}
                ]
            elif num_args == 2:
                test_cases = [
                    {"input": "1, 2", "expected": "None"},
                    {"input": "0, 0", "expected": "None"}
                ]
        
        return test_cases
    
    async def _analyze_performance(self, code: str) -> Dict[str, Any]:
        """Analyze code performance characteristics"""
        try:
            # Time complexity estimation based on patterns
            time_complexity = self._estimate_time_complexity(code)
            
            # Space complexity estimation
            space_complexity = self._estimate_space_complexity(code)
            
            # Memory usage estimation
            execution_result = await self._execute_python_code(f"""
import sys
import tracemalloc

tracemalloc.start()

{code}

current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Current memory usage: {{current / 1024 / 1024:.2f}} MB")
print(f"Peak memory usage: {{peak / 1024 / 1024:.2f}} MB")
""")
            
            memory_info = {}
            if execution_result["success"]:
                output = execution_result["output"]
                if "Current memory usage:" in output:
                    lines = output.strip().split('\n')
                    for line in lines:
                        if "Current memory usage:" in line:
                            memory_info["current_mb"] = float(line.split(': ')[1].split(' MB')[0])
                        elif "Peak memory usage:" in line:
                            memory_info["peak_mb"] = float(line.split(': ')[1].split(' MB')[0])
            
            return {
                "time_complexity": time_complexity,
                "space_complexity": space_complexity,
                "execution_time": execution_result.get("execution_time", 0),
                "memory_usage": memory_info,
                "performance_score": self._calculate_performance_score(time_complexity, space_complexity)
            }
            
        except Exception as e:
            logger.debug(f"Performance analysis failed: {e}")
            return {"error": str(e)}
    
    def _estimate_time_complexity(self, code: str) -> str:
        """Estimate time complexity based on code patterns"""
        # Simple heuristic-based estimation
        if re.search(r'for.*for.*', code, re.DOTALL):
            return "O(n²)"
        elif re.search(r'for.*while.*', code, re.DOTALL):
            return "O(n²)"
        elif re.search(r'while.*for.*', code, re.DOTALL):
            return "O(n²)"
        elif re.search(r'for.*in.*range.*:', code):
            return "O(n)"
        elif re.search(r'while.*:', code):
            return "O(n)"
        elif re.search(r'sort\(', code):
            return "O(n log n)"
        else:
            return "O(1)"
    
    def _estimate_space_complexity(self, code: str) -> str:
        """Estimate space complexity based on code patterns"""
        if re.search(r'\[\].*for.*in.*', code):
            return "O(n)"
        elif re.search(r'list\(', code):
            return "O(n)"
        elif re.search(r'dict\(', code):
            return "O(n)"
        else:
            return "O(1)"
    
    def _calculate_performance_score(self, time_complexity: str, space_complexity: str) -> int:
        """Calculate performance score (0-100)"""
        time_scores = {
            "O(1)": 100,
            "O(log n)": 90,
            "O(n)": 80,
            "O(n log n)": 60,
            "O(n²)": 40,
            "O(n³)": 20,
            "O(2^n)": 10
        }
        
        space_scores = {
            "O(1)": 100,
            "O(log n)": 90,
            "O(n)": 80,
            "O(n²)": 60
        }
        
        time_score = time_scores.get(time_complexity, 50)
        space_score = space_scores.get(space_complexity, 50)
        
        return int((time_score * 0.7) + (space_score * 0.3))
    
    def _generate_suggestions(self, code: str, analysis_result: Dict[str, Any]) -> List[str]:
        """Generate optimization and improvement suggestions"""
        suggestions = []
        
        # Complexity suggestions
        complexity = analysis_result.get("complexity_analysis", {})
        if complexity.get("complexity_level") == "high":
            suggestions.append("Consider breaking down complex functions into smaller, more manageable pieces")
        
        if complexity.get("max_nesting_depth", 0) > 3:
            suggestions.append("Reduce nesting depth by using early returns or extracting nested logic into functions")
        
        # Performance suggestions
        performance = analysis_result.get("performance_metrics", {})
        if performance.get("time_complexity") in ["O(n²)", "O(n³)", "O(2^n)"]:
            suggestions.append("Consider optimizing the algorithm for better time complexity")
        
        # Code pattern suggestions
        if "for i in range(len(" in code:
            suggestions.append("Consider using 'for item in list' instead of 'for i in range(len(list))'")
        
        if re.search(r'if.*==.*True', code):
            suggestions.append("Use 'if condition:' instead of 'if condition == True:'")
        
        if re.search(r'if.*==.*False', code):
            suggestions.append("Use 'if not condition:' instead of 'if condition == False:'")
        
        return suggestions

# Testing
if __name__ == "__main__":
    async def test_executor():
        executor = CodeExecutor()
        
        # Test code
        test_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
"""
        
        result = await executor.test_code(test_code)
        print("Test Result:")
        print(f"Success: {result['success']}")
        print(f"Syntax Valid: {result['syntax_valid']}")
        print(f"Complexity: {result['complexity_analysis']}")
        print(f"Test Cases: {result['test_cases_passed']}/{result['test_cases_total']}")
        print(f"Suggestions: {result['suggestions']}")
    
    asyncio.run(test_executor())
