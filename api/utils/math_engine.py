"""
Mathematical reasoning engine for the AI Agent
Provides calculus, statistics, and algorithm analysis capabilities
"""

import numpy as np
import scipy.stats as stats
from scipy.optimize import minimize
from scipy.integrate import quad
from scipy.optimize import approx_fprime
import sympy as sp
from typing import Dict, List, Any, Optional, Tuple, Callable
import re
import ast
import logging
import math

logger = logging.getLogger(__name__)

class MathEngine:
    """
    Mathematical reasoning and analysis engine
    
    Features:
    - Calculus operations (derivatives, integrals)
    - Statistical analysis and hypothesis testing
    - Algorithm complexity analysis
    - Probability calculations
    - Optimization problems
    - Neural network math (backpropagation, gradients)
    """
    
    def __init__(self):
        self.symbols = {}
        self.functions = {}
        
        # Common mathematical constants
        self.constants = {
            'pi': math.pi,
            'e': math.e,
            'golden_ratio': (1 + math.sqrt(5)) / 2,
            'euler_gamma': 0.5772156649015329
        }
    
    async def analyze_algorithm(self, code: str) -> Dict[str, Any]:
        """
        Comprehensive mathematical analysis of an algorithm
        
        Args:
            code: Algorithm code to analyze
            
        Returns:
            Mathematical analysis including complexity, optimization insights
        """
        try:
            analysis = {
                "time_complexity": self._analyze_time_complexity(code),
                "space_complexity": self._analyze_space_complexity(code),
                "mathematical_operations": self._identify_math_operations(code),
                "optimization_opportunities": self._find_optimization_opportunities(code),
                "numerical_stability": self._analyze_numerical_stability(code),
                "algorithmic_insights": self._generate_algorithmic_insights(code)
            }
            
            # Add complexity score
            analysis["complexity_score"] = self._calculate_complexity_score(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Algorithm analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_time_complexity(self, code: str) -> Dict[str, Any]:
        """Analyze time complexity with mathematical justification"""
        complexity_analysis = {
            "estimated_complexity": "O(1)",
            "justification": "",
            "detailed_analysis": [],
            "mathematical_proof": ""
        }
        
        try:
            # Parse the code to analyze loops and recursion
            tree = ast.parse(code)
            
            # Count nested loops
            max_loop_depth = self._calculate_max_loop_depth(tree)
            
            # Check for recursion
            has_recursion = self._has_recursion(tree)
            
            # Analyze specific patterns
            if has_recursion:
                if "fibonacci" in code.lower():
                    complexity_analysis.update({
                        "estimated_complexity": "O(2^n)",
                        "justification": "Fibonacci recursion creates exponential branching",
                        "mathematical_proof": "T(n) = T(n-1) + T(n-2) + O(1), solving recurrence gives O(φ^n) where φ ≈ 1.618"
                    })
                elif "factorial" in code.lower():
                    complexity_analysis.update({
                        "estimated_complexity": "O(n)",
                        "justification": "Linear recursion with single recursive call",
                        "mathematical_proof": "T(n) = T(n-1) + O(1) = O(n)"
                    })
                else:
                    complexity_analysis.update({
                        "estimated_complexity": "O(n)",
                        "justification": "General recursive pattern detected"
                    })
            
            elif max_loop_depth == 1:
                complexity_analysis.update({
                    "estimated_complexity": "O(n)",
                    "justification": "Single loop iteration over input",
                    "mathematical_proof": "Single loop: Σ(i=1 to n) 1 = n = O(n)"
                })
            
            elif max_loop_depth == 2:
                complexity_analysis.update({
                    "estimated_complexity": "O(n²)",
                    "justification": "Nested loops create quadratic growth",
                    "mathematical_proof": "Nested loops: Σ(i=1 to n) Σ(j=1 to n) 1 = n² = O(n²)"
                })
            
            elif max_loop_depth >= 3:
                complexity_analysis.update({
                    "estimated_complexity": f"O(n^{max_loop_depth})",
                    "justification": f"Deep nesting ({max_loop_depth} levels) creates polynomial growth",
                    "mathematical_proof": f"{max_loop_depth} nested loops = O(n^{max_loop_depth})"
                })
            
            # Check for sorting algorithms
            if re.search(r'\.sort\(\)|sorted\(', code):
                complexity_analysis.update({
                    "estimated_complexity": "O(n log n)",
                    "justification": "Built-in sorting algorithms use optimal comparison-based sorting",
                    "mathematical_proof": "Optimal comparison sorting: Ω(n log n) lower bound"
                })
            
            return complexity_analysis
            
        except Exception as e:
            logger.debug(f"Time complexity analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_space_complexity(self, code: str) -> Dict[str, Any]:
        """Analyze space complexity with mathematical justification"""
        space_analysis = {
            "estimated_complexity": "O(1)",
            "justification": "",
            "memory_allocations": [],
            "recursive_stack_depth": 0
        }
        
        try:
            # Check for list comprehensions and data structure allocations
            if re.search(r'\[.*for.*in.*\]', code):
                space_analysis.update({
                    "estimated_complexity": "O(n)",
                    "justification": "List comprehension creates new list of size n"
                })
            
            # Check for recursive functions (stack space)
            if self._has_recursion(ast.parse(code)):
                max_recursion_depth = self._estimate_recursion_depth(code)
                space_analysis.update({
                    "estimated_complexity": f"O({max_recursion_depth})",
                    "justification": f"Recursive calls use stack space up to depth {max_recursion_depth}",
                    "recursive_stack_depth": max_recursion_depth
                })
            
            # Check for data structure creation
            data_structures = re.findall(r'(list|dict|set)\(', code)
            if data_structures:
                space_analysis["memory_allocations"] = data_structures
                if len(data_structures) > 0:
                    space_analysis.update({
                        "estimated_complexity": "O(n)",
                        "justification": f"Creates {len(data_structures)} data structure(s)"
                    })
            
            return space_analysis
            
        except Exception as e:
            logger.debug(f"Space complexity analysis failed: {e}")
            return {"error": str(e)}
    
    def _identify_math_operations(self, code: str) -> List[Dict[str, Any]]:
        """Identify mathematical operations in the code"""
        operations = []
        
        # Arithmetic operations
        arithmetic_ops = {
            '+': 'addition',
            '-': 'subtraction', 
            '*': 'multiplication',
            '/': 'division',
            '//': 'integer_division',
            '%': 'modulo',
            '**': 'exponentiation'
        }
        
        for op, name in arithmetic_ops.items():
            if op in code:
                operations.append({
                    "operation": name,
                    "symbol": op,
                    "complexity": "O(1)",
                    "mathematical_properties": self._get_operation_properties(name)
                })
        
        # Mathematical functions
        math_functions = [
            'sqrt', 'pow', 'exp', 'log', 'sin', 'cos', 'tan',
            'abs', 'min', 'max', 'sum', 'factorial'
        ]
        
        for func in math_functions:
            if func in code:
                operations.append({
                    "operation": func,
                    "type": "mathematical_function",
                    "complexity": self._get_function_complexity(func)
                })
        
        return operations
    
    def _find_optimization_opportunities(self, code: str) -> List[Dict[str, Any]]:
        """Find mathematical optimization opportunities"""
        opportunities = []
        
        # Check for expensive operations in loops
        if re.search(r'for.*:.*\*\*', code):
            opportunities.append({
                "type": "exponentiation_in_loop",
                "suggestion": "Consider precomputing or using logarithms",
                "mathematical_basis": "log(a^b) = b*log(a), potentially faster for repeated operations"
            })
        
        # Check for repeated calculations
        if code.count('sqrt(') > 1:
            opportunities.append({
                "type": "repeated_sqrt",
                "suggestion": "Cache square root calculations",
                "mathematical_basis": "√x computation can be expensive, O(log log n) with Newton's method"
            })
        
        # Check for fibonacci-like patterns
        if re.search(r'fibonacci|fib', code, re.IGNORECASE) and 'memo' not in code:
            opportunities.append({
                "type": "fibonacci_optimization",
                "suggestion": "Use memoization or dynamic programming",
                "mathematical_basis": "Reduces O(2^n) to O(n) using overlapping subproblems principle"
            })
        
        # Check for sorting opportunities
        if re.search(r'for.*for.*if.*<', code):
            opportunities.append({
                "type": "potential_sorting",
                "suggestion": "Consider using built-in sorting algorithms",
                "mathematical_basis": "Comparison-based sorting lower bound is Ω(n log n)"
            })
        
        return opportunities
    
    def _analyze_numerical_stability(self, code: str) -> Dict[str, Any]:
        """Analyze numerical stability of the algorithm"""
        stability_analysis = {
            "stability_score": 1.0,
            "potential_issues": [],
            "recommendations": []
        }
        
        # Check for division operations
        if '/' in code and 'zero' not in code.lower():
            stability_analysis["potential_issues"].append("Division by zero not explicitly handled")
            stability_analysis["stability_score"] *= 0.8
        
        # Check for floating point comparisons
        if re.search(r'==.*\d+\.\d+', code):
            stability_analysis["potential_issues"].append("Direct floating-point equality comparison")
            stability_analysis["recommendations"].append("Use absolute difference comparison: abs(a - b) < epsilon")
            stability_analysis["stability_score"] *= 0.7
        
        # Check for large number operations
        if '**' in code:
            stability_analysis["potential_issues"].append("Exponentiation may cause overflow")
            stability_analysis["recommendations"].append("Consider using logarithms for large exponentials")
            stability_analysis["stability_score"] *= 0.9
        
        return stability_analysis
    
    def _generate_algorithmic_insights(self, code: str) -> List[str]:
        """Generate algorithmic insights and mathematical connections"""
        insights = []
        
        # Pattern recognition
        if 'fibonacci' in code.lower():
            insights.append("Fibonacci sequence follows the golden ratio: F(n)/F(n-1) → φ as n→∞")
            insights.append("Closed form: F(n) = (φⁿ - ψⁿ)/√5 where φ = (1+√5)/2, ψ = (1-√5)/2")
        
        if 'factorial' in code.lower():
            insights.append("Factorial growth: n! = Γ(n+1), grows faster than exponential")
            insights.append("Stirling's approximation: n! ≈ √(2πn)(n/e)ⁿ")
        
        if re.search(r'sort|sorted', code):
            insights.append("Optimal comparison sorting requires Ω(n log n) comparisons")
            insights.append("Information theoretic lower bound: log₂(n!) ≈ n log n - n log e")
        
        if 'prime' in code.lower():
            insights.append("Prime testing: AKS algorithm is O(log⁶ n), Miller-Rabin is probabilistic O(k log³ n)")
            insights.append("Prime Number Theorem: π(n) ≈ n/ln(n)")
        
        return insights
    
    async def analyze_problem(self, problem: str) -> Dict[str, Any]:
        """Analyze a mathematical problem and provide insights"""
        try:
            analysis = {
                "problem_type": self._classify_problem_type(problem),
                "mathematical_concepts": self._identify_concepts(problem),
                "solution_approaches": self._suggest_solution_approaches(problem),
                "complexity_considerations": self._analyze_problem_complexity(problem),
                "mathematical_formulation": self._formulate_mathematically(problem)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Problem analysis failed: {e}")
            return {"error": str(e)}
    
    def calculate_derivative(self, function_str: str, variable: str = 'x') -> str:
        """Calculate symbolic derivative"""
        try:
            x = sp.Symbol(variable)
            expr = sp.sympify(function_str)
            derivative = sp.diff(expr, x)
            return str(derivative)
        except Exception as e:
            return f"Error calculating derivative: {e}"
    
    def calculate_integral(self, function_str: str, variable: str = 'x', limits: Tuple = None) -> str:
        """Calculate symbolic or definite integral"""
        try:
            x = sp.Symbol(variable)
            expr = sp.sympify(function_str)
            
            if limits:
                a, b = limits
                integral = sp.integrate(expr, (x, a, b))
            else:
                integral = sp.integrate(expr, x)
            
            return str(integral)
        except Exception as e:
            return f"Error calculating integral: {e}"
    
    def analyze_statistical_data(self, data: List[float]) -> Dict[str, Any]:
        """Comprehensive statistical analysis"""
        if not data:
            return {"error": "No data provided"}
        
        try:
            data_array = np.array(data)
            
            analysis = {
                "descriptive_stats": {
                    "mean": float(np.mean(data_array)),
                    "median": float(np.median(data_array)),
                    "mode": float(stats.mode(data_array)[0]),
                    "std_dev": float(np.std(data_array)),
                    "variance": float(np.var(data_array)),
                    "skewness": float(stats.skew(data_array)),
                    "kurtosis": float(stats.kurtosis(data_array)),
                    "range": float(np.max(data_array) - np.min(data_array)),
                    "iqr": float(np.percentile(data_array, 75) - np.percentile(data_array, 25))
                },
                "distribution_tests": {
                    "normality_test": self._test_normality(data_array),
                    "outliers": self._detect_outliers(data_array)
                },
                "confidence_intervals": {
                    "95_percent_ci": self._calculate_confidence_interval(data_array, 0.95)
                }
            }
            
            return analysis
            
        except Exception as e:
            return {"error": str(e)}
    
    def neural_network_math(self, operation: str, **kwargs) -> Dict[str, Any]:
        """Neural network mathematical operations"""
        try:
            if operation == "backpropagation":
                return self._demonstrate_backpropagation(**kwargs)
            elif operation == "gradient_descent":
                return self._demonstrate_gradient_descent(**kwargs)
            elif operation == "activation_functions":
                return self._analyze_activation_functions()
            elif operation == "loss_functions":
                return self._analyze_loss_functions()
            else:
                return {"error": f"Unknown operation: {operation}"}
                
        except Exception as e:
            return {"error": str(e)}
    
    # Helper methods
    def _calculate_max_loop_depth(self, tree: ast.AST) -> int:
        """Calculate maximum loop nesting depth"""
        def get_loop_depth(node, current_depth=0):
            max_depth = current_depth
            
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.For, ast.While)):
                    depth = get_loop_depth(child, current_depth + 1)
                    max_depth = max(max_depth, depth)
                else:
                    depth = get_loop_depth(child, current_depth)
                    max_depth = max(max_depth, depth)
            
            return max_depth
        
        return get_loop_depth(tree)
    
    def _has_recursion(self, tree: ast.AST) -> bool:
        """Check if code contains recursion"""
        function_names = set()
        calls = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_names.add(node.name)
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                calls.add(node.func.id)
        
        return bool(function_names.intersection(calls))
    
    def _estimate_recursion_depth(self, code: str) -> str:
        """Estimate maximum recursion depth"""
        if 'fibonacci' in code.lower():
            return "n"
        elif 'factorial' in code.lower():
            return "n"
        else:
            return "log n"
    
    def _get_operation_properties(self, operation: str) -> Dict[str, Any]:
        """Get mathematical properties of operations"""
        properties = {
            "addition": {
                "commutative": True,
                "associative": True,
                "identity_element": 0,
                "inverse": "subtraction"
            },
            "multiplication": {
                "commutative": True,
                "associative": True,
                "identity_element": 1,
                "inverse": "division"
            },
            "exponentiation": {
                "commutative": False,
                "associative": False,
                "identity_element": "depends on base/exponent",
                "inverse": "logarithm"
            }
        }
        
        return properties.get(operation, {})
    
    def _get_function_complexity(self, function: str) -> str:
        """Get computational complexity of mathematical functions"""
        complexities = {
            "sqrt": "O(log log n)",
            "exp": "O(log n)",
            "log": "O(log n)",
            "sin": "O(1)",
            "cos": "O(1)",
            "tan": "O(1)",
            "factorial": "O(n)",
            "sum": "O(n)",
            "min": "O(n)",
            "max": "O(n)"
        }
        
        return complexities.get(function, "O(1)")
    
    def _calculate_complexity_score(self, analysis: Dict[str, Any]) -> float:
        """Calculate overall complexity score"""
        time_complexity = analysis.get("time_complexity", {}).get("estimated_complexity", "O(1)")
        space_complexity = analysis.get("space_complexity", {}).get("estimated_complexity", "O(1)")
        
        # Score based on complexity (0-100, lower is better)
        complexity_scores = {
            "O(1)": 10,
            "O(log n)": 20,
            "O(n)": 40,
            "O(n log n)": 60,
            "O(n²)": 80,
            "O(n³)": 90,
            "O(2^n)": 100
        }
        
        time_score = complexity_scores.get(time_complexity, 50)
        space_score = complexity_scores.get(space_complexity, 25)
        
        return (time_score * 0.7) + (space_score * 0.3)
    
    def _classify_problem_type(self, problem: str) -> str:
        """Classify the type of mathematical problem"""
        problem_lower = problem.lower()
        
        if any(word in problem_lower for word in ['sort', 'order', 'arrange']):
            return "sorting"
        elif any(word in problem_lower for word in ['search', 'find', 'locate']):
            return "searching"
        elif any(word in problem_lower for word in ['optimize', 'minimize', 'maximize']):
            return "optimization"
        elif any(word in problem_lower for word in ['graph', 'tree', 'node', 'edge']):
            return "graph_theory"
        elif any(word in problem_lower for word in ['probability', 'random', 'statistics']):
            return "probability_statistics"
        elif any(word in problem_lower for word in ['integral', 'derivative', 'calculus']):
            return "calculus"
        else:
            return "general_algorithm"
    
    def _identify_concepts(self, problem: str) -> List[str]:
        """Identify mathematical concepts in the problem"""
        concepts = []
        problem_lower = problem.lower()
        
        concept_keywords = {
            "recursion": ["recursive", "recursion", "divide and conquer"],
            "dynamic_programming": ["memoization", "overlapping", "subproblems"],
            "graph_theory": ["graph", "tree", "node", "edge", "path"],
            "number_theory": ["prime", "factorial", "fibonacci", "gcd"],
            "probability": ["probability", "random", "chance", "expected"],
            "statistics": ["average", "mean", "median", "variance"],
            "calculus": ["derivative", "integral", "limit", "continuous"],
            "complexity": ["time", "space", "efficiency", "optimization"]
        }
        
        for concept, keywords in concept_keywords.items():
            if any(keyword in problem_lower for keyword in keywords):
                concepts.append(concept)
        
        return concepts
    
    def _suggest_solution_approaches(self, problem: str) -> List[Dict[str, str]]:
        """Suggest mathematical solution approaches"""
        approaches = []
        problem_type = self._classify_problem_type(problem)
        
        approach_map = {
            "sorting": [
                {"name": "Merge Sort", "complexity": "O(n log n)", "description": "Divide and conquer approach"},
                {"name": "Quick Sort", "complexity": "O(n log n) average", "description": "Partition-based sorting"},
                {"name": "Heap Sort", "complexity": "O(n log n)", "description": "Heap-based sorting"}
            ],
            "searching": [
                {"name": "Binary Search", "complexity": "O(log n)", "description": "Divide and conquer for sorted data"},
                {"name": "Hash Table", "complexity": "O(1) average", "description": "Hash-based lookup"},
                {"name": "Linear Search", "complexity": "O(n)", "description": "Sequential search"}
            ],
            "optimization": [
                {"name": "Gradient Descent", "complexity": "O(n*iterations)", "description": "Iterative optimization"},
                {"name": "Dynamic Programming", "complexity": "O(n*m)", "description": "Optimal substructure"},
                {"name": "Greedy Algorithm", "complexity": "O(n log n)", "description": "Locally optimal choices"}
            ]
        }
        
        return approach_map.get(problem_type, [])
    
    def _analyze_problem_complexity(self, problem: str) -> Dict[str, str]:
        """Analyze the complexity characteristics of a problem"""
        problem_type = self._classify_problem_type(problem)
        
        complexity_analysis = {
            "sorting": {
                "lower_bound": "Ω(n log n)",
                "explanation": "Comparison-based sorting information-theoretic lower bound"
            },
            "searching": {
                "lower_bound": "Ω(log n)",
                "explanation": "Binary search in sorted array is optimal"
            },
            "optimization": {
                "lower_bound": "Problem-dependent",
                "explanation": "Depends on problem structure and constraints"
            }
        }
        
        return complexity_analysis.get(problem_type, {
            "lower_bound": "Unknown",
            "explanation": "Problem type not recognized"
        })
    
    def _formulate_mathematically(self, problem: str) -> str:
        """Provide mathematical formulation of the problem"""
        problem_type = self._classify_problem_type(problem)
        
        formulations = {
            "sorting": "Given array A[1..n], find permutation π such that A[π(1)] ≤ A[π(2)] ≤ ... ≤ A[π(n)]",
            "searching": "Given sorted array A[1..n] and target x, find index i such that A[i] = x",
            "optimization": "Find x* = argmin f(x) subject to constraints g(x) ≤ 0, h(x) = 0"
        }
        
        return formulations.get(problem_type, "Mathematical formulation depends on specific problem structure")
    
    def _test_normality(self, data: np.ndarray) -> Dict[str, Any]:
        """Test if data follows normal distribution"""
        statistic, p_value = stats.shapiro(data)
        
        return {
            "test": "Shapiro-Wilk",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "is_normal": p_value > 0.05,
            "interpretation": "Normal distribution" if p_value > 0.05 else "Not normal distribution"
        }
    
    def _detect_outliers(self, data: np.ndarray) -> List[float]:
        """Detect outliers using IQR method"""
        Q1 = np.percentile(data, 25)
        Q3 = np.percentile(data, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data < lower_bound) | (data > upper_bound)]
        return outliers.tolist()
    
    def _calculate_confidence_interval(self, data: np.ndarray, confidence: float) -> Tuple[float, float]:
        """Calculate confidence interval for mean"""
        n = len(data)
        mean = np.mean(data)
        std_err = stats.sem(data)
        
        # Use t-distribution for small samples
        if n < 30:
            t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
            margin_error = t_value * std_err
        else:
            z_value = stats.norm.ppf((1 + confidence) / 2)
            margin_error = z_value * std_err
        
        return (float(mean - margin_error), float(mean + margin_error))
    
    def _demonstrate_backpropagation(self, **kwargs) -> Dict[str, Any]:
        """Demonstrate backpropagation mathematics"""
        return {
            "concept": "Backpropagation",
            "mathematical_foundation": "Chain rule of calculus",
            "formula": "∂L/∂w = (∂L/∂y) * (∂y/∂z) * (∂z/∂w)",
            "explanation": "Gradient flows backward through network using chain rule",
            "example": {
                "loss_function": "L = (y - ŷ)²/2",
                "derivative": "∂L/∂ŷ = -(y - ŷ)",
                "weight_update": "w_new = w_old - η * ∂L/∂w"
            }
        }
    
    def _demonstrate_gradient_descent(self, **kwargs) -> Dict[str, Any]:
        """Demonstrate gradient descent mathematics"""
        return {
            "concept": "Gradient Descent",
            "mathematical_foundation": "Optimization using gradients",
            "formula": "θ_{t+1} = θ_t - η∇f(θ_t)",
            "explanation": "Iteratively move in direction of steepest descent",
            "variants": {
                "SGD": "Stochastic gradient descent - single sample",
                "Mini-batch": "Small batch of samples",
                "Adam": "Adaptive learning rate with momentum"
            }
        }
    
    def _analyze_activation_functions(self) -> Dict[str, Any]:
        """Analyze activation functions and their derivatives"""
        return {
            "sigmoid": {
                "function": "σ(x) = 1/(1 + e^(-x))",
                "derivative": "σ'(x) = σ(x)(1 - σ(x))",
                "range": "(0, 1)",
                "issues": "Vanishing gradient problem"
            },
            "relu": {
                "function": "ReLU(x) = max(0, x)",
                "derivative": "ReLU'(x) = 1 if x > 0, else 0",
                "range": "[0, +∞)",
                "issues": "Dead neurons problem"
            },
            "tanh": {
                "function": "tanh(x) = (e^x - e^(-x))/(e^x + e^(-x))",
                "derivative": "tanh'(x) = 1 - tanh²(x)",
                "range": "(-1, 1)",
                "issues": "Still suffers from vanishing gradients"
            }
        }
    
    def _analyze_loss_functions(self) -> Dict[str, Any]:
        """Analyze common loss functions"""
        return {
            "mse": {
                "function": "MSE = (1/n)Σ(y_i - ŷ_i)²",
                "derivative": "∂MSE/∂ŷ = -2(y - ŷ)/n",
                "use_case": "Regression problems",
                "properties": "Convex, differentiable"
            },
            "cross_entropy": {
                "function": "CE = -Σy_i log(ŷ_i)",
                "derivative": "∂CE/∂ŷ = -y/ŷ",
                "use_case": "Classification problems",
                "properties": "Convex for linear models"
            }
        }

# Testing
if __name__ == "__main__":
    engine = MathEngine()
    
    # Test derivative calculation
    derivative = engine.calculate_derivative("x**2 + 3*x + 2", "x")
    print(f"Derivative of x² + 3x + 2: {derivative}")
    
    # Test statistical analysis
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    stats_result = engine.analyze_statistical_data(data)
    print(f"Statistical analysis: {stats_result}")
    
    # Test algorithm analysis
    code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    
    import asyncio
    result = asyncio.run(engine.analyze_algorithm(code))
    print(f"Algorithm analysis: {result}")
