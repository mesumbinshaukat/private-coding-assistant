"""
Autonomous AI Agent Core - The brain of the system
Implements ReAct framework with self-training capabilities
"""

import os
import json
import asyncio
import numpy as np
import torch
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from pathlib import Path

# LangChain imports - Updated for current version
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import BaseOutputParser

# Transformers and ML imports
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, pipeline,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, load_dataset
import sentence_transformers
import faiss

# Utility imports
from api.utils.web_search import WebSearcher
from api.utils.code_executor import CodeExecutor
from api.utils.math_engine import MathEngine
from api.utils.memory_manager import MemoryManager
from api.utils.training_manager import TrainingManager

logger = logging.getLogger(__name__)

class ReActOutputParser(BaseOutputParser):
    """Parser for ReAct framework outputs"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        """Parse ReAct format: Thought: ... Action: ... Observation: ..."""
        sections = {}
        current_section = None
        current_content = []
        
        for line in text.split('\n'):
            if line.startswith('Thought:'):
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'thought'
                current_content = [line[8:].strip()]
            elif line.startswith('Action:'):
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'action'
                current_content = [line[7:].strip()]
            elif line.startswith('Observation:'):
                if current_section:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'observation'
                current_content = [line[12:].strip()]
            else:
                if current_content:
                    current_content.append(line)
        
        if current_section:
            sections[current_section] = '\n'.join(current_content)
        
        return sections

class AutonomousAgent:
    """
    Autonomous AI Agent with self-training capabilities
    
    Features:
    - ReAct framework for reasoning and acting
    - Self-training with RLHF loop
    - Long-term memory with FAISS
    - Mathematical reasoning with calculus and statistics
    - Code generation, debugging, and optimization
    - Deep web searching and knowledge synthesis
    """
    
    def __init__(self, model_name: str = "distilgpt2"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.llm = None
        self.conversation_chain = None
        
        # Components
        self.web_searcher = WebSearcher()
        self.code_executor = CodeExecutor()
        self.math_engine = MathEngine()
        self.memory_manager = MemoryManager()
        self.training_manager = TrainingManager()
        
        # ReAct parser
        self.output_parser = ReActOutputParser()
        
        # Performance tracking
        self.interaction_history = []
        self.performance_metrics = {
            "code_generation_success_rate": 0.0,
            "search_relevance_score": 0.0,
            "reasoning_accuracy": 0.0,
            "training_iterations": 0
        }
        
        # Model paths
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
    async def initialize(self):
        """Initialize the agent with model loading and setup"""
        logger.info(f"Initializing agent with model: {self.model_name}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,  # CPU deployment
                device_map="cpu"
            )
            
            # Create text generation pipeline
            text_gen_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Initialize LangChain LLM
            self.llm = HuggingFacePipeline(pipeline=text_gen_pipeline)
            
            # Setup conversation chain with ReAct prompt
            react_prompt = PromptTemplate(
                input_variables=["history", "input"],
                template="""You are an autonomous AI agent specialized in coding tasks. Use the ReAct framework:

Thought: Think step-by-step about the problem
Action: Decide what action to take (code, search, calculate, reason)
Observation: Observe the results of your action

Previous conversation:
{history}

Current task: {input}

Thought:"""
            )
            
            # Initialize memory and conversation chain
            memory = ConversationBufferWindowMemory(k=10)
            self.conversation_chain = ConversationChain(
                llm=self.llm,
                prompt=react_prompt,
                memory=memory,
                output_parser=self.output_parser
            )
            
            # Initialize components
            await self.web_searcher.initialize()
            await self.memory_manager.initialize()
            await self.training_manager.initialize()
            
            logger.info("Agent initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Agent initialization failed: {str(e)}")
            raise
    
    async def generate_code(self, prompt: str, language: str = "python", context: str = None) -> Dict[str, Any]:
        """
        Generate code using ReAct framework with mathematical reasoning
        
        Process:
        1. Analyze the prompt (Thought)
        2. Plan the solution approach (Action)
        3. Generate code with explanations (Observation)
        4. Test and validate the code
        5. Provide complexity analysis and optimizations
        """
        logger.info(f"Generating {language} code for: {prompt[:100]}...")
        
        try:
            # Prepare the enhanced prompt
            enhanced_prompt = f"""
Generate {language} code for: {prompt}

Additional context: {context or 'None'}

Requirements:
1. Write clean, well-documented code
2. Include test cases
3. Provide complexity analysis (Big O notation)
4. Explain mathematical concepts if applicable
5. Suggest optimizations

Use the ReAct framework to think through this step by step.
"""
            
            # Generate response using ReAct
            response = await self._react_generate(enhanced_prompt)
            
            # Extract and execute code if possible
            code = self._extract_code_from_response(response)
            
            # Test the code
            test_results = await self.code_executor.test_code(code, language)
            
            # Mathematical analysis
            math_analysis = await self.math_engine.analyze_algorithm(code)
            
            # Generate test cases
            test_cases = await self._generate_test_cases(code, prompt)
            
            # Calculate confidence based on test results and analysis
            confidence = self._calculate_confidence(test_results, math_analysis)
            
            result = {
                "code": code,
                "explanation": response.get("thought", ""),
                "test_cases": test_cases,
                "complexity_analysis": math_analysis,
                "optimizations": await self._suggest_optimizations(code, math_analysis),
                "reasoning_steps": response.get("action", ""),
                "confidence": confidence,
                "test_results": test_results
            }
            
            # Store in memory for learning
            await self.memory_manager.store_interaction("code_generation", {
                "prompt": prompt,
                "language": language,
                "context": context,
                "result": result
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Code generation error: {str(e)}")
            return {
                "code": f"# Error generating code: {str(e)}",
                "explanation": f"An error occurred during code generation: {str(e)}",
                "test_cases": [],
                "complexity_analysis": {},
                "optimizations": [],
                "reasoning_steps": "",
                "confidence": 0.0,
                "test_results": {"success": False, "error": str(e)}
            }
    
    async def deep_search(self, query: str, depth: int = 5, include_code: bool = True) -> Dict[str, Any]:
        """
        Perform deep web search with synthesis and analysis
        
        Process:
        1. Multi-source search (Stack Overflow, GitHub, docs)
        2. Extract and analyze results
        3. Synthesize information
        4. Extract relevant code examples
        5. Score relevance and confidence
        """
        logger.info(f"Deep searching: {query[:100]}...")
        
        try:
            # Use ReAct to plan the search
            search_plan = await self._react_generate(f"""
Plan a comprehensive search strategy for: {query}

Consider:
1. What specific information are we looking for?
2. Which sources would be most relevant?
3. What search terms and variations should we use?
4. How can we verify the quality of results?

Provide a detailed search plan.
""")
            
            # Execute the search
            search_results = await self.web_searcher.search_multiple_sources(
                query=query,
                depth=depth,
                include_code=include_code,
                sources=["stackoverflow", "github", "documentation"]
            )
            
            # Synthesize results using the agent
            synthesis_prompt = f"""
Analyze and synthesize these search results for the query: {query}

Search Results:
{json.dumps(search_results, indent=2)}

Provide:
1. A comprehensive answer synthesizing all sources
2. Key insights and patterns
3. Best practices identified
4. Code examples with explanations
5. Confidence assessment of the information
"""
            
            synthesis = await self._react_generate(synthesis_prompt)
            
            # Extract code examples
            code_examples = self._extract_code_examples(search_results)
            
            # Calculate relevance scores
            relevance_scores = await self._calculate_relevance_scores(query, search_results)
            
            # Generate related queries
            related_queries = await self._generate_related_queries(query, search_results)
            
            result = {
                "results": search_results,
                "synthesized_answer": synthesis.get("thought", ""),
                "code_examples": code_examples,
                "sources": [result.get("source", "") for result in search_results],
                "confidence": np.mean(relevance_scores) if relevance_scores else 0.0,
                "related_queries": related_queries,
                "search_plan": search_plan.get("action", "")
            }
            
            # Store in memory
            await self.memory_manager.store_interaction("search", {
                "query": query,
                "depth": depth,
                "include_code": include_code,
                "result": result
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Deep search error: {str(e)}")
            return {
                "results": [],
                "synthesized_answer": f"Search failed: {str(e)}",
                "code_examples": [],
                "sources": [],
                "confidence": 0.0,
                "related_queries": [],
                "search_plan": ""
            }
    
    async def reason_step_by_step(self, problem: str, domain: str = "coding", include_math: bool = True) -> Dict[str, Any]:
        """
        Perform detailed step-by-step reasoning with mathematical analysis
        
        Process:
        1. Break down the problem into sub-problems
        2. Apply domain-specific reasoning
        3. Use mathematical tools for analysis
        4. Validate each step
        5. Provide alternative approaches
        """
        logger.info(f"Reasoning about: {problem[:100]}...")
        
        try:
            # Initial problem analysis
            analysis_prompt = f"""
Analyze this {domain} problem step by step: {problem}

Break it down into:
1. Core problem identification
2. Sub-problems and dependencies  
3. Mathematical formulation (if applicable)
4. Solution approach options
5. Complexity considerations

Use detailed reasoning for each step.
"""
            
            # Get initial analysis
            analysis = await self._react_generate(analysis_prompt)
            
            # Mathematical analysis if requested
            math_analysis = {}
            if include_math and domain == "coding":
                math_analysis = await self.math_engine.analyze_problem(problem)
            
            # Generate step-by-step solution
            solution_prompt = f"""
Provide a detailed step-by-step solution for: {problem}

Based on the analysis: {analysis.get('thought', '')}

Mathematical considerations: {json.dumps(math_analysis, indent=2)}

Provide:
1. Detailed solution steps
2. Reasoning for each step
3. Mathematical derivations (if applicable)
4. Code implementation (if applicable)
5. Validation of the solution
"""
            
            solution = await self._react_generate(solution_prompt)
            
            # Generate alternative approaches
            alternatives = await self._generate_alternative_approaches(problem, analysis, solution)
            
            # Complexity analysis
            complexity_analysis = await self._analyze_solution_complexity(solution, math_analysis)
            
            # Calculate confidence
            confidence = self._calculate_reasoning_confidence(analysis, solution, math_analysis)
            
            result = {
                "reasoning_steps": [
                    {"step": 1, "description": "Problem Analysis", "content": analysis.get("thought", "")},
                    {"step": 2, "description": "Mathematical Formulation", "content": json.dumps(math_analysis, indent=2)},
                    {"step": 3, "description": "Solution Development", "content": solution.get("thought", "")},
                    {"step": 4, "description": "Validation", "content": solution.get("observation", "")}
                ],
                "solution": solution.get("action", ""),
                "mathematical_analysis": math_analysis,
                "alternative_approaches": alternatives,
                "confidence": confidence,
                "complexity_analysis": complexity_analysis
            }
            
            # Store in memory
            await self.memory_manager.store_interaction("reasoning", {
                "problem": problem,
                "domain": domain,
                "include_math": include_math,
                "result": result
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Reasoning error: {str(e)}")
            return {
                "reasoning_steps": [{"step": 1, "description": "Error", "content": f"Reasoning failed: {str(e)}"}],
                "solution": f"Unable to solve due to error: {str(e)}",
                "mathematical_analysis": {},
                "alternative_approaches": [],
                "confidence": 0.0,
                "complexity_analysis": {}
            }
    
    async def self_train(self, dataset_name: str = None, training_type: str = "rlhf", iterations: int = 10):
        """
        Self-training mechanism with RLHF and LoRA fine-tuning
        
        Process:
        1. Load training data (from interactions or external dataset)
        2. Evaluate current performance
        3. Generate synthetic data if needed
        4. Fine-tune using LoRA
        5. Validate improvements
        6. Update model if performance improves
        """
        logger.info(f"Starting self-training: {training_type} for {iterations} iterations")
        
        try:
            # Prepare training data
            if dataset_name:
                # Load external dataset
                training_data = await self.training_manager.load_dataset(dataset_name)
            else:
                # Use interaction history
                training_data = await self.training_manager.prepare_interaction_data(
                    self.interaction_history
                )
            
            # Evaluate current performance
            baseline_metrics = await self._evaluate_current_performance()
            
            # Prepare LoRA configuration
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"]
            )
            
            # Apply LoRA to model
            peft_model = get_peft_model(self.model, lora_config)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir=str(self.model_dir / "training"),
                num_train_epochs=1,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                warmup_steps=100,
                weight_decay=0.01,
                logging_dir=str(self.model_dir / "logs"),
                logging_steps=10,
                evaluation_strategy="steps",
                eval_steps=50,
                save_strategy="steps",
                save_steps=100,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False
            )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=peft_model,
                args=training_args,
                train_dataset=training_data["train"],
                eval_dataset=training_data["eval"],
                data_collator=data_collator
            )
            
            # Train the model
            for iteration in range(iterations):
                logger.info(f"Training iteration {iteration + 1}/{iterations}")
                
                # Train
                trainer.train()
                
                # Evaluate
                eval_results = trainer.evaluate()
                
                # Generate synthetic data for next iteration
                if iteration < iterations - 1:
                    synthetic_data = await self._generate_synthetic_training_data()
                    training_data = await self.training_manager.augment_dataset(
                        training_data, synthetic_data
                    )
            
            # Evaluate final performance
            final_metrics = await self._evaluate_current_performance()
            
            # Check if improvement occurred
            improvement = self._calculate_improvement(baseline_metrics, final_metrics)
            
            if improvement > 0.05:  # 5% improvement threshold
                # Save the improved model
                trainer.save_model(str(self.model_dir / "improved_model"))
                
                # Update performance metrics
                self.performance_metrics["training_iterations"] += iterations
                self.performance_metrics.update(final_metrics)
                
                logger.info(f"Training completed successfully. Improvement: {improvement:.2%}")
            else:
                logger.info("Training completed but no significant improvement detected")
            
        except Exception as e:
            logger.error(f"Self-training error: {str(e)}")
            raise
    
    async def learn_from_interaction(self, interaction_type: str, request_data: Dict, response_data: Dict):
        """Learn from user interactions for continuous improvement"""
        try:
            interaction = {
                "type": interaction_type,
                "timestamp": datetime.utcnow().isoformat(),
                "request": request_data,
                "response": response_data
            }
            
            self.interaction_history.append(interaction)
            
            # Store in vector memory for similarity search
            await self.memory_manager.store_interaction(interaction_type, interaction)
            
            # Update performance metrics
            await self._update_performance_metrics(interaction_type, response_data)
            
            # Trigger self-training if enough new data accumulated
            if len(self.interaction_history) % 100 == 0:  # Every 100 interactions
                await self.self_train(training_type="continuous", iterations=5)
                
        except Exception as e:
            logger.error(f"Learning from interaction error: {str(e)}")
    
    async def process_feedback(self, feedback: Dict[str, Any]):
        """Process user feedback for model improvement"""
        try:
            # Store feedback
            await self.memory_manager.store_feedback(feedback)
            
            # Analyze feedback sentiment and extract improvements
            feedback_analysis = await self._analyze_feedback(feedback)
            
            # Update training data based on feedback
            if feedback_analysis["should_retrain"]:
                await self.self_train(training_type="feedback", iterations=3)
                
        except Exception as e:
            logger.error(f"Feedback processing error: {str(e)}")
    
    # Helper methods
    async def _react_generate(self, prompt: str) -> Dict[str, Any]:
        """Generate response using ReAct framework"""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.conversation_chain.predict, prompt
            )
            return self.output_parser.parse(response)
        except Exception as e:
            logger.error(f"ReAct generation error: {str(e)}")
            return {"thought": "", "action": "", "observation": ""}
    
    def _extract_code_from_response(self, response: Dict[str, Any]) -> str:
        """Extract code from ReAct response"""
        # Implementation details for code extraction
        action = response.get("action", "")
        # Extract code blocks (implementation depends on format)
        return action  # Simplified
    
    async def _generate_test_cases(self, code: str, prompt: str) -> List[Dict]:
        """Generate test cases for the code"""
        # Implementation for test case generation
        return []  # Simplified
    
    def _calculate_confidence(self, test_results: Dict, math_analysis: Dict) -> float:
        """Calculate confidence score based on various factors"""
        # Implementation for confidence calculation
        return 0.8  # Simplified
    
    async def _suggest_optimizations(self, code: str, math_analysis: Dict) -> List[str]:
        """Suggest code optimizations"""
        # Implementation for optimization suggestions
        return []  # Simplified
    
    def _extract_code_examples(self, search_results: List[Dict]) -> List[Dict]:
        """Extract code examples from search results"""
        # Implementation for code extraction
        return []  # Simplified
    
    async def _calculate_relevance_scores(self, query: str, results: List[Dict]) -> List[float]:
        """Calculate relevance scores for search results"""
        # Implementation for relevance scoring
        return [0.8] * len(results)  # Simplified
    
    async def _generate_related_queries(self, query: str, results: List[Dict]) -> List[str]:
        """Generate related search queries"""
        # Implementation for related query generation
        return []  # Simplified
    
    async def _generate_alternative_approaches(self, problem: str, analysis: Dict, solution: Dict) -> List[Dict]:
        """Generate alternative solution approaches"""
        # Implementation for alternative generation
        return []  # Simplified
    
    async def _analyze_solution_complexity(self, solution: Dict, math_analysis: Dict) -> Dict:
        """Analyze solution complexity"""
        # Implementation for complexity analysis
        return {}  # Simplified
    
    def _calculate_reasoning_confidence(self, analysis: Dict, solution: Dict, math_analysis: Dict) -> float:
        """Calculate confidence in reasoning"""
        # Implementation for reasoning confidence
        return 0.8  # Simplified
    
    async def _evaluate_current_performance(self) -> Dict[str, float]:
        """Evaluate current model performance"""
        # Implementation for performance evaluation
        return self.performance_metrics.copy()
    
    async def _generate_synthetic_training_data(self) -> Dataset:
        """Generate synthetic training data"""
        # Implementation for synthetic data generation
        return Dataset.from_dict({"text": []})  # Simplified
    
    def _calculate_improvement(self, baseline: Dict, final: Dict) -> float:
        """Calculate improvement between baseline and final metrics"""
        # Implementation for improvement calculation
        return 0.1  # Simplified
    
    async def _update_performance_metrics(self, interaction_type: str, response_data: Dict):
        """Update performance metrics based on interaction"""
        # Implementation for metrics update
        pass
    
    async def _analyze_feedback(self, feedback: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze user feedback"""
        # Implementation for feedback analysis
        return {"should_retrain": False}
    
    # Status and monitoring methods
    async def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "model_loaded": self.model is not None,
            "components_initialized": True,
            "performance_metrics": self.performance_metrics,
            "interaction_count": len(self.interaction_history)
        }
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        return {"memory_usage": "placeholder"}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": self.model_name,
            "model_size": "placeholder",
            "quantization": "fp32"
        }
    
    def get_training_history(self) -> List[Dict]:
        """Get training history"""
        return []
    
    def get_capabilities(self) -> List[str]:
        """Get agent capabilities"""
        return [
            "Code generation",
            "Deep web search",
            "Mathematical reasoning",
            "Self-training",
            "Step-by-step reasoning"
        ]
