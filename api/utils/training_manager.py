"""
Training management utilities for the AI Agent
Handles self-training, RLHF, and LoRA fine-tuning
"""

import os
import json
import asyncio
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging

# Transformers and PEFT imports
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from datasets import Dataset, load_dataset

# For RLHF
try:
    from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
    from trl.core import LengthSampler
except ImportError:
    PPOTrainer = None
    logger.warning("TRL not available, RLHF training disabled")

logger = logging.getLogger(__name__)

class TrainingManager:
    """
    Comprehensive training management system
    
    Features:
    - LoRA (Low-Rank Adaptation) fine-tuning
    - RLHF (Reinforcement Learning from Human Feedback)
    - Continuous learning from interactions
    - Performance evaluation and validation
    - Model versioning and rollback
    - Synthetic data generation
    """
    
    def __init__(self, model_name: str = "distilgpt2"):
        self.model_name = model_name
        self.base_model = None
        self.tokenizer = None
        
        # Training configurations
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        # Training history and metrics
        self.training_history = []
        self.current_model_version = 0
        self.performance_metrics = {}
        
        # Storage paths
        self.training_dir = Path("training")
        self.training_dir.mkdir(exist_ok=True)
        
        self.models_dir = self.training_dir / "models"
        self.models_dir.mkdir(exist_ok=True)
        
        self.datasets_dir = self.training_dir / "datasets"
        self.datasets_dir.mkdir(exist_ok=True)
    
    async def initialize(self):
        """Initialize the training manager"""
        logger.info("Initializing TrainingManager...")
        
        try:
            # Load base model and tokenizer for training operations
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load training history if exists
            await self._load_training_history()
            
            logger.info("TrainingManager initialized successfully")
            
        except Exception as e:
            logger.error(f"TrainingManager initialization failed: {e}")
            raise
    
    async def load_dataset(self, dataset_name: str) -> Dict[str, Dataset]:
        """
        Load dataset from Hugging Face or local storage
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            Dictionary with train/eval datasets
        """
        try:
            logger.info(f"Loading dataset: {dataset_name}")
            
            # Check if it's a local dataset first
            local_dataset_path = self.datasets_dir / f"{dataset_name}.json"
            
            if local_dataset_path.exists():
                # Load local dataset
                with open(local_dataset_path, 'r') as f:
                    data = json.load(f)
                
                dataset = Dataset.from_dict(data)
                
                # Split into train/eval
                train_test_split = dataset.train_test_split(test_size=0.2)
                
                return {
                    "train": train_test_split["train"],
                    "eval": train_test_split["test"]
                }
            
            else:
                # Try to load from Hugging Face
                if dataset_name == "codeparrot/github-code":
                    # Special handling for code dataset
                    dataset = load_dataset("codeparrot/github-code", split="train", streaming=True)
                    
                    # Take a subset for training (due to memory constraints)
                    dataset_list = []
                    for i, example in enumerate(dataset):
                        if i >= 1000:  # Limit to 1000 examples
                            break
                        if example.get("language") == "Python":
                            dataset_list.append({"text": example["code"]})
                    
                    dataset = Dataset.from_list(dataset_list)
                
                elif dataset_name == "roneneldan/TinyStories":
                    dataset = load_dataset("roneneldan/TinyStories", split="train[:1000]")
                    # Convert to text format
                    dataset = dataset.map(lambda x: {"text": x["text"]})
                
                else:
                    # Generic dataset loading
                    dataset = load_dataset(dataset_name, split="train[:1000]")
                    
                    # Ensure text field exists
                    if "text" not in dataset.column_names:
                        # Try to find a text-like field
                        text_fields = [col for col in dataset.column_names if "text" in col.lower() or "content" in col.lower()]
                        if text_fields:
                            dataset = dataset.rename_column(text_fields[0], "text")
                        else:
                            raise ValueError(f"No text field found in dataset {dataset_name}")
                
                # Split dataset
                train_test_split = dataset.train_test_split(test_size=0.2)
                
                # Save locally for future use
                await self._save_dataset_locally(dataset_name, {
                    "train": train_test_split["train"].to_dict(),
                    "eval": train_test_split["test"].to_dict()
                })
                
                return {
                    "train": train_test_split["train"],
                    "eval": train_test_split["test"]
                }
        
        except Exception as e:
            logger.error(f"Failed to load dataset {dataset_name}: {e}")
            
            # Return a minimal synthetic dataset as fallback
            return await self._create_synthetic_dataset()
    
    async def prepare_interaction_data(self, interactions: List[Dict[str, Any]]) -> Dict[str, Dataset]:
        """
        Prepare interaction data for training
        
        Args:
            interactions: List of interaction data
            
        Returns:
            Dictionary with train/eval datasets
        """
        try:
            # Extract training examples from interactions
            training_examples = []
            
            for interaction in interactions:
                if interaction.get("type") == "code_generation":
                    data = interaction.get("response", {})
                    request = interaction.get("request", {})
                    
                    # Create training examples
                    prompt = request.get("prompt", "")
                    code = data.get("code", "")
                    explanation = data.get("explanation", "")
                    
                    if prompt and code:
                        # Format as instruction-following example
                        text = f"### Instruction:\n{prompt}\n\n### Response:\n{code}\n\n### Explanation:\n{explanation}"
                        training_examples.append({"text": text})
                
                elif interaction.get("type") == "reasoning":
                    data = interaction.get("response", {})
                    request = interaction.get("request", {})
                    
                    problem = request.get("problem", "")
                    solution = data.get("solution", "")
                    
                    if problem and solution:
                        text = f"### Problem:\n{problem}\n\n### Solution:\n{solution}"
                        training_examples.append({"text": text})
            
            if not training_examples:
                # Create minimal synthetic data if no valid interactions
                return await self._create_synthetic_dataset()
            
            # Create dataset
            dataset = Dataset.from_list(training_examples)
            
            # Split into train/eval
            if len(training_examples) > 10:
                train_test_split = dataset.train_test_split(test_size=0.2)
                return {
                    "train": train_test_split["train"],
                    "eval": train_test_split["test"]
                }
            else:
                # Too few examples, use all for training
                return {
                    "train": dataset,
                    "eval": dataset.select(range(min(2, len(dataset))))  # Small eval set
                }
        
        except Exception as e:
            logger.error(f"Failed to prepare interaction data: {e}")
            return await self._create_synthetic_dataset()
    
    async def fine_tune_with_lora(self, 
                                dataset: Dict[str, Dataset],
                                training_args: Optional[TrainingArguments] = None) -> Dict[str, Any]:
        """
        Fine-tune model using LoRA
        
        Args:
            dataset: Training and evaluation datasets
            training_args: Custom training arguments
            
        Returns:
            Training results and metrics
        """
        try:
            logger.info("Starting LoRA fine-tuning...")
            
            # Load base model for training
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            
            # Apply LoRA
            peft_model = get_peft_model(model, self.lora_config)
            
            # Tokenize datasets
            tokenized_datasets = await self._tokenize_datasets(dataset)
            
            # Default training arguments
            if training_args is None:
                training_args = TrainingArguments(
                    output_dir=str(self.models_dir / f"lora_model_v{self.current_model_version + 1}"),
                    num_train_epochs=3,
                    per_device_train_batch_size=2,
                    per_device_eval_batch_size=2,
                    warmup_steps=100,
                    weight_decay=0.01,
                    logging_dir=str(self.training_dir / "logs"),
                    logging_steps=10,
                    evaluation_strategy="steps",
                    eval_steps=50,
                    save_strategy="steps",
                    save_steps=100,
                    load_best_model_at_end=True,
                    metric_for_best_model="eval_loss",
                    greater_is_better=False,
                    dataloader_pin_memory=False,  # For CPU training
                    report_to=[]  # Disable wandb/tensorboard
                )
            
            # Data collator
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
                pad_to_multiple_of=8
            )
            
            # Initialize trainer
            trainer = Trainer(
                model=peft_model,
                args=training_args,
                train_dataset=tokenized_datasets["train"],
                eval_dataset=tokenized_datasets["eval"],
                data_collator=data_collator,
                tokenizer=self.tokenizer
            )
            
            # Train the model
            training_start = datetime.utcnow()
            
            train_result = trainer.train()
            
            training_end = datetime.utcnow()
            training_duration = (training_end - training_start).total_seconds()
            
            # Evaluate the model
            eval_result = trainer.evaluate()
            
            # Save the trained model
            model_path = self.models_dir / f"lora_model_v{self.current_model_version + 1}"
            trainer.save_model(str(model_path))
            
            # Update version
            self.current_model_version += 1
            
            # Record training history
            training_record = {
                "version": self.current_model_version,
                "timestamp": training_start.isoformat(),
                "duration_seconds": training_duration,
                "training_args": training_args.to_dict(),
                "train_result": {
                    "train_loss": train_result.training_loss,
                    "train_runtime": train_result.training_runtime,
                    "train_samples_per_second": train_result.train_samples_per_second,
                    "total_flos": train_result.total_flos
                },
                "eval_result": eval_result,
                "model_path": str(model_path),
                "dataset_size": {
                    "train": len(tokenized_datasets["train"]),
                    "eval": len(tokenized_datasets["eval"])
                }
            }
            
            self.training_history.append(training_record)
            await self._save_training_history()
            
            logger.info(f"LoRA fine-tuning completed. Model version: {self.current_model_version}")
            
            return training_record
        
        except Exception as e:
            logger.error(f"LoRA fine-tuning failed: {e}")
            raise
    
    async def rlhf_training(self, 
                          interactions: List[Dict[str, Any]],
                          reward_model: Optional[Any] = None) -> Dict[str, Any]:
        """
        Perform RLHF training (simplified version)
        
        Args:
            interactions: List of interactions with feedback
            reward_model: Optional reward model
            
        Returns:
            Training results
        """
        try:
            if PPOTrainer is None:
                logger.warning("RLHF training not available (TRL not installed)")
                # Fallback to supervised fine-tuning with filtered data
                return await self._supervised_feedback_training(interactions)
            
            logger.info("Starting RLHF training...")
            
            # Extract positive and negative examples based on feedback
            positive_examples = []
            negative_examples = []
            
            for interaction in interactions:
                if interaction.get("type") == "feedback":
                    content = interaction.get("content", {})
                    sentiment = self._analyze_feedback_sentiment(content)
                    
                    if sentiment == "positive":
                        positive_examples.append(interaction)
                    elif sentiment == "negative":
                        negative_examples.append(interaction)
            
            # For now, implement a simplified RLHF approach
            # In a full implementation, you would use PPOTrainer
            
            # Create reward signal based on feedback
            reward_data = []
            for example in positive_examples:
                reward_data.append({
                    "text": self._extract_text_from_interaction(example),
                    "reward": 1.0
                })
            
            for example in negative_examples:
                reward_data.append({
                    "text": self._extract_text_from_interaction(example),
                    "reward": -1.0
                })
            
            if reward_data:
                # Convert to dataset and fine-tune
                dataset = Dataset.from_list(reward_data)
                
                # Prepare for training (simplified approach)
                training_dataset = {"train": dataset, "eval": dataset.select(range(min(5, len(dataset))))}
                
                # Use LoRA fine-tuning with reward-weighted loss
                result = await self.fine_tune_with_lora(training_dataset)
                result["training_type"] = "rlhf_simplified"
                
                return result
            else:
                logger.warning("No feedback data available for RLHF training")
                return {"error": "No feedback data available"}
        
        except Exception as e:
            logger.error(f"RLHF training failed: {e}")
            return {"error": str(e)}
    
    async def augment_dataset(self, 
                            original_dataset: Dict[str, Dataset],
                            synthetic_data: Dataset) -> Dict[str, Dataset]:
        """
        Augment dataset with synthetic data
        
        Args:
            original_dataset: Original training dataset
            synthetic_data: Synthetic data to add
            
        Returns:
            Augmented dataset
        """
        try:
            from datasets import concatenate_datasets
            
            # Concatenate original and synthetic data
            augmented_train = concatenate_datasets([original_dataset["train"], synthetic_data])
            
            # Keep original eval set
            augmented_eval = original_dataset["eval"]
            
            logger.info(f"Dataset augmented: {len(original_dataset['train'])} → {len(augmented_train)} examples")
            
            return {
                "train": augmented_train,
                "eval": augmented_eval
            }
        
        except Exception as e:
            logger.error(f"Dataset augmentation failed: {e}")
            return original_dataset
    
    async def evaluate_model_performance(self, 
                                       model_path: str,
                                       test_dataset: Dataset) -> Dict[str, Any]:
        """
        Evaluate model performance on test dataset
        
        Args:
            model_path: Path to the model to evaluate
            test_dataset: Test dataset
            
        Returns:
            Performance metrics
        """
        try:
            # Load model for evaluation
            base_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="cpu"
            )
            
            # Load LoRA weights if available
            if Path(model_path).exists():
                model = PeftModel.from_pretrained(base_model, model_path)
            else:
                model = base_model
            
            model.eval()
            
            # Evaluate on test examples
            total_loss = 0.0
            num_examples = 0
            
            with torch.no_grad():
                for example in test_dataset:
                    text = example["text"]
                    
                    # Tokenize
                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=512
                    )
                    
                    # Forward pass
                    outputs = model(**inputs, labels=inputs["input_ids"])
                    loss = outputs.loss
                    
                    total_loss += loss.item()
                    num_examples += 1
                    
                    # Limit evaluation for performance
                    if num_examples >= 100:
                        break
            
            avg_loss = total_loss / num_examples if num_examples > 0 else float('inf')
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            
            metrics = {
                "avg_loss": avg_loss,
                "perplexity": perplexity,
                "num_examples": num_examples,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            logger.info(f"Model evaluation completed: perplexity={perplexity:.2f}")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return {"error": str(e)}
    
    # Private methods
    async def _tokenize_datasets(self, datasets: Dict[str, Dataset]) -> Dict[str, Dataset]:
        """Tokenize datasets for training"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
        
        tokenized_datasets = {}
        for split, dataset in datasets.items():
            tokenized_datasets[split] = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=dataset.column_names
            )
        
        return tokenized_datasets
    
    async def _create_synthetic_dataset(self) -> Dict[str, Dataset]:
        """Create a minimal synthetic dataset for training"""
        synthetic_examples = [
            {"text": "def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)"},
            {"text": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"},
            {"text": "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr"},
            {"text": "def binary_search(arr, x):\n    low = 0\n    high = len(arr) - 1\n    while low <= high:\n        mid = (high + low) // 2\n        if arr[mid] < x:\n            low = mid + 1\n        elif arr[mid] > x:\n            high = mid - 1\n        else:\n            return mid\n    return -1"},
            {"text": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)"}
        ]
        
        dataset = Dataset.from_list(synthetic_examples)
        
        return {
            "train": dataset,
            "eval": dataset.select([0, 1])  # Use first 2 for eval
        }
    
    async def _save_dataset_locally(self, dataset_name: str, dataset_dict: Dict):
        """Save dataset locally for future use"""
        try:
            dataset_path = self.datasets_dir / f"{dataset_name}.json"
            with open(dataset_path, 'w') as f:
                json.dump(dataset_dict, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to save dataset locally: {e}")
    
    async def _save_training_history(self):
        """Save training history to file"""
        try:
            history_path = self.training_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump({
                    "current_version": self.current_model_version,
                    "history": self.training_history,
                    "performance_metrics": self.performance_metrics
                }, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to save training history: {e}")
    
    async def _load_training_history(self):
        """Load training history from file"""
        try:
            history_path = self.training_dir / "training_history.json"
            if history_path.exists():
                with open(history_path, 'r') as f:
                    data = json.load(f)
                
                self.current_model_version = data.get("current_version", 0)
                self.training_history = data.get("history", [])
                self.performance_metrics = data.get("performance_metrics", {})
                
                logger.info(f"Loaded training history: {len(self.training_history)} training sessions")
        except Exception as e:
            logger.debug(f"Failed to load training history: {e}")
    
    async def _supervised_feedback_training(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback supervised training using feedback"""
        try:
            # Filter positive interactions for training
            positive_interactions = []
            
            for interaction in interactions:
                if interaction.get("type") == "feedback":
                    content = interaction.get("content", {})
                    if self._analyze_feedback_sentiment(content) == "positive":
                        # Find the original interaction that was rated positively
                        original_interaction = content.get("original_interaction", {})
                        if original_interaction:
                            positive_interactions.append(original_interaction)
            
            if positive_interactions:
                # Prepare dataset from positive interactions
                dataset = await self.prepare_interaction_data(positive_interactions)
                
                # Fine-tune with LoRA
                result = await self.fine_tune_with_lora(dataset)
                result["training_type"] = "supervised_feedback"
                
                return result
            else:
                return {"error": "No positive feedback available for training"}
        
        except Exception as e:
            logger.error(f"Supervised feedback training failed: {e}")
            return {"error": str(e)}
    
    def _analyze_feedback_sentiment(self, feedback: Dict[str, Any]) -> str:
        """Analyze feedback sentiment"""
        content = json.dumps(feedback).lower()
        
        positive_words = ["good", "great", "excellent", "helpful", "correct", "accurate", "useful", "perfect"]
        negative_words = ["bad", "wrong", "incorrect", "unhelpful", "poor", "terrible", "useless", "awful"]
        
        positive_count = sum(1 for word in positive_words if word in content)
        negative_count = sum(1 for word in negative_words if word in content)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _extract_text_from_interaction(self, interaction: Dict[str, Any]) -> str:
        """Extract text content from interaction for training"""
        if interaction.get("type") == "code_generation":
            request = interaction.get("request", {})
            response = interaction.get("response", {})
            
            prompt = request.get("prompt", "")
            code = response.get("code", "")
            
            return f"### Instruction:\n{prompt}\n\n### Response:\n{code}"
        
        elif interaction.get("type") == "reasoning":
            request = interaction.get("request", {})
            response = interaction.get("response", {})
            
            problem = request.get("problem", "")
            solution = response.get("solution", "")
            
            return f"### Problem:\n{problem}\n\n### Solution:\n{solution}"
        
        else:
            return json.dumps(interaction)[:500]  # Fallback
    
    # Public utility methods
    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history"""
        return self.training_history
    
    def get_current_version(self) -> int:
        """Get current model version"""
        return self.current_model_version
    
    def get_available_models(self) -> List[str]:
        """Get list of available trained models"""
        models = []
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                models.append(model_dir.name)
        return models

# Testing
if __name__ == "__main__":
    async def test_training_manager():
        manager = TrainingManager()
        await manager.initialize()
        
        # Test dataset loading
        try:
            dataset = await manager.load_dataset("roneneldan/TinyStories")
            print(f"Loaded dataset with {len(dataset['train'])} training examples")
        except Exception as e:
            print(f"Dataset loading failed, using synthetic: {e}")
            dataset = await manager._create_synthetic_dataset()
        
        # Test LoRA fine-tuning (commented out for testing without GPU)
        # result = await manager.fine_tune_with_lora(dataset)
        # print(f"Training result: {result}")
        
        print("TrainingManager test completed")
    
    asyncio.run(test_training_manager())
