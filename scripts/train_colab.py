#!/usr/bin/env python3
"""
Google Colab Training Script for Autonomous AI Agent

This script is designed to run in Google Colab for training and fine-tuning
the AI agent model using free GPU resources.
"""

import os
import json
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
import gc
import zipfile
import requests
from google.colab import files, drive
import matplotlib.pyplot as plt
import seaborn as sns

# Colab-specific imports
try:
    from google.colab import auth, drive
    COLAB_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False
    print("⚠️ Google Colab not detected. This script is optimized for Colab.")

# ML and Training imports
from datasets import Dataset, load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, 
    TrainingArguments, Trainer, DataCollatorForLanguageModeling,
    pipeline, BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import wandb
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import nltk
from nltk.translate.bleu_score import sentence_bleu

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ColabTrainer:
    """Training manager for Google Colab environment"""
    
    def __init__(self, model_name: str = "distilgpt2", use_quantization: bool = True):
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Training configuration
        self.training_config = {
            "max_length": 512,
            "learning_rate": 5e-5,
            "batch_size": 4,
            "gradient_accumulation_steps": 4,
            "num_epochs": 3,
            "warmup_steps": 100,
            "logging_steps": 10,
            "save_steps": 500,
            "eval_steps": 500,
        }
        
        # LoRA configuration
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj", "c_fc"]  # GPT-2 specific
        )
        
        print(f"🚀 ColabTrainer initialized")
        print(f"📱 Device: {self.device}")
        print(f"🤖 Model: {model_name}")
        print(f"⚡ Quantization: {use_quantization}")
        
    def setup_colab_environment(self):
        """Set up the Colab environment"""
        print("🔧 Setting up Colab environment...")
        
        # Mount Google Drive
        if COLAB_AVAILABLE:
            try:
                drive.mount('/content/drive')
                print("✅ Google Drive mounted")
                
                # Create working directory
                work_dir = Path("/content/drive/MyDrive/autonomous_ai_agent")
                work_dir.mkdir(exist_ok=True)
                os.chdir(work_dir)
                print(f"📁 Working directory: {work_dir}")
                
            except Exception as e:
                print(f"⚠️ Drive mount failed: {e}")
                work_dir = Path("/content/autonomous_ai_agent")
                work_dir.mkdir(exist_ok=True)
                os.chdir(work_dir)
        
        # Install additional packages
        packages = [
            "accelerate",
            "bitsandbytes",
            "peft",
            "wandb",
            "nltk",
            "evaluate"
        ]
        
        for package in packages:
            try:
                os.system(f"pip install {package} -q")
                print(f"✅ Installed {package}")
            except:
                print(f"⚠️ Failed to install {package}")
        
        # Download NLTK data
        try:
            import nltk
            nltk.download('punkt', quiet=True)
            nltk.download('bleu', quiet=True)
            print("✅ NLTK data downloaded")
        except:
            print("⚠️ NLTK download failed")
        
        # Initialize wandb (optional)
        try:
            wandb.login()
            print("✅ Weights & Biases logged in")
        except:
            print("⚠️ Wandb login failed (optional)")
    
    def load_and_prepare_model(self) -> Tuple[Any, Any]:
        """Load and prepare model for training"""
        print(f"🔄 Loading model: {self.model_name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Configure quantization if needed
        if self.use_quantization and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            # Prepare for LoRA
            model = prepare_model_for_kbit_training(model)
            
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
        
        # Apply LoRA
        model = get_peft_model(model, self.lora_config)
        model.print_trainable_parameters()
        
        print("✅ Model loaded and prepared for training")
        return model, tokenizer
    
    def prepare_coding_dataset(self, dataset_name: str = "codeparrot/github-code", 
                             num_samples: int = 10000) -> Dataset:
        """Prepare coding dataset for training"""
        print(f"📚 Preparing dataset: {dataset_name}")
        
        try:
            # Load dataset
            if dataset_name == "codeparrot/github-code":
                # Use a subset for faster loading
                dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]", streaming=False)
            elif dataset_name == "bigcode/the-stack":
                # Filter for Python files only
                dataset = load_dataset(
                    dataset_name, 
                    data_dir="data/python", 
                    split=f"train[:{num_samples}]",
                    streaming=False
                )
            else:
                dataset = load_dataset(dataset_name, split=f"train[:{num_samples}]")
            
            print(f"✅ Dataset loaded: {len(dataset)} samples")
            
            # Preprocess for code generation training
            def preprocess_function(examples):
                # Format as instruction-following
                if "code" in examples:
                    texts = []
                    for code in examples["code"]:
                        if code and len(code.strip()) > 50:  # Filter very short code
                            text = f"### Code:\n{code}\n### End"
                            texts.append(text)
                    return {"text": texts}
                else:
                    # Fallback for different dataset formats
                    return {"text": [str(ex)[:1000] for ex in examples.values() if isinstance(ex, str)]}
            
            processed_dataset = dataset.map(
                preprocess_function,
                batched=True,
                remove_columns=dataset.column_names
            )
            
            # Filter out empty or too short examples
            processed_dataset = processed_dataset.filter(
                lambda x: x["text"] and len(x["text"]) > 100
            )
            
            print(f"✅ Dataset processed: {len(processed_dataset)} samples")
            return processed_dataset
            
        except Exception as e:
            print(f"❌ Dataset preparation failed: {e}")
            # Create synthetic dataset as fallback
            return self._create_synthetic_dataset()
    
    def _create_synthetic_dataset(self) -> Dataset:
        """Create a synthetic coding dataset for testing"""
        print("🔄 Creating synthetic dataset...")
        
        synthetic_examples = [
            "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
            "class Node:\n    def __init__(self, data):\n        self.data = data\n        self.next = None",
            "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
            "def merge_sort(arr):\n    if len(arr) <= 1:\n        return arr\n    mid = len(arr) // 2\n    left = merge_sort(arr[:mid])\n    right = merge_sort(arr[mid:])\n    return merge(left, right)",
            "import requests\n\ndef fetch_data(url):\n    response = requests.get(url)\n    if response.status_code == 200:\n        return response.json()\n    return None"
        ]
        
        # Expand with variations
        expanded_examples = []
        for example in synthetic_examples:
            for i in range(5):  # Create 5 variations of each
                expanded_examples.append(f"### Code:\n{example}\n### End")
        
        return Dataset.from_dict({"text": expanded_examples})
    
    def tokenize_dataset(self, dataset: Dataset, tokenizer: Any) -> Dataset:
        """Tokenize the dataset"""
        print("🔤 Tokenizing dataset...")
        
        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=self.training_config["max_length"],
                return_overflowing_tokens=False,
            )
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing"
        )
        
        print(f"✅ Dataset tokenized: {len(tokenized_dataset)} samples")
        return tokenized_dataset
    
    def setup_training_arguments(self, output_dir: str = "./results") -> TrainingArguments:
        """Set up training arguments"""
        
        return TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.training_config["num_epochs"],
            per_device_train_batch_size=self.training_config["batch_size"],
            per_device_eval_batch_size=self.training_config["batch_size"],
            gradient_accumulation_steps=self.training_config["gradient_accumulation_steps"],
            learning_rate=self.training_config["learning_rate"],
            warmup_steps=self.training_config["warmup_steps"],
            logging_steps=self.training_config["logging_steps"],
            save_steps=self.training_config["save_steps"],
            eval_steps=self.training_config["eval_steps"],
            evaluation_strategy="steps",
            save_strategy="steps",
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            remove_unused_columns=False,
            report_to="wandb" if "wandb" in os.environ else None,
            run_name=f"autonomous-agent-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            fp16=torch.cuda.is_available(),
            gradient_checkpointing=True,
            optim="adamw_torch"
        )
    
    def compute_metrics(self, eval_pred):
        """Compute training metrics"""
        predictions, labels = eval_pred
        
        # Calculate perplexity
        losses = []
        for i in range(len(predictions)):
            loss = torch.nn.functional.cross_entropy(
                torch.tensor(predictions[i]), 
                torch.tensor(labels[i]), 
                ignore_index=-100
            )
            losses.append(loss.item())
        
        avg_loss = np.mean(losses)
        perplexity = np.exp(avg_loss)
        
        return {
            "eval_loss": avg_loss,
            "perplexity": perplexity
        }
    
    def train_model(self, model: Any, tokenizer: Any, train_dataset: Dataset, 
                   eval_dataset: Optional[Dataset] = None) -> Dict[str, Any]:
        """Train the model"""
        print("🚀 Starting training...")
        
        # Setup data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        )
        
        # Split dataset if no eval dataset provided
        if eval_dataset is None:
            split_dataset = train_dataset.train_test_split(test_size=0.1)
            train_dataset = split_dataset["train"]
            eval_dataset = split_dataset["test"]
        
        # Setup training arguments
        training_args = self.setup_training_arguments()
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Start training
        print("⏳ Training in progress...")
        
        try:
            training_result = trainer.train()
            
            # Save the model
            trainer.save_model("./final_model")
            tokenizer.save_pretrained("./final_model")
            
            print("✅ Training completed successfully!")
            
            # Generate training report
            training_report = {
                "training_loss": training_result.training_loss,
                "eval_loss": trainer.evaluate()["eval_loss"],
                "train_runtime": training_result.training_loss,
                "train_samples_per_second": training_result.training_loss,
                "model_name": self.model_name,
                "training_config": self.training_config,
                "timestamp": datetime.now().isoformat()
            }
            
            # Save training report
            with open("training_report.json", "w") as f:
                json.dump(training_report, f, indent=2)
            
            return training_report
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return {"error": str(e)}
    
    def evaluate_model(self, model: Any, tokenizer: Any, test_prompts: List[str]) -> Dict[str, Any]:
        """Evaluate the trained model"""
        print("📊 Evaluating model...")
        
        # Create text generation pipeline
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        
        evaluation_results = []
        
        for prompt in test_prompts:
            try:
                # Generate text
                outputs = generator(prompt, max_new_tokens=100, num_return_sequences=1)
                generated_text = outputs[0]['generated_text']
                
                # Extract only the generated part
                generated_part = generated_text[len(prompt):].strip()
                
                evaluation_results.append({
                    "prompt": prompt,
                    "generated": generated_part,
                    "full_output": generated_text,
                    "length": len(generated_part)
                })
                
            except Exception as e:
                evaluation_results.append({
                    "prompt": prompt,
                    "error": str(e)
                })
        
        return {
            "evaluation_results": evaluation_results,
            "total_prompts": len(test_prompts),
            "successful_generations": len([r for r in evaluation_results if "error" not in r])
        }
    
    def create_training_visualizations(self, training_logs: List[Dict]) -> None:
        """Create visualizations of training progress"""
        print("📈 Creating training visualizations...")
        
        if not training_logs:
            print("⚠️ No training logs available for visualization")
            return
        
        # Extract metrics
        steps = [log.get("step", 0) for log in training_logs if "loss" in log]
        losses = [log.get("loss", 0) for log in training_logs if "loss" in log]
        eval_losses = [log.get("eval_loss", 0) for log in training_logs if "eval_loss" in log]
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Training loss
        if losses:
            axes[0, 0].plot(steps[:len(losses)], losses)
            axes[0, 0].set_title("Training Loss")
            axes[0, 0].set_xlabel("Steps")
            axes[0, 0].set_ylabel("Loss")
            axes[0, 0].grid(True)
        
        # Validation loss
        if eval_losses:
            eval_steps = steps[:len(eval_losses)]
            axes[0, 1].plot(eval_steps, eval_losses, color='orange')
            axes[0, 1].set_title("Validation Loss")
            axes[0, 1].set_xlabel("Steps")
            axes[0, 1].set_ylabel("Loss")
            axes[0, 1].grid(True)
        
        # Learning rate (if available)
        learning_rates = [log.get("learning_rate", 0) for log in training_logs if "learning_rate" in log]
        if learning_rates:
            axes[1, 0].plot(steps[:len(learning_rates)], learning_rates, color='green')
            axes[1, 0].set_title("Learning Rate")
            axes[1, 0].set_xlabel("Steps")
            axes[1, 0].set_ylabel("Learning Rate")
            axes[1, 0].grid(True)
        
        # GPU memory usage (if available)
        gpu_memory = [log.get("train_memory", 0) for log in training_logs if "train_memory" in log]
        if gpu_memory:
            axes[1, 1].plot(steps[:len(gpu_memory)], gpu_memory, color='red')
            axes[1, 1].set_title("GPU Memory Usage")
            axes[1, 1].set_xlabel("Steps")
            axes[1, 1].set_ylabel("Memory (MB)")
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig("training_visualization.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Training visualizations saved")
    
    def package_trained_model(self) -> str:
        """Package the trained model for download"""
        print("📦 Packaging trained model...")
        
        # Create zip file with model and artifacts
        zip_filename = f"autonomous_agent_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip"
        
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add model files
            model_dir = Path("./final_model")
            if model_dir.exists():
                for file_path in model_dir.rglob("*"):
                    if file_path.is_file():
                        zipf.write(file_path, f"model/{file_path.relative_to(model_dir)}")
            
            # Add training artifacts
            artifacts = [
                "training_report.json",
                "training_visualization.png"
            ]
            
            for artifact in artifacts:
                if os.path.exists(artifact):
                    zipf.write(artifact, f"artifacts/{artifact}")
            
            # Add README
            readme_content = f"""# Autonomous AI Agent - Trained Model

Training completed: {datetime.now().isoformat()}
Model base: {self.model_name}
Training method: LoRA fine-tuning

## Files included:
- model/: Trained model files
- artifacts/: Training logs and visualizations

## Usage:
1. Extract the zip file
2. Load the model using transformers library
3. Use with the main application

## Model loading example:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("{self.model_name}")
model = PeftModel.from_pretrained(base_model, "path/to/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/model")
```
"""
            zipf.writestr("README.md", readme_content)
        
        print(f"✅ Model packaged: {zip_filename}")
        return zip_filename

def run_complete_training():
    """Run the complete training pipeline"""
    print("🎯 Starting complete training pipeline...\n")
    
    # Initialize trainer
    trainer = ColabTrainer(model_name="distilgpt2", use_quantization=True)
    
    # Setup environment
    trainer.setup_colab_environment()
    
    # Load model
    model, tokenizer = trainer.load_and_prepare_model()
    
    # Prepare dataset
    dataset = trainer.prepare_coding_dataset("codeparrot/github-code", num_samples=5000)
    tokenized_dataset = trainer.tokenize_dataset(dataset, tokenizer)
    
    # Train model
    training_result = trainer.train_model(model, tokenizer, tokenized_dataset)
    
    if "error" not in training_result:
        # Evaluate model
        test_prompts = [
            "def fibonacci(n):",
            "class BinaryTree:",
            "def quicksort(arr):",
            "import numpy as np",
            "def calculate_mean(data):"
        ]
        
        evaluation_result = trainer.evaluate_model(model, tokenizer, test_prompts)
        
        # Create visualizations (if training logs available)
        # trainer.create_training_visualizations([])
        
        # Package model
        zip_filename = trainer.package_trained_model()
        
        print("\n🎉 Training pipeline completed successfully!")
        print(f"📦 Model package: {zip_filename}")
        print("📊 Evaluation results:")
        print(json.dumps(evaluation_result, indent=2))
        
        # Download model package in Colab
        if COLAB_AVAILABLE:
            try:
                files.download(zip_filename)
                print("📥 Model package download started")
            except:
                print("⚠️ Automatic download failed. Please download manually.")
    
    else:
        print(f"❌ Training failed: {training_result['error']}")

def quick_test_training():
    """Quick test with synthetic data"""
    print("🧪 Running quick test training...\n")
    
    trainer = ColabTrainer(model_name="distilgpt2", use_quantization=False)
    
    # Use synthetic dataset for quick test
    dataset = trainer._create_synthetic_dataset()
    
    # Load model
    model, tokenizer = trainer.load_and_prepare_model()
    
    # Tokenize
    tokenized_dataset = trainer.tokenize_dataset(dataset, tokenizer)
    
    # Quick training (1 epoch, small batch)
    trainer.training_config.update({
        "num_epochs": 1,
        "batch_size": 2,
        "logging_steps": 5,
        "save_steps": 50,
        "eval_steps": 50
    })
    
    # Train
    training_result = trainer.train_model(model, tokenizer, tokenized_dataset)
    
    print(f"🎯 Quick test result: {training_result}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Colab Training Script")
    parser.add_argument("--mode", choices=["full", "test"], default="full",
                       help="Training mode: full or test")
    parser.add_argument("--model", default="distilgpt2", 
                       help="Base model name")
    parser.add_argument("--samples", type=int, default=5000,
                       help="Number of training samples")
    
    # In Colab, we can't use command line args, so we'll use environment variables
    mode = os.environ.get("TRAINING_MODE", "full")
    
    print(f"🚀 Starting Colab training in {mode} mode")
    
    if mode == "test":
        quick_test_training()
    else:
        run_complete_training()
