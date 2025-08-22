#!/usr/bin/env python3
"""
Advanced Training Examples for Autonomous AI Agent

This script demonstrates advanced training scenarios including:
- Custom dataset training
- RLHF training with feedback loops
- Performance monitoring and evaluation
- Model versioning and rollback
"""

import requests
import json
import time
import asyncio
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from datetime import datetime, timedelta
import numpy as np

# Configuration
API_BASE = "https://your-deployment.vercel.app"
API_TOKEN = "autonomous-ai-agent-2024"

class AdvancedTrainingClient:
    """Advanced client for training scenarios"""
    
    def __init__(self, base_url: str = API_BASE, token: str = API_TOKEN):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        self.training_history = []
        self.performance_metrics = []
    
    def _make_request(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict[str, Any]:
        """Make HTTP request with error handling"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url, headers=self.headers, timeout=60)
            elif method == "POST":
                response = requests.post(url, headers=self.headers, json=data, timeout=60)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def start_training_session(self, session_config: Dict[str, Any]) -> str:
        """Start a comprehensive training session"""
        session_id = f"training_{int(time.time())}"
        
        print(f"🚀 Starting training session: {session_id}")
        print(f"📋 Configuration:")
        for key, value in session_config.items():
            print(f"  {key}: {value}")
        
        self.training_history.append({
            "session_id": session_id,
            "config": session_config,
            "start_time": datetime.now(),
            "status": "started"
        })
        
        return session_id
    
    def custom_dataset_training(self, dataset_config: Dict[str, Any]) -> Dict[str, Any]:
        """Train with custom dataset"""
        print(f"\n📚 Training with dataset: {dataset_config.get('name', 'Custom')}")
        
        result = self._make_request("/train", "POST", {
            "dataset_name": dataset_config.get("name"),
            "training_type": dataset_config.get("type", "fine_tune"),
            "iterations": dataset_config.get("iterations", 10)
        })
        
        if "error" not in result:
            print(f"✅ Training started successfully")
            print(f"   Type: {result.get('training_type')}")
            print(f"   Iterations: {result.get('iterations')}")
            print(f"   Duration: {result.get('estimated_duration')}")
        else:
            print(f"❌ Training failed: {result.get('error')}")
        
        return result
    
    def rlhf_training_with_feedback(self, feedback_data: List[Dict]) -> Dict[str, Any]:
        """Perform RLHF training with provided feedback"""
        print(f"\n🧠 Starting RLHF training with {len(feedback_data)} feedback samples")
        
        # Submit feedback first
        for i, feedback in enumerate(feedback_data):
            print(f"📝 Submitting feedback {i+1}/{len(feedback_data)}")
            
            feedback_result = self._make_request("/feedback", "POST", feedback)
            if "error" in feedback_result:
                print(f"⚠️  Feedback {i+1} failed: {feedback_result['error']}")
            
            time.sleep(1)  # Rate limiting
        
        # Trigger RLHF training
        print("🔄 Triggering RLHF training...")
        result = self._make_request("/train", "POST", {
            "training_type": "rlhf",
            "iterations": min(len(feedback_data), 10)
        })
        
        return result
    
    def evaluate_performance(self, test_cases: List[Dict]) -> Dict[str, Any]:
        """Evaluate agent performance on test cases"""
        print(f"\n📊 Evaluating performance on {len(test_cases)} test cases")
        
        results = []
        total_time = 0
        successful_cases = 0
        
        for i, test_case in enumerate(test_cases):
            print(f"🧪 Test case {i+1}/{len(test_cases)}: {test_case['description']}")
            
            start_time = time.time()
            
            if test_case['type'] == 'code_generation':
                response = self._make_request("/generate", "POST", {
                    "prompt": test_case['prompt'],
                    "language": test_case.get('language', 'python')
                })
            elif test_case['type'] == 'reasoning':
                response = self._make_request("/reason", "POST", {
                    "problem": test_case['problem'],
                    "domain": test_case.get('domain', 'coding')
                })
            elif test_case['type'] == 'search':
                response = self._make_request("/search", "POST", {
                    "query": test_case['query'],
                    "depth": test_case.get('depth', 5)
                })
            else:
                response = {"error": "Unknown test case type"}
            
            end_time = time.time()
            execution_time = end_time - start_time
            total_time += execution_time
            
            success = "error" not in response and response.get('confidence', 0) > 0.5
            if success:
                successful_cases += 1
            
            results.append({
                "test_case": test_case,
                "response": response,
                "execution_time": execution_time,
                "success": success,
                "confidence": response.get('confidence', 0)
            })
            
            print(f"   ⏱️  Time: {execution_time:.2f}s")
            print(f"   {'✅' if success else '❌'} {'Success' if success else 'Failed'}")
            print(f"   📈 Confidence: {response.get('confidence', 0):.2%}")
            
            time.sleep(2)  # Rate limiting
        
        performance_summary = {
            "total_cases": len(test_cases),
            "successful_cases": successful_cases,
            "success_rate": successful_cases / len(test_cases),
            "average_time": total_time / len(test_cases),
            "average_confidence": np.mean([r['confidence'] for r in results]),
            "results": results
        }
        
        self.performance_metrics.append({
            "timestamp": datetime.now(),
            "metrics": performance_summary
        })
        
        print(f"\n📈 Performance Summary:")
        print(f"   Success Rate: {performance_summary['success_rate']:.2%}")
        print(f"   Average Time: {performance_summary['average_time']:.2f}s")
        print(f"   Average Confidence: {performance_summary['average_confidence']:.2%}")
        
        return performance_summary
    
    def monitor_training_progress(self, session_id: str, duration_minutes: int = 30):
        """Monitor training progress over time"""
        print(f"\n📡 Monitoring training progress for {duration_minutes} minutes")
        
        start_time = time.time()
        end_time = start_time + (duration_minutes * 60)
        
        progress_data = []
        
        while time.time() < end_time:
            status = self._make_request("/status")
            
            if "error" not in status:
                current_metrics = {
                    "timestamp": datetime.now(),
                    "agent_status": status.get('agent_status', {}),
                    "memory_usage": status.get('memory_usage', {}),
                    "training_history": status.get('training_history', [])
                }
                
                progress_data.append(current_metrics)
                
                # Print current status
                agent_status = current_metrics['agent_status']
                memory_usage = current_metrics['memory_usage']
                
                print(f"⏰ {datetime.now().strftime('%H:%M:%S')} - Status Update:")
                print(f"   Model loaded: {agent_status.get('model_loaded', False)}")
                print(f"   Interactions: {agent_status.get('interaction_count', 0)}")
                print(f"   Memory usage: {memory_usage.get('memory_utilization', 0):.1%}")
            
            time.sleep(60)  # Check every minute
        
        return progress_data
    
    def comparative_training_study(self, training_configs: List[Dict]) -> Dict[str, Any]:
        """Compare different training configurations"""
        print(f"\n🔬 Comparative Training Study - {len(training_configs)} configurations")
        
        study_results = []
        
        for i, config in enumerate(training_configs):
            print(f"\n--- Configuration {i+1}/{len(training_configs)} ---")
            print(f"Name: {config.get('name', f'Config {i+1}')}")
            
            # Get baseline performance
            baseline_status = self._make_request("/status")
            baseline_metrics = self._extract_performance_metrics(baseline_status)
            
            # Start training
            training_result = self._make_request("/train", "POST", config)
            
            if "error" in training_result:
                print(f"❌ Training failed: {training_result['error']}")
                continue
            
            print(f"✅ Training started: {training_result.get('status')}")
            
            # Wait for training to complete (simplified)
            time.sleep(30)  # In real scenario, poll for completion
            
            # Get post-training performance
            post_status = self._make_request("/status")
            post_metrics = self._extract_performance_metrics(post_status)
            
            # Calculate improvement
            improvement = self._calculate_improvement(baseline_metrics, post_metrics)
            
            study_results.append({
                "config": config,
                "baseline_metrics": baseline_metrics,
                "post_metrics": post_metrics,
                "improvement": improvement,
                "training_result": training_result
            })
            
            print(f"📊 Improvement: {improvement:.2%}")
        
        # Find best configuration
        best_config = max(study_results, key=lambda x: x['improvement'])
        
        print(f"\n🏆 Best Configuration: {best_config['config'].get('name')}")
        print(f"   Improvement: {best_config['improvement']:.2%}")
        
        return {
            "study_results": study_results,
            "best_config": best_config,
            "summary": self._generate_study_summary(study_results)
        }
    
    def continuous_learning_simulation(self, interactions: List[Dict], learning_frequency: int = 10):
        """Simulate continuous learning from interactions"""
        print(f"\n🔄 Continuous Learning Simulation")
        print(f"   Interactions: {len(interactions)}")
        print(f"   Learning frequency: Every {learning_frequency} interactions")
        
        performance_over_time = []
        interaction_count = 0
        
        for interaction in interactions:
            interaction_count += 1
            
            # Simulate interaction
            if interaction['type'] == 'code_generation':
                response = self._make_request("/generate", "POST", {
                    "prompt": interaction['prompt'],
                    "language": interaction.get('language', 'python')
                })
            else:
                continue  # Skip non-code generation for simplicity
            
            # Simulate feedback based on expected quality
            if "error" not in response:
                confidence = response.get('confidence', 0)
                
                # Generate synthetic feedback
                rating = int(confidence * 5)  # Convert confidence to 1-5 rating
                feedback = {
                    "rating": rating,
                    "comments": f"Interaction {interaction_count} feedback",
                    "specific_feedback": {
                        "code_quality": "high" if confidence > 0.8 else "medium",
                        "explanation_clarity": "clear" if confidence > 0.7 else "unclear"
                    }
                }
                
                self._make_request("/feedback", "POST", feedback)
            
            # Trigger learning periodically
            if interaction_count % learning_frequency == 0:
                print(f"🧠 Triggering learning after {interaction_count} interactions")
                
                training_result = self._make_request("/train", "POST", {
                    "training_type": "continuous",
                    "iterations": 3
                })
                
                # Evaluate current performance
                test_case = {
                    "description": f"Performance check at {interaction_count} interactions",
                    "type": "code_generation",
                    "prompt": "Write a function to calculate factorial recursively",
                    "language": "python"
                }
                
                perf_result = self.evaluate_performance([test_case])
                performance_over_time.append({
                    "interaction_count": interaction_count,
                    "success_rate": perf_result['success_rate'],
                    "average_confidence": perf_result['average_confidence']
                })
                
                print(f"📈 Current performance: {perf_result['success_rate']:.2%} success rate")
            
            time.sleep(1)  # Simulate interaction spacing
        
        return {
            "total_interactions": interaction_count,
            "learning_sessions": len(performance_over_time),
            "performance_trajectory": performance_over_time,
            "final_performance": performance_over_time[-1] if performance_over_time else None
        }
    
    def _extract_performance_metrics(self, status: Dict) -> Dict[str, float]:
        """Extract performance metrics from status"""
        if "error" in status:
            return {"error": True}
        
        agent_status = status.get('agent_status', {})
        performance_metrics = agent_status.get('performance_metrics', {})
        
        return {
            "code_generation_success_rate": performance_metrics.get('code_generation_success_rate', 0),
            "search_relevance_score": performance_metrics.get('search_relevance_score', 0),
            "reasoning_accuracy": performance_metrics.get('reasoning_accuracy', 0),
            "interaction_count": agent_status.get('interaction_count', 0)
        }
    
    def _calculate_improvement(self, baseline: Dict, post: Dict) -> float:
        """Calculate overall improvement percentage"""
        if "error" in baseline or "error" in post:
            return 0.0
        
        metrics = ['code_generation_success_rate', 'search_relevance_score', 'reasoning_accuracy']
        improvements = []
        
        for metric in metrics:
            if metric in baseline and metric in post:
                if baseline[metric] > 0:
                    improvement = (post[metric] - baseline[metric]) / baseline[metric]
                    improvements.append(improvement)
        
        return np.mean(improvements) if improvements else 0.0
    
    def _generate_study_summary(self, study_results: List[Dict]) -> Dict[str, Any]:
        """Generate summary of comparative study"""
        improvements = [r['improvement'] for r in study_results]
        
        return {
            "total_configurations": len(study_results),
            "average_improvement": np.mean(improvements),
            "best_improvement": max(improvements),
            "worst_improvement": min(improvements),
            "std_improvement": np.std(improvements),
            "successful_configs": len([r for r in study_results if r['improvement'] > 0])
        }
    
    def generate_training_report(self, session_id: str) -> str:
        """Generate comprehensive training report"""
        report = f"""
# Advanced Training Report
Session ID: {session_id}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Session Summary
- Training sessions completed: {len(self.training_history)}
- Performance evaluations: {len(self.performance_metrics)}

## Performance Trends
"""
        
        if self.performance_metrics:
            latest_metrics = self.performance_metrics[-1]['metrics']
            report += f"""
- Latest success rate: {latest_metrics['success_rate']:.2%}
- Average response time: {latest_metrics['average_time']:.2f}s
- Average confidence: {latest_metrics['average_confidence']:.2%}
"""
        
        report += f"""

## Recommendations
1. Continue RLHF training with diverse feedback
2. Focus on improving low-confidence responses
3. Monitor performance degradation over time
4. Consider periodic full retraining

## Next Steps
- Implement automated performance monitoring
- Set up continuous learning pipeline
- Establish performance benchmarks
- Create feedback collection system
"""
        
        return report

def create_sample_test_cases() -> List[Dict]:
    """Create sample test cases for evaluation"""
    return [
        {
            "description": "Simple algorithm implementation",
            "type": "code_generation",
            "prompt": "Write a Python function to implement binary search",
            "language": "python"
        },
        {
            "description": "Complex data structure",
            "type": "code_generation", 
            "prompt": "Create a balanced binary search tree class with insertion and deletion",
            "language": "python"
        },
        {
            "description": "Algorithm optimization problem",
            "type": "reasoning",
            "problem": "How to optimize a nested loop algorithm from O(n²) to O(n log n)?",
            "domain": "algorithms"
        },
        {
            "description": "Web search for best practices",
            "type": "search",
            "query": "Python async programming best practices performance optimization",
            "depth": 5
        }
    ]

def create_sample_feedback_data() -> List[Dict]:
    """Create sample feedback data for RLHF training"""
    return [
        {
            "rating": 5,
            "comments": "Excellent code with clear explanations",
            "specific_feedback": {
                "code_quality": "high",
                "explanation_clarity": "very_clear",
                "usefulness": "very_useful"
            }
        },
        {
            "rating": 4,
            "comments": "Good implementation but could use better comments",
            "specific_feedback": {
                "code_quality": "medium",
                "explanation_clarity": "clear",
                "usefulness": "useful"
            }
        },
        {
            "rating": 3,
            "comments": "Algorithm works but not optimal",
            "specific_feedback": {
                "code_quality": "medium",
                "explanation_clarity": "unclear",
                "usefulness": "somewhat_useful"
            }
        }
    ]

def create_training_configurations() -> List[Dict]:
    """Create different training configurations for comparison"""
    return [
        {
            "name": "Standard RLHF",
            "training_type": "rlhf",
            "iterations": 5,
            "dataset_name": None
        },
        {
            "name": "Code-focused Fine-tuning",
            "training_type": "fine_tune",
            "iterations": 10,
            "dataset_name": "codeparrot/github-code"
        },
        {
            "name": "Continuous Learning",
            "training_type": "continuous",
            "iterations": 3,
            "dataset_name": None
        },
        {
            "name": "Story-based Pre-training",
            "training_type": "fine_tune",
            "iterations": 8,
            "dataset_name": "roneneldan/TinyStories"
        }
    ]

def create_sample_interactions() -> List[Dict]:
    """Create sample interactions for continuous learning"""
    return [
        {
            "type": "code_generation",
            "prompt": "Write a function to reverse a string",
            "language": "python"
        },
        {
            "type": "code_generation", 
            "prompt": "Implement merge sort algorithm",
            "language": "python"
        },
        {
            "type": "code_generation",
            "prompt": "Create a class for a linked list",
            "language": "python"
        },
        {
            "type": "code_generation",
            "prompt": "Write a function to find prime numbers",
            "language": "python"
        },
        {
            "type": "code_generation",
            "prompt": "Implement depth-first search for graphs",
            "language": "python"
        }
    ] * 4  # Repeat to have 20 interactions

def main():
    """Main advanced training demonstration"""
    print("🧠 Autonomous AI Agent - Advanced Training Examples")
    print("This script demonstrates sophisticated training scenarios.")
    
    client = AdvancedTrainingClient()
    
    # Start training session
    session_config = {
        "session_type": "advanced_demonstration",
        "focus_areas": ["code_generation", "reasoning", "continuous_learning"],
        "duration": "demo",
        "evaluation_metrics": ["success_rate", "response_time", "confidence"]
    }
    
    session_id = client.start_training_session(session_config)
    
    try:
        # 1. Performance Baseline
        print("\n" + "="*60)
        print("  1. ESTABLISHING PERFORMANCE BASELINE")
        print("="*60)
        
        test_cases = create_sample_test_cases()
        baseline_performance = client.evaluate_performance(test_cases)
        
        # 2. Custom Dataset Training
        print("\n" + "="*60)
        print("  2. CUSTOM DATASET TRAINING")
        print("="*60)
        
        dataset_configs = [
            {
                "name": "codeparrot/github-code",
                "type": "fine_tune",
                "iterations": 5
            },
            {
                "name": "roneneldan/TinyStories", 
                "type": "fine_tune",
                "iterations": 3
            }
        ]
        
        for config in dataset_configs:
            client.custom_dataset_training(config)
            time.sleep(10)  # Wait between trainings
        
        # 3. RLHF Training
        print("\n" + "="*60)
        print("  3. RLHF TRAINING WITH FEEDBACK")
        print("="*60)
        
        feedback_data = create_sample_feedback_data()
        rlhf_result = client.rlhf_training_with_feedback(feedback_data)
        
        # 4. Comparative Study
        print("\n" + "="*60)
        print("  4. COMPARATIVE TRAINING STUDY")
        print("="*60)
        
        training_configs = create_training_configurations()
        study_results = client.comparative_training_study(training_configs)
        
        # 5. Continuous Learning
        print("\n" + "="*60)
        print("  5. CONTINUOUS LEARNING SIMULATION")
        print("="*60)
        
        interactions = create_sample_interactions()
        continuous_results = client.continuous_learning_simulation(interactions, learning_frequency=5)
        
        # 6. Final Evaluation
        print("\n" + "="*60)
        print("  6. FINAL PERFORMANCE EVALUATION")
        print("="*60)
        
        final_performance = client.evaluate_performance(test_cases)
        
        # Calculate overall improvement
        overall_improvement = (final_performance['success_rate'] - baseline_performance['success_rate']) / baseline_performance['success_rate'] if baseline_performance['success_rate'] > 0 else 0
        
        print(f"\n🎯 TRAINING SESSION SUMMARY")
        print(f"   Baseline Success Rate: {baseline_performance['success_rate']:.2%}")
        print(f"   Final Success Rate: {final_performance['success_rate']:.2%}")
        print(f"   Overall Improvement: {overall_improvement:.2%}")
        
        # Generate Report
        print("\n" + "="*60)
        print("  7. GENERATING TRAINING REPORT")
        print("="*60)
        
        report = client.generate_training_report(session_id)
        
        # Save report to file
        with open(f"training_report_{session_id}.md", "w") as f:
            f.write(report)
        
        print(f"📄 Training report saved to: training_report_{session_id}.md")
        
        print(f"\n✅ Advanced training session completed successfully!")
        print(f"   Session ID: {session_id}")
        print(f"   Total duration: {(datetime.now() - client.training_history[0]['start_time']).total_seconds():.0f} seconds")
        
    except Exception as e:
        print(f"\n💥 Training session error: {e}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Training session interrupted. Goodbye!")
    except ImportError as e:
        print(f"📦 Missing dependency: {e}")
        print("Install with: pip install matplotlib numpy")
    except Exception as e:
        print(f"\n💥 An unexpected error occurred: {e}")
        print("Please check your configuration and try again.")
