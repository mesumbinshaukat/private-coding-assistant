#!/usr/bin/env python3
"""
Desktop Integration Examples for Autonomous AI Agent

This script demonstrates how to integrate the AI agent with desktop applications,
including file processing, batch operations, and workflow automation.
"""

import os
import sys
import json
import time
import requests
import argparse
from pathlib import Path
from typing import Dict, Any, List
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading

# Configuration
API_BASE = "https://your-deployment.vercel.app"
API_TOKEN = "autonomous-ai-agent-2024"

class DesktopAIIntegration:
    """Desktop integration wrapper for the AI agent"""
    
    def __init__(self, base_url: str = API_BASE, token: str = API_TOKEN):
        self.base_url = base_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    def _make_request(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict[str, Any]:
        """Make HTTP request to the API"""
        url = f"{self.base_url}{endpoint}"
        
        try:
            if method == "GET":
                response = self.session.get(url, timeout=30)
            elif method == "POST":
                response = self.session.post(url, json=data, timeout=30)
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def process_file(self, file_path: str, operation: str) -> Dict[str, Any]:
        """Process a single file with AI assistance"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return {"error": f"Failed to read file: {e}"}
        
        file_ext = Path(file_path).suffix.lower()
        language_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.cs': 'csharp',
            '.go': 'go',
            '.rs': 'rust'
        }
        
        language = language_map.get(file_ext, 'python')
        
        if operation == "analyze":
            return self._analyze_code(content, language, file_path)
        elif operation == "optimize":
            return self._optimize_code(content, language, file_path)
        elif operation == "document":
            return self._document_code(content, language, file_path)
        elif operation == "debug":
            return self._debug_code(content, language, file_path)
        elif operation == "test":
            return self._generate_tests(content, language, file_path)
        else:
            return {"error": f"Unknown operation: {operation}"}
    
    def _analyze_code(self, content: str, language: str, file_path: str) -> Dict[str, Any]:
        """Analyze code complexity and structure"""
        prompt = f"""
Analyze this {language} code from {file_path}:

{content}

Provide:
1. Code complexity analysis (time/space complexity)
2. Code quality assessment
3. Potential issues or bugs
4. Suggestions for improvement
5. Design pattern analysis
"""
        
        return self._make_request("/reason", "POST", {
            "problem": prompt,
            "domain": "code_analysis",
            "include_math": True
        })
    
    def _optimize_code(self, content: str, language: str, file_path: str) -> Dict[str, Any]:
        """Optimize code for better performance"""
        prompt = f"""
Optimize this {language} code for better performance:

{content}

Focus on:
1. Algorithm efficiency improvements
2. Memory usage optimization
3. Code readability enhancements
4. Best practices implementation
"""
        
        return self._make_request("/generate", "POST", {
            "prompt": prompt,
            "language": language,
            "context": f"Optimizing code from {file_path}"
        })
    
    def _document_code(self, content: str, language: str, file_path: str) -> Dict[str, Any]:
        """Generate documentation for code"""
        prompt = f"""
Generate comprehensive documentation for this {language} code:

{content}

Include:
1. Function/class descriptions
2. Parameter documentation
3. Return value descriptions
4. Usage examples
5. API documentation if applicable
"""
        
        return self._make_request("/generate", "POST", {
            "prompt": prompt,
            "language": "markdown",
            "context": f"Documenting {language} code from {file_path}"
        })
    
    def _debug_code(self, content: str, language: str, file_path: str) -> Dict[str, Any]:
        """Debug and fix code issues"""
        prompt = f"""
Debug this {language} code and fix any issues:

{content}

Focus on:
1. Syntax errors
2. Logic errors
3. Runtime exceptions
4. Edge case handling
5. Error handling improvements
"""
        
        return self._make_request("/generate", "POST", {
            "prompt": prompt,
            "language": language,
            "context": f"Debugging code from {file_path}"
        })
    
    def _generate_tests(self, content: str, language: str, file_path: str) -> Dict[str, Any]:
        """Generate test cases for code"""
        prompt = f"""
Generate comprehensive test cases for this {language} code:

{content}

Include:
1. Unit tests for all functions/methods
2. Edge case testing
3. Error condition testing
4. Integration tests if applicable
5. Test setup and teardown
"""
        
        test_framework_map = {
            'python': 'pytest',
            'javascript': 'jest',
            'java': 'junit',
            'csharp': 'nunit'
        }
        
        framework = test_framework_map.get(language, 'unittest')
        
        return self._make_request("/generate", "POST", {
            "prompt": prompt,
            "language": language,
            "context": f"Generating {framework} tests for {file_path}"
        })
    
    def batch_process_directory(self, directory: str, operation: str, file_patterns: List[str] = None) -> Dict[str, Any]:
        """Process all files in a directory"""
        if file_patterns is None:
            file_patterns = ['*.py', '*.js', '*.ts', '*.java', '*.cpp', '*.c']
        
        directory_path = Path(directory)
        if not directory_path.exists():
            return {"error": f"Directory does not exist: {directory}"}
        
        results = []
        processed_files = []
        
        # Find all matching files
        for pattern in file_patterns:
            for file_path in directory_path.rglob(pattern):
                if file_path.is_file():
                    processed_files.append(file_path)
        
        print(f"Found {len(processed_files)} files to process")
        
        # Process each file
        for i, file_path in enumerate(processed_files):
            print(f"Processing {i+1}/{len(processed_files)}: {file_path.name}")
            
            result = self.process_file(str(file_path), operation)
            results.append({
                "file": str(file_path),
                "result": result,
                "success": "error" not in result
            })
            
            # Rate limiting
            time.sleep(2)
        
        successful = len([r for r in results if r["success"]])
        
        return {
            "total_files": len(processed_files),
            "successful_files": successful,
            "failed_files": len(processed_files) - successful,
            "success_rate": successful / len(processed_files) if processed_files else 0,
            "results": results
        }
    
    def create_project_summary(self, project_path: str) -> Dict[str, Any]:
        """Create a comprehensive project summary"""
        project_path = Path(project_path)
        
        if not project_path.exists():
            return {"error": f"Project path does not exist: {project_path}"}
        
        # Analyze project structure
        file_types = {}
        total_lines = 0
        total_files = 0
        
        for file_path in project_path.rglob("*"):
            if file_path.is_file():
                total_files += 1
                ext = file_path.suffix.lower()
                
                if ext not in file_types:
                    file_types[ext] = {"count": 0, "lines": 0}
                
                file_types[ext]["count"] += 1
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = len(f.readlines())
                        file_types[ext]["lines"] += lines
                        total_lines += lines
                except:
                    pass  # Skip binary or unreadable files
        
        # Get AI analysis of main files
        main_files = []
        for pattern in ['main.py', 'index.js', 'app.py', 'server.py', 'README.md']:
            for file_path in project_path.rglob(pattern):
                if file_path.is_file():
                    main_files.append(str(file_path))
        
        ai_analysis = None
        if main_files:
            # Analyze the first main file found
            analysis_result = self.process_file(main_files[0], "analyze")
            if "error" not in analysis_result:
                ai_analysis = analysis_result
        
        return {
            "project_path": str(project_path),
            "total_files": total_files,
            "total_lines": total_lines,
            "file_types": file_types,
            "main_files": main_files,
            "ai_analysis": ai_analysis,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def interactive_code_assistant(self):
        """Interactive command-line code assistant"""
        print("🤖 Interactive Code Assistant")
        print("Commands: analyze, optimize, document, debug, test, summary, quit")
        
        while True:
            try:
                command = input("\n> ").strip().lower()
                
                if command == "quit":
                    break
                elif command == "analyze":
                    file_path = input("Enter file path: ").strip()
                    result = self.process_file(file_path, "analyze")
                    self._print_result(result)
                elif command == "optimize":
                    file_path = input("Enter file path: ").strip()
                    result = self.process_file(file_path, "optimize")
                    self._print_result(result)
                elif command == "document":
                    file_path = input("Enter file path: ").strip()
                    result = self.process_file(file_path, "document")
                    self._print_result(result)
                elif command == "debug":
                    file_path = input("Enter file path: ").strip()
                    result = self.process_file(file_path, "debug")
                    self._print_result(result)
                elif command == "test":
                    file_path = input("Enter file path: ").strip()
                    result = self.process_file(file_path, "test")
                    self._print_result(result)
                elif command == "summary":
                    project_path = input("Enter project directory: ").strip()
                    result = self.create_project_summary(project_path)
                    self._print_result(result)
                else:
                    print("Unknown command. Available: analyze, optimize, document, debug, test, summary, quit")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
        
        print("👋 Goodbye!")
    
    def _print_result(self, result: Dict[str, Any]):
        """Print result in a formatted way"""
        if "error" in result:
            print(f"❌ Error: {result['error']}")
            return
        
        print("✅ Result:")
        
        if "code" in result:
            print("📝 Generated Code:")
            print("-" * 40)
            print(result["code"])
            print("-" * 40)
        
        if "explanation" in result:
            print("\n💡 Explanation:")
            print(result["explanation"])
        
        if "reasoning_steps" in result:
            print("\n🧠 Reasoning Steps:")
            for step in result["reasoning_steps"]:
                print(f"Step {step.get('step', '?')}: {step.get('description', 'Unknown')}")
                print(f"  {step.get('content', 'No content')}")
        
        if "solution" in result:
            print("\n🎯 Solution:")
            print(result["solution"])

class DesktopGUI:
    """Simple GUI for desktop integration"""
    
    def __init__(self, ai_client: DesktopAIIntegration):
        self.ai_client = ai_client
        self.root = tk.Tk()
        self.root.title("Autonomous AI Agent - Desktop Integration")
        self.root.geometry("800x600")
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the GUI interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="🤖 AI Code Assistant", font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # File selection
        ttk.Label(main_frame, text="Select File:").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        file_frame = ttk.Frame(main_frame)
        file_frame.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        
        self.file_path_var = tk.StringVar()
        self.file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=50)
        self.file_entry.grid(row=0, column=0, padx=(0, 5))
        
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=1)
        
        # Operation selection
        ttk.Label(main_frame, text="Operation:").grid(row=2, column=0, sticky=tk.W, pady=5)
        
        self.operation_var = tk.StringVar(value="analyze")
        operations = ["analyze", "optimize", "document", "debug", "test"]
        
        operation_frame = ttk.Frame(main_frame)
        operation_frame.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        for i, op in enumerate(operations):
            ttk.Radiobutton(operation_frame, text=op.title(), variable=self.operation_var, value=op).grid(row=0, column=i, padx=5)
        
        # Process button
        ttk.Button(main_frame, text="Process File", command=self.process_file).grid(row=3, column=0, columnspan=2, pady=20)
        
        # Batch processing
        ttk.Separator(main_frame, orient='horizontal').grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        ttk.Label(main_frame, text="Batch Processing:", font=("Arial", 12, "bold")).grid(row=5, column=0, columnspan=2, pady=5)
        
        batch_frame = ttk.Frame(main_frame)
        batch_frame.grid(row=6, column=0, columnspan=2, pady=5)
        
        ttk.Label(batch_frame, text="Directory:").grid(row=0, column=0, padx=5)
        
        self.dir_path_var = tk.StringVar()
        self.dir_entry = ttk.Entry(batch_frame, textvariable=self.dir_path_var, width=40)
        self.dir_entry.grid(row=0, column=1, padx=5)
        
        ttk.Button(batch_frame, text="Browse", command=self.browse_directory).grid(row=0, column=2, padx=5)
        ttk.Button(batch_frame, text="Process Directory", command=self.process_directory).grid(row=0, column=3, padx=5)
        
        # Results area
        ttk.Label(main_frame, text="Results:", font=("Arial", 12, "bold")).grid(row=7, column=0, sticky=tk.W, pady=(20, 5))
        
        # Create text widget with scrollbar
        text_frame = ttk.Frame(main_frame)
        text_frame.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.results_text = tk.Text(text_frame, wrap=tk.WORD, height=20, width=80)
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=self.results_text.yview)
        
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(8, weight=1)
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)
    
    def browse_file(self):
        """Open file browser dialog"""
        file_path = filedialog.askopenfilename(
            title="Select a code file",
            filetypes=[
                ("Python files", "*.py"),
                ("JavaScript files", "*.js"),
                ("TypeScript files", "*.ts"),
                ("Java files", "*.java"),
                ("C++ files", "*.cpp"),
                ("C files", "*.c"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            self.file_path_var.set(file_path)
    
    def browse_directory(self):
        """Open directory browser dialog"""
        dir_path = filedialog.askdirectory(title="Select a directory")
        
        if dir_path:
            self.dir_path_var.set(dir_path)
    
    def process_file(self):
        """Process selected file"""
        file_path = self.file_path_var.get()
        operation = self.operation_var.get()
        
        if not file_path:
            messagebox.showerror("Error", "Please select a file first")
            return
        
        if not os.path.exists(file_path):
            messagebox.showerror("Error", "Selected file does not exist")
            return
        
        # Show processing message
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"🔄 Processing {operation} for {os.path.basename(file_path)}...\n\n")
        self.root.update()
        
        # Process in background thread
        def process_thread():
            try:
                result = self.ai_client.process_file(file_path, operation)
                
                # Update GUI in main thread
                self.root.after(0, lambda: self.display_result(result, file_path, operation))
                
            except Exception as e:
                self.root.after(0, lambda: self.display_error(str(e)))
        
        threading.Thread(target=process_thread, daemon=True).start()
    
    def process_directory(self):
        """Process all files in directory"""
        dir_path = self.dir_path_var.get()
        operation = self.operation_var.get()
        
        if not dir_path:
            messagebox.showerror("Error", "Please select a directory first")
            return
        
        if not os.path.exists(dir_path):
            messagebox.showerror("Error", "Selected directory does not exist")
            return
        
        # Show processing message
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"🔄 Batch processing {operation} for directory: {dir_path}...\n\n")
        self.root.update()
        
        # Process in background thread
        def process_thread():
            try:
                result = self.ai_client.batch_process_directory(dir_path, operation)
                
                # Update GUI in main thread
                self.root.after(0, lambda: self.display_batch_result(result, dir_path, operation))
                
            except Exception as e:
                self.root.after(0, lambda: self.display_error(str(e)))
        
        threading.Thread(target=process_thread, daemon=True).start()
    
    def display_result(self, result: Dict[str, Any], file_path: str, operation: str):
        """Display processing result"""
        self.results_text.delete(1.0, tk.END)
        
        if "error" in result:
            self.results_text.insert(tk.END, f"❌ Error processing {os.path.basename(file_path)}:\n{result['error']}\n\n")
            return
        
        self.results_text.insert(tk.END, f"✅ {operation.title()} completed for {os.path.basename(file_path)}\n")
        self.results_text.insert(tk.END, "="*60 + "\n\n")
        
        if "code" in result:
            self.results_text.insert(tk.END, "📝 Generated Code:\n")
            self.results_text.insert(tk.END, "-"*40 + "\n")
            self.results_text.insert(tk.END, result["code"] + "\n")
            self.results_text.insert(tk.END, "-"*40 + "\n\n")
        
        if "explanation" in result:
            self.results_text.insert(tk.END, "💡 Explanation:\n")
            self.results_text.insert(tk.END, result["explanation"] + "\n\n")
        
        if "reasoning_steps" in result:
            self.results_text.insert(tk.END, "🧠 Reasoning Steps:\n")
            for step in result["reasoning_steps"]:
                self.results_text.insert(tk.END, f"Step {step.get('step', '?')}: {step.get('description', 'Unknown')}\n")
                self.results_text.insert(tk.END, f"  {step.get('content', 'No content')}\n\n")
        
        if "solution" in result:
            self.results_text.insert(tk.END, "🎯 Solution:\n")
            self.results_text.insert(tk.END, result["solution"] + "\n\n")
    
    def display_batch_result(self, result: Dict[str, Any], dir_path: str, operation: str):
        """Display batch processing result"""
        self.results_text.delete(1.0, tk.END)
        
        if "error" in result:
            self.results_text.insert(tk.END, f"❌ Error processing directory:\n{result['error']}\n\n")
            return
        
        self.results_text.insert(tk.END, f"📁 Batch {operation} completed for: {os.path.basename(dir_path)}\n")
        self.results_text.insert(tk.END, "="*60 + "\n\n")
        
        self.results_text.insert(tk.END, f"📊 Summary:\n")
        self.results_text.insert(tk.END, f"  Total files: {result['total_files']}\n")
        self.results_text.insert(tk.END, f"  Successful: {result['successful_files']}\n")
        self.results_text.insert(tk.END, f"  Failed: {result['failed_files']}\n")
        self.results_text.insert(tk.END, f"  Success rate: {result['success_rate']:.1%}\n\n")
        
        if result.get('results'):
            self.results_text.insert(tk.END, "📋 File Results:\n")
            for file_result in result['results'][:10]:  # Show first 10
                status = "✅" if file_result['success'] else "❌"
                file_name = os.path.basename(file_result['file'])
                self.results_text.insert(tk.END, f"  {status} {file_name}\n")
            
            if len(result['results']) > 10:
                self.results_text.insert(tk.END, f"  ... and {len(result['results']) - 10} more files\n")
    
    def display_error(self, error: str):
        """Display error message"""
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"💥 An error occurred:\n{error}\n")
    
    def run(self):
        """Start the GUI"""
        self.root.mainloop()

def main():
    """Main function for desktop integration examples"""
    parser = argparse.ArgumentParser(description="Desktop Integration Examples for Autonomous AI Agent")
    parser.add_argument("--mode", choices=["cli", "gui", "batch"], default="cli",
                      help="Interface mode: cli (interactive), gui (graphical), or batch (file processing)")
    parser.add_argument("--file", help="File to process (for batch mode)")
    parser.add_argument("--operation", choices=["analyze", "optimize", "document", "debug", "test"], 
                      default="analyze", help="Operation to perform (for batch mode)")
    parser.add_argument("--directory", help="Directory to process (for batch mode)")
    
    args = parser.parse_args()
    
    # Initialize AI client
    ai_client = DesktopAIIntegration()
    
    # Test connection
    print("🔗 Testing API connection...")
    test_result = ai_client._make_request("/")
    if "error" in test_result:
        print(f"❌ Connection failed: {test_result['error']}")
        print("Please check your API endpoint and token configuration.")
        return
    else:
        print("✅ Connected successfully!")
    
    if args.mode == "cli":
        # Interactive command-line mode
        ai_client.interactive_code_assistant()
        
    elif args.mode == "gui":
        # Graphical user interface mode
        try:
            gui = DesktopGUI(ai_client)
            gui.run()
        except ImportError:
            print("❌ GUI mode requires tkinter. Please install it or use CLI mode.")
            
    elif args.mode == "batch":
        # Batch processing mode
        if args.file:
            # Process single file
            print(f"🔄 Processing file: {args.file}")
            result = ai_client.process_file(args.file, args.operation)
            ai_client._print_result(result)
            
        elif args.directory:
            # Process directory
            print(f"🔄 Processing directory: {args.directory}")
            result = ai_client.batch_process_directory(args.directory, args.operation)
            
            print(f"📊 Batch processing completed:")
            print(f"  Total files: {result['total_files']}")
            print(f"  Successful: {result['successful_files']}")
            print(f"  Failed: {result['failed_files']}")
            print(f"  Success rate: {result['success_rate']:.1%}")
            
        else:
            print("❌ Batch mode requires either --file or --directory argument")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Operation cancelled. Goodbye!")
    except Exception as e:
        print(f"\n💥 An unexpected error occurred: {e}")
        print("Please check your configuration and try again.")
