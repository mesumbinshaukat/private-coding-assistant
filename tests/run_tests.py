#!/usr/bin/env python3
"""
Test runner script for the Autonomous AI Agent
Provides convenient ways to run different test suites
"""

import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and handle output"""
    print(f"\n{'='*60}")
    if description:
        print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"\n✅ {description or 'Command'} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {description or 'Command'} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Command not found: {cmd[0]}")
        print("Please ensure pytest is installed: pip install pytest pytest-asyncio pytest-cov")
        return False

def run_all_tests():
    """Run all tests with coverage"""
    cmd = [
        "python", "-m", "pytest",
        "tests/",
        "-v",
        "--cov=api",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "--tb=short"
    ]
    return run_command(cmd, "All tests with coverage")

def run_unit_tests():
    """Run only unit tests"""
    cmd = [
        "python", "-m", "pytest",
        "tests/",
        "-v", 
        "-m", "unit",
        "--tb=short"
    ]
    return run_command(cmd, "Unit tests")

def run_integration_tests():
    """Run only integration tests"""
    cmd = [
        "python", "-m", "pytest",
        "tests/",
        "-v",
        "-m", "integration", 
        "--tb=short"
    ]
    return run_command(cmd, "Integration tests")

def run_api_tests():
    """Run API endpoint tests"""
    cmd = [
        "python", "-m", "pytest",
        "tests/test_api.py",
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, "API endpoint tests")

def run_agent_tests():
    """Run agent core tests"""
    cmd = [
        "python", "-m", "pytest", 
        "tests/test_agent.py",
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, "Agent core tests")

def run_utils_tests():
    """Run utility module tests"""
    cmd = [
        "python", "-m", "pytest",
        "tests/test_utils.py", 
        "-v",
        "--tb=short"
    ]
    return run_command(cmd, "Utility module tests")

def run_performance_tests():
    """Run performance tests"""
    cmd = [
        "python", "-m", "pytest",
        "tests/",
        "-v",
        "-m", "performance",
        "--tb=short"
    ]
    return run_command(cmd, "Performance tests")

def run_specific_test(test_pattern):
    """Run tests matching a specific pattern"""
    cmd = [
        "python", "-m", "pytest",
        "tests/",
        "-v",
        "-k", test_pattern,
        "--tb=short"
    ]
    return run_command(cmd, f"Tests matching pattern: {test_pattern}")

def run_quick_tests():
    """Run quick tests (exclude slow tests)"""
    cmd = [
        "python", "-m", "pytest",
        "tests/",
        "-v",
        "-m", "not slow",
        "--tb=short"
    ]
    return run_command(cmd, "Quick tests (excluding slow tests)")

def run_coverage_report():
    """Generate detailed coverage report"""
    cmd = [
        "python", "-m", "pytest",
        "tests/",
        "--cov=api",
        "--cov-report=html:htmlcov",
        "--cov-report=term-missing",
        "--cov-report=xml",
        "--cov-fail-under=80"
    ]
    success = run_command(cmd, "Coverage report generation")
    
    if success:
        print("\n📊 Coverage reports generated:")
        print("  - HTML report: htmlcov/index.html")
        print("  - XML report: coverage.xml")
        print("  - Terminal report: shown above")
    
    return success

def lint_code():
    """Run code linting"""
    print("\n🔍 Running code linting...")
    
    # Try different linters
    linters = [
        (["python", "-m", "flake8", "api/", "tests/"], "Flake8"),
        (["python", "-m", "pylint", "api/"], "Pylint"),
        (["python", "-m", "black", "--check", "api/", "tests/"], "Black formatting check")
    ]
    
    results = []
    for cmd, name in linters:
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                print(f"✅ {name}: No issues found")
                results.append(True)
            else:
                print(f"⚠️  {name}: Issues found")
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(result.stderr)
                results.append(False)
        except FileNotFoundError:
            print(f"⚠️  {name}: Not installed (skipping)")
            results.append(True)  # Don't fail if linter not installed
    
    return all(results)

def check_dependencies():
    """Check if required test dependencies are installed"""
    print("\n🔧 Checking test dependencies...")
    
    required_packages = [
        "pytest",
        "pytest-asyncio", 
        "pytest-cov",
        "pytest-mock"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package}: Installed")
        except ImportError:
            print(f"❌ {package}: Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n✅ All test dependencies are installed")
    return True

def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Test runner for Autonomous AI Agent")
    parser.add_argument(
        "command", 
        nargs="?",
        choices=[
            "all", "unit", "integration", "api", "agent", "utils", 
            "performance", "quick", "coverage", "lint", "deps"
        ],
        default="all",
        help="Test suite to run"
    )
    parser.add_argument(
        "--pattern", "-k",
        help="Run tests matching this pattern"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--no-coverage",
        action="store_true", 
        help="Skip coverage reporting"
    )
    
    args = parser.parse_args()
    
    # Change to the project root directory
    project_root = Path(__file__).parent.parent
    import os
    os.chdir(project_root)
    
    print("🤖 Autonomous AI Agent Test Suite")
    print(f"📂 Running from: {project_root}")
    
    # Check dependencies first
    if args.command == "deps":
        return 0 if check_dependencies() else 1
    
    if not check_dependencies():
        print("\n❌ Missing dependencies. Run with 'deps' to check what's needed.")
        return 1
    
    # Run the requested test suite
    success = True
    
    if args.pattern:
        success = run_specific_test(args.pattern)
    elif args.command == "all":
        success = run_all_tests()
    elif args.command == "unit":
        success = run_unit_tests()
    elif args.command == "integration":
        success = run_integration_tests()
    elif args.command == "api":
        success = run_api_tests()
    elif args.command == "agent":
        success = run_agent_tests()
    elif args.command == "utils":
        success = run_utils_tests()
    elif args.command == "performance":
        success = run_performance_tests()
    elif args.command == "quick":
        success = run_quick_tests()
    elif args.command == "coverage":
        success = run_coverage_report()
    elif args.command == "lint":
        success = lint_code()
    
    # Final summary
    print(f"\n{'='*60}")
    if success:
        print("🎉 All tests completed successfully!")
        print("✅ Ready for deployment")
    else:
        print("❌ Some tests failed")
        print("🔧 Please fix the issues before deployment")
    print('='*60)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
