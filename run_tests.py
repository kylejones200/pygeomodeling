#!/usr/bin/env python3
"""
Test runner script for SPE9 Geomodeling package.

This script provides various options for running tests with different configurations.
"""

import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description=""):
    """Run a command and return the result."""
    if description:
        print(f"\n🔄 {description}")
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✅ Success: {description}")
        if result.stdout:
            print(result.stdout)
    else:
        print(f"❌ Failed: {description}")
        if result.stderr:
            print("STDERR:", result.stderr)
        if result.stdout:
            print("STDOUT:", result.stdout)
    
    return result.returncode == 0


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="Run tests for SPE9 Geomodeling package")
    
    parser.add_argument(
        "--quick", "-q", 
        action="store_true", 
        help="Run quick tests only (skip slow tests)"
    )
    parser.add_argument(
        "--coverage", "-c", 
        action="store_true", 
        help="Run tests with coverage report"
    )
    parser.add_argument(
        "--unit", "-u", 
        action="store_true", 
        help="Run unit tests only"
    )
    parser.add_argument(
        "--integration", "-i", 
        action="store_true", 
        help="Run integration tests only"
    )
    parser.add_argument(
        "--parallel", "-p", 
        action="store_true", 
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Verbose output"
    )
    parser.add_argument(
        "--file", "-f", 
        type=str, 
        help="Run specific test file"
    )
    parser.add_argument(
        "--function", "-t", 
        type=str, 
        help="Run specific test function"
    )
    
    args = parser.parse_args()
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test path
    if args.file:
        cmd.append(f"tests/{args.file}")
    else:
        cmd.append("tests/")
    
    # Add specific function if specified
    if args.function:
        if args.file:
            cmd[-1] += f"::{args.function}"
        else:
            print("Error: --function requires --file to be specified")
            return 1
    
    # Add options based on arguments
    if args.verbose:
        cmd.extend(["-v", "--tb=long"])
    else:
        cmd.extend(["--tb=short"])
    
    if args.coverage:
        cmd.extend([
            "--cov=spe9_geomodeling",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov",
            "--cov-report=xml:coverage.xml"
        ])
    
    if args.quick:
        cmd.extend(["-m", "not slow"])
    
    if args.unit:
        cmd.extend(["-m", "unit"])
    
    if args.integration:
        cmd.extend(["-m", "integration"])
    
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    # Add some default options
    cmd.extend([
        "--strict-markers",
        "--strict-config",
        "--disable-warnings"
    ])
    
    print("🧪 SPE9 Geomodeling Test Runner")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("tests").exists():
        print("❌ Error: tests directory not found. Run from project root.")
        return 1
    
    if not Path("spe9_geomodeling").exists():
        print("❌ Error: spe9_geomodeling package not found. Run from project root.")
        return 1
    
    # Run the tests
    success = run_command(cmd, "Running tests")
    
    if success:
        print("\n🎉 All tests passed!")
        
        if args.coverage:
            print("\n📊 Coverage report generated:")
            print("  - Terminal: see output above")
            print("  - HTML: open htmlcov/index.html")
            print("  - XML: coverage.xml")
    else:
        print("\n💥 Some tests failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
