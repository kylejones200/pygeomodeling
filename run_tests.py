#!/usr/bin/env python3
"""
Test runner script for pygeomodeling package.

This script provides various options for running tests with different configurations.
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def run_command(cmd, description=""):
    """Run a command and return the result."""
    if description:
        logger.info("\n%s", description)

    logger.info("Running: %s", ' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        logger.info("Success: %s", description)
        if result.stdout:
            logger.info(result.stdout)
    else:
        logger.error("Failed: %s", description)
        if result.stderr:
            logger.error("STDERR: %s", result.stderr)
        if result.stdout:
            logger.info("STDOUT: %s", result.stdout)

    return result.returncode == 0


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Run tests for SPE9 Geomodeling package"
    )

    parser.add_argument(
        "--quick",
        "-q",
        action="store_true",
        help="Run quick tests only (skip slow tests)",
    )
    parser.add_argument(
        "--coverage", "-c", action="store_true", help="Run tests with coverage report"
    )
    parser.add_argument("--unit", "-u", action="store_true", help="Run unit tests only")
    parser.add_argument(
        "--integration", "-i", action="store_true", help="Run integration tests only"
    )
    parser.add_argument(
        "--parallel", "-p", action="store_true", help="Run tests in parallel"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--file", "-f", type=str, help="Run specific test file")
    parser.add_argument("--function", "-t", type=str, help="Run specific test function")

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
            logger.error("Error: --function requires --file to be specified")
            return 1

    # Add options based on arguments
    if args.verbose:
        cmd.extend(["-v", "--tb=long"])
    else:
        cmd.extend(["--tb=short"])

    if args.coverage:
        cmd.extend(
            [
                "--cov=pygeomodeling",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-report=xml:coverage.xml",
            ]
        )

    if args.quick:
        cmd.extend(["-m", "not slow"])

    if args.unit:
        cmd.extend(["-m", "unit"])

    if args.integration:
        cmd.extend(["-m", "integration"])

    if args.parallel:
        cmd.extend(["-n", "auto"])

    # Add some default options
    cmd.extend(["--strict-markers", "--strict-config", "--disable-warnings"])

    logger.info("SPE9 Geomodeling Test Runner")

    # Check if we're in the right directory
    if not Path("tests").exists():
        logger.error("Error: tests directory not found. Run from project root.")
        return 1

    if not Path("pygeomodeling").exists():
        logger.error("Error: pygeomodeling package not found. Run from project root.")
        return 1

    # Run the tests
    success = run_command(cmd, "Running tests")

    if success:
        logger.info("All tests passed!")

        if args.coverage:
            logger.info("Coverage report generated:")
            logger.info("  - Terminal: see output above")
            logger.info("  - HTML: open htmlcov/index.html")
            logger.info("  - XML: coverage.xml")
    else:
        logger.error("Some tests failed!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
