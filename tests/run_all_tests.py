#!/usr/bin/env python3
"""
Comprehensive test runner for the SE SME Agent system.
Runs all test categories with proper reporting and coverage.
"""
import pytest
import sys
import os
from pathlib import Path
import argparse
import subprocess
import time


def run_test_category(category, verbose=False, coverage=False):
    """Run tests for a specific category."""
    test_dir = Path(__file__).parent / category
    if not test_dir.exists():
        print(f"Warning: Test directory {test_dir} does not exist")
        return False

    print(f"\n{'='*60}")
    print(f"Running {category.upper()} tests")
    print(f"{'='*60}")

    args = ["-v"] if verbose else []
    if coverage:
        args.extend(["--cov=.", "--cov-report=term-missing"])

    args.append(str(test_dir))

    start_time = time.time()
    result = pytest.main(args)
    end_time = time.time()

    print(
        f"\n{category.upper()} tests completed in {end_time - start_time:.2f} seconds"
    )
    print(f"Result: {'PASSED' if result == 0 else 'FAILED'}")

    return result == 0


def run_specific_tests(test_patterns, verbose=False, coverage=False):
    """Run specific test files or patterns."""
    print(f"\n{'='*60}")
    print(f"Running specific tests: {', '.join(test_patterns)}")
    print(f"{'='*60}")

    args = ["-v"] if verbose else []
    if coverage:
        args.extend(["--cov=.", "--cov-report=term-missing"])

    args.extend(test_patterns)

    start_time = time.time()
    result = pytest.main(args)
    end_time = time.time()

    print(f"\nSpecific tests completed in {end_time - start_time:.2f} seconds")
    print(f"Result: {'PASSED' if result == 0 else 'FAILED'}")

    return result == 0


def generate_test_report():
    """Generate a comprehensive test report."""
    print(f"\n{'='*60}")
    print("GENERATING COMPREHENSIVE TEST REPORT")
    print(f"{'='*60}")

    # Run with coverage and generate HTML report
    args = [
        "--cov=.",
        "--cov-report=html:htmlcov",
        "--cov-report=xml:coverage.xml",
        "--cov-report=term-missing",
        "--junit-xml=test-results.xml",
        "-v",
        "tests/",
    ]

    result = pytest.main(args)

    if result == 0:
        print("\n✅ All tests passed!")
        print("📊 Coverage report generated in htmlcov/")
        print("📋 JUnit XML report generated: test-results.xml")
    else:
        print("\n❌ Some tests failed!")
        print("📊 Coverage report still generated in htmlcov/")

    return result == 0


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description="SE SME Agent Test Runner")
    parser.add_argument(
        "--category",
        choices=["unit", "integration", "e2e", "performance", "security", "monitoring"],
        help="Run tests for a specific category",
    )
    parser.add_argument("--all", action="store_true", help="Run all test categories")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run only unit and integration tests (skip slow tests)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate comprehensive test report with coverage",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument(
        "--coverage", action="store_true", help="Run with coverage reporting"
    )
    parser.add_argument(
        "tests", nargs="*", help="Specific test files or patterns to run"
    )

    args = parser.parse_args()

    # Change to the tests directory
    os.chdir(Path(__file__).parent)

    success = True

    if args.report:
        success = generate_test_report()
    elif args.category:
        success = run_test_category(args.category, args.verbose, args.coverage)
    elif args.all:
        categories = [
            "unit",
            "integration",
            "e2e",
            "performance",
            "security",
            "monitoring",
        ]
        for category in categories:
            if not run_test_category(category, args.verbose, args.coverage):
                success = False
    elif args.fast:
        fast_categories = ["unit", "integration"]
        for category in fast_categories:
            if not run_test_category(category, args.verbose, args.coverage):
                success = False
    elif args.tests:
        success = run_specific_tests(args.tests, args.verbose, args.coverage)
    else:
        # Default: run unit and integration tests
        print("No specific category selected. Running unit and integration tests...")
        for category in ["unit", "integration"]:
            if not run_test_category(category, args.verbose, args.coverage):
                success = False

    # Print summary
    print(f"\n{'='*60}")
    print("TEST EXECUTION SUMMARY")
    print(f"{'='*60}")

    if success:
        print("🎉 All selected tests PASSED!")
        sys.exit(0)
    else:
        print("💥 Some tests FAILED!")
        sys.exit(1)


if __name__ == "__main__":
    main()
