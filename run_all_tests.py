#!/usr/bin/env python3
import unittest
import sys
import os

if __name__ == '__main__':
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Set up the test discovery
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(
        start_dir=os.path.join(script_dir, 'tests'),
        pattern='*_test.py',  # This will find files like analyzer_test.py
        top_level_dir=script_dir
    )
    
    # Also add test_parser.py which doesn't follow the *_test.py pattern
    parser_tests = test_loader.discover(
        start_dir=os.path.join(script_dir, 'tests'),
        pattern='test_*.py',  # This will find files like test_parser.py
        top_level_dir=script_dir
    )
    test_suite.addTests(parser_tests)
    
    # Run the tests with verbose output
    print(f"Running all tests in {os.path.join(script_dir, 'tests')}...")
    result = unittest.TextTestRunner(verbosity=2).run(test_suite)
    
    # Print a summary
    print(f"\nTest Summary:")
    print(f"  Ran {result.testsRun} tests")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Skipped: {len(result.skipped)}")
    
    # Return non-zero exit code if tests failed
    sys.exit(not result.wasSuccessful()) 