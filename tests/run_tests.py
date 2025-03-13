#!/usr/bin/env python3
import unittest
import sys
from tests.test_parser import TestWhatsAppParser

if __name__ == '__main__':
    # Create a test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestWhatsAppParser)
    
    # Run the tests with verbose output
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Return non-zero exit code if tests failed
    sys.exit(not result.wasSuccessful()) 