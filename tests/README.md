# Chat Insight Tests

This directory contains tests for the Chat Insight application.

## Test Coverage

The tests cover the following functionality:

1. **App Tests** (`app_test.py`): Tests for the FastAPI application routes and functionality.
2. **Analyzer Tests** (`analyzer_test.py`): Tests for the chat analysis functionality, including:
   - Visualization limits for group chats (top 20 participants)
   - Emoji analysis limits (top 5 participants)
3. **Template Tests** (`template_test.py`): Tests for the Jinja2 templates, including:
   - Info-text display for group chats
   - No info-text display for individual chats

## Running the Tests

### Prerequisites

Install the required dependencies:

```bash
pip install -r tests/requirements.txt
```

### Running All Tests

```bash
pytest tests/
```

### Running Specific Test Files

```bash
pytest tests/app_test.py
pytest tests/analyzer_test.py
pytest tests/template_test.py
```

### Running Specific Test Functions

```bash
pytest tests/analyzer_test.py::TestVisualizationLimits::test_message_length_distribution_limit
```

## Test Structure

- **Fixtures**: Reusable test components like mock data and configurations
- **Mocks**: Simulated components to isolate tests from external dependencies
- **Assertions**: Verification of expected behavior

## Adding New Tests

When adding new features to the application, please add corresponding tests to maintain code quality and prevent regressions.
