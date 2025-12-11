# Testing Guide

This document describes how to run and write tests for the LedgerFox bot.

## Setup

1. Install test dependencies:
```bash
pip install pytest pytest-asyncio pytest-cov pytest-mock
```

2. Run all tests:
```bash
pytest
```

3. Run with coverage:
```bash
pytest --cov=ledger_fox_bot --cov-report=html
```

4. Run specific test file:
```bash
pytest tests/test_utils.py
```

5. Run specific test:
```bash
pytest tests/test_utils.py::TestNormalizeStoreName::test_remove_legal_entity_prefixes
```

## Test Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── test_utils.py            # Unit tests for utility functions
├── test_supabase_gateway.py # Tests for database operations
└── test_handlers.py         # Tests for Telegram handlers
```

## Test Categories

### Unit Tests
Test individual functions in isolation:
- `normalize_store_name`
- `parse_datetime_flexible`
- `safe_float`
- `is_url`
- `parse_manual_expense`

Run unit tests:
```bash
pytest -m unit
```

### Integration Tests
Test interactions between components:
- Database operations
- API calls (with mocks)
- End-to-end flows

Run integration tests:
```bash
pytest -m integration
```

### Async Tests
Tests for async functions:
- Telegram handlers
- Database async operations

Mark async tests with `@pytest.mark.asyncio`

## Writing Tests

### Example Unit Test

```python
def test_normalize_store_name():
    result = normalize_store_name('ТОО "АлмаСтор"')
    assert "ТОО" in result
```

### Example Async Test

```python
@pytest.mark.asyncio
async def test_delete_expense_handler():
    # Setup mocks
    mock_message = Mock()
    mock_message.answer = AsyncMock()
    
    # Test
    await handler(mock_message, state)
    
    # Assert
    mock_message.answer.assert_called_once()
```

### Using Fixtures

```python
def test_with_fixture(mock_user, sample_receipt_data):
    assert mock_user.id == 123456789
    assert sample_receipt_data["store"] == 'ТОО "АлмаСтор"'
```

## Mocking External Services

### Mock Supabase

```python
@patch('ledger_fox_bot.SupabaseClient')
def test_with_mock_supabase(mock_client):
    # Setup mock
    mock_client.return_value.table.return_value.select.return_value.execute.return_value.data = []
    
    # Test
    result = gateway.fetch_expenses_list(user_id=123)
    
    # Assert
    assert result == []
```

### Mock Telegram API

```python
@pytest.fixture
def mock_message():
    message = Mock(spec=Message)
    message.answer = AsyncMock()
    return message
```

## CI/CD

Tests run automatically on:
- Push to `main` or `develop` branches
- Pull requests

See `.github/workflows/tests.yml` for configuration.

## Coverage Goals

- Aim for >80% code coverage
- Focus on critical paths:
  - Receipt parsing
  - Expense deletion
  - Report generation
  - Database operations

## Best Practices

1. **Test one thing at a time**: Each test should verify one behavior
2. **Use descriptive names**: Test names should describe what they test
3. **Arrange-Act-Assert**: Structure tests clearly
4. **Mock external dependencies**: Don't make real API calls in tests
5. **Test edge cases**: Empty strings, None values, invalid inputs
6. **Keep tests fast**: Unit tests should run in milliseconds

## Troubleshooting

### Tests fail with import errors
Make sure you're running from the project root:
```bash
cd /path/to/ledgerfox
pytest
```

### Async tests not running
Install `pytest-asyncio`:
```bash
pip install pytest-asyncio
```

### Coverage not working
Install `pytest-cov`:
```bash
pip install pytest-cov
```

