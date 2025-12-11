"""
Pytest configuration and fixtures
"""
import pytest
import os
from unittest.mock import Mock, AsyncMock
from aiogram.types import User, Chat, Message as TgMessage
from datetime import datetime

# Set test environment variables
os.environ.setdefault("TESTING", "1")


@pytest.fixture
def mock_user():
    """Create a mock Telegram user"""
    user = Mock(spec=User)
    user.id = 123456789
    user.username = "test_user"
    user.first_name = "Test"
    user.last_name = "User"
    user.is_bot = False
    return user


@pytest.fixture
def mock_chat():
    """Create a mock Telegram chat"""
    chat = Mock(spec=Chat)
    chat.id = 123456789
    chat.type = "private"
    return chat


@pytest.fixture
def mock_message(mock_user, mock_chat):
    """Create a mock Telegram message"""
    message = Mock(spec=TgMessage)
    message.message_id = 1
    message.from_user = mock_user
    message.chat = mock_chat
    message.text = None
    message.photo = None
    message.document = None
    message.answer = AsyncMock()
    message.reply = AsyncMock()
    return message


@pytest.fixture
def mock_callback_query(mock_user, mock_chat):
    """Create a mock callback query"""
    callback = Mock()
    callback.id = "test_callback_id"
    callback.from_user = mock_user
    callback.data = None
    callback.message = Mock()
    callback.message.chat = mock_chat
    callback.message.answer = AsyncMock()
    callback.answer = AsyncMock()
    return callback


@pytest.fixture
def mock_supabase_client():
    """Create a mock Supabase client"""
    client = Mock()
    client.table = Mock(return_value=Mock())
    return client


@pytest.fixture
def sample_receipt_data():
    """Sample receipt data for testing"""
    return {
        "id": 1,
        "user_id": 123456789,
        "store": 'ТОО "АлмаСтор"',
        "total": 5420.00,
        "currency": "KZT",
        "purchased_at": "2025-12-15T18:45:00Z",
        "items": [
            {"name": "Товар 1", "quantity": 2, "price": 2710.00, "category": "Продукты"}
        ],
        "receipt_hash": "test_hash_123",
    }


@pytest.fixture
def sample_expense_data():
    """Sample expense data for testing"""
    return {
        "id": 1,
        "user_id": 123456789,
        "source": "receipt",
        "store": 'ТОО "АлмаСтор"',
        "amount": 5420.00,
        "currency": "KZT",
        "date": "2025-12-15",
        "receipt_id": 1,
        "category": "Продукты",
    }

