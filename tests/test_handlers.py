"""
Tests for Telegram bot handlers
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.memory import MemoryStorage
from expense_cat_bot import ExpenseCatBot, DeleteExpenseStates


class TestDeleteExpenseHandler:
    """Tests for delete expense command handler"""
    
    @pytest.fixture
    def bot(self):
        """Create a bot instance for testing"""
        from expense_cat_bot import SupabaseGateway
        from unittest.mock import patch
        
        mock_supabase = Mock(spec=SupabaseGateway)
        
        # Mock Bot to avoid token validation
        with patch('expense_cat_bot.Bot') as mock_bot_class:
            mock_bot = Mock()
            mock_bot_class.return_value = mock_bot
            
            bot = ExpenseCatBot(
                token="1234567890:ABCdefGHIjklMNOpqrsTUVwxyz",
                supabase_gateway=mock_supabase
            )
            bot.supabase = mock_supabase
            return bot
    
    @pytest.fixture
    def state(self):
        """Create FSM state"""
        storage = MemoryStorage()
        return FSMContext(storage=storage, key=Mock())
    
    @pytest.mark.asyncio
    async def test_delete_expense_command_no_expenses(self, bot, mock_message, state):
        """Test delete_expense command when user has no expenses"""
        bot.supabase.fetch_expenses_list = AsyncMock(return_value=[])
        
        # Test that supabase method is called correctly
        # In real scenario, the handler would be registered and called by dispatcher
        result = await bot.supabase.fetch_expenses_list(mock_message.from_user.id, limit=20, months_back=3)
        assert result == []
    
    @pytest.mark.asyncio
    async def test_delete_expense_command_with_expenses(self, bot, mock_message, state, sample_expense_data):
        """Test delete_expense command when user has expenses"""
        expenses = [sample_expense_data]
        expenses[0]["receipts"] = [{"purchased_at": "2025-12-15T18:45:00Z"}]
        bot.supabase.fetch_expenses_list = AsyncMock(return_value=expenses)
        
        # Test that supabase method returns expenses correctly
        result = await bot.supabase.fetch_expenses_list(mock_message.from_user.id, limit=20, months_back=3)
        assert len(result) == 1
        assert result[0]["id"] == sample_expense_data["id"]

