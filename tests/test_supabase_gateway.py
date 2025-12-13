"""
Tests for SupabaseGateway class
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
from expense_cat_bot import SupabaseGateway


class TestSupabaseGateway:
    """Tests for SupabaseGateway"""
    
    @pytest.fixture
    def gateway(self):
        """Create a SupabaseGateway instance with mocked client"""
        with patch('expense_cat_bot.create_client') as mock_create_client:
            mock_client = Mock()
            mock_create_client.return_value = mock_client
            gateway = SupabaseGateway(
                url="https://test.supabase.co",
                service_key="test_key"
            )
            gateway._client = mock_client
            return gateway
    
    def test_fetch_expenses_list_sync(self, gateway, sample_expense_data):
        """Test fetching expenses list"""
        # Create a proper result object with data attribute
        class MockResult:
            def __init__(self, data):
                self.data = data
        
        # Mock the Supabase query chain - need to handle all method calls
        mock_table = Mock()
        mock_query = Mock()
        
        # Setup the chain: table -> select -> eq -> order -> limit -> execute
        mock_table.select.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.order.return_value = mock_query
        mock_query.limit.return_value = mock_query
        # Important: execute must return MockResult, not Mock
        mock_query.execute.return_value = MockResult([sample_expense_data])
        
        gateway._client.table.return_value = mock_table
        
        # Call with months_back=0 to avoid gte() call
        result = gateway._fetch_expenses_list_sync(user_id=123456789, limit=10, months_back=0)
        
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["id"] == sample_expense_data["id"]
        gateway._client.table.assert_called_once()
    
    def test_fetch_expenses_list_with_date_filter(self, gateway):
        """Test fetching expenses with date filter"""
        mock_table = Mock()
        mock_query = Mock()
        mock_result = Mock()
        
        mock_table.select.return_value = mock_query
        mock_query.eq.return_value = mock_query
        mock_query.gte.return_value = mock_query
        mock_query.order.return_value = mock_query
        mock_query.limit.return_value = mock_query
        mock_query.execute.return_value = mock_result
        mock_result.data = []
        
        gateway._client.table.return_value = mock_table
        
        result = gateway._fetch_expenses_list_sync(
            user_id=123456789, 
            limit=10, 
            months_back=3
        )
        
        # Verify gte was called for date filtering
        mock_query.gte.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_delete_expense_sync(self, gateway, sample_expense_data):
        """Test deleting expense with cascade"""
        # Mock expense query
        mock_expense_table = Mock()
        mock_expense_query = Mock()
        mock_expense_result = Mock()
        mock_expense_result.data = [{
            "receipt_id": 1,
            "bank_transaction_id": None,
            "source": "receipt"
        }]
        
        mock_expense_table.select.return_value = mock_expense_query
        mock_expense_query.eq.return_value = mock_expense_query
        mock_expense_query.execute.return_value = mock_expense_result
        
        # Mock receipt deletion
        mock_receipt_table = Mock()
        mock_receipt_query = Mock()
        mock_receipt_result = Mock()
        mock_receipt_result.data = [{"id": 1}]
        
        mock_receipt_table.delete.return_value = mock_receipt_query
        mock_receipt_query.eq.return_value = mock_receipt_query
        mock_receipt_query.execute.return_value = mock_receipt_result
        
        # Mock expense deletion
        mock_expense_delete_table = Mock()
        mock_expense_delete_query = Mock()
        mock_expense_delete_result = Mock()
        mock_expense_delete_result.data = [{"id": 1}]
        
        mock_expense_delete_table.delete.return_value = mock_expense_delete_query
        mock_expense_delete_query.eq.return_value = mock_expense_delete_query
        mock_expense_delete_query.execute.return_value = mock_expense_delete_result
        
        # Setup table returns - need to handle multiple calls to expenses table
        call_count = {"expenses": 0}
        def table_side_effect(table_name):
            if table_name == gateway.expenses_table:
                call_count["expenses"] += 1
                if call_count["expenses"] == 1:
                    return mock_expense_table  # First call: select
                else:
                    return mock_expense_delete_table  # Second call: delete
            elif table_name == gateway.receipts_table:
                return mock_receipt_table
            return Mock()
        
        gateway._client.table.side_effect = table_side_effect
        
        result = gateway._delete_expense_sync(user_id=123456789, expense_id=1)
        
        assert result == True
        # Verify receipt was deleted (cascade)
        mock_receipt_table.delete.assert_called_once()

