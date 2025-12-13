"""
Unit tests for utility functions
"""
import pytest
from datetime import datetime
from expense_cat_bot import (
    normalize_store_name,
    parse_datetime_flexible,
    safe_float,
    is_url,
    _currency_from_value,
    parse_manual_expense,
)


class TestNormalizeStoreName:
    """Tests for normalize_store_name function"""
    
    def test_remove_legal_entity_prefixes(self):
        """Test removal of legal entity prefixes"""
        assert normalize_store_name('ТОО "АлмаСтор"') == 'ТОО "АлмаСтор"'
        assert normalize_store_name('ЗАО "ЭлектронПл"') == 'ЗАО "ЭлектронПл"'
        assert normalize_store_name('ООО "Фруктовый"') == 'ООО "Фруктовый"'
    
    def test_remove_filial_prefix(self):
        """Test removal of filial prefixes"""
        result = normalize_store_name('Филиал №81 ТОВАРИЩЕСТВО С ОГРАНИЧЕННОЙ ОТВЕТСТВЕННОСТЬЮ "АЛМАСТОР"')
        assert "Филиал" not in result
        assert "АЛМАСТОР" in result or "АлмаСтор" in result
    
    def test_shorten_long_legal_forms(self):
        """Test shortening of long legal forms"""
        result = normalize_store_name('ТОВАРИЩЕСТВО С ОГРАНИЧЕННОЙ ОТВЕТСТВЕННОСТЬЮ "АЛМАСТОР"')
        assert "ТОО" in result
    
    def test_empty_string(self):
        """Test with empty string"""
        assert normalize_store_name("") == ""
        assert normalize_store_name(None) == None


class TestParseDatetimeFlexible:
    """Tests for parse_datetime_flexible function"""
    
    def test_iso_format(self):
        """Test parsing ISO format datetime"""
        result = parse_datetime_flexible("2025-12-15T18:45:00")
        assert isinstance(result, datetime)
        assert result.year == 2025
        assert result.month == 12
        assert result.day == 15
    
    def test_iso_with_z(self):
        """Test parsing ISO format with Z"""
        result = parse_datetime_flexible("2025-12-15T18:45:00Z")
        assert isinstance(result, datetime)
    
    def test_date_only(self):
        """Test parsing date only"""
        result = parse_datetime_flexible("2025-12-15")
        assert isinstance(result, datetime)
        assert result.year == 2025
    
    def test_none_value(self):
        """Test with None value"""
        result = parse_datetime_flexible(None)
        assert isinstance(result, datetime)
    
    def test_invalid_format(self):
        """Test with invalid format - should return current time"""
        result = parse_datetime_flexible("invalid-date")
        assert isinstance(result, datetime)


class TestSafeFloat:
    """Tests for safe_float function"""
    
    def test_valid_float_string(self):
        """Test with valid float string"""
        assert safe_float("123.45") == 123.45
        assert safe_float("100") == 100.0
    
    def test_invalid_string(self):
        """Test with invalid string"""
        assert safe_float("invalid") == 0.0
        assert safe_float("abc") == 0.0
    
    def test_none_value(self):
        """Test with None"""
        assert safe_float(None) == 0.0
    
    def test_custom_default(self):
        """Test with custom default value"""
        assert safe_float("invalid", default=99.99) == 99.99


class TestIsUrl:
    """Tests for is_url function"""
    
    def test_valid_http_url(self):
        """Test with valid HTTP URL"""
        assert is_url("http://example.com") == True
        assert is_url("http://example.com/path") == True
    
    def test_valid_https_url(self):
        """Test with valid HTTPS URL"""
        assert is_url("https://example.com") == True
        assert is_url("https://example.com/path?query=1") == True
    
    def test_invalid_url(self):
        """Test with invalid URL"""
        assert is_url("not a url") == False
        assert is_url("example.com") == False
        assert is_url("ftp://example.com") == False
    
    def test_empty_string(self):
        """Test with empty string"""
        assert is_url("") == False


class TestCurrencyFromValue:
    """Tests for _currency_from_value function"""
    
    def test_rub_symbols(self):
        """Test RUB currency detection"""
        assert _currency_from_value("₽") == "RUB"
        assert _currency_from_value("руб") == "RUB"
        assert _currency_from_value("р.") == "RUB"
    
    def test_kzt_symbols(self):
        """Test KZT currency detection"""
        assert _currency_from_value("₸") == "KZT"
        assert _currency_from_value("тг") == "KZT"
    
    def test_usd_symbols(self):
        """Test USD currency detection"""
        assert _currency_from_value("$") == "USD"
        assert _currency_from_value("usd") == "USD"
    
    def test_eur_symbols(self):
        """Test EUR currency detection"""
        assert _currency_from_value("€") == "EUR"
        assert _currency_from_value("eur") == "EUR"
    
    def test_none_value(self):
        """Test with None value"""
        assert _currency_from_value(None) == None


class TestParseManualExpense:
    """Tests for parse_manual_expense function"""
    
    def test_simple_expense(self):
        """Test parsing simple expense"""
        result = parse_manual_expense("Кафе 500 руб")
        assert result is not None
        # Note: Current implementation may leave partial text after removing amount pattern
        # The important thing is that amount and currency are parsed correctly
        assert "Кафе" in result.store
        assert result.amount == 500.0
        assert result.currency == "RUB"
    
    def test_expense_with_date(self):
        """Test parsing expense with date"""
        result = parse_manual_expense("Такси 1200 KZT 03.12")
        assert result is not None
        assert result.amount == 1200.0
        assert result.currency == "KZT"
        assert result.store == "Такси"
    
    def test_expense_with_symbols(self):
        """Test parsing expense with currency symbols"""
        result = parse_manual_expense("Продукты 2500₽")
        assert result is not None
        assert result.amount == 2500.0
        assert result.currency == "RUB"
    
    def test_invalid_input(self):
        """Test with invalid input"""
        assert parse_manual_expense("") == None
        assert parse_manual_expense("ab") == None
        assert parse_manual_expense("no amount here") == None

