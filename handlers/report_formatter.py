"""
Report formatter - логика форматирования отчетов.
Изолирует всю логику форматирования текстовых отчетов.
"""
from typing import Dict, Any, Optional
from datetime import datetime
import logging

# Импорты из основного модуля
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from expense_cat_bot import normalize_store_name


class ReportFormatter:
    """Форматтер отчетов - изолированная логика форматирования"""
    
    CURRENCY_SYMBOLS = {
        "RUB": "₽",
        "KZT": "₸",
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "GEL": "₾",
    }
    
    def format_report(self, report: Dict[str, Any], currency: Optional[str] = None) -> str:
        """
        Форматирует отчет с разбивкой по категориям, топ категорий/магазинов.
        Поддерживает мультивалютные отчеты.
        
        Args:
            report: Словарь с данными отчета
            currency: Опциональная валюта для фильтрации (если None, показываются все валюты)
        """
        if not report:
            logging.warning("format_report: empty report provided")
            return "📊 Нет данных за выбранный период."
        
        period = report.get("period", "")
        currencies_data = report.get("currencies_data", {})
        
        # Если указана конкретная валюта, фильтруем данные
        if currency and currency != "all":
            if currency in currencies_data:
                currencies_data = {currency: currencies_data[currency]}
            else:
                logging.info(f"format_report: currency {currency} not found in report")
                return f"📊 Отчёт за {period}\n\n💰 Нет данных для валюты {currency} за выбранный период."
        
        if not currencies_data:
            logging.info(f"format_report: no currencies data for period {period}")
            return f"📊 Отчёт за {period}\n\n💰 Всего расходов: 0.00\n\nНет данных за выбранный период."
        
        display_period = self._format_period(period)
        lines = [f"📊 Отчёт за {display_period}"]
        
        # Всего расходов - всегда показываем по каждой валюте отдельно
        lines.append("💰 Всего расходов:")
        if currencies_data:
            for currency_code in sorted(currencies_data.keys()):
                currency_info = currencies_data[currency_code]
                total = currency_info.get("total", 0.0)
                symbol = self.CURRENCY_SYMBOLS.get(currency_code, currency_code)
                lines.append(f"  {symbol} {total:.2f}")
        else:
            lines.append("  0.00")
        lines.append("")
        
        # Топы по каждой валюте отдельно
        most_expensive_by_currency = report.get("most_expensive_by_currency", {})
        all_currencies = set(currencies_data.keys()) | set(most_expensive_by_currency.keys())
        
        for currency_code in sorted(all_currencies):
            currency_tops = most_expensive_by_currency.get(currency_code, {})
            symbol = self.CURRENCY_SYMBOLS.get(currency_code, currency_code)
            
            # Самая дорогая покупка для этой валюты
            item_info = currency_tops.get("item", {})
            if item_info.get("name") and item_info.get("price", 0) > 0:
                lines.extend(self._format_most_expensive_item(
                    item_info, symbol, currencies_data
                ))
            
            # Самый дорогой расход для этой валюты
            expense_info = currency_tops.get("expense", {})
            if expense_info.get("amount", 0) > 0:
                lines.extend(self._format_most_expensive_expense(
                    expense_info, symbol, currencies_data
                ))
        
        # Формируем отчет по каждой валюте отдельно
        for currency_code in sorted(currencies_data.keys()):
            currency_info = currencies_data[currency_code]
            total = currency_info.get("total", 0.0)
            by_category = currency_info.get("by_category", {})
            symbol = self.CURRENCY_SYMBOLS.get(currency_code, currency_code)
            
            # Заголовок для валюты (если несколько валют)
            if len(currencies_data) > 1:
                lines.append(f"━━━ {symbol} ━━━")
                lines.append(f"💰 Итого: {total:.2f} {symbol}")
            lines.append("")
            
            # Разбивка по категориям для этой валюты
            if by_category:
                lines.append("📂 По категориям:")
                sorted_categories = sorted(by_category.items(), key=lambda x: x[1], reverse=True)
                for category, amount in sorted_categories[:10]:  # Топ 10
                    percentage = (amount / total * 100) if total > 0 else 0
                    lines.append(f"  • {category}: {amount:.2f} {symbol} ({percentage:.1f}%)")
                lines.append("")
        
        return "\n".join(lines)
    
    def _format_period(self, period: str) -> str:
        """Форматирует период для отображения"""
        if " - " in period:
            try:
                start_str, end_str = period.split(" - ")
                start_date = datetime.strptime(start_str, "%Y-%m-%d")
                end_date = datetime.strptime(end_str, "%Y-%m-%d")
                return f"{start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}"
            except:
                return period
        elif len(period) == 7 and period[4] == "-":
            try:
                date_obj = datetime.strptime(period, "%Y-%m")
                months = ["январь", "февраль", "март", "апрель", "май", "июнь",
                         "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]
                month_name = months[date_obj.month - 1]
                return f"{month_name} {date_obj.year}"
            except:
                return period
        return period
    
    def _format_most_expensive_item(
        self, 
        item_info: Dict[str, Any], 
        symbol: str,
        currencies_data: Dict[str, Any]
    ) -> List[str]:
        """Форматирует самую дорогую покупку"""
        lines = []
        item_name = item_info.get("name", "Неизвестно")
        item_price = item_info.get("price", 0.0)
        item_store = item_info.get("store", "Неизвестно")
        item_date = item_info.get("date", "")
        
        date_str = self._format_date(item_date)
        store_name = item_store[:30] if len(item_store) > 30 else item_store
        
        if len(currencies_data) > 1:
            lines.append(f"💎 Самая дорогая покупка ({symbol}):")
        else:
            lines.append("💎 Самая дорогая покупка:")
        
        if date_str:
            lines.append(f"  {item_name} - {item_price:.2f} {symbol} ({store_name}, {date_str})")
        else:
            lines.append(f"  {item_name} - {item_price:.2f} {symbol} ({store_name})")
        lines.append("")
        
        return lines
    
    def _format_most_expensive_expense(
        self,
        expense_info: Dict[str, Any],
        symbol: str,
        currencies_data: Dict[str, Any]
    ) -> List[str]:
        """Форматирует самый дорогой расход"""
        lines = []
        exp_amount = expense_info.get("amount", 0.0)
        exp_store = expense_info.get("store", "Неизвестно")
        exp_date = expense_info.get("date", "")
        
        date_str = self._format_date(exp_date)
        store_name = exp_store[:30] if len(exp_store) > 30 else exp_store
        
        if len(currencies_data) > 1:
            lines.append(f"💸 Самый дорогой расход ({symbol}):")
        else:
            lines.append("💸 Самый дорогой расход:")
        
        if date_str:
            lines.append(f"  {exp_amount:.2f} {symbol} - {store_name} ({date_str})")
        else:
            lines.append(f"  {exp_amount:.2f} {symbol} - {store_name}")
        lines.append("")
        
        return lines
    
    def _format_date(self, date_str: str) -> str:
        """Форматирует дату для отображения"""
        if not date_str:
            return ""
        try:
            if "T" in date_str:
                date_obj = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            else:
                date_obj = datetime.strptime(date_str[:10], "%Y-%m-%d")
            return date_obj.strftime("%d.%m.%Y")
        except:
            return date_str[:10] if len(date_str) >= 10 else date_str

