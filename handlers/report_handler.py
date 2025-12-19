"""
Report handler - логика работы с отчетами.
Изолирует всю логику формирования и обработки отчетов.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import logging
import time

from aiogram.types import CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.context import FSMContext


class ReportHandler:
    """Обработчик отчетов - изолированная логика работы с отчетами"""
    
    def __init__(self, supabase_gateway):
        self.supabase = supabase_gateway
    
    async def handle_report_period_callback(
        self, 
        callback: CallbackQuery, 
        state: FSMContext,
        callback_data: str
    ) -> Optional[Dict[str, Any]]:
        """
        Обрабатывает выбор периода для отчета.
        Возвращает словарь с результатом или None если нужно показать меню выбора валюты.
        """
        now = datetime.utcnow()
        period = None
        start_date = None
        end_date = None
        
        if callback_data == "report_current_month":
            period = now.strftime("%Y-%m")
        elif callback_data == "report_last_month":
            last_month = (now.replace(day=1) - timedelta(days=1))
            period = last_month.strftime("%Y-%m")
        elif callback_data == "report_current_week":
            days_since_monday = now.weekday()
            start_date = (now - timedelta(days=days_since_monday)).strftime("%Y-%m-%d")
            end_date = now.strftime("%Y-%m-%d")
        elif callback_data == "report_last_week":
            days_since_monday = now.weekday()
            week_start = now - timedelta(days=days_since_monday + 7)
            week_end = now - timedelta(days=days_since_monday + 1)
            start_date = week_start.strftime("%Y-%m-%d")
            end_date = week_end.strftime("%Y-%m-%d")
        elif callback_data == "report_current_year":
            start_date = now.replace(month=1, day=1).strftime("%Y-%m-%d")
            end_date = now.strftime("%Y-%m-%d")
        elif callback_data == "report_custom":
            # Произвольный период - возвращаем None, чтобы показать запрос даты
            return {"action": "request_start_date"}
        else:
            return {"action": "error", "message": "❌ Неизвестный период для отчета."}
        
        # Получаем отчет
        if not callback.from_user:
            return {"action": "error", "message": "❌ Ошибка: не удалось определить пользователя."}
        
        report_start = time.perf_counter()
        report = await self.supabase.fetch_monthly_report(
            callback.from_user.id,
            period=period,
            start_date=start_date,
            end_date=end_date
        )
        report_time = time.perf_counter() - report_start
        logging.info(f"⏱️ [PERF] Report fetched in {report_time*1000:.1f}ms ({report_time:.2f}s)")
        
        if not report:
            return {"action": "error", "message": "📊 Нет данных за выбранный период."}
        
        currencies_data = report.get("currencies_data", {})
        currencies_list = list(currencies_data.keys())
        
        logging.info(f"📊 [REPORT_HANDLER] Found currencies: {currencies_list}, count: {len(currencies_list)}")
        
        # Если несколько валют - возвращаем информацию для показа меню выбора
        if len(currencies_list) > 1:
            return {
                "action": "select_currency",
                "report": report,
                "currencies": currencies_list,
                "currencies_data": currencies_data,
                "period": period,
                "start_date": start_date,
                "end_date": end_date
            }
        
        # Если одна валюта - возвращаем готовый отчет
        return {
            "action": "show_report",
            "report": report
        }
    
    def create_currency_selection_keyboard(
        self, 
        currencies_list: List[str], 
        currencies_data: Dict[str, Any]
    ) -> InlineKeyboardMarkup:
        """Создает клавиатуру для выбора валюты"""
        currency_symbols = {
            "RUB": "₽",
            "KZT": "₸",
            "USD": "$",
            "EUR": "€",
            "GBP": "£",
            "GEL": "₾",
        }
        
        keyboard_buttons = []
        for currency in sorted(currencies_list):
            symbol = currency_symbols.get(currency, currency)
            total = currencies_data[currency].get("total", 0.0)
            keyboard_buttons.append([
                InlineKeyboardButton(
                    text=f"{symbol} {total:.2f}",
                    callback_data=f"report_currency_{currency}"
                )
            ])
        
        keyboard_buttons.append([
            InlineKeyboardButton(
                text="🌍 Общий отчет (все валюты)",
                callback_data="report_currency_all"
            )
        ])
        
        return InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)
    
    async def handle_currency_selection(
        self,
        callback: CallbackQuery,
        state: FSMContext,
        selected_currency: str,
        report: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Обрабатывает выбор валюты для отчета.
        Возвращает отфильтрованный отчет или общий отчет.
        """
        if selected_currency == "all":
            return {
                "action": "show_report",
                "report": report
            }
        
        # Фильтруем данные по выбранной валюте
        filtered_report = {
            "period": report.get("period", ""),
            "currencies_data": {
                selected_currency: report.get("currencies_data", {}).get(selected_currency, {})
            },
            "most_expensive_by_currency": {
                selected_currency: report.get("most_expensive_by_currency", {}).get(selected_currency, {})
            }
        }
        
        return {
            "action": "show_report",
            "report": filtered_report
        }

