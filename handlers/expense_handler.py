"""
Expense handler - логика обработки ручных расходов.
Изолирует логику парсинга и обработки расходов.
"""
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.fsm.context import FSMContext

# Импорты из основного модуля
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from expense_cat_bot import parse_manual_expense, build_manual_expense_payload, ParsedManualExpense


class ExpenseHandler:
    """Обработчик расходов - изолированная логика обработки ручных расходов"""
    
    def __init__(self, supabase_gateway):
        self.supabase = supabase_gateway
    
    async def parse_and_confirm_expense(
        self,
        message: Message,
        state: FSMContext,
        text: str,
        default_currency: str = "RUB"
    ) -> Optional[Dict[str, Any]]:
        """
        Парсит текст расхода и возвращает данные для подтверждения.
        Возвращает None если не удалось распознать.
        """
        parsed = parse_manual_expense(text, default_currency)
        if not parsed:
            return None
        
        logging.info(f"📝 [EXPENSE_HANDLER] Parsed expense: {parsed.description}, {parsed.amount} {parsed.currency}")
        
        # Сохраняем в состояние
        await state.update_data(
            parsed_expense=parsed,
            expense_text=text
        )
        
        return {
            "parsed": parsed,
            "confirmation_text": self._build_confirmation_text(parsed),
            "keyboard": self._build_confirmation_keyboard()
        }
    
    def _build_confirmation_text(self, parsed: ParsedManualExpense) -> str:
        """Формирует текст подтверждения"""
        currency_symbols = {
            "RUB": "₽",
            "KZT": "₸",
            "USD": "$",
            "EUR": "€",
            "GBP": "£",
            "GEL": "₾",
        }
        currency_symbol = currency_symbols.get(parsed.currency, parsed.currency)
        
        return (
            f"📝 <b>Расход:</b> {parsed.description}\n"
            f"💰 <b>Сумма:</b> {parsed.amount:.2f} {currency_symbol}\n"
            f"📅 <b>Дата:</b> {parsed.occurred_at.strftime('%d.%m.%Y')}\n\n"
            f"Всё верно?"
        )
    
    def _build_confirmation_keyboard(self) -> InlineKeyboardMarkup:
        """Создает клавиатуру подтверждения"""
        return InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="✅ Да, верно", callback_data="expense_confirm_parsed"),
                InlineKeyboardButton(text="❌ Нет, исправить", callback_data="expense_cancel")
            ]
        ])
    
    async def save_expense(
        self,
        callback: CallbackQuery,
        state: FSMContext,
        category: str
    ) -> Dict[str, Any]:
        """
        Сохраняет расход в базу данных.
        Возвращает результат сохранения.
        """
        data = await state.get_data()
        parsed = data.get("parsed_expense")
        
        if not parsed:
            return {
                "success": False,
                "message": "❌ Ошибка: данные расхода не найдены. Начните заново с /expense"
            }
        
        parsed.category = category
        
        try:
            payload = build_manual_expense_payload(callback.from_user.id, parsed)
            result = await self.supabase.record_expense(payload, check_duplicates=True)
            
            currency_symbols = {
                "RUB": "₽",
                "KZT": "₸",
                "USD": "$",
                "EUR": "€",
                "GBP": "£",
                "GEL": "₾",
            }
            currency_symbol = currency_symbols.get(parsed.currency, parsed.currency)
            
            success_message = (
                f"✅ <b>Расход добавлен!</b>\n\n"
                f"📝 {parsed.description}\n"
                f"💰 {parsed.amount:.2f} {currency_symbol}\n"
                f"📂 {category}\n"
                f"📅 {parsed.occurred_at.strftime('%d.%m.%Y')}"
            )
            
            logging.info(f"Manual expense saved: user={callback.from_user.id}, amount={parsed.amount}, category={category}")
            
            return {
                "success": True,
                "message": success_message
            }
        except Exception as exc:
            logging.exception(f"Error saving manual expense: {exc}")
            return {
                "success": False,
                "message": f"❌ Ошибка при сохранении расхода: {str(exc)[:200]}"
            }

