#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ExpenseCatBot - Telegram бот для учета расходов по чекам и выпискам.
Поддерживает распознавание чеков через OpenAI и локальный OCR (Tesseract/PaddleOCR).
"""

import asyncio
import base64
import csv
import hashlib
import io
import json
import logging
import mimetypes
import os
import re
import sys
import time

# Устанавливаем кодировку UTF-8 для вывода
if sys.stdout.encoding != 'utf-8':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    BufferedInputFile, Message, CallbackQuery, InlineKeyboardMarkup, 
    InlineKeyboardButton, BotCommand, LabeledPrice, PreCheckoutQuery, 
    SuccessfulPayment
)
from dotenv import load_dotenv
import requests

try:
    import pdfplumber
    PDF_SUPPORT = True
except ModuleNotFoundError:  # pragma: no cover - import guard
    pdfplumber = None  # type: ignore
    PDF_SUPPORT = False

try:
    import pytesseract
    from pytesseract import Output
    from PIL import Image, ImageOps, ImageFilter, ImageDraw, ImageFont
    TESSERACT_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - import guard
    pytesseract = None  # type: ignore
    Output = None  # type: ignore
    Image = None  # type: ignore
    ImageOps = None  # type: ignore
    ImageFilter = None  # type: ignore
    TESSERACT_AVAILABLE = False

try:
    from pillow_heif import read_heif
    HEIF_SUPPORT = True
except ModuleNotFoundError:  # pragma: no cover - import guard
    read_heif = None  # type: ignore
    HEIF_SUPPORT = False

try:
    from supabase import Client, create_client
    SUPABASE_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - import guard
    Client = Any  # type: ignore
    create_client = None  # type: ignore
    SUPABASE_AVAILABLE = False

try:
    from deep_translator import GoogleTranslator
    TRANSLATION_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - import guard
    GoogleTranslator = None  # type: ignore
    TRANSLATION_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover - import guard
    PaddleOCR = None  # type: ignore
    PADDLEOCR_AVAILABLE = False

# Настройка пути к zbar библиотеке для macOS
if os.name == "posix" and os.path.exists("/opt/homebrew/lib/libzbar.dylib"):
    os.environ.setdefault("DYLD_LIBRARY_PATH", "/opt/homebrew/lib")

try:
    from pyzbar.pyzbar import decode as pyzbar_decode
    QR_READER_AVAILABLE = True
except (ModuleNotFoundError, ImportError, OSError):  # pragma: no cover - import guard
    # ImportError может возникнуть если zbar библиотека не установлена системно
    pyzbar_decode = None  # type: ignore
    QR_READER_AVAILABLE = False
    logging.warning("pyzbar недоступен (установите zbar: brew install zbar на macOS)")

try:
    from qreader import QReader
    QREADER_AVAILABLE = True
    qreader_instance = QReader()
except (ModuleNotFoundError, ImportError):  # pragma: no cover - import guard
    QReader = None  # type: ignore
    qreader_instance = None  # type: ignore
    QREADER_AVAILABLE = False
    logging.warning("qreader недоступен (установите: pip install qreader)")


load_dotenv()
log_level_name = os.getenv("EXPENSECAT_LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)
logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")
log_dir = Path(os.getenv("EXPENSECAT_LOG_DIR", "logs"))
log_dir.mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(log_dir / "ocr_debug.log")
file_handler.setLevel(log_level)
file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logging.getLogger().addHandler(file_handler)
logging.info("ExpenseCatBot logging configured at %s", log_level_name)
logging.info("Active preprocess pipeline hash marker: portrait-fix-v3")


class ReceiptStates(StatesGroup):
    waiting_for_photo = State()
    waiting_for_confirmation = State()


class StatementStates(StatesGroup):
    waiting_for_statement = State()


class DeleteStates(StatesGroup):
    waiting_for_confirmation = State()


class ReportStates(StatesGroup):
    waiting_for_start_date = State()
    waiting_for_end_date = State()
    waiting_for_single_date = State()
    waiting_for_currency = State()


class ExportStates(StatesGroup):
    waiting_for_start_date = State()
    waiting_for_end_date = State()


class FeedbackStates(StatesGroup):
    waiting_for_feedback_type = State()
    waiting_for_feedback_text = State()


class DeleteExpenseStates(StatesGroup):
    waiting_for_confirmation = State()


class SetupStates(StatesGroup):
    waiting_for_currency = State()


class ExpenseStates(StatesGroup):
    waiting_for_expense_text = State()
    waiting_for_category = State()
    waiting_for_confirmation = State()


@dataclass
class ParsedReceiptItem:
    name: str
    quantity: float
    price: float
    category: Optional[str] = None


@dataclass
class ParsedReceipt:
    store: str
    total: float
    currency: str
    purchased_at: datetime
    tax_amount: Optional[float]
    items: List[ParsedReceiptItem]
    merchant_address: Optional[str] = None
    external_id: Optional[str] = None


@dataclass
class ParsedBankTransaction:
    amount: float
    currency: str
    merchant: str
    booked_at: datetime
    description: str
    source_id: str


@dataclass
class ParsedManualExpense:
    description: str  # Описание расхода (например, "кофе", "автосервис", "такси")
    amount: float
    currency: str
    occurred_at: datetime
    category: Optional[str] = None  # Категория расхода
    store: Optional[str] = None  # Магазин (опционально, если указан)
    note: Optional[str] = None


@dataclass
class ProcessingResult:
    success: bool
    summary: Optional[str] = None
    error: Optional[str] = None
    parsed_receipt: Optional[ParsedReceipt] = None
    receipt_payload: Optional[Dict[str, Any]] = None
    qr_url_found_but_failed: Optional[str] = None  # URL QR-кода, если он был найден, но данные не получены
    qr_codes: Optional[List[Dict[str, Any]]] = None  # Найденные QR-коды
    qr_parsing_info: Optional[Dict[str, Dict[str, Any]]] = None  # Информация о парсинге QR-кодов
    qr_time: Optional[float] = None  # Время чтения QR-кодов
    openai_time: Optional[float] = None  # Время OpenAI запроса
    file_bytes: Optional[bytes] = None  # Байты файла для возможного сохранения при отклонении
    mime_type: Optional[str] = None  # MIME тип файла
    recognition_method: Optional[str] = None  # Способ распознавания: 'qr', 'openai_photo', 'openai_qr_data'


@dataclass
class Snapshot:
    step: str
    pil_image: "Image.Image"
    description: str


@dataclass
class Snapshot:
    step: str
    pil_image: "Image.Image"
    description: str


# Базовый промпт для извлечения данных чека (общий для изображений и URL)
RECEIPT_BASE_PROMPT = (
    "Верни только JSON без комментариев со следующей структурой: "
    "{\"store\": str, \"merchant_address\": str | null, "
        "\"purchased_at\": iso8601 datetime (UTC или локальная зона), \"currency\": ISO4217, "
        "\"total\": float, \"tax_amount\": float | null, "
        "\"items\": [{\"name\": str, \"quantity\": float, \"price\": float, \"category\": str | null}]}."
    "\n\nЯЗЫК ЧЕКА (по валюте):"
    "\n- Если в чеке валюта тенге (ТНГ, TNG, KZT) — названия и текст могут быть на русском и казахском, учитывай оба языка при чтении."
    "\n- Если в чеке рубли (руб, RUB) — язык текста русский."
    "\n\nКРИТИЧЕСКИ ВАЖНО - как читать чеки:"
    "\nНа чеках обычно есть колонки: название товара | ЦЕНА ЗА ЕДИНИЦУ | КОЛИЧЕСТВО | ИТОГО ЗА ПОЗИЦИЮ"
    "\n- \"quantity\" - это КОЛИЧЕСТВО товара (сколько штук/кг/литров куплено). Примеры: 2, 3, 1.5, 0.5, 3.452"
    "\n- \"price\" - это ИТОГОВАЯ сумма за всю позицию (цена за единицу × количество). Примеры: если купили 2 штуки по 5.49, то price = 10.98"
    "\n\nШАБЛОНЫ ПОЗИЦИЙ В КАЗАХСТАНСКИХ ЧЕКАХ (KZT/ТНГ):"
    "\nЧеки из разных магазинов могут иметь разные форматы позиций. Определи формат и используй соответствующий шаблон:"
    "\n\nФОРМАТ 1: А-Store и похожие магазины (многострочный формат с вариациями):"
    "\n- ВЕСОВЫЕ товары (с единицей (кг)):"
    "\n  Строка 1: [Номер] [Название (кг)]"
    "\n  Строка 2: [Код товара] [Цена за единицу] [Вес] [Итого] ТНГА"
    "\n  Пример: '1 СТЕЙК КАZBEEF ЧАК АЙ РОЛ ЗАМ ВЕС (кг)' → строка 2: '2214925 11960. 0.380 4545.00 ТНГА'"
    "\n  → name='СТЕЙК КАZBEEF ЧАК АЙ РОЛ ЗАМ ВЕС', quantity=0.380, price=4545.00"
    "\n  Особенности: НЕТ символа * перед количеством, НЕТ символа = перед итогом (или есть без пробела)"
    "\n- ШТУЧНЫЕ товары (с единицей (шт)):"
    "\n  Строка 1: [Номер] [Название (шт)]"
    "\n  Строка 2: [Код товара] [Цена за единицу] *[Количество] =[Итого] ТНГА"
    "\n  Строка 3 (опционально): 'ntin: [NTIN код]' - ИГНОРИРУЙ эту строку при извлечении названия"
    "\n  Пример: '2 ТЕФТЕЛИ МЕАТ ТО ЕАТ КУРИНЫЕ 500ГР (шт)' → строка 2: '710497213769 1875.0 *2 =3750.00 ТНГА'"
    "\n  → name='ТЕФТЕЛИ МЕАТ ТО ЕАТ КУРИНЫЕ 500ГР', quantity=2, price=3750.00"
    "\n  Особенности: ЕСТЬ символ * перед количеством, ЕСТЬ символ = перед итогом"
    "\n\nФОРМАТ 2: Magnum Cash&Carry и похожие магазины (четырехстрочный формат):"
    "\n  Строка 1: [13-значный код товара] [Название товара]"
    "\n  Строка 2: [Количество] X [Цена за единицу]"
    "\n  Строка 3: 'Сома' (служебная строка - ИГНОРИРУЙ)"
    "\n  Строка 4: =[Итого]_Б"
    "\n  Пример:"
    "\n    Строка 1: '4870003986870 КОЧЕ СЫРА ЖАРҚЫН 3,9% 0,45Л Ж/Б'"
    "\n    Строка 2: '2.000 X 685.00'"
    "\n    Строка 3: 'Сома'"
    "\n    Строка 4: '=1370.00_Б'"
    "\n  → name='КОЧЕ СЫРА ЖАРҚЫН 3,9% 0,45Л Ж/Б', quantity=2.000, price=1370.00"
    "\n  Особенности: Код товара всегда 13 цифр в начале первой строки, количество в формате X.XXX, разделитель X между количеством и ценой"
    "\n\nВАЖНО при распознавании:"
    "\n- Определи формат чека по структуре позиций"
    "\n- Для формата 1: различай весовые (кг) и штучные (шт) товары по наличию символов * и ="
    "\n- Для формата 2: извлекай данные из строк 1, 2 и 4, игнорируй строку 'Сома'"
    "\n- Код товара (штрихкод, NTIN) НЕ является частью названия товара - исключи его из name"
    "\n- Строки 'ntin: ...' и 'Сома' - служебные, не включай их в название товара"
    "\n\nФОРМАТ «ДВЕ СТРОКИ НА ПОЗИЦИЮ» (формат различается по магазинам):"
    "\n- В одних чеках ПЕРВАЯ строка = название товара, ВТОРАЯ = код (штрихкод), цена за единицу, количество, ИТОГО за позицию."
    "\n- В других чеках ПЕРВАЯ строка = код товара (числа/штрихкод), ВТОРАЯ = название, цены, количество, итог."
    "\n- В \"name\" всегда вноси только НАЗВАНИЕ товара (текст, по которому понятно, что за товар). Код/штрихкод (длинное число, ntin и т.п.) в \"name\" НЕ попадает!"
    "\n- В \"price\" клади ТОЛЬКО итоговую сумму за позицию (обычно последнее число перед ТНГ/руб или с \"=\"), НЕ цену за единицу и НЕ код!"
    "\n- КРИТИЧНО: итог для каждой позиции бери ТОЛЬКО из второй строки ИМЕННО ЭТОЙ позиции (число после \"=\" или последнее число перед ТНГ на той же строке). НЕ подставляй сумму из соседней позиции! Например: чай с =675.00 ТНГ → price=675; следующая позиция молоко с =810.00 ТНГ → price=810. У каждой позиции своя сумма."
    "\n- В \"quantity\" — количество (кг, шт и т.д.) для этой позиции."
    "\n- Строки вида \"ntin: ...\" и служебные коды — не название и не цена, игнорируй."
    "\n- Пример (название в первой строке): строка 1 \"СТЕЙК КАZBEEF ЧАК АЙ РОЛ ЗАМ ВЕС (кг)\", строка 2 \"2214925 11960. 0.380 4545.00 ТНГ\" → name=название из строки 1, quantity=0.380, price=4545.00."
    "\n- Пример (код в первой строке): строка 1 \"2214925\" или \"710497213769\", строка 2 \"ТЕФТЕЛИ КУРИНЫЕ 500ГР 1875.0 *2 =3750.00 ТНГ\" → name=название из строки 2 (ТЕФТЕЛИ...), quantity=2, price=3750.00."
    "\n\nПРАВИЛА ИЗВЛЕЧЕНИЯ:"
    "\n1. Если на чеке написано \"Товар × 2 = 10.98\", то quantity = 2, price = 10.98"
    "\n2. Если на чеке написано \"Товар 5.49 × 2 = 10.98\", то quantity = 2, price = 10.98 (НЕ 5.49!)"
    "\n3. Если на чеке написано \"Товар 3.452 кг × 34.89 = 120.44\", то quantity = 3.452, price = 120.44"
    "\n4. Если на чеке только одна колонка с суммой (без количества), то quantity = 1, price = эта сумма"
    "\n5. НИКОГДА не путай цену за единицу с количеством! Количество обычно меньше итоговой суммы (кроме случаев когда quantity = 1)"
    "\n\nЦифры 6 и 2 на чеках часто выглядят похоже (шрифт, печать, фото). Внимательно различай: 6 — замкнутый верх, 2 — без верхней дуги. Если сумма позиций не сходится с итогом чека — проверь, не перепутаны ли 6 и 2 в ценах."
    "\n\nКРИТИЧЕСКИ ВАЖНО - НЕ ПОДГОНЯЙ ЦЕНЫ ПОД ОБЩУЮ СУММУ:"
    "\n- Извлекай цены ТОЧНО так, как они указаны на чеке, БЕЗ изменений"
    "\n- Если сумма позиций не совпадает с итогом чека — это нормально, НЕ исправляй цены"
    "\n- НЕ изменяй цены товаров, чтобы сумма позиций совпала с итогом"
    "\n- НЕ распределяй разницу между позициями"
    "\n- Извлекай данные как есть, даже если есть расхождения"
    "\n\nКРИТИЧЕСКИ ВАЖНО - обработка одинаковых и похожих позиций:"
    "\n- Если в чеке есть ДВЕ ОДИНАКОВЫЕ позиции (например, \"Молоко\" появляется дважды), НЕ ОБЪЕДИНЯЙ их в одну!"
    "\n- Если в чеке есть ДВЕ ПОХОЖИЕ позиции с одинаковым брендом (например, \"Bonduelle Кукуруза\" и \"Bonduelle Горошек\"), НЕ ОБЪЕДИНЯЙ их в одну!"
    "\n- Каждая строка в чеке должна быть ОТДЕЛЬНЫМ элементом в массиве items"
    "\n- Даже если товары одинаковые или похожие, сохраняй их как разные элементы массива"
    "\n- Пример 1: если в чеке \"Молоко × 1 = 50\" и \"Молоко × 1 = 50\", то items должен содержать ДВА элемента: [{\"name\": \"Молоко\", \"quantity\": 1, \"price\": 50}, {\"name\": \"Молоко\", \"quantity\": 1, \"price\": 50}]"
    "\n- Пример 2: если в чеке \"Bonduelle Кукуруза × 1 = 595\" и \"Bonduelle Горошек × 1 = 595\", то items должен содержать ДВА элемента: [{\"name\": \"Bonduelle Кукуруза\", \"quantity\": 1, \"price\": 595}, {\"name\": \"Bonduelle Горошек\", \"quantity\": 1, \"price\": 595}]"
    "\n- НЕ ДЕЛАЙ так: [{\"name\": \"Молоко\", \"quantity\": 2, \"price\": 100}] - это НЕПРАВИЛЬНО!"
    "\n- НЕ ДЕЛАЙ так: [{\"name\": \"Bonduelle\", \"quantity\": 2, \"price\": 1190}] - это НЕПРАВИЛЬНО!"
    "\n- ВАЖНО: Каждая отдельная строка в чеке = отдельный элемент в массиве items. Не объединяй строки, даже если они похожи!"
    "\n\nКРИТИЧЕСКИ ВАЖНО - названия товаров:"
    "\n- Сохраняй ПОЛНОЕ название товара из чека, НЕ ОБРЕЗАЙ его!"
    "\n- Если в чеке написано \"Птицынно-куриный тандалмалы бодене жумырткасы\", то name должен быть \"Птицынно-куриный тандалмалы бодене жумырткасы\", а НЕ \"Птицынно-куриный тандалмалы\""
    "\n- Полное название важно для правильного определения категории товара"
    "\n- НЕ сокращай названия товаров, даже если они длинные"
    "\n- НЕ обрезай названия товаров до определенной длины"
    "\n\nДля каждого товара определи категорию на основе его ПОЛНОГО названия. "
    "Используй категории: "
    "\"Продукты\", \"Мясо/Рыба\", \"Молочные продукты\", \"Хлеб/Выпечка\", "
    "\"Овощи/Фрукты\", \"Напитки\", \"Алкоголь\", \"Сладости\", \"Кондитерские изделия\", "
    "\"Одежда\", \"Аксессуары\", \"Бытовая химия\", \"Косметика/Гигиена\", "
    "\"Электроника\", \"Компьютеры/Телефоны\", \"Мебель\", \"Дом/Интерьер\", "
    "\"Ресторан/Кафе\", \"Доставка еды\", \"Фастфуд\", \"Транспорт\", \"Такси\", \"Парковка\", "
    "\"Бензин/Топливо\", \"Медицина\", \"Лекарства\", "
    "\"Образование\", \"Книги\", \"Канцтовары\", \"Игрушки\", \"Детские товары\", "
    "\"Развлечения\", \"Кино\", \"Театр\", \"Концерты\", \"Спорт\", "
    "\"Путешествия\", \"Отель\", \"Авиабилеты\", \"Железнодорожные билеты\", "
    "\"Коммунальные\", \"Электричество\", \"Вода\", \"Газ\", \"Отопление\", "
    "\"Интернет/Связь\", \"Мобильная связь\", \"Подписки\", \"Стриминг\", "
    "\"Страхование\", \"Налоги\", \"Штрафы\", \"Банковские услуги\", "
    "\"Ремонт\", \"Строительные материалы\", \"Инструменты\", \"Садоводство\", "
    "\"Животные\", \"Корм для животных\", \"Ветеринария\", \"Другое\". "
    "Если категория не очевидна, используй \"Другое\"."
)

RECEIPT_EXTRACTION_PROMPT = os.getenv(
    "RECEIPT_OCR_PROMPT",
    f"Ты извлекаешь данные из фото кассовых чеков. {RECEIPT_BASE_PROMPT}",
).strip()

RECEIPT_DATA_STRUCTURING_PROMPT = os.getenv(
    "RECEIPT_DATA_PROMPT",
    f"Ты получаешь данные чека, которые были извлечены с веб-страницы. Структурируй и улучши эти данные. ОБЯЗАТЕЛЬНО определи категорию для КАЖДОГО товара на основе его названия. Если в данных нет категорий или они null, ты должен их добавить. {RECEIPT_BASE_PROMPT}",
).strip()
RECEIPT_MODEL = os.getenv("RECEIPT_OCR_MODEL", "gpt-4o").strip()
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
RECEIPT_TEMPERATURE = float(os.getenv("RECEIPT_OCR_TEMPERATURE", "0.1"))
DEFAULT_RECEIPT_TIMEOUT = int(os.getenv("RECEIPT_OCR_TIMEOUT", "120"))
RECEIPT_PIPELINE_MODE = os.getenv("RECEIPT_PIPELINE_MODE", "text").strip().lower()
USE_OPENAI_FOR_RECEIPTS = RECEIPT_PIPELINE_MODE == "ai"
RECEIPT_FALLBACK_LANG = os.getenv("RECEIPT_FALLBACK_LANG", "rus+kaz+eng").strip() or "rus+kaz+eng"
TESSERACT_CONFIG = os.getenv("TESSERACT_CONFIG", "--oem 3 --psm 6").strip()
# Метод поиска углов: "brightness" (анализ яркости), "contour" (контуры), "text" (текстовые проекции)
CORNER_DETECTION_METHOD = os.getenv("CORNER_DETECTION_METHOD", "brightness").strip().lower()
# OCR движок: "tesseract", "paddleocr" или "both" (для сравнения обоих)
OCR_ENGINE = os.getenv("OCR_ENGINE", "both").strip().lower()

_receipt_parser: Optional["ReceiptParserAI"] = None
_paddleocr_instance: Optional[Any] = None


def get_paddleocr_instance():
    """Получает или создает экземпляр PaddleOCR (ленивая инициализация)."""
    global _paddleocr_instance
    if not PADDLEOCR_AVAILABLE or PaddleOCR is None:
        return None
    if _paddleocr_instance is None:
        try:
            # Используем многоязычную модель (поддерживает русский, казахский, английский)
            # Пробуем сначала китайскую модель (многоязычную), она более универсальная
            logging.info("Initializing PaddleOCR...")
            
            # Проверяем, есть ли ошибка подключения к хостам моделей
            network_error_keywords = [
                "No model hoster is available",
                "network connection",
                "HuggingFace",
                "ModelScope",
                "AIStudio",
                "BOS"
            ]
            
            try:
                _paddleocr_instance = PaddleOCR(
                    use_angle_cls=True,
                    lang='ch',  # Китайская модель включает русский, английский и другие языки
                    use_gpu=False,  # Используем CPU, можно включить GPU если доступен
                    show_log=False
                )
                logging.info("PaddleOCR initialized with Chinese (multilingual) model")
            except Exception as exc_ch:
                error_msg = str(exc_ch)
                # Проверяем, является ли это ошибкой подключения к хостам моделей
                is_network_error = any(keyword in error_msg for keyword in network_error_keywords)
                
                if is_network_error:
                    logging.warning(
                        f"PaddleOCR cannot connect to model hosts: {error_msg}\n"
                        f"PaddleOCR will be disabled. Using Tesseract instead.\n"
                        f"To fix: ensure internet connection or download models manually."
                    )
                    # Помечаем PaddleOCR как недоступный для этой сессии
                    _paddleocr_instance = False  # Используем False вместо None для обозначения ошибки сети
                    return None
                
                logging.warning(f"Failed to initialize PaddleOCR with 'ch' model: {exc_ch}, trying 'ru'...")
                try:
                    _paddleocr_instance = PaddleOCR(
                        use_angle_cls=True,
                        lang='ru',  # Русская модель
                        use_gpu=False,
                        show_log=False
                    )
                    logging.info("PaddleOCR initialized with Russian model")
                except Exception as exc_ru:
                    error_msg_ru = str(exc_ru)
                    is_network_error_ru = any(keyword in error_msg_ru for keyword in network_error_keywords)
                    
                    if is_network_error_ru:
                        logging.warning(
                            f"PaddleOCR cannot connect to model hosts: {error_msg_ru}\n"
                            f"PaddleOCR will be disabled. Using Tesseract instead."
                        )
                        _paddleocr_instance = False
                        return None
                    
                    logging.error(f"Failed to initialize PaddleOCR with 'ru' model: {exc_ru}")
                    # Пробуем без use_angle_cls
                    try:
                        _paddleocr_instance = PaddleOCR(
                            use_angle_cls=False,
                            lang='ch',
                            use_gpu=False,
                            show_log=False
                        )
                        logging.info("PaddleOCR initialized without angle classification")
                    except Exception as exc_no_cls:
                        error_msg_no_cls = str(exc_no_cls)
                        is_network_error_no_cls = any(keyword in error_msg_no_cls for keyword in network_error_keywords)
                        
                        if is_network_error_no_cls:
                            logging.warning(
                                f"PaddleOCR cannot connect to model hosts: {error_msg_no_cls}\n"
                                f"PaddleOCR will be disabled. Using Tesseract instead."
                            )
                            _paddleocr_instance = False
                            return None
                        
                        logging.error(f"Failed to initialize PaddleOCR without cls: {exc_no_cls}")
                        # Не пробрасываем исключение, просто возвращаем None
                        _paddleocr_instance = False
                        return None
            
            if _paddleocr_instance and _paddleocr_instance is not False:
                logging.info("PaddleOCR initialized successfully")
        except Exception as exc:
            error_msg = str(exc)
            is_network_error = any(keyword in error_msg for keyword in network_error_keywords)
            
            if is_network_error:
                logging.warning(
                    f"PaddleOCR cannot connect to model hosts: {error_msg}\n"
                    f"PaddleOCR will be disabled. Using Tesseract instead."
                )
            else:
                logging.error(f"Failed to initialize PaddleOCR: {exc}")
                _paddleocr_instance = False
            return None
    
    # Если _paddleocr_instance == False, значит была ошибка сети, возвращаем None
    if _paddleocr_instance is False:
        return None
    
    return _paddleocr_instance


class ReceiptParsingError(RuntimeError):
    """Raised when AI receipt parsing fails."""


class StatementParsingError(RuntimeError):
    """Raised when bank statement parsing fails."""


def format_user_friendly_error(exc: Exception) -> str:
    """
    Преобразует технические ошибки в понятные сообщения для пользователя.
    """
    error_str = str(exc)
    error_type = type(exc).__name__
    
    # Проверяем тип исключения
    import requests
    if isinstance(exc, requests.exceptions.Timeout):
        return "Превышено время ожидания ответа от сервера. Пожалуйста, попробуйте отправить фото чека еще раз."
    
    if isinstance(exc, requests.exceptions.ConnectionError):
        return "Не удалось подключиться к серверу распознавания. Пожалуйста, попробуйте позже."
    
    if isinstance(exc, requests.exceptions.RequestException):
        return "Ошибка при отправке запроса. Пожалуйста, попробуйте позже."
    
    # Ошибки таймаута (по тексту)
    if "timeout" in error_str.lower() or "timed out" in error_str.lower() or "read timeout" in error_str.lower():
        return "Превышено время ожидания ответа от сервера. Пожалуйста, попробуйте отправить фото чека еще раз."
    
    # Ошибки соединения
    if "connection" in error_str.lower() or "HTTPSConnectionPool" in error_str or "ConnectionPool" in error_str:
        return "Не удалось подключиться к серверу распознавания. Пожалуйста, попробуйте позже."
    
    # Ошибки сети
    if "network" in error_str.lower() or "socket" in error_str.lower() or "dns" in error_str.lower():
        return "Ошибка сетевого соединения. Проверьте подключение к интернету и попробуйте еще раз."
    
    # Ошибки OpenAI API
    if "openai" in error_str.lower() or "api.openai.com" in error_str.lower():
        if "rate limit" in error_str.lower():
            return "Превышен лимит запросов. Пожалуйста, подождите немного и попробуйте еще раз."
        if "401" in error_str or "unauthorized" in error_str.lower():
            return "Ошибка авторизации. Обратитесь в поддержку через /feedback"
        if "429" in error_str:
            return "Слишком много запросов. Пожалуйста, подождите немного и попробуйте еще раз."
        if "500" in error_str or "502" in error_str or "503" in error_str:
            return "Сервер временно недоступен. Пожалуйста, попробуйте позже."
        return "Ошибка при распознавании чека. Пожалуйста, попробуйте отправить фото чека еще раз."
    
    # Ошибки обработки изображения
    if "image" in error_str.lower() or "pil" in error_str.lower() or "pillow" in error_str.lower():
        return "Не удалось обработать изображение. Убедитесь, что файл является корректным изображением."
    
    # Ошибки JSON парсинга
    if "json" in error_str.lower() or "parse" in error_str.lower():
        return "Не удалось распознать данные чека. Пожалуйста, попробуйте отправить фото чека еще раз."
    
    # Если это уже дружелюбное сообщение (не содержит технических деталей), возвращаем как есть
    if not any(keyword in error_str.lower() for keyword in [
        "error", "exception", "failed", "timeout", "connection", "http", "api", 
        "traceback", "stack", "trace", "file", "line", "code", "status"
    ]):
        return error_str
    
    # Общие ошибки - показываем дружелюбное сообщение без технических деталей
    return "Не удалось распознать чек. Пожалуйста, попробуйте отправить фото чека еще раз или убедитесь, что на фото четко виден кассовый чек."


class ReceiptParserAI:
    """Minimal wrapper around OpenAI Chat Completions for receipt extraction."""

    def __init__(
        self,
        api_key: str,
        model: str = RECEIPT_MODEL,
        base_url: str = OPENAI_BASE_URL,
        prompt: str = RECEIPT_EXTRACTION_PROMPT,
        data_prompt: str = RECEIPT_DATA_STRUCTURING_PROMPT,
        temperature: float = RECEIPT_TEMPERATURE,
        timeout: int = DEFAULT_RECEIPT_TIMEOUT,
    ) -> None:
        if not api_key:
            raise ReceiptParsingError("OPENAI_API_KEY is required for receipt parsing.")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.prompt = prompt
        self.data_prompt = data_prompt
        self.temperature = temperature
        self.timeout = timeout
        self._session = requests.Session()

    async def parse(
        self, 
        file_bytes: bytes, 
        mime_type: str, 
        qr_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Отправляет изображение в OpenAI и возвращает полный JSON response без изменений.
        Если передан qr_data, отправляются данные для структурирования (без изображения).
        """
        # Если есть данные из QR-кода, проверяем, содержат ли они позиции товаров
        if qr_data:
            items_in_qr = qr_data.get("items") or []
            # Если в данных из QR-кода нет позиций или только одна позиция "Без позиций", используем изображение
            if not items_in_qr or (len(items_in_qr) == 1 and items_in_qr[0].get("name") in ["Без позиций", "Покупка"]):
                logging.warning(f"⚠️ Данные из QR-кода не содержат позиций товаров (items: {len(items_in_qr)}), используем изображение для распознавания")
                if not mime_type.startswith("image/"):
                    raise ReceiptParsingError("На данный момент поддерживаются только изображения чеков.")
                data_url = build_data_url(file_bytes, mime_type)
                payload = self._build_payload(data_url)
            else:
                logging.info(f"✅ Используем данные из QR-кода для структурирования (найдено {len(items_in_qr)} позиций), изображение НЕ отправляется")
            payload = self._build_payload("", qr_data=qr_data)
        else:
            # Если QR-кода нет, используем изображение
            if not mime_type.startswith("image/"):
                raise ReceiptParsingError("На данный момент поддерживаются только изображения чеков.")
        data_url = build_data_url(file_bytes, mime_type)
        payload = self._build_payload(data_url)
        
        # Логируем запрос в OpenAI
        # Создаем копию payload для логирования с обрезанным data_url для читаемости
        payload_for_log = json.loads(json.dumps(payload))
        if "messages" in payload_for_log:
            for msg in payload_for_log["messages"]:
                if "content" in msg and isinstance(msg["content"], list):
                    for content_item in msg["content"]:
                        if content_item.get("type") == "image_url" and "image_url" in content_item:
                            url = content_item["image_url"].get("url", "")
                            if len(url) > 100:
                                content_item["image_url"]["url"] = url[:100] + f"... (truncated, total length: {len(url)})"
        
        logging.info(f"OpenAI request payload: {json.dumps(payload_for_log, ensure_ascii=False, indent=2)}")
        logging.info(f"System prompt length: {len(self.prompt)} chars, prompt_cache_key: {payload.get('prompt_cache_key', 'not set')}")
        
        response_json = await asyncio.to_thread(self._post_payload, payload)
        
        # Логируем весь body ответа
        logging.info(f"OpenAI full response body: {json.dumps(response_json, ensure_ascii=False, indent=2)}")
        
        return response_json

    async def improve_receipt_data(self, receipt_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Улучшает данные чека через OpenAI без отправки изображения.
        Используется когда данные уже получены из QR-кода, но нужно улучшить категории и структуру.
        """
        # Создаем промпт для улучшения данных
        improvement_prompt = (
            "Ты улучшаешь данные чека, которые уже были извлечены из QR-кода. "
            "Верни улучшенный JSON без комментариев со следующей структурой: "
            '{"store": str, "merchant_address": str | null, '
            '"purchased_at": iso8601 datetime, "currency": ISO4217, '
            '"total": float, "tax_amount": float | null, '
            '"items": [{"name": str, "quantity": float, "price": float, "category": str | null}]}. '
            "Улучши категории товаров на основе их названий. "
            "Используй категории: "
            "\"Продукты\", \"Мясо/Рыба\", \"Молочные продукты\", \"Хлеб/Выпечка\", "
            "\"Овощи/Фрукты\", \"Напитки\", \"Алкоголь\", \"Сладости\", \"Кондитерские изделия\", "
            "\"Одежда\", \"Аксессуары\", \"Бытовая химия\", \"Косметика/Гигиена\", "
            "\"Электроника\", \"Компьютеры/Телефоны\", \"Мебель\", \"Дом/Интерьер\", "
            "\"Ресторан/Кафе\", \"Доставка еды\", \"Фастфуд\", \"Транспорт\", \"Такси\", \"Парковка\", "
            "\"Бензин/Топливо\", \"Медицина\", \"Лекарства\", "
            "\"Образование\", \"Книги\", \"Канцтовары\", \"Игрушки\", \"Детские товары\", "
            "\"Развлечения\", \"Кино\", \"Театр\", \"Концерты\", \"Спорт\", "
            "\"Путешествия\", \"Отель\", \"Авиабилеты\", \"Железнодорожные билеты\", "
            "\"Коммунальные\", \"Электричество\", \"Вода\", \"Газ\", \"Отопление\", "
            "\"Интернет/Связь\", \"Мобильная связь\", \"Подписки\", \"Стриминг\", "
            "\"Страхование\", \"Налоги\", \"Штрафы\", \"Банковские услуги\", "
            "\"Ремонт\", \"Строительные материалы\", \"Инструменты\", \"Садоводство\", "
            "\"Животные\", \"Корм для животных\", \"Ветеринария\", \"Другое\". "
            "Если категория не очевидна, используй \"Другое\". "
            "Исправь названия товаров, если они некорректны. "
            "Сохрани все исходные данные, только улучши их."
        )
        
        # Формируем сообщение с данными чека
        receipt_json = json.dumps(receipt_data, ensure_ascii=False, indent=2)
        user_message = f"Улучши данные чека:\n\n{receipt_json}"
        
        payload = {
            "model": self.model,
            "response_format": {"type": "json_object"},
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "system",
                    "content": improvement_prompt,
                },
                {
                    "role": "user",
                    "content": user_message,
                },
            ],
        }
        
        # Добавляем prompt_cache_key
        import hashlib
        prompt_hash = hashlib.md5(improvement_prompt.encode()).hexdigest()[:16]
        payload["prompt_cache_key"] = f"receipt_improve_{prompt_hash}"
        
        logging.info(f"Отправляем данные чека в OpenAI для улучшения (без изображения)")
        logging.info(f"OpenAI improvement request payload: {json.dumps(payload, ensure_ascii=False, indent=2)}")
        
        response_json = await asyncio.to_thread(self._post_payload, payload)
        
        logging.info(f"OpenAI improvement response: {json.dumps(response_json, ensure_ascii=False, indent=2)}")
        
        return response_json

    def _build_payload(
        self, 
        data_url: str, 
        qr_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        # OpenAI автоматически кэширует системные промпты >= 1024 токенов
        # Используем единый базовый промпт для лучшего кэширования
        # Разница только в начале промпта (про фото или про URL), остальное одинаково
        
        import hashlib
        import json
        
        # Добавляем текущую дату в промпт, чтобы OpenAI знал, какой год использовать
        current_date = datetime.utcnow().strftime("%Y-%m-%d")
        current_year = datetime.utcnow().year
        date_context = f"\n\nВАЖНО: Сегодняшняя дата: {current_date} (год {current_year}). Если на чеке указана дата без года (например, '6 января'), используй текущий год ({current_year}) или предыдущий ({current_year - 1}), если дата еще не наступила в текущем году."
        
        # Если есть данные из QR-кода, отправляем их для структурирования
        if qr_data is not None:
            user_text = (
                f"Вот данные чека, которые были извлечены с веб-страницы:\n\n"
                f"{json.dumps(qr_data, ensure_ascii=False, indent=2)}\n\n"
                f"ВАЖНО: Структурируй эти данные и ОБЯЗАТЕЛЬНО определи категорию для КАЖДОГО товара на основе его названия. "
                f"Если в данных категории отсутствуют или равны null, ты должен их добавить. "
                f"Используй категории из списка: "
                f"Продукты, Мясо/Рыба, Молочные продукты, Хлеб/Выпечка, "
                f"Овощи/Фрукты, Напитки, Алкоголь, Сладости, Кондитерские изделия, "
                f"Одежда, Аксессуары, Бытовая химия, Косметика/Гигиена, "
                f"Электроника, Компьютеры/Телефоны, Мебель, Дом/Интерьер, "
                f"Ресторан/Кафе, Доставка еды, Фастфуд, Транспорт, Такси, Парковка, "
                f"Бензин/Топливо, Медицина, Лекарства, "
                f"Образование, Книги, Канцтовары, Игрушки, Детские товары, "
                f"Развлечения, Кино, Театр, Концерты, Спорт, "
                f"Путешествия, Отель, Авиабилеты, Железнодорожные билеты, "
                f"Коммунальные, Электричество, Вода, Газ, Отопление, "
                f"Интернет/Связь, Мобильная связь, Подписки, Стриминг, "
                f"Страхование, Налоги, Штрафы, Банковские услуги, "
                f"Ремонт, Строительные материалы, Инструменты, Садоводство, "
                f"Животные, Корм для животных, Ветеринария, Другое.\n\n"
                f"КРИТИЧЕСКИ ВАЖНО - обработка одинаковых и похожих позиций:"
                f"\n- Если в данных есть ДВЕ ОДИНАКОВЫЕ позиции (например, \"Молоко\" появляется дважды), НЕ ОБЪЕДИНЯЙ их в одну!"
                f"\n- Если в данных есть ДВЕ ПОХОЖИЕ позиции с одинаковым брендом (например, \"Bonduelle Кукуруза\" и \"Bonduelle Горошек\"), НЕ ОБЪЕДИНЯЙ их в одну!"
                f"\n- Каждая позиция должна быть ОТДЕЛЬНЫМ элементом в массиве items"
                f"\n- Даже если товары одинаковые или похожие, сохраняй их как разные элементы массива"
                f"\n- Пример 1: если в данных \"Молоко × 1 = 50\" и \"Молоко × 1 = 50\", то items должен содержать ДВА элемента: [{{\"name\": \"Молоко\", \"quantity\": 1, \"price\": 50}}, {{\"name\": \"Молоко\", \"quantity\": 1, \"price\": 50}}]"
                f"\n- Пример 2: если в данных \"Bonduelle Кукуруза × 1 = 595\" и \"Bonduelle Горошек × 1 = 595\", то items должен содержать ДВА элемента: [{{\"name\": \"Bonduelle Кукуруза\", \"quantity\": 1, \"price\": 595}}, {{\"name\": \"Bonduelle Горошек\", \"quantity\": 1, \"price\": 595}}]"
                f"\n- НЕ ДЕЛАЙ так: [{{\"name\": \"Молоко\", \"quantity\": 2, \"price\": 100}}] - это НЕПРАВИЛЬНО!"
                f"\n- НЕ ДЕЛАЙ так: [{{\"name\": \"Bonduelle\", \"quantity\": 2, \"price\": 1190}}] - это НЕПРАВИЛЬНО!"
                f"\n- ВАЖНО: Каждая отдельная строка в чеке = отдельный элемент в массиве items. Не объединяй строки, даже если они похожи!"
                f"\n\nКРИТИЧЕСКИ ВАЖНО - названия товаров:"
                f"\n- Сохраняй ПОЛНОЕ название товара из данных, НЕ ОБРЕЗАЙ его!"
                f"\n- Если в данных написано \"Птицынно-куриный тандалмалы бодене жумырткасы\", то name должен быть \"Птицынно-куриный тандалмалы бодене жумырткасы\", а НЕ \"Птицынно-куриный тандалмалы\""
                f"\n- Полное название важно для правильного определения категории товара"
                f"\n- НЕ сокращай названия товаров, даже если они длинные"
                f"\n- НЕ обрезай названия товаров до определенной длины"
            )
            logging.info(f"Отправляем данные из QR-кода в OpenAI для структурирования")
            
            # Используем промпт для структурирования данных
            system_prompt = self.data_prompt + date_context
            payload = {
                "model": self.model,
                "response_format": {"type": "json_object"},
                "temperature": self.temperature,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_text,
                    },
                ],
            }
        else:
            # Если нет URL, отправляем изображение как обычно
            user_text = "Извлеки данные чека и верни JSON."
            system_prompt = self.prompt + date_context
            payload = {
                "model": self.model,
                "response_format": {"type": "json_object"},
                "temperature": self.temperature,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                                "text": user_text,
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url},
                        },
                    ],
                },
            ],
        }
        
        # Добавляем prompt_cache_key для лучшего кэширования
        # Используем хеш системного промпта как ключ кэша
        # Оба промпта имеют одинаковую базовую часть (RECEIPT_BASE_PROMPT), что улучшает кэширование
        prompt_hash = hashlib.md5(system_prompt.encode()).hexdigest()[:16]
        payload["prompt_cache_key"] = f"receipt_{prompt_hash}"
        
        return payload

    def _post_payload(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Логируем URL и headers (без API ключа)
        headers_for_log = {k: (v[:20] + "..." if k == "Authorization" else v) for k, v in headers.items()}
        logging.info(f"OpenAI request URL: {url}")
        logging.info(f"OpenAI request headers: {json.dumps(headers_for_log, ensure_ascii=False)}")
        try:
            resp = self._session.post(url, headers=headers, json=payload, timeout=self.timeout)
        except Exception as exc:
            # Преобразуем технические ошибки в понятные сообщения
            user_friendly_msg = format_user_friendly_error(exc)
            raise ReceiptParsingError(user_friendly_msg) from exc
        
        if resp.status_code >= 400:
            # Преобразуем ошибки API в понятные сообщения
            error_text = resp.text[:500] if resp.text else ""
            if resp.status_code == 429:
                user_msg = "Слишком много запросов. Пожалуйста, подождите немного и попробуйте еще раз."
            elif resp.status_code == 401:
                user_msg = "Ошибка авторизации. Обратитесь в поддержку через /feedback"
            elif resp.status_code >= 500:
                user_msg = "Сервер временно недоступен. Пожалуйста, попробуйте позже."
            elif "rate limit" in error_text.lower():
                user_msg = "Превышен лимит запросов. Пожалуйста, подождите немного и попробуйте еще раз."
            else:
                user_msg = "Ошибка при распознавании чека. Пожалуйста, попробуйте отправить фото чека еще раз."
            raise ReceiptParsingError(user_msg)
        return resp.json()


class SupabaseGateway:
    """Async helper around Supabase client used by ExpenseCatBot."""
    # TODO: добавить таблицы категорий, бюджетов и повторяющихся платежей.

    def __init__(
        self,
        url: str,
        service_key: str,
        receipts_table: str = "receipts",
        bank_table: str = "bank_transactions",
        expenses_table: str = "expenses",
        settings_table: str = "user_settings",
        feedback_table: str = "feedback",
        limits_table: str = "user_limits",
    ) -> None:
        if not SUPABASE_AVAILABLE or create_client is None:
            raise RuntimeError("Supabase client is not installed. Run `pip install supabase`.")
        self._client: Client = create_client(url, service_key)
        self.receipts_table = receipts_table
        self.bank_table = bank_table
        self.expenses_table = expenses_table
        self.settings_table = settings_table
        self.feedback_table = feedback_table
        self.limits_table = limits_table
        
        # Кэш лимитов пользователей: {user_id: (limits_dict, timestamp)}
        # TTL кэша: 60 секунд
        self._limits_cache: Dict[int, tuple[Dict[str, Any], float]] = {}
        self._limits_cache_ttl = 60.0  # секунды

    async def check_receipt_exists(self, receipt_hash: str) -> bool:
        """Проверяет, существует ли чек с данным хешем."""
        try:
            result = await asyncio.to_thread(
                lambda: self._client.table(self.receipts_table)
                .select("id")
                .eq("receipt_hash", receipt_hash)
                .limit(1)
                .execute()
            )
            return len(result.data) > 0 if result.data else False
        except Exception as exc:
            logging.warning(f"Ошибка при проверке дубликата чека: {exc}")
            return False

    async def upsert_receipt(self, payload: Dict[str, Any]) -> tuple[Dict[str, Any], bool]:
        """
        Сохраняет или обновляет чек в базе.
        Возвращает (данные чека, is_duplicate) где is_duplicate=True если чек уже существовал.
        """
        start_time = time.perf_counter()
        receipt_hash = payload.get("receipt_hash")
        logging.info("Upserting receipt %s", receipt_hash)
        
        # Проверяем, существует ли чек
        check_start = time.perf_counter()
        is_duplicate = await self.check_receipt_exists(receipt_hash)
        check_time = time.perf_counter() - check_start
        if check_time > 0.1:
            logging.info(f"⏱️ [PERF] Проверка дубликата чека: {check_time*1000:.1f}ms")
        
        if is_duplicate:
            logging.info("Receipt with hash %s already exists, will update if data differs", receipt_hash)
        
        upsert_start = time.perf_counter()
        stored_receipt = await asyncio.to_thread(
            self._table_upsert,
            self.receipts_table,
            payload,
            on_conflict="receipt_hash",
        )
        upsert_time = time.perf_counter() - upsert_start
        logging.info(f"⏱️ [PERF] Сохранение чека в БД: {upsert_time*1000:.1f}ms")
        
        # Проверяем, что получили реальную запись из базы (с id)
        if stored_receipt and stored_receipt.get("id"):
            if is_duplicate:
                logging.info("Receipt updated/retrieved: id=%s, hash=%s", stored_receipt.get("id"), receipt_hash)
            else:
                logging.info("Receipt created: id=%s, hash=%s", stored_receipt.get("id"), receipt_hash)
        else:
            logging.warning("Upsert returned receipt without id, using payload as fallback")
        
        total_time = time.perf_counter() - start_time
        logging.info(f"⏱️ [PERF] upsert_receipt всего: {total_time*1000:.1f}ms")
        
        return stored_receipt, is_duplicate

    async def upsert_bank_transactions(self, payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not payloads:
            return []
        logging.info("Upserting %d bank transactions", len(payloads))
        return await asyncio.to_thread(
            self._table_upsert_many,
            self.bank_table,
            payloads,
            "transaction_hash",
        )

    async def check_duplicate_expense(self, user_id: int, date: str, amount: float, currency: str, tolerance_days: int = 1, tolerance_percent: float = 0.01) -> bool:
        """
        Проверяет, есть ли уже расход с похожей датой и суммой.
        tolerance_days - допустимое отклонение по дате (дни)
        tolerance_percent - допустимое отклонение по сумме (например, 0.01 = 1%)
        """
        try:
            # Парсим дату (может быть ISO строка или просто дата)
            try:
                if 'T' in date or '+' in date or 'Z' in date:
                    date_obj = datetime.fromisoformat(date.replace('Z', '+00:00'))
                else:
                    # Просто дата без времени
                    date_obj = datetime.fromisoformat(date)
            except (ValueError, AttributeError):
                # Если не удалось распарсить, пробуем другой формат
                try:
                    date_obj = datetime.strptime(date[:10], "%Y-%m-%d")
                except:
                    logging.warning(f"Could not parse date: {date}")
                    return False
            
            date_center = date_obj.date() if hasattr(date_obj, 'date') else date_obj
            date_start = date_center - timedelta(days=tolerance_days)
            date_end = date_center + timedelta(days=tolerance_days)
            
            # Вычисляем диапазон сумм
            amount_min = amount * (1 - tolerance_percent)
            amount_max = amount * (1 + tolerance_percent)
            
            # Ищем похожие расходы (по дате и сумме)
            # Используем gte/lte для даты и суммы
            result = (
                self._client.table(self.expenses_table)
                .select("id,date,amount,currency,source")
                .eq("user_id", user_id)
                .eq("currency", currency)
                .gte("amount", amount_min)
                .lte("amount", amount_max)
                .gte("date", date_start.isoformat())
                .lte("date", date_end.isoformat())
                .execute()
            )
            
            if result.data:
                # Проверяем каждый найденный расход
                for existing in result.data:
                    existing_date_str = existing.get("date", "")
                    existing_amount = existing.get("amount", 0.0)
                    existing_source = existing.get("source", "")
                    
                    if not existing_date_str or not existing_amount:
                        continue
                    
                    try:
                        # Парсим дату существующего расхода
                        if 'T' in existing_date_str or '+' in existing_date_str or 'Z' in existing_date_str:
                            existing_date_obj = datetime.fromisoformat(existing_date_str.replace('Z', '+00:00'))
                        else:
                            existing_date_obj = datetime.fromisoformat(existing_date_str)
                        
                        existing_date = existing_date_obj.date() if hasattr(existing_date_obj, 'date') else existing_date_obj
                        
                        # Проверяем, что дата и сумма в пределах допуска
                        if date_start <= existing_date <= date_end and amount_min <= existing_amount <= amount_max:
                            logging.info(
                                f"Found duplicate expense: existing={existing_source} date={existing_date_str} amount={existing_amount}, "
                                f"new date={date} amount={amount}"
                            )
                            return True
                    except Exception as e:
                        logging.debug(f"Error parsing existing date {existing_date_str}: {e}")
                        continue
            
            return False
        except Exception as exc:
            logging.exception(f"Error checking duplicate expense: {exc}")
            return False

    async def record_expense(self, payload: Dict[str, Any], check_duplicates: bool = True) -> Dict[str, Any]:
        """
        Сохраняет расход в базу данных.
        check_duplicates - проверять ли дубликаты по дате и сумме перед сохранением.
        Для мануальных расходов (source="manual") проверка дубликатов отключена,
        так как хеш уже уникален (включает описание и дату/время).
        """
        user_id = payload.get("user_id")
        date = payload.get("date")
        amount = payload.get("amount")
        currency = payload.get("currency")
        source = payload.get("source")
        
        # Для мануальных расходов не проверяем дубликаты, так как хеш уникален
        # (включает описание и дату/время), и пользователь может вводить одинаковые расходы
        if source == "manual":
            check_duplicates = False
            logging.info(f"Skipping duplicate check for manual expense: description={payload.get('note', 'N/A')}")
        
        # Проверяем дубликаты, если включено
        if check_duplicates and user_id and date and amount and currency:
            is_duplicate = await self.check_duplicate_expense(user_id, date, amount, currency)
            if is_duplicate:
                logging.info(
                    f"Skipping duplicate expense: user={user_id} date={date} amount={amount} {currency}"
                )
                # Возвращаем пустой словарь или None, чтобы показать, что запись не была создана
                return {"duplicate": True}
        
        logging.info(
            "Recording expense user=%s source=%s",
            payload.get("user_id"),
            payload.get("source"),
        )
        return await asyncio.to_thread(
            self._table_upsert,
            self.expenses_table,
            payload,
            on_conflict="expense_hash",
        )

    async def fetch_monthly_report(
        self, 
        user_id: int, 
        period: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        logging.info("Fetching report for user=%s period=%s start_date=%s end_date=%s", 
                    user_id, period, start_date, end_date)
        return await asyncio.to_thread(
            self._fetch_report_sync,
            user_id,
            period,
            start_date,
            end_date,
        )

    async def export_expenses_csv(
        self, 
        user_id: int, 
        period: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> str:
        logging.info("Exporting expenses for user=%s period=%s start_date=%s end_date=%s", 
                     user_id, period or "all", start_date or "none", end_date or "none")
        return await asyncio.to_thread(
            self._export_expenses_csv_sync,
            user_id,
            period,
            start_date,
            end_date,
        )

    async def delete_all_user_data(self, user_id: int) -> Dict[str, int]:
        """
        Удаляет все данные пользователя из всех таблиц.
        Возвращает словарь с количеством удаленных записей по таблицам.
        """
        logging.warning(f"Deleting all data for user={user_id}")
        return await asyncio.to_thread(
            self._delete_all_user_data_sync,
            user_id,
        )

    def _table_upsert(self, table: str, payload: Dict[str, Any], on_conflict: str) -> Dict[str, Any]:
        try:
            result = (
                self._client.table(table)
                .upsert(payload, on_conflict=on_conflict, returning="representation")
                .execute()
            )
            if result.data and len(result.data) > 0:
                record_id = result.data[0].get('id')
                # Проверяем, что получили реальную запись с id
                if record_id:
                    logging.info(f"✅ Успешно сохранено в {table}: id={record_id}")
                    # Проверяем, что items сохранились корректно (для таблицы receipts)
                    if table == self.receipts_table and "items" in payload:
                        saved_items = result.data[0].get("items", [])
                        payload_items = payload.get("items", [])
                        if len(saved_items) != len(payload_items):
                            logging.warning(f"⚠️ Количество items изменилось при сохранении: было {len(payload_items)}, стало {len(saved_items)}")
                        else:
                            logging.info(f"✅ Все {len(saved_items)} позиций сохранены корректно")
                    return result.data[0]
                else:
                    logging.warning(f"⚠️ Supabase вернул запись без id для {table}, используем payload")
                    return payload
            else:
                logging.warning(f"⚠️ Supabase вернул пустой результат для {table}, используем payload")
                return payload
        except Exception as exc:
            # Если ошибка связана с отсутствующей колонкой category, пробуем без неё
            error_str = str(exc)
            if "category" in error_str.lower() and "column" in error_str.lower():
                logging.warning(f"⚠️ Колонка 'category' отсутствует в таблице {table}, пробуем без неё")
                payload_without_category = {k: v for k, v in payload.items() if k != "category"}
                try:
                    result = (
                        self._client.table(table)
                        .upsert(payload_without_category, on_conflict=on_conflict, returning="representation")
                        .execute()
                    )
                    if result.data and len(result.data) > 0:
                        logging.info(f"✅ Успешно сохранено в {table} (без category): {result.data[0].get('id')}")
                        return result.data[0]
                    else:
                        logging.warning(f"⚠️ Supabase вернул пустой результат для {table}, используем payload")
                        return payload_without_category
                except Exception as retry_exc:
                    logging.exception(f"❌ Ошибка при повторной попытке сохранения в {table}: {retry_exc}")
                    logging.error(f"Payload: {json.dumps(payload_without_category, ensure_ascii=False, default=str)}")
                    raise
            
            logging.exception(f"❌ Ошибка при сохранении в {table}: {exc}")
            logging.error(f"Payload: {json.dumps(payload, ensure_ascii=False, default=str)}")
            raise

    def _table_upsert_many(
        self,
        table: str,
        payloads: List[Dict[str, Any]],
        on_conflict: str,
    ) -> List[Dict[str, Any]]:
        result = (
            self._client.table(table)
            .upsert(payloads, on_conflict=on_conflict, returning="representation")
            .execute()
        )
        return result.data if result.data else payloads

    def _fetch_report_sync(
        self, 
        user_id: int, 
        period: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Получает отчет за период.
        Если указан period (формат YYYY-MM), используется он.
        Если указаны start_date и end_date (формат YYYY-MM-DD), используется диапазон дат.
        Категории берутся из items чеков, а не из категории расхода.
        """
        start_time = time.perf_counter()
        logging.info(f"📊 [REPORT_FETCH] Starting report fetch: user_id={user_id}, period={period}, start_date={start_date}, end_date={end_date}")
        
        # Оптимизация: выбираем только нужные поля вместо * для ускорения запроса
        # Это особенно важно, если в таблице есть большие JSONB поля или много колонок
        query = (
            self._client.table(self.expenses_table)
            .select("id, user_id, amount, currency, date, receipt_id, category, store, source")
            .eq("user_id", user_id)
        )
        logging.info(f"📊 [REPORT_FETCH] Base query created for user_id={user_id}")
        
        if period:
            # Фильтр по периоду (месяц) - используем поле period, которое не изменилось
            logging.info(f"📊 [REPORT_FETCH] Using period filter: {period}")
            query = query.ilike("period", period)
        elif start_date and end_date:
            # Фильтр по диапазону дат (date теперь TIMESTAMPTZ)
            # Для корректного сравнения с TIMESTAMPTZ нужно добавить время
            # Начало дня: 00:00:00, конец дня: 23:59:59.999 (включая весь день)
            start_datetime = f"{start_date}T00:00:00Z"
            # Для конца дня используем начало следующего дня (не включая его) для надежности
            # Это гарантирует, что все траты за end_date будут включены
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
            end_datetime = (end_date_obj + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
            logging.info(f"📊 [REPORT_FETCH] Using date range filter: {start_datetime} to {end_datetime} (exclusive)")
            query = query.gte("date", start_datetime).lt("date", end_datetime)
        else:
            # Если ничего не указано, берем текущий месяц
            period = datetime.utcnow().strftime("%Y-%m")
            logging.info(f"📊 [REPORT_FETCH] No filter specified, using current month: {period}")
            query = query.ilike("period", period)
        
        logging.info(f"📊 [REPORT_FETCH] Executing query...")
        query_start = time.perf_counter()
        try:
            result = query.execute()
            data = result.data or []
            query_time = time.perf_counter() - query_start
            logging.info(f"📊 [REPORT_FETCH] Query executed in {query_time*1000:.1f}ms, got {len(data)} records")
        except Exception as exc:
            logging.exception(f"📊 [REPORT_FETCH] ❌ Error executing query: {exc}")
            logging.error(f"📊 [REPORT_FETCH] Error type: {type(exc).__name__}")
            logging.error(f"📊 [REPORT_FETCH] Error args: {exc.args}")
            data = []
        
        # Группируем данные по валютам
        data_by_currency: Dict[str, List[Dict[str, Any]]] = {}
        currency_samples = {}  # Для логирования примеров валют
        
        for entry in data:
            currency = entry.get("currency")
            # Логируем первые несколько примеров валют для диагностики
            if len(currency_samples) < 5:
                currency_samples[entry.get("id", "unknown")] = {
                    "currency_raw": currency,
                    "currency_type": type(currency).__name__ if currency else "None"
                }
            
            # Если валюта не указана или пустая, используем RUB по умолчанию
            if not currency or (isinstance(currency, str) and currency.strip() == ""):
                currency = "RUB"
            else:
                currency = str(currency).upper().strip()  # Нормализуем валюту
            
            if currency not in data_by_currency:
                data_by_currency[currency] = []
            data_by_currency[currency].append(entry)
        
        logging.info(f"📊 [REPORT_FETCH] Currency samples: {currency_samples}")
        logging.info(f"📊 [REPORT_FETCH] Grouped data by currency: {list(data_by_currency.keys())}, counts: {[(k, len(v)) for k, v in data_by_currency.items()]}")
        
        # Получаем все receipt_id из expenses (уникальные)
        receipt_ids = list(set(entry.get("receipt_id") for entry in data if entry.get("receipt_id")))
        
        # Получаем чеки для извлечения категорий из items и поиска самой дорогой покупки
        receipts_data = {}
        receipts_full_data = {}
        if receipt_ids:
            receipts_query_start = time.perf_counter()
            # Оптимизация: выбираем только нужные поля
            receipts_query = (
                self._client.table(self.receipts_table)
                .select("id, items, store, purchased_at, currency")
                .in_("id", receipt_ids)
            )
            receipts_result = receipts_query.execute().data or []
            receipts_query_time = time.perf_counter() - receipts_query_start
            logging.info(f"📊 [REPORT_FETCH] Receipts query executed in {receipts_query_time*1000:.1f}ms, got {len(receipts_result)} receipts")
            
            receipts_data = {r.get("id"): r.get("items", []) for r in receipts_result}
            receipts_full_data = {
                r.get("id"): {
                    "items": r.get("items", []),
                    "store": r.get("store", ""),
                    "purchased_at": r.get("purchased_at", ""),
                    "currency": r.get("currency", "RUB")
                }
                for r in receipts_result
            }
        
        # Обрабатываем данные по каждой валюте отдельно
        currencies_data = {}
        for currency, currency_data in data_by_currency.items():
            total = sum(entry.get("amount", 0.0) for entry in currency_data)
            
            # Разбивка по категориям товаров из items чеков
            categories = {}
            expenses_without_receipt = []
            
            for entry in currency_data:
                receipt_id = entry.get("receipt_id")
                if receipt_id and receipt_id in receipts_data:
                    # Берем категории из items чека
                    items = receipts_data[receipt_id]
                    if items and isinstance(items, list) and len(items) > 0:
                        # Есть items - суммируем по категориям товаров
                        for item in items:
                            if isinstance(item, dict):
                                item_category = item.get("category")
                                item_price = float(item.get("price", 0.0))
                                if item_category:
                                    categories[item_category] = categories.get(item_category, 0.0) + item_price
                                else:
                                    # Если категории нет, используем "Другое"
                                    categories["Другое"] = categories.get("Другое", 0.0) + item_price
                    else:
                        # Нет items в чеке - используем категорию из expense или "Другое"
                        expenses_without_receipt.append(entry)
                else:
                    # Нет чека - используем категорию из expense или "Другое"
                    expenses_without_receipt.append(entry)
            
            # Обрабатываем расходы без чеков или без items
            for entry in expenses_without_receipt:
                category = entry.get("category") or "Другое"
                amount = entry.get("amount", 0.0)
                categories[category] = categories.get(category, 0.0) + amount
            
            # Разбивка по магазинам
            stores = {}
            for entry in currency_data:
                store = entry.get("store") or "Без названия"
                amount = entry.get("amount", 0.0)
                stores[store] = stores.get(store, 0.0) + amount
            
            # Разбивка по дням (убрана из отчета, но оставляем для совместимости)
            daily = {}
            for entry in currency_data:
                date_str = entry.get("date", "")
                if date_str:
                    day = date_str[:10]  # YYYY-MM-DD
                    amount = entry.get("amount", 0.0)
                    daily[day] = daily.get(day, 0.0) + amount
            
            currencies_data[currency] = {
                "total": total,
                "by_category": categories,
                "by_store": stores,
                "by_day": daily,
                "entries": currency_data,
            }
            logging.info(f"📊 [REPORT_FETCH] Processed currency {currency}: total={total}, entries={len(currency_data)}")
        
        logging.info(f"📊 [REPORT_FETCH] Final currencies_data keys: {list(currencies_data.keys())}, count: {len(currencies_data)}")
        
        # Поиск самой дорогой покупки и самого дорогого расхода для каждой валюты отдельно
        # Структура: {currency: {"item": {...}, "expense": {...}}}
        most_expensive_by_currency = {}
        
        # Сначала ищем в items чеков
        for receipt_id, receipt_info in receipts_full_data.items():
            items = receipt_info.get("items", [])
            currency = receipt_info.get("currency", "RUB")
            if items and isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        item_price = float(item.get("price", 0.0))
                        # Инициализируем структуру для валюты, если её еще нет
                        if currency not in most_expensive_by_currency:
                            most_expensive_by_currency[currency] = {
                                "item": {"price": 0.0, "name": None, "store": None, "date": None},
                                "expense": {"amount": 0.0, "store": None, "date": None}
                            }
                        # Обновляем самую дорогую покупку для этой валюты
                        if item_price > most_expensive_by_currency[currency]["item"]["price"]:
                            most_expensive_by_currency[currency]["item"] = {
                                "price": item_price,
                                "name": item.get("name", "Неизвестно"),
                                "store": receipt_info.get("store", "Неизвестно"),
                                "date": receipt_info.get("purchased_at", "")
                            }
        
        # Теперь ищем в ручных расходах (expenses)
        # Учитываем ручные расходы как "покупки" для определения самой дорогой покупки
        for entry in data:
            amount = float(entry.get("amount", 0.0))
            currency = entry.get("currency", "RUB")
            store = entry.get("store", "Неизвестно")
            expense_date = entry.get("date", "")
            
            # Инициализируем структуру для валюты, если её еще нет
            if currency not in most_expensive_by_currency:
                most_expensive_by_currency[currency] = {
                    "item": {"price": 0.0, "name": None, "store": None, "date": None},
                    "expense": {"amount": 0.0, "store": None, "date": None}
                }
            
            # Обновляем самую дорогую покупку для этой валюты (если ручной расход больше)
            if amount > most_expensive_by_currency[currency]["item"]["price"]:
                most_expensive_by_currency[currency]["item"] = {
                    "price": amount,
                    "name": store,  # Для ручных расходов используем store как название покупки
                    "store": store,
                    "date": expense_date
                }
            
            # Обновляем самый дорогой расход для этой валюты
            if amount > most_expensive_by_currency[currency]["expense"]["amount"]:
                most_expensive_by_currency[currency]["expense"] = {
                    "amount": amount,
                    "store": store,
                    "date": expense_date
                }
        
        # Определяем период для отображения
        display_period = period
        if start_date and end_date:
            display_period = f"{start_date} - {end_date}"
        elif not display_period:
            display_period = datetime.utcnow().strftime("%Y-%m")
        
        total_time = time.perf_counter() - start_time
        logging.info(f"⏱️ [PERF] _fetch_report_sync: {total_time*1000:.1f}ms ({total_time:.2f}s)")
        
        return {
            "period": display_period,
            "currencies_data": currencies_data,
            "most_expensive_by_currency": most_expensive_by_currency,  # Топы по каждой валюте отдельно
        }

    def _export_expenses_csv_sync(
        self, 
        user_id: int, 
        period: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> str:
        start_time = time.perf_counter()
        query = self._client.table(self.expenses_table).select("*").eq("user_id", user_id)
        if period:
            query = query.ilike("period", f"{period}%")
        elif start_date and end_date:
            # Фильтр по диапазону дат (date теперь TIMESTAMPTZ)
            # Для корректного сравнения с TIMESTAMPTZ нужно добавить время
            # Начало дня: 00:00:00, конец дня: используем начало следующего дня (не включая его)
            # Это гарантирует, что все траты за end_date будут включены
            start_datetime = f"{start_date}T00:00:00Z"
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
            end_datetime = (end_date_obj + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
            logging.info(f"📊 [EXPORT_CSV] Using date range filter: {start_datetime} to {end_datetime} (exclusive)")
            query = query.gte("date", start_datetime).lt("date", end_datetime)
        elif start_date:
            # Для начала периода используем начало дня
            start_datetime = f"{start_date}T00:00:00Z"
            query = query.gte("date", start_datetime)
        elif end_date:
            # Для конца периода используем начало следующего дня (не включая его)
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
            end_datetime = (end_date_obj + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z")
            query = query.lt("date", end_datetime)
        data = query.execute().data or []
        if not data:
            return "id,store,currency,date,source,category,note,bank_transaction_id,receipt_id,amount\n"
        
        # Поля, которые нужно исключить
        excluded_fields = {"created_at", "expense_hash", "period", "updated_at", "user_id", "status"}
        
        # Получаем все поля из данных, исключая ненужные
        all_fields = {field for row in data for field in row.keys() if field not in excluded_fields}
        
        # Определяем порядок полей: id в начале, amount в конце, остальные по алфавиту
        priority_fields = ["id"]
        end_fields = ["amount"]
        middle_fields = sorted(all_fields - set(priority_fields) - set(end_fields))
        fieldnames = priority_fields + middle_fields + end_fields
        
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            # Форматируем данные для записи
            formatted_row = {}
            for key in fieldnames:
                value = row.get(key)
                # Если это поле date, форматируем с временем
                if key == "date" and value:
                    if isinstance(value, str):
                        # Если дата в формате ISO с временем, оставляем как есть
                        # Если только дата, добавляем время 00:00:00
                        if "T" in value:
                            # Уже есть время, оставляем как есть
                            pass
                        elif " " in value:
                            # Есть пробел, значит формат "YYYY-MM-DD HH:MM:SS"
                            pass
                        elif len(value) == 10:
                            # Только дата, добавляем время
                            value = value + "T00:00:00Z"
                    elif hasattr(value, 'strftime'):
                        # Если это объект datetime/date, форматируем с временем
                        if hasattr(value, 'hour'):
                            # Это datetime, есть время
                            value = value.strftime("%Y-%m-%dT%H:%M:%S")
                        else:
                            # Это date, нет времени
                            value = value.strftime("%Y-%m-%dT00:00:00")
                formatted_row[key] = value
            writer.writerow(formatted_row)
        
        total_time = time.perf_counter() - start_time
        logging.info(f"⏱️ [PERF] _export_expenses_csv_sync: {total_time*1000:.1f}ms ({total_time:.2f}s), записей: {len(data)}")
        
        return buffer.getvalue()

    def _delete_all_user_data_sync(self, user_id: int) -> Dict[str, int]:
        """
        Синхронное удаление всех данных пользователя из всех таблиц.
        """
        result = {}
        
        # Удаляем из expenses
        try:
            expenses_result = (
                self._client.table(self.expenses_table)
                .delete()
                .eq("user_id", user_id)
                .execute()
            )
            result["expenses"] = len(expenses_result.data) if expenses_result.data else 0
            logging.info(f"Deleted {result['expenses']} expenses for user={user_id}")
        except Exception as exc:
            logging.exception(f"Error deleting expenses for user={user_id}: {exc}")
            result["expenses"] = 0
        
        # Удаляем из receipts
        try:
            receipts_result = (
                self._client.table(self.receipts_table)
                .delete()
                .eq("user_id", user_id)
                .execute()
            )
            result["receipts"] = len(receipts_result.data) if receipts_result.data else 0
            logging.info(f"Deleted {result['receipts']} receipts for user={user_id}")
        except Exception as exc:
            logging.exception(f"Error deleting receipts for user={user_id}: {exc}")
            result["receipts"] = 0
        
        # Удаляем из bank_transactions
        try:
            bank_result = (
                self._client.table(self.bank_table)
                .delete()
                .eq("user_id", user_id)
                .execute()
            )
            result["bank_transactions"] = len(bank_result.data) if bank_result.data else 0
            logging.info(f"Deleted {result['bank_transactions']} bank transactions for user={user_id}")
        except Exception as exc:
            logging.exception(f"Error deleting bank transactions for user={user_id}: {exc}")
            result["bank_transactions"] = 0
        
        total_deleted = sum(result.values())
        logging.warning(f"Total deleted records for user={user_id}: {total_deleted}")
        return result

    def _fetch_receipts_list_sync(self, user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Получает список чеков пользователя для отображения"""
        try:
            result = (
                self._client.table(self.receipts_table)
                .select("id, store, total, currency, purchased_at")
                .eq("user_id", user_id)
                .order("purchased_at", desc=True)
                .limit(limit)
                .execute()
            )
            return result.data or []
        except Exception as exc:
            logging.exception(f"Error fetching receipts list for user={user_id}: {exc}")
            return []

    def _fetch_expenses_list_sync(self, user_id: int, limit: int = 50, months_back: int = 3) -> List[Dict[str, Any]]:
        """Получает список расходов пользователя для отображения"""
        try:
            # Получаем expenses с joined данными из receipts для получения purchased_at с временем
            query = (
                self._client.table(self.expenses_table)
                .select("id, store, amount, currency, date, source, category, receipt_id, note, receipts(purchased_at)")
                .eq("user_id", user_id)
            )
            
            # Фильтруем по периоду (последние N месяцев)
            if months_back > 0:
                from datetime import datetime, timedelta
                cutoff_date = (datetime.utcnow() - timedelta(days=months_back * 30)).strftime("%Y-%m-%d")
                query = query.gte("date", cutoff_date)
            
            result = (
                query
                .order("date", desc=True)
                .limit(limit)
                .execute()
            )
            return result.data or []
        except Exception as exc:
            logging.exception(f"Error fetching expenses list for user={user_id}: {exc}")
            return []

    def _delete_expense_sync(self, user_id: int, expense_id: int) -> bool:
        """Удаляет конкретный расход пользователя с каскадным удалением источника"""
        try:
            # Сначала получаем expense, чтобы узнать источник
            expense_result = (
                self._client.table(self.expenses_table)
                .select("receipt_id, bank_transaction_id, source")
                .eq("id", expense_id)
                .eq("user_id", user_id)
                .execute()
            )
            
            if not expense_result.data or len(expense_result.data) == 0:
                logging.warning(f"Expense id={expense_id} not found or not owned by user={user_id}")
                return False
            
            expense = expense_result.data[0]
            receipt_id = expense.get("receipt_id")
            bank_transaction_id = expense.get("bank_transaction_id")
            source = expense.get("source", "")
            
            # Удаляем источник (receipt или bank_transaction) перед удалением expense
            if receipt_id:
                try:
                    receipt_result = (
                        self._client.table(self.receipts_table)
                        .delete()
                        .eq("id", receipt_id)
                        .eq("user_id", user_id)
                        .execute()
                    )
                    if receipt_result.data:
                        logging.info(f"Deleted receipt id={receipt_id} (cascade from expense {expense_id})")
                except Exception as exc:
                    logging.exception(f"Error deleting receipt id={receipt_id} (cascade): {exc}")
            
            if bank_transaction_id:
                try:
                    bank_result = (
                        self._client.table(self.bank_table)
                        .delete()
                        .eq("id", bank_transaction_id)
                        .eq("user_id", user_id)
                        .execute()
                    )
                    if bank_result.data:
                        logging.info(f"Deleted bank_transaction id={bank_transaction_id} (cascade from expense {expense_id})")
                except Exception as exc:
                    logging.exception(f"Error deleting bank_transaction id={bank_transaction_id} (cascade): {exc}")
            
            # Теперь удаляем сам expense
            result = (
                self._client.table(self.expenses_table)
                .delete()
                .eq("id", expense_id)
                .eq("user_id", user_id)  # Проверка безопасности: только свои расходы
                .execute()
            )
            deleted = len(result.data) if result.data else 0
            if deleted > 0:
                logging.info(f"Deleted expense id={expense_id} for user={user_id} (source={source})")
                return True
            else:
                logging.warning(f"Expense id={expense_id} not found or not owned by user={user_id}")
                return False
        except Exception as exc:
            logging.exception(f"Error deleting expense id={expense_id} for user={user_id}: {exc}")
            return False

    async def fetch_receipts_list(self, user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
        """Асинхронная обертка для получения списка чеков"""
        return await asyncio.to_thread(self._fetch_receipts_list_sync, user_id, limit)

    async def fetch_expenses_list(self, user_id: int, limit: int = 50, months_back: int = 3) -> List[Dict[str, Any]]:
        """Асинхронная обертка для получения списка расходов"""
        return await asyncio.to_thread(self._fetch_expenses_list_sync, user_id, limit, months_back)

    async def delete_expense(self, user_id: int, expense_id: int) -> bool:
        """Асинхронная обертка для удаления расхода"""
        return await asyncio.to_thread(self._delete_expense_sync, user_id, expense_id)
    
    async def save_feedback(
        self,
        user_id: int,
        username: Optional[str],
        first_name: Optional[str],
        feedback_type: str,
        feedback_text: str
    ) -> Dict[str, Any]:
        """Сохраняет отзыв в базу данных"""
        payload = {
            "user_id": user_id,
            "username": username,
            "first_name": first_name,
            "feedback_type": feedback_type,
            "feedback_text": feedback_text,
        }
        return await asyncio.to_thread(
            self._table_insert,
            self.feedback_table,
            payload,
        )
    
    async def save_receipt_recognition_stat(
        self,
        user_id: int,
        recognition_method: str,
        success: bool,
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Сохраняет статистику распознавания чека"""
        payload = {
            "user_id": user_id,
            "recognition_method": recognition_method,
            "success": success,
            "error_message": error_message,
        }
        try:
            return await asyncio.to_thread(
                self._table_insert,
                "receipt_recognition_stats",
                payload,
            )
        except Exception as exc:
            # Логируем ошибку, но не прерываем выполнение
            logging.warning(f"Failed to save receipt recognition stat: {exc}")
            return {}
    
    async def get_receipt_stats(self, user_id: int) -> Dict[str, Any]:
        """Получает статистику распознавания чеков для пользователя"""
        try:
            result = await asyncio.to_thread(
                lambda: self._client.table("receipt_stats_by_user")
                .select("*")
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            if result.data and len(result.data) > 0:
                return result.data[0]
            return {}
        except Exception as exc:
            logging.warning(f"Failed to get receipt stats: {exc}")
            return {}
    
    def _table_insert(self, table: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Вставляет новую запись в таблицу"""
        result = self._client.table(table).insert(payload).execute()
        if not result.data:
            raise RuntimeError(f"Failed to insert into {table}")
        return result.data[0]

    def _get_user_settings_sync(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Получает настройки пользователя"""
        try:
            result = (
                self._client.table(self.settings_table)
                .select("default_currency")
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            if result.data and len(result.data) > 0:
                return result.data[0]
            return None
        except Exception as exc:
            logging.exception(f"Error fetching user settings for user={user_id}: {exc}")
            return None

    async def get_user_settings(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Асинхронная обертка для получения настроек пользователя"""
        return await asyncio.to_thread(self._get_user_settings_sync, user_id)

    def _set_user_default_currency_sync(self, user_id: int, currency: str) -> bool:
        """Устанавливает валюту по умолчанию для пользователя"""
        try:
            payload = {
                "user_id": user_id,
                "default_currency": currency.upper(),
            }
            result = (
                self._client.table(self.settings_table)
                .upsert(payload, on_conflict="user_id", returning="representation")
                .execute()
            )
            if result.data and len(result.data) > 0:
                logging.info(f"Set default currency {currency} for user={user_id}")
                return True
            return False
        except Exception as exc:
            logging.exception(f"Error setting default currency for user={user_id}: {exc}")
            return False

    async def set_user_default_currency(self, user_id: int, currency: str) -> bool:
        """Асинхронная обертка для установки валюты по умолчанию"""
        return await asyncio.to_thread(self._set_user_default_currency_sync, user_id, currency)

    def _check_user_has_data_sync(self, user_id: int) -> bool:
        """Проверяет, есть ли у пользователя какие-либо данные (чеки, расходы)"""
        try:
            # Проверяем наличие расходов
            expenses_result = (
                self._client.table(self.expenses_table)
                .select("id")
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            if expenses_result.data and len(expenses_result.data) > 0:
                return True
            
            # Проверяем наличие чеков
            receipts_result = (
                self._client.table(self.receipts_table)
                .select("id")
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            if receipts_result.data and len(receipts_result.data) > 0:
                return True
            
            return False
        except Exception as exc:
            logging.exception(f"Error checking user data for user={user_id}: {exc}")
            return False

    async def check_user_has_data(self, user_id: int) -> bool:
        """Асинхронная обертка для проверки наличия данных у пользователя"""
        return await asyncio.to_thread(self._check_user_has_data_sync, user_id)
    
    def _get_cached_limits(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Получает лимиты из кэша, если они еще актуальны"""
        if user_id in self._limits_cache:
            limits, timestamp = self._limits_cache[user_id]
            if time.perf_counter() - timestamp < self._limits_cache_ttl:
                return limits
            else:
                # Кэш устарел, удаляем
                del self._limits_cache[user_id]
        return None
    
    def _set_cached_limits(self, user_id: int, limits: Dict[str, Any]) -> None:
        """Сохраняет лимиты в кэш"""
        self._limits_cache[user_id] = (limits, time.perf_counter())
    
    def _invalidate_limits_cache(self, user_id: int) -> None:
        """Инвалидирует кэш лимитов для пользователя"""
        if user_id in self._limits_cache:
            del self._limits_cache[user_id]
    
    def _get_or_create_user_limits_sync(self, user_id: int) -> Dict[str, Any]:
        """Получает или создает запись лимитов для пользователя (с кэшированием)"""
        # Проверяем кэш
        cached_limits = self._get_cached_limits(user_id)
        if cached_limits is not None:
            return cached_limits
        
        try:
            result = (
                self._client.table(self.limits_table)
                .select("*")
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            if result.data and len(result.data) > 0:
                limits = result.data[0]
                # Сохраняем в кэш
                self._set_cached_limits(user_id, limits)
                return limits
            
            # Создаем новую запись с дефолтными значениями
            default_limits = {
                "user_id": user_id,
                "receipts_count": 0,
                "limit_receipts": 20,  # Trial: 20 чеков (одноразово), Standard: 50 чеков/месяц, Pro: 100 чеков/месяц, Premium: безлимит
                "subscription_type": "trial",
                "expires_at": None,
            }
            insert_result = (
                self._client.table(self.limits_table)
                .insert(default_limits)
                .execute()
            )
            if insert_result.data and len(insert_result.data) > 0:
                limits = insert_result.data[0]
                # Сохраняем в кэш
                self._set_cached_limits(user_id, limits)
                return limits
            # Сохраняем дефолтные значения в кэш
            self._set_cached_limits(user_id, default_limits)
            return default_limits
        except Exception as exc:
            logging.exception(f"Error getting/creating user limits for user={user_id}: {exc}")
            # Возвращаем дефолтные значения при ошибке
            default_limits = {
                "user_id": user_id,
                "receipts_count": 0,
                "limit_receipts": 20,
                "subscription_type": "trial",
                "expires_at": None,
            }
            # Не кэшируем при ошибке
            return default_limits
    
    async def get_or_create_user_limits(self, user_id: int) -> Dict[str, Any]:
        """Асинхронная обертка для получения/создания лимитов пользователя"""
        return await asyncio.to_thread(self._get_or_create_user_limits_sync, user_id)
    
    def _check_receipt_limit_sync(self, user_id: int) -> tuple[bool, Dict[str, Any]]:
        """Проверяет, не превышен ли лимит чеков для пользователя"""
        limits = self._get_or_create_user_limits_sync(user_id)
        receipts_count = limits.get("receipts_count", 0)
        limit_receipts = limits.get("limit_receipts", 20)
        subscription_type = limits.get("subscription_type", "trial")
        
        # Проверяем подписку Premium (безлимит)
        if subscription_type == "premium":
            # Проверяем срок действия подписки
            expires_at = limits.get("expires_at")
            if expires_at:
                from datetime import datetime
                try:
                    if isinstance(expires_at, str):
                        expires_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                    else:
                        expires_dt = expires_at
                    if expires_dt < datetime.utcnow().replace(tzinfo=expires_dt.tzinfo):
                        # Подписка истекла, возвращаемся к пробному периоду
                        subscription_type = "trial"
                        limit_receipts = 20
                        # Обновляем лимит в базе
                        self._client.table(self.limits_table).update({
                            "subscription_type": "trial",
                            "limit_receipts": 20
                        }).eq("user_id", user_id).execute()
                        # Инвалидируем кэш
                        self._invalidate_limits_cache(user_id)
                    else:
                        # Premium подписка активна - безлимит
                        return True, limits
                except:
                    pass
            else:
                # Premium без expires_at - безлимит
                return True, limits
        
        # Проверяем срок действия подписки для других типов
        expires_at = limits.get("expires_at")
        if expires_at:
            from datetime import datetime
            try:
                if isinstance(expires_at, str):
                    expires_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                else:
                    expires_dt = expires_at
                if expires_dt < datetime.utcnow().replace(tzinfo=expires_dt.tzinfo):
                    # Подписка истекла, возвращаемся к пробному периоду
                    subscription_type = "trial"
                    limit_receipts = 20
                    # Обновляем лимит в базе
                    self._client.table(self.limits_table).update({
                        "subscription_type": "trial",
                        "limit_receipts": 20
                    }).eq("user_id", user_id).execute()
            except:
                pass
        
        # Устанавливаем лимит в зависимости от типа подписки
        # Trial: 20 чеков (одноразово, не обновляется)
        # Standard: 50 чеков в месяц
        # Pro: 100 чеков в месяц
        # Premium: безлимит (уже обработано выше)
        if subscription_type == "standard":
            if limit_receipts != 50:
                limit_receipts = 50
                # Обновляем лимит в базе
                try:
                    self._client.table(self.limits_table).update({
                        "limit_receipts": 50
                    }).eq("user_id", user_id).execute()
                except:
                    pass
        elif subscription_type == "pro":
            if limit_receipts != 100:
                limit_receipts = 100
                # Обновляем лимит в базе
                try:
                    self._client.table(self.limits_table).update({
                        "limit_receipts": 100
                    }).eq("user_id", user_id).execute()
                except:
                    pass
        elif subscription_type == "trial":
            if limit_receipts != 20:
                limit_receipts = 20
                # Обновляем лимит в базе
                try:
                    self._client.table(self.limits_table).update({
                        "limit_receipts": 20
                    }).eq("user_id", user_id).execute()
                except:
                    pass
        
        # Проверяем лимит
        if receipts_count >= limit_receipts:
            return False, limits
        
        return True, limits
    
    async def check_receipt_limit(self, user_id: int) -> tuple[bool, Dict[str, Any]]:
        """Асинхронная обертка для проверки лимита чеков"""
        return await asyncio.to_thread(self._check_receipt_limit_sync, user_id)
    
    def _increment_receipt_count_sync(self, user_id: int) -> None:
        """Увеличивает счетчик чеков пользователя (оптимизированная версия с кэшем)"""
        try:
            # Пытаемся получить текущее значение из кэша
            cached_limits = self._get_cached_limits(user_id)
            current_count = None
            
            if cached_limits:
                current_count = cached_limits.get("receipts_count", 0)
            else:
                # Если нет в кэше, делаем SELECT только для receipts_count
                select_result = (
                    self._client.table(self.limits_table)
                    .select("receipts_count")
                    .eq("user_id", user_id)
                    .limit(1)
                    .execute()
                )
                
                if select_result.data and len(select_result.data) > 0:
                    current_count = select_result.data[0].get("receipts_count", 0)
            
            if current_count is not None:
                # Запись существует - обновляем
                new_count = current_count + 1
                update_result = (
                    self._client.table(self.limits_table)
                    .update({"receipts_count": new_count})
                    .eq("user_id", user_id)
                    .execute()
                )
                if update_result.data:
                    # Обновляем кэш
                    if cached_limits:
                        cached_limits["receipts_count"] = new_count
                        self._set_cached_limits(user_id, cached_limits)
                    logging.info(f"Incremented receipt count for user={user_id}: {current_count} -> {new_count}")
            else:
                # Записи нет - создаем новую с receipts_count = 1
                default_limits = {
                    "user_id": user_id,
                    "receipts_count": 1,
                    "limit_receipts": 20,
                    "subscription_type": "trial",
                    "expires_at": None,
                }
                insert_result = (
                    self._client.table(self.limits_table)
                    .insert(default_limits)
                    .execute()
                )
                if insert_result.data:
                    # Сохраняем в кэш
                    self._set_cached_limits(user_id, insert_result.data[0])
                    logging.info(f"Incremented receipt count for user={user_id}: 0 -> 1 (new record)")
        except Exception as exc:
            logging.exception(f"Error incrementing receipt count for user={user_id}: {exc}")
            # Инвалидируем кэш при ошибке
            self._invalidate_limits_cache(user_id)
    
    async def increment_receipt_count(self, user_id: int) -> None:
        """Асинхронная обертка для увеличения счетчика чеков"""
        await asyncio.to_thread(self._increment_receipt_count_sync, user_id)
    
    def _activate_subscription_sync(self, user_id: int, subscription_type: str, months: int = 1) -> Dict[str, Any]:
        """Активирует подписку для пользователя (pro или premium)"""
        from datetime import datetime, timedelta
        
        expires_at = datetime.utcnow() + timedelta(days=30 * months)
        
        # Определяем лимит в зависимости от типа подписки
        if subscription_type == "standard":
            limit_receipts = 50
        elif subscription_type == "pro":
            limit_receipts = 100
        elif subscription_type == "premium":
            limit_receipts = None  # Безлимит (будет храниться как NULL или большое число)
        else:
            raise ValueError(f"Invalid subscription type: {subscription_type}")
        
        try:
            # Обновляем или создаем запись с подпиской
            update_data = {
                "user_id": user_id,
                "subscription_type": subscription_type,
                "expires_at": expires_at.isoformat() + "Z",
                "receipts_count": 0,  # Сбрасываем счетчик при активации подписки
            }
            
            if limit_receipts is not None:
                update_data["limit_receipts"] = limit_receipts
            
            result = (
                self._client.table(self.limits_table)
                .upsert(update_data)
                .execute()
            )
            if result.data and len(result.data) > 0:
                # Инвалидируем кэш и обновляем его новыми данными
                self._invalidate_limits_cache(user_id)
                self._set_cached_limits(user_id, result.data[0])
                return result.data[0]
            return {
                "user_id": user_id,
                "subscription_type": subscription_type,
                "limit_receipts": limit_receipts,
                "expires_at": expires_at.isoformat() + "Z",
            }
        except Exception as exc:
            logging.exception(f"Error activating {subscription_type} subscription for user={user_id}: {exc}")
            raise
    
    async def activate_subscription(self, user_id: int, subscription_type: str, months: int = 1) -> Dict[str, Any]:
        """Асинхронная обертка для активации подписки"""
        return await asyncio.to_thread(self._activate_subscription_sync, user_id, subscription_type, months)
    
    # Оставляем старый метод для обратной совместимости
    async def activate_premium_subscription(self, user_id: int, months: int = 1) -> Dict[str, Any]:
        """Асинхронная обертка для активации Premium подписки (deprecated, используйте activate_subscription)"""
        return await self.activate_subscription(user_id, "premium", months)
    
    def _save_rejected_receipt_photo_sync(
        self, 
        user_id: int, 
        file_bytes: bytes, 
        reason: str,
        mime_type: str = "image/jpeg"
    ) -> Optional[str]:
        """
        Сохраняет фото отклоненного чека в Supabase Storage.
        Использует service_role ключ для авторизации (передается при создании SupabaseGateway).
        """
        """
        Сохраняет фото отклоненного/невалидного чека в Supabase Storage.
        Возвращает путь к файлу или None при ошибке.
        """
        try:
            from datetime import datetime
            import hashlib
            
            # Создаем уникальное имя файла: user_id_timestamp_hash.ext
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            file_hash = hashlib.md5(file_bytes).hexdigest()[:8]
            
            # Определяем расширение файла
            ext_map = {
                "image/jpeg": "jpg",
                "image/jpg": "jpg",
                "image/png": "png",
                "image/webp": "webp",
                "image/heic": "heic",
                "image/heif": "heif",
            }
            ext = ext_map.get(mime_type.lower(), "jpg")
            
            # Путь в Storage: rejected_receipts/user_id/YYYYMMDD_HHMMSS_hash.ext
            file_path = f"rejected_receipts/{user_id}/{timestamp}_{file_hash}.{ext}"
            
            # Пытаемся загрузить файл
            try:
                # Пробуем разные варианты bucket (приоритет: rejected-receipts)
                # Пытаемся загрузить напрямую, так как list_buckets() может не работать из-за прав доступа
                bucket_candidates = ["rejected-receipts", "rejected_receipts", "receipts"]
                bucket_name = None
                upload_error = None
                
                # Пробуем загрузить в каждый bucket по очереди
                for candidate in bucket_candidates:
                    try:
                        storage_api = self._client.storage.from_(candidate)
                        result = storage_api.upload(
                            path=file_path,
                            file=file_bytes,
                            file_options={
                                "content-type": mime_type,
                                "upsert": False  # Не перезаписывать существующие файлы
                            }
                        )
                        bucket_name = candidate
                        logging.info(f"✅ Successfully uploaded to bucket: {bucket_name}")
                        break
                    except Exception as upload_exc:
                        upload_error = upload_exc
                        error_msg = str(upload_exc)
                        # Логируем детали ошибки для диагностики
                        if hasattr(upload_exc, 'message'):
                            error_msg = upload_exc.message
                        elif isinstance(upload_exc, dict):
                            error_msg = str(upload_exc)
                        logging.warning(f"Failed to upload to bucket '{candidate}': {error_msg}")
                        continue
                
                if not bucket_name:
                    # Если не удалось загрузить ни в один bucket, пробуем получить список для диагностики
                    try:
                        buckets_response = self._client.storage.list_buckets()
                        bucket_names = []
                        if buckets_response:
                            if isinstance(buckets_response, list):
                                buckets_list = buckets_response
                            elif hasattr(buckets_response, 'data'):
                                buckets_list = buckets_response.data or []
                            elif hasattr(buckets_response, '__iter__'):
                                buckets_list = list(buckets_response)
                            else:
                                buckets_list = []
                            
                            for b in buckets_list:
                                if isinstance(b, str):
                                    bucket_names.append(b)
                                elif hasattr(b, 'name'):
                                    bucket_names.append(b.name)
                                elif isinstance(b, dict):
                                    bucket_names.append(b.get('name', ''))
                                elif hasattr(b, 'id'):
                                    bucket_names.append(str(b.id))
                        logging.warning(
                            f"Failed to upload to any bucket. Available buckets: {bucket_names}. "
                            f"Last error: {upload_error}. Please create 'rejected-receipts' bucket manually."
                        )
                    except Exception as list_exc:
                        logging.warning(
                            f"Failed to upload to any bucket and cannot list buckets. "
                            f"Last upload error: {upload_error}. List error: {list_exc}. "
                            f"Please create 'rejected-receipts' bucket manually in Supabase Dashboard."
                        )
                    return None
                
                # Сохраняем метаданные в отдельную таблицу (опционально)
                # Можно добавить таблицу rejected_receipts для отслеживания
                
                logging.info(
                    f"Saved rejected receipt photo: user_id={user_id}, "
                    f"reason={reason}, path={file_path}, size={len(file_bytes)} bytes"
                )
                
                return file_path
                
            except Exception as storage_exc:
                logging.exception(f"Error uploading to storage: {storage_exc}")
                return None
                
        except Exception as exc:
            logging.exception(f"Error saving rejected receipt photo for user={user_id}: {exc}")
            return None
    
    async def save_rejected_receipt_photo(
        self,
        user_id: int,
        file_bytes: bytes,
        reason: str,
        mime_type: str = "image/jpeg"
    ) -> Optional[str]:
        """
        Асинхронная обертка для сохранения фото отклоненного чека.
        reason: 'rejected_by_user' или 'validation_error'
        """
        return await asyncio.to_thread(
            self._save_rejected_receipt_photo_sync,
            user_id,
            file_bytes,
            reason,
            mime_type
        )


def truncate_message_for_telegram(text: str, max_length: int = 4000) -> str:
    """
    Обрезает сообщение до максимальной длины для Telegram.
    Telegram имеет лимит 4096 символов, оставляем запас.
    """
    if len(text) <= max_length:
        return text
    
    # Проверяем, содержит ли сообщение таблицу товаров
    has_table = "ИТОГО" in text or "Всего товаров:" in text
    
    # Обрезаем и добавляем индикатор обрезки
    truncated = text[:max_length - 150]  # Оставляем больше места для предупреждения
    # Пытаемся обрезать по последнему переносу строки
    last_newline = truncated.rfind('\n')
    if last_newline > max_length - 200:
        truncated = truncated[:last_newline]
    
    # Проверяем, не обрезали ли мы таблицу товаров
    if has_table:
        # Ищем информацию о количестве товаров в оригинальном тексте
        import re
        items_count_match = re.search(r'Всего товаров:\s*(\d+)', text)
        items_sum_match = re.search(r'Сумма всех позиций:\s*([\d.]+)', text)
        
        if items_count_match or items_sum_match:
            # Таблица была обрезана, но информация о товарах есть
            warning = "\n\n⚠️ Таблица обрезана из-за большого количества товаров."
            if items_count_match:
                warning += f"\nВсего товаров в чеке: {items_count_match.group(1)}"
            if items_sum_match:
                warning += f"\nСумма всех позиций: {items_sum_match.group(1)}"
            warning += "\nВалидация суммы выполнена по всем товарам."
            return truncated + warning
        elif "ИТОГО" in text and "ИТОГО" not in truncated:
            # Таблица была обрезана до строки ИТОГО
            return truncated + "\n\n⚠️ Таблица обрезана из-за большого количества товаров.\nВалидация суммы выполнена по всем товарам."
    
    return truncated + "\n\n... (сообщение обрезано, слишком длинное)"


def truncate_html_safely(text: str, max_length: int) -> str:
    """
    Безопасно обрезает HTML-текст, закрывая незакрытые теги.
    """
    if len(text) <= max_length:
        return text
    
    # Список самозакрывающихся тегов (не требуют закрывающего тега)
    self_closing_tags = {'br', 'hr', 'img', 'input', 'meta', 'link', 'area', 'base', 'col', 'embed', 'source', 'track', 'wbr'}
    
    # Находим все открытые теги до места обрезки
    truncated = text[:max_length - 20]
    
    # Находим все открытые теги в обрезанном тексте
    import re
    open_tags = []
    tag_pattern = r'<(\/?)(\w+)(?:\s[^>]*)?>'
    
    for match in re.finditer(tag_pattern, truncated):
        is_closing = match.group(1) == '/'
        tag_name = match.group(2).lower()
        
        if tag_name in self_closing_tags:
            continue
            
        if is_closing:
            # Удаляем соответствующий открывающий тег из стека
            if open_tags and open_tags[-1] == tag_name:
                open_tags.pop()
        else:
            # Добавляем открывающий тег в стек
            open_tags.append(tag_name)
    
    # Закрываем все незакрытые теги в обратном порядке
    closing_tags = ''.join(f'</{tag}>' for tag in reversed(open_tags))
    
    return truncated + "\n\n... (обрезано)" + closing_tags


class RateLimitMiddleware:
    """Middleware для защиты от DDoS и злоупотреблений через rate limiting."""
    
    def __init__(
        self,
        max_requests_per_minute: int = 15,
        max_requests_per_hour: int = 50,
        max_file_requests_per_minute: int = 10,
    ):
        self.max_requests_per_minute = max_requests_per_minute
        self.max_requests_per_hour = max_requests_per_hour
        self.max_file_requests_per_minute = max_file_requests_per_minute
        
        # Хранилище запросов: user_id -> список временных меток
        self._requests_per_user: Dict[int, List[datetime]] = defaultdict(list)
        self._file_requests_per_user: Dict[int, List[datetime]] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def __call__(self, handler, event, data):
        """Проверяет rate limit перед обработкой события."""
        if not hasattr(event, 'from_user') or not event.from_user:
            return await handler(event, data)
        
        user_id = event.from_user.id
        now = datetime.utcnow()
        
        async with self._lock:
            # Очищаем старые записи (старше часа)
            self._cleanup_old_requests(user_id, now)
            
            # Проверяем лимит на файлы (фото, документы)
            is_file_request = (
                hasattr(event, 'photo') and event.photo or
                hasattr(event, 'document') and event.document
            )
            
            if is_file_request:
                # Более строгий лимит для файлов
                file_requests = self._file_requests_per_user[user_id]
                recent_file_requests = [
                    req_time for req_time in file_requests
                    if (now - req_time).total_seconds() < 60
                ]
                
                if len(recent_file_requests) >= self.max_file_requests_per_minute:
                    logging.warning(
                        f"Rate limit exceeded for file requests: user_id={user_id}, "
                        f"requests={len(recent_file_requests)}/{self.max_file_requests_per_minute}"
                    )
                    if hasattr(event, 'answer'):
                        await event.answer(
                            "⚠️ Слишком много запросов. Пожалуйста, подождите немного."
                        )
                    return
            
                file_requests.append(now)
                self._file_requests_per_user[user_id] = recent_file_requests + [now]
            
            # Проверяем общий лимит запросов в минуту
            requests = self._requests_per_user[user_id]
            recent_requests = [
                req_time for req_time in requests
                if (now - req_time).total_seconds() < 60
            ]
            
            if len(recent_requests) >= self.max_requests_per_minute:
                logging.warning(
                    f"Rate limit exceeded per minute: user_id={user_id}, "
                    f"requests={len(recent_requests)}/{self.max_requests_per_minute}"
                )
                if hasattr(event, 'answer'):
                    await event.answer(
                        "⚠️ Слишком много запросов в минуту. Пожалуйста, подождите."
                    )
                return
            
            # Проверяем лимит запросов в час
            hourly_requests = [
                req_time for req_time in requests
                if (now - req_time).total_seconds() < 3600
            ]
            
            if len(hourly_requests) >= self.max_requests_per_hour:
                logging.warning(
                    f"Rate limit exceeded per hour: user_id={user_id}, "
                    f"requests={len(hourly_requests)}/{self.max_requests_per_hour}"
                )
                if hasattr(event, 'answer'):
                    await event.answer(
                        "⚠️ Превышен лимит запросов в час. Пожалуйста, попробуйте позже."
                    )
                return
            
            requests.append(now)
            self._requests_per_user[user_id] = hourly_requests + [now]
        
        return await handler(event, data)
    
    def _cleanup_old_requests(self, user_id: int, now: datetime) -> None:
        """Удаляет старые записи (старше часа) для экономии памяти."""
        # Очищаем общие запросы
        requests = self._requests_per_user[user_id]
        self._requests_per_user[user_id] = [
            req_time for req_time in requests
            if (now - req_time).total_seconds() < 3600
        ]
        
        # Очищаем запросы файлов
        file_requests = self._file_requests_per_user[user_id]
        self._file_requests_per_user[user_id] = [
            req_time for req_time in file_requests
            if (now - req_time).total_seconds() < 3600
        ]


class ExpenseCatBot:
    """Telegram bot orchestrating OCR, bank parsing, and Supabase storage."""

    def __init__(
        self, 
        token: str, 
        supabase_gateway: Optional[SupabaseGateway] = None,
        feedback_chat_id: Optional[str] = None,
        failed_receipts_chat_id: Optional[str] = None,
        admin_user_ids: Optional[set[int]] = None
    ) -> None:
        self.bot = Bot(token=token)
        self.dp = Dispatcher()
        self.router = Router(name="expensecat")
        self.supabase = supabase_gateway
        self.feedback_chat_id = feedback_chat_id  # ID канала/чата для отзывов
        self.failed_receipts_chat_id = failed_receipts_chat_id  # ID канала для фейлов чеков
        self.admin_user_ids = admin_user_ids or set()  # Список ID администраторов
        self._media_group_cache: Dict[str, List[Message]] = {}
        self._media_group_tasks: Dict[str, asyncio.Task] = {}
        
        # Добавляем rate limiting middleware для защиты от DDoS
        rate_limit_middleware = RateLimitMiddleware(
            max_requests_per_minute=15,
            max_requests_per_hour=50,
            max_file_requests_per_minute=10,
        )
        self.router.message.middleware(rate_limit_middleware)
        self.router.callback_query.middleware(rate_limit_middleware)
        logging.info("Rate limiting middleware enabled for DDoS protection")
        
        self.dp.include_router(self.router)
        self._register_handlers()
    
    async def _process_report_request(
        self,
        user_id: int,
        message_or_callback: Message | CallbackQuery,
        state: FSMContext,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: Optional[str] = None
    ) -> None:
        """
        Единый метод для обработки запросов отчетов.
        Принимает даты начала и конца (или период) и обрабатывает отчет:
        - Если несколько валют - показывает меню выбора валюты
        - Если одна валюта - сразу показывает отчет
        """
        try:
            # Получаем отчет
            logging.info(f"Fetching report for user {user_id}: period={period}, start_date={start_date}, end_date={end_date}")
            report_start = time.perf_counter()
            report = await self.supabase.fetch_monthly_report(
                user_id,
                period=period,
                start_date=start_date,
                end_date=end_date
            )
            report_time = time.perf_counter() - report_start
            logging.info(f"⏱️ [PERF] Report fetched in {report_time*1000:.1f}ms ({report_time:.2f}s)")
            
            if not report:
                logging.warning(f"Empty report returned for user {user_id}")
                message = message_or_callback.message if isinstance(message_or_callback, CallbackQuery) else message_or_callback
                await message.answer("📊 Нет данных за выбранный период.")
                return
            
            currencies_data = report.get("currencies_data", {})
            currencies_list = list(currencies_data.keys())
            
            logging.info(f"📊 [REPORT] Found currencies in report: {currencies_list}, count: {len(currencies_list)}")
            
            # Если несколько валют - предлагаем выбрать валюту или общий отчет
            if len(currencies_list) > 1:
                logging.info(f"📊 [REPORT] Multiple currencies detected, showing currency selection menu")
                # Сохраняем отчет в состояние для последующего использования
                await state.update_data(
                    report_data=report,
                    report_period=period,
                    report_start_date=start_date,
                    report_end_date=end_date
                )
                
                # Создаем клавиатуру с валютами и опцией "общий отчет"
                currency_symbols = {
                    "RUB": "₽",
                    "KZT": "₸",
                    "USD": "$",
                    "EUR": "€",
                    "GBP": "£",
                    "GEL": "₾",
                }
                
                keyboard_buttons = []
                # Добавляем кнопки для каждой валюты
                for currency in sorted(currencies_list):
                    symbol = currency_symbols.get(currency, currency)
                    total = currencies_data[currency].get("total", 0.0)
                    keyboard_buttons.append([
                        InlineKeyboardButton(
                            text=f"{symbol} {total:.2f}",
                            callback_data=f"report_currency_{currency}"
                        )
                    ])
                
                # Добавляем кнопку "общий отчет"
                keyboard_buttons.append([
                    InlineKeyboardButton(
                        text="🌍 Общий отчет (все валюты)",
                        callback_data="report_currency_all"
                    )
                ])
                
                keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)
                
                message = message_or_callback.message if isinstance(message_or_callback, CallbackQuery) else message_or_callback
                await message.answer(
                    "💰 За выбранный период есть траты в разных валютах.\n"
                    "Выберите валюту для отчета или общий отчет:",
                    reply_markup=keyboard
                )
                await state.set_state(ReportStates.waiting_for_currency)
                return
            
            # Если одна валюта - сразу показываем отчет
            logging.info(f"Formatting report: period={report.get('period')}, currencies={currencies_list}")
            report_text = format_report(report)
            
            # Обрезаем если слишком длинный
            truncated_report = truncate_message_for_telegram(report_text)
            message = message_or_callback.message if isinstance(message_or_callback, CallbackQuery) else message_or_callback
            await message.answer(truncated_report)
            await state.clear()
            logging.info(f"✅ Report sent successfully to user {user_id}")
        except Exception as exc:
            logging.exception(f"Error in _process_report_request: {exc}")
            try:
                message = message_or_callback.message if isinstance(message_or_callback, CallbackQuery) else message_or_callback
                # Отправляем техническую ошибку в технический чат
                if message.from_user:
                    await self._send_tech_error_to_channel(
                        user_id=message.from_user.id,
                        username=message.from_user.username,
                        first_name=message.from_user.first_name,
                        error=exc,
                        context="Получение отчета",
                        additional_info=f"start_date={start_date}, end_date={end_date}, period={period}"
                    )
                # Пользователю отправляем дружелюбное сообщение
                await message.answer("❌ Ошибка при получении отчета. Пожалуйста, попробуйте позже или обратитесь в поддержку через /feedback")
            except Exception as send_exc:
                logging.error(f"Failed to send error message to user: {send_exc}")
            await state.clear()
    
    async def _send_feedback_to_channel(
        self,
        user_id: int,
        username: Optional[str],
        first_name: Optional[str],
        feedback_type: str,
        type_name: str,
        emoji: str,
        feedback_text: str,
        photo_bytes: Optional[bytes] = None
    ) -> None:
        """Отправляет отзыв в канал обратной связи"""
        # Проверяем, что канал настроен
        if not self.feedback_chat_id:
            logging.warning("Feedback channel not configured (FEEDBACK_CHAT_ID not set)")
            return
        
        # Формируем сообщение
        user_info = f"ID: {user_id}"
        if username:
            user_info += f" (@{username})"
        if first_name:
            user_info += f" - {first_name}"
        
        message_text = (
            f"{emoji} <b>Новый отзыв: {type_name}</b>\n\n"
            f"👤 Пользователь: {user_info}\n"
            f"📝 Тип: {type_name}\n\n"
            f"💬 Сообщение:\n{feedback_text}"
        )
        
        # Отправляем в канал
        try:
            # Пробуем разные форматы ID
            chat_id = self.feedback_chat_id
            
            # Если это числовой ID, пробуем как int
            if chat_id.startswith("-") or chat_id.isdigit():
                try:
                    chat_id_int = int(chat_id)
                    # Если есть фото, отправляем с фото
                    if photo_bytes:
                        from aiogram.types import BufferedInputFile
                        # Обрезаем caption до 1024 символов (лимит Telegram для подписей к фото)
                        MAX_CAPTION_LENGTH = 1024
                        photo_caption = message_text
                        if len(photo_caption) > MAX_CAPTION_LENGTH:
                            # Безопасно обрезаем HTML, закрывая незакрытые теги
                            photo_caption = truncate_html_safely(photo_caption, MAX_CAPTION_LENGTH)
                            logging.warning(f"Feedback caption too long ({len(message_text)} chars), truncated to {len(photo_caption)} chars")
                        
                        photo_file = BufferedInputFile(photo_bytes, filename="feedback_photo.jpg")
                        await self.bot.send_photo(
                            chat_id=chat_id_int,
                            photo=photo_file,
                            caption=photo_caption,
                            parse_mode="HTML"
                        )
                    else:
                        # Для текстовых сообщений лимит 4096 символов
                        MAX_MESSAGE_LENGTH = 4096
                        text_message = message_text
                        if len(text_message) > MAX_MESSAGE_LENGTH:
                            # Безопасно обрезаем HTML, закрывая незакрытые теги
                            text_message = truncate_html_safely(text_message, MAX_MESSAGE_LENGTH)
                            logging.warning(f"Feedback message too long ({len(message_text)} chars), truncated to {len(text_message)} chars")
                        
                        await self.bot.send_message(
                            chat_id=chat_id_int,
                            text=text_message,
                            parse_mode="HTML"
                        )
                    logging.info(f"✅ Feedback sent to channel {chat_id_int}")
                    return
                except Exception as int_exc:
                    logging.warning(f"Failed to send as int {chat_id_int}: {int_exc}")
            
            # Пробуем как строку (для username каналов типа @channel_name)
            if photo_bytes:
                from aiogram.types import BufferedInputFile
                # Обрезаем caption до 1024 символов (лимит Telegram для подписей к фото)
                MAX_CAPTION_LENGTH = 1024
                photo_caption = message_text
                if len(photo_caption) > MAX_CAPTION_LENGTH:
                    # Безопасно обрезаем HTML, закрывая незакрытые теги
                    photo_caption = truncate_html_safely(photo_caption, MAX_CAPTION_LENGTH)
                    logging.warning(f"Feedback caption too long ({len(message_text)} chars), truncated to {len(photo_caption)} chars")
                
                photo_file = BufferedInputFile(photo_bytes, filename="feedback_photo.jpg")
                await self.bot.send_photo(
                    chat_id=chat_id,
                    photo=photo_file,
                    caption=photo_caption,
                    parse_mode="HTML"
                )
            else:
                # Для текстовых сообщений лимит 4096 символов
                MAX_MESSAGE_LENGTH = 4096
                text_message = message_text
                if len(text_message) > MAX_MESSAGE_LENGTH:
                    # Безопасно обрезаем HTML, закрывая незакрытые теги
                    text_message = truncate_html_safely(text_message, MAX_MESSAGE_LENGTH)
                    logging.warning(f"Feedback message too long ({len(message_text)} chars), truncated to {len(text_message)} chars")
                
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=text_message,
                    parse_mode="HTML"
                )
            logging.info(f"✅ Feedback sent to channel {chat_id}")
        except Exception as exc:
            # Если канал не найден, логируем подробную ошибку
            error_msg = str(exc)
            logging.error(
                f"❌ Failed to send feedback to channel {self.feedback_chat_id}: {error_msg}\n"
                f"   Проверьте:\n"
                f"   1. Бот добавлен в канал как администратор?\n"
                f"   2. ID канала правильный? (для приватных каналов нужен ID вида -100...)\n"
                f"   3. Для публичных каналов можно использовать @channel_name"
            )
    
    async def _send_failed_receipt_to_channel(
        self,
        user_id: int,
        username: Optional[str],
        first_name: Optional[str],
        reason: str,
        file_path: Optional[str] = None,
        file_bytes: Optional[bytes] = None,
        mime_type: Optional[str] = None,
        error_message: Optional[str] = None,
        receipt_table: Optional[str] = None
    ) -> None:
        """Отправляет информацию о фейле чека в канал с фото"""
        if not self.failed_receipts_chat_id:
            logging.warning("failed_receipts_chat_id not configured, skipping failed receipt notification")
            return
        
        logging.info(f"📤 Preparing to send failed receipt to channel: user_id={user_id}, reason={reason}, has_file_bytes={file_bytes is not None and isinstance(file_bytes, bytes) and len(file_bytes) > 0 if file_bytes else False}")
        
        # Формируем сообщение
        user_info = f"ID: {user_id}"
        if username:
            user_info += f" (@{username})"
        if first_name:
            user_info += f" - {first_name}"
        
        reason_names = {
            "rejected_by_user": "Отклонен пользователем",
            "validation_error": "Ошибка валидации"
        }
        reason_name = reason_names.get(reason, reason)
        
        emoji = "❌" if reason == "rejected_by_user" else "⚠️"
        
        caption = (
            f"{emoji} <b>Фейл чека: {reason_name}</b>\n\n"
            f"👤 Пользователь: {user_info}\n"
            f"📋 Причина: {reason_name}\n"
        )
        
        if file_path:
            caption += f"📁 Файл: <code>{file_path}</code>\n"
        
        if error_message:
            caption += f"\n💬 Ошибка:\n<code>{error_message[:500]}</code>"
        
        # Добавляем результаты калькуляции, если они есть
        if receipt_table:
            # Обрезаем таблицу, если она слишком длинная
            # Для подписей к фото лимит 1024 символа, для текстовых сообщений 4096
            max_table_length = 2000  # Оставляем место для остального текста
            if len(receipt_table) > max_table_length:
                receipt_table = receipt_table[:max_table_length] + "\n\n... (обрезано)"
            caption += f"\n\n📊 <b>Результаты калькуляции:</b>\n<pre>{receipt_table}</pre>"
        
        # Сохраняем полный текст для текстовых сообщений
        full_caption = caption
        
        # Отправляем в канал
        try:
            chat_id = self.failed_receipts_chat_id
            
            # Определяем chat_id как int или str
            chat_id_int = None
            if chat_id.startswith("-") or chat_id.isdigit():
                try:
                    chat_id_int = int(chat_id)
                except ValueError:
                    pass
            
            # Если есть фото, отправляем его с подписью
            if file_bytes and isinstance(file_bytes, bytes) and len(file_bytes) > 0:
                # Обрезаем caption до 1024 символов (лимит Telegram для подписей к фото)
                MAX_CAPTION_LENGTH = 1024
                photo_caption = caption
                if len(photo_caption) > MAX_CAPTION_LENGTH:
                    # Безопасно обрезаем HTML, закрывая незакрытые теги
                    photo_caption = truncate_html_safely(photo_caption, MAX_CAPTION_LENGTH)
                    logging.warning(f"Caption too long ({len(caption)} chars), truncated to {len(photo_caption)} chars")
                
                # Определяем расширение файла из mime_type
                ext_map = {
                    "image/jpeg": "jpg",
                    "image/jpg": "jpg",
                    "image/png": "png",
                    "image/webp": "webp",
                    "image/heic": "heic",
                    "image/heif": "heif",
                }
                ext = ext_map.get(mime_type or "image/jpeg", "jpg")
                filename = f"rejected_receipt_{user_id}_{reason}.{ext}"
                
                logging.info(f"📤 Sending failed receipt photo to channel: {len(file_bytes)} bytes, mime_type={mime_type}, caption_length={len(photo_caption)}")
                photo = BufferedInputFile(file_bytes, filename=filename)
                
                if chat_id_int is not None:
                    await self.bot.send_photo(
                        chat_id=chat_id_int,
                        photo=photo,
                        caption=photo_caption,
                        parse_mode="HTML"
                    )
                    logging.info(f"✅ Failed receipt photo sent to channel {chat_id_int}")
                else:
                    await self.bot.send_photo(
                        chat_id=chat_id,
                        photo=photo,
                        caption=photo_caption,
                        parse_mode="HTML"
                    )
                    logging.info(f"✅ Failed receipt photo sent to channel {chat_id}")
            else:
                # Если фото нет, отправляем только текст
                logging.info(f"⚠️ No file_bytes provided or invalid, sending text-only message. file_bytes type: {type(file_bytes)}, is None: {file_bytes is None}, length: {len(file_bytes) if file_bytes else 0}")
                # Для текстовых сообщений лимит 4096 символов
                MAX_MESSAGE_LENGTH = 4096
                text_message = full_caption
                if len(text_message) > MAX_MESSAGE_LENGTH:
                    # Безопасно обрезаем HTML, закрывая незакрытые теги
                    text_message = truncate_html_safely(text_message, MAX_MESSAGE_LENGTH)
                    logging.warning(f"Message too long ({len(full_caption)} chars), truncated to {len(text_message)} chars")
                
                if chat_id_int is not None:
                    await self.bot.send_message(
                        chat_id=chat_id_int,
                        text=text_message,
                        parse_mode="HTML"
                    )
                    logging.info(f"✅ Failed receipt info sent to channel {chat_id_int}")
                else:
                    await self.bot.send_message(
                        chat_id=chat_id,
                        text=text_message,
                        parse_mode="HTML"
                    )
                    logging.info(f"✅ Failed receipt info sent to channel {chat_id}")
        except Exception as exc:
            error_msg = str(exc)
            logging.error(
                f"❌ Failed to send failed receipt info to channel {self.failed_receipts_chat_id}: {error_msg}\n"
                f"   Проверьте:\n"
                f"   1. Бот добавлен в канал как администратор?\n"
                f"   2. ID канала правильный? (для приватных каналов нужен ID вида -100...)\n"
                f"   3. Для публичных каналов можно использовать @channel_name"
            )

    async def _send_tech_error_to_channel(
        self,
        user_id: int,
        username: Optional[str],
        first_name: Optional[str],
        error: Exception,
        context: str,
        additional_info: Optional[str] = None
    ) -> None:
        """Отправляет техническую ошибку в технический чат"""
        if not self.failed_receipts_chat_id:
            logging.warning("failed_receipts_chat_id not configured, skipping tech error notification")
            return
        
        try:
            # Формируем сообщение
            user_info = f"ID: {user_id}"
            if username:
                user_info += f" (@{username})"
            if first_name:
                user_info += f" - {first_name}"
            
            error_type = type(error).__name__
            error_str = str(error)
            
            message_text = (
                f"🔧 <b>Техническая ошибка</b>\n\n"
                f"👤 Пользователь: {user_info}\n"
                f"📋 Контекст: {context}\n"
                f"❌ Тип ошибки: <code>{error_type}</code>\n"
                f"💬 Сообщение:\n<code>{error_str[:2000]}</code>\n"
            )
            
            if additional_info:
                message_text += f"\n📝 Дополнительная информация:\n<code>{additional_info[:1000]}</code>"
            
            # Обрезаем до лимита Telegram (4096 символов)
            MAX_MESSAGE_LENGTH = 4096
            if len(message_text) > MAX_MESSAGE_LENGTH:
                message_text = message_text[:MAX_MESSAGE_LENGTH - 50] + "\n\n... (обрезано)"
            
            chat_id = self.failed_receipts_chat_id
            
            # Определяем chat_id как int или str
            chat_id_int = None
            if chat_id.startswith("-") or chat_id.isdigit():
                try:
                    chat_id_int = int(chat_id)
                except ValueError:
                    pass
            
            if chat_id_int is not None:
                await self.bot.send_message(
                    chat_id=chat_id_int,
                    text=message_text,
                    parse_mode="HTML"
                )
                logging.info(f"✅ Tech error sent to channel {chat_id_int}")
            else:
                await self.bot.send_message(
                    chat_id=chat_id,
                    text=message_text,
                    parse_mode="HTML"
                )
                logging.info(f"✅ Tech error sent to channel {chat_id}")
        except Exception as exc:
            error_msg = str(exc)
            logging.error(
                f"❌ Failed to send tech error to channel {self.failed_receipts_chat_id}: {error_msg}\n"
                f"   Проверьте:\n"
                f"   1. Бот добавлен в канал как администратор?\n"
                f"   2. ID канала правильный? (для приватных каналов нужен ID вида -100...)\n"
                f"   3. Для публичных каналов можно использовать @channel_name"
            )

    def _create_category_keyboard(self) -> InlineKeyboardMarkup:
        """Создает клавиатуру с категориями расходов"""
        categories = [
            ["Продукты", "Мясо/Рыба", "Молочные продукты"],
            ["Хлеб/Выпечка", "Овощи/Фрукты", "Напитки"],
            ["Алкоголь", "Сладости", "Ресторан/Кафе"],
            ["Доставка еды", "Фастфуд", "Транспорт"],
            ["Такси", "Парковка", "Бензин/Топливо"],
            ["Медицина", "Одежда", "Бытовая химия"],
            ["Косметика/Гигиена", "Электроника", "Образование"],
            ["Книги", "Развлечения", "Спорт"],
            ["Путешествия", "Коммунальные", "Интернет/Связь"],
            ["Ремонт", "Другое"]
        ]
        
        keyboard_buttons = []
        for row in categories:
            button_row = []
            for category in row:
                button_row.append(InlineKeyboardButton(
                    text=category,
                    callback_data=f"expense_category_{category}"
                ))
            keyboard_buttons.append(button_row)
        
        return InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)

    def _create_currency_keyboard(self) -> InlineKeyboardMarkup:
        """Создает клавиатуру для выбора валюты"""
        return InlineKeyboardMarkup(inline_keyboard=[
            [
                InlineKeyboardButton(text="₽ RUB", callback_data="setup_currency_RUB"),
                InlineKeyboardButton(text="₸ KZT", callback_data="setup_currency_KZT"),
            ],
            [
                InlineKeyboardButton(text="$ USD", callback_data="setup_currency_USD"),
                InlineKeyboardButton(text="€ EUR", callback_data="setup_currency_EUR"),
            ],
            [
                InlineKeyboardButton(text="£ GBP", callback_data="setup_currency_GBP"),
                InlineKeyboardButton(text="₾ GEL", callback_data="setup_currency_GEL"),
            ],
        ])

    @classmethod
    def from_env(cls) -> "ExpenseCatBot":
        token = os.getenv("EXPENSECAT_BOT_TOKEN")
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        feedback_chat_id = os.getenv("FEEDBACK_CHAT_ID")  # ID канала/чата для отзывов
        failed_receipts_chat_id = os.getenv("FAILED_RECEIPTS_CHAT_ID")  # ID канала для фейлов чеков
        
        # Список администраторов (ID пользователей через запятую)
        admin_user_ids_str = os.getenv("ADMIN_USER_IDS", "").strip()
        admin_user_ids = set()
        if admin_user_ids_str:
            try:
                admin_user_ids = {int(uid.strip()) for uid in admin_user_ids_str.split(",") if uid.strip()}
                logging.info(f"Admin user IDs configured: {admin_user_ids}")
            except ValueError:
                logging.warning(f"Invalid ADMIN_USER_IDS format: {admin_user_ids_str}")
        else:
            logging.info("No ADMIN_USER_IDS configured, stats command will be disabled")
        if not token:
            raise RuntimeError("EXPENSECAT_BOT_TOKEN is required to run ExpenseCatBot.")
        gateway = None
        if supabase_url and supabase_key:
            gateway = SupabaseGateway(url=supabase_url, service_key=supabase_key)
        else:
            logging.warning(
                "Supabase credentials not found. Persistence features are disabled until configured."
            )
        if feedback_chat_id:
            logging.info(f"Feedback channel configured: {feedback_chat_id}")
        if failed_receipts_chat_id:
            logging.info(f"Failed receipts channel configured: {failed_receipts_chat_id}")
        return cls(
            token=token, 
            supabase_gateway=gateway, 
            feedback_chat_id=feedback_chat_id,
            failed_receipts_chat_id=failed_receipts_chat_id,
            admin_user_ids=admin_user_ids
        )

    async def run(self) -> None:
        logging.info("Starting ExpenseCatBot")
        
        # Настраиваем меню команд
        commands = [
            BotCommand(command="expense", description="Добавить расход вручную"),
            BotCommand(command="report", description="Получить отчёт"),
            BotCommand(command="statement", description="Импортировать выписку"),
            BotCommand(command="export", description="Экспорт данных в CSV"),
            BotCommand(command="limits", description="Проверить лимиты и подписку"),
            BotCommand(command="subscribe", description="Оформить подписку"),
            BotCommand(command="feedback", description="Обратная связь (ошибки, предложения)"),
            BotCommand(command="settings", description="Настройки (валюта по умолчанию)"),
            # Команда /stats скрыта от обычных пользователей (только для администраторов)
            BotCommand(command="delete_expense", description="Удалить расход"),
            BotCommand(command="delete_all", description="Удалить все данные"),
        ]
        await self.bot.set_my_commands(commands)
        logging.info("Bot commands menu configured")
        
        # Логируем информацию о доступных OCR движках
        ocr_status = []
        if TESSERACT_AVAILABLE:
            ocr_status.append("Tesseract ✓")
        if PADDLEOCR_AVAILABLE:
            ocr_status.append("PaddleOCR ✓")
        logging.info(f"Available OCR engines: {', '.join(ocr_status) if ocr_status else 'None'}")
        logging.info(f"Selected OCR engine: {OCR_ENGINE}")
        await self.dp.start_polling(
            self.bot, allowed_updates=self.dp.resolve_used_update_types()
        )

    def _register_handlers(self) -> None:
        @self.router.message(CommandStart())
        async def handle_start(message: Message, state: FSMContext) -> None:
            await state.clear()
            
            # Проверяем, есть ли у пользователя настройки
            if self.supabase and message.from_user:
                settings = await self.supabase.get_user_settings(message.from_user.id)
                
                # Если настроек нет и нет данных - предлагаем выбрать валюту
                if not settings:
                    has_data = await self.supabase.check_user_has_data(message.from_user.id)
                    if not has_data:
                        # Первый запуск - предлагаем выбрать валюту
                        keyboard = self._create_currency_keyboard()
                        await message.answer(
                            "👋 Привет! Я помогу вам вести учёт расходов.\n\n"
                            "✨ Что я умею:\n"
                            "📸 Распознаю чеки по фото (OCR + QR-коды)\n"
                            "📝 Добавляю расходы вручную\n"
                            "🏦 Импортирую выписки из банка\n"
                            "📊 Создаю отчёты по категориям и магазинам\n"
                            "💰 Поддерживаю несколько валют\n"
                            "📤 Экспортирую данные в CSV\n\n"
                            "💡 Умное распознавание:\n"
                            "Я автоматически определяю, что вы хотите сделать, по типу отправленного контента:\n"
                            "• 📸 Отправьте фото → я распознаю чек\n"
                            "• 📝 Отправьте текст с суммой (например, \"кофе 1300\" или \"автосервис 10к\") → я добавлю расход\n"
                            "• 📄 Отправьте файл банковской выписки (CSV/PDF/XLSX) → я импортирую транзакции\n\n"
                            "Команды не обязательны — просто отправьте нужный контент!\n\n"
                            "💰 Для начала выберите валюту по умолчанию:",
                            reply_markup=keyboard
                        )
                        await state.set_state(SetupStates.waiting_for_currency)
                        return
            
            # Обычное приветствие для существующих пользователей
            await message.answer(
                "👋 Привет! Я помогу вам вести учёт расходов.\n\n"
                "💡 Умное распознавание:\n"
                "Я автоматически определяю, что вы хотите сделать, по типу отправленного контента:\n"
                "• 📸 Отправьте фото → я распознаю чек\n"
                "• 📝 Отправьте текст с суммой (например, \"кофе 1300\" или \"автосервис 10к\") → я добавлю расход\n"
                "• 📄 Отправьте файл банковской выписки (CSV/PDF/XLSX) → я импортирую транзакции\n\n"
                "Команды не обязательны — просто отправьте нужный контент!\n\n"
                "✨ Основные возможности:\n"
                "📊 Получайте отчёты по категориям, магазинам и периодам\n"
                "💰 Поддерживаю несколько валют\n"
                "📤 Экспортируйте данные в CSV\n\n"
                "📋 Команды:\n"
                "/report — получить отчёт\n"
                "/export — экспорт в CSV\n"
                "/subscribe — оформить подписку\n"
                "/delete_expense — удалить расход\n"
                "/settings — настройки"
            )

        @self.router.message(Command("cancel"))
        async def handle_cancel(message: Message, state: FSMContext) -> None:
            await state.clear()
            await message.answer("Ок, отменили.")
        
        @self.router.message(Command("limits"))
        async def handle_limits(message: Message) -> None:
            """Обработчик команды /limits - показывает информацию о лимитах и подписке"""
            if not message.from_user:
                return
            
            if not self.supabase:
                await message.answer("❌ База данных не подключена.")
                return
            
            limits = await self.supabase.get_or_create_user_limits(message.from_user.id)
            subscription_type = limits.get("subscription_type", "trial")
            receipts_count = limits.get("receipts_count", 0)
            limit_receipts = limits.get("limit_receipts", 20)
            expires_at = limits.get("expires_at")
            
            # Определяем название тарифа и эмодзи
            subscription_info = {
                "trial": ("Trial", "🆓", "бесплатно"),
                "standard": ("Standard", "📦", "100 ⭐/мес"),
                "pro": ("Pro", "⭐", "200 ⭐/мес"),
                "premium": ("Premium", "👑", "500 ⭐/мес"),
            }
            sub_name, sub_emoji, sub_price = subscription_info.get(subscription_type, ("Unknown", "❓", "?"))
            
            # Формируем сообщение
            status_text = f"{sub_emoji} <b>Ваш тариф: {sub_name}</b>\n"
            if subscription_type != "trial":
                status_text += f"💰 Стоимость: {sub_price}\n"
            status_text += "\n"
            
            # Информация о лимитах
            if subscription_type == "premium":
                status_text += f"📊 <b>Чеков использовано:</b> {receipts_count}\n"
                status_text += f"♾️ <b>Лимит:</b> Безлимит\n"
            else:
                status_text += f"📊 <b>Чеков использовано:</b> {receipts_count}/{limit_receipts}\n"
                status_text += f"📈 <b>Осталось:</b> {max(0, limit_receipts - receipts_count)} чеков\n"
            
            # Информация о сроке действия подписки
            if expires_at and subscription_type != "trial":
                try:
                    if isinstance(expires_at, str):
                        expires_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                    else:
                        expires_dt = expires_at
                    now = datetime.utcnow().replace(tzinfo=expires_dt.tzinfo)
                    
                    if expires_dt > now:
                        expires_str = expires_dt.strftime("%d.%m.%Y")
                        days_left = (expires_dt - now).days
                        status_text += f"\n📅 <b>Подписка действует до:</b> {expires_str}\n"
                        if days_left > 0:
                            status_text += f"⏰ <b>Осталось дней:</b> {days_left}\n"
                    else:
                        status_text += f"\n⚠️ <b>Подписка истекла</b>\n"
                        status_text += f"Используйте /subscribe для продления\n"
                except Exception as e:
                    logging.warning(f"Error parsing expires_at: {e}")
            
            # Предупреждение о приближении к лимиту
            if subscription_type != "premium" and receipts_count >= limit_receipts * 0.8:
                remaining = limit_receipts - receipts_count
                if remaining <= 0:
                    status_text += f"\n⚠️ <b>Лимит исчерпан!</b>\n"
                    status_text += f"Используйте /subscribe для увеличения лимита\n"
                elif remaining <= 5:
                    status_text += f"\n⚠️ Осталось мало чеков! Используйте /subscribe\n"
            
            await message.answer(status_text, parse_mode="HTML")
        
        @self.router.message(Command("stats"))
        async def handle_stats(message: Message) -> None:
            """Обработчик команды /stats - показывает статистику распознавания чеков (только для администраторов)"""
            if not message.from_user:
                await message.answer("Не удалось определить пользователя.")
                return
            
            # Проверка доступа: только администраторы могут видеть статистику
            if message.from_user.id not in self.admin_user_ids:
                logging.info(f"User {message.from_user.id} attempted to access /stats (not admin)")
                await message.answer("❌ Эта команда недоступна.")
                return
            
            logging.info(f"Admin user {message.from_user.id} accessed /stats")
            
            if not self.supabase:
                await message.answer("База данных недоступна.")
                return
            
            try:
                stats = await self.supabase.get_receipt_stats(message.from_user.id)
                
                if not stats or stats.get("total_count", 0) == 0:
                    await message.answer(
                        "📊 Статистика распознавания чеков\n\n"
                        "Пока нет данных о распознавании чеков.\n"
                        "Отправьте фото чека, чтобы начать собирать статистику."
                    )
                    return
                
                total_successful = stats.get("total_successful", 0)
                total_failed = stats.get("total_failed", 0)
                total_count = stats.get("total_count", 0)
                overall_success_rate = stats.get("overall_success_rate", 0)
                
                qr_successful = stats.get("qr_successful", 0)
                qr_failed = stats.get("qr_failed", 0)
                qr_total = qr_successful + qr_failed
                
                openai_photo_successful = stats.get("openai_photo_successful", 0)
                openai_photo_failed = stats.get("openai_photo_failed", 0)
                openai_photo_total = openai_photo_successful + openai_photo_failed
                
                openai_qr_data_successful = stats.get("openai_qr_data_successful", 0)
                openai_qr_data_failed = stats.get("openai_qr_data_failed", 0)
                openai_qr_data_total = openai_qr_data_successful + openai_qr_data_failed
                
                response = (
                    f"📊 <b>Статистика распознавания чеков</b>\n\n"
                    f"📈 <b>Общая статистика:</b>\n"
                    f"✅ Успешно: {total_successful}\n"
                    f"❌ Неуспешно: {total_failed}\n"
                    f"📊 Всего: {total_count}\n"
                    f"📈 Успешность: {overall_success_rate}%\n\n"
                )
                
                if qr_total > 0:
                    qr_success_rate = round(100.0 * qr_successful / qr_total, 2) if qr_total > 0 else 0
                    response += (
                        f"🔍 <b>По QR-коду:</b>\n"
                        f"✅ Успешно: {qr_successful}\n"
                        f"❌ Неуспешно: {qr_failed}\n"
                        f"📊 Всего: {qr_total}\n"
                        f"📈 Успешность: {qr_success_rate}%\n\n"
                    )
                
                if openai_photo_total > 0:
                    openai_photo_success_rate = round(100.0 * openai_photo_successful / openai_photo_total, 2) if openai_photo_total > 0 else 0
                    response += (
                        f"🖼️ <b>По фото (OpenAI):</b>\n"
                        f"✅ Успешно: {openai_photo_successful}\n"
                        f"❌ Неуспешно: {openai_photo_failed}\n"
                        f"📊 Всего: {openai_photo_total}\n"
                        f"📈 Успешность: {openai_photo_success_rate}%\n\n"
                    )
                
                if openai_qr_data_total > 0:
                    openai_qr_data_success_rate = round(100.0 * openai_qr_data_successful / openai_qr_data_total, 2) if openai_qr_data_total > 0 else 0
                    response += (
                        f"🔗 <b>По QR-данным (OpenAI):</b>\n"
                        f"✅ Успешно: {openai_qr_data_successful}\n"
                        f"❌ Неуспешно: {openai_qr_data_failed}\n"
                        f"📊 Всего: {openai_qr_data_total}\n"
                        f"📈 Успешность: {openai_qr_data_success_rate}%\n"
                    )
                
                await message.answer(response, parse_mode="HTML")
            except Exception as exc:
                logging.exception(f"Error getting receipt stats: {exc}")
                # Отправляем техническую ошибку в технический чат
                if message.from_user:
                    await self._send_tech_error_to_channel(
                        user_id=message.from_user.id,
                        username=message.from_user.username,
                        first_name=message.from_user.first_name,
                        error=exc,
                        context="Получение статистики чеков"
                    )
                # Пользователю отправляем дружелюбное сообщение
                await message.answer("Ошибка при получении статистики. Попробуйте позже или обратитесь в поддержку через /feedback.")
        
        @self.router.message(Command("subscribe"))
        async def handle_subscribe(message: Message) -> None:
            """Обработчик команды /subscribe - показывает биллборд с тарифами"""
            if not message.from_user:
                return
            
            # Проверяем текущую подписку
            current_subscription = "trial"
            expires_at = None
            if self.supabase:
                limits = await self.supabase.get_or_create_user_limits(message.from_user.id)
                current_subscription = limits.get("subscription_type", "trial")
                expires_at = limits.get("expires_at")
                
                # Проверяем, не истекла ли подписка
                if expires_at and current_subscription in ("pro", "premium"):
                    try:
                        if isinstance(expires_at, str):
                            expires_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                        else:
                            expires_dt = expires_at
                        if expires_dt <= datetime.utcnow().replace(tzinfo=expires_dt.tzinfo):
                            current_subscription = "trial"
                    except:
                        pass
            
            # Формируем биллборд с тарифами
            billboard_text = (
                "💎 <b>Тарифы</b>\n\n"
                
                "🆓 <b>Trial</b> (текущий тариф)\n"
                "• 20 чеков бесплатно\n"
                "• Все базовые функции\n"
                "• Отчеты и экспорт\n\n"
                
                "📦 <b>Standard</b> — 100 ⭐/месяц\n"
                "• 50 чеков в месяц\n"
                "• Все функции бота\n\n"
                
                "⭐ <b>Pro</b> — 200 ⭐/месяц\n"
                "• 100 чеков в месяц\n"
                "• Все функции бота\n\n"
                
                "👑 <b>Premium</b> — 500 ⭐/месяц\n"
                "• Безлимит чеков\n"
                "• Все функции бота\n\n"
            )
            
            # Добавляем информацию о текущей подписке
            if current_subscription == "trial":
                billboard_text += "📊 <i>Ваш текущий тариф: Trial</i>\n\n"
            elif current_subscription == "standard" and expires_at:
                try:
                    if isinstance(expires_at, str):
                        expires_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                    else:
                        expires_dt = expires_at
                    if expires_dt > datetime.utcnow().replace(tzinfo=expires_dt.tzinfo):
                        expires_str = expires_dt.strftime("%d.%m.%Y")
                        billboard_text += f"📦 <i>У вас активна Standard подписка до {expires_str}</i>\n\n"
                    else:
                        billboard_text += "📊 <i>Ваш текущий тариф: Trial</i>\n\n"
                except:
                    billboard_text += "📊 <i>Ваш текущий тариф: Trial</i>\n\n"
            elif current_subscription == "pro" and expires_at:
                try:
                    if isinstance(expires_at, str):
                        expires_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                    else:
                        expires_dt = expires_at
                    if expires_dt > datetime.utcnow().replace(tzinfo=expires_dt.tzinfo):
                        expires_str = expires_dt.strftime("%d.%m.%Y")
                        billboard_text += f"⭐ <i>У вас активна Pro подписка до {expires_str}</i>\n\n"
                    else:
                        billboard_text += "📊 <i>Ваш текущий тариф: Trial</i>\n\n"
                except:
                    billboard_text += "📊 <i>Ваш текущий тариф: Trial</i>\n\n"
            elif current_subscription == "premium" and expires_at:
                try:
                    if isinstance(expires_at, str):
                        expires_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                    else:
                        expires_dt = expires_at
                    if expires_dt > datetime.utcnow().replace(tzinfo=expires_dt.tzinfo):
                        expires_str = expires_dt.strftime("%d.%m.%Y")
                        billboard_text += f"👑 <i>У вас активна Premium подписка до {expires_str}</i>\n\n"
                    else:
                        billboard_text += "📊 <i>Ваш текущий тариф: Trial</i>\n\n"
                except:
                    billboard_text += "📊 <i>Ваш текущий тариф: Trial</i>\n\n"
            
            billboard_text += "Выберите тариф для оплаты:"
            
            # Создаем кнопки для выбора тарифа (каждая кнопка на отдельной строке)
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="📦 Standard — 100 ⭐/мес", callback_data="subscribe_standard"),
                ],
                [
                    InlineKeyboardButton(text="⭐ Pro — 200 ⭐/мес", callback_data="subscribe_pro"),
                ],
                [
                    InlineKeyboardButton(text="👑 Premium — 500 ⭐/мес", callback_data="subscribe_premium"),
                ]
            ])
            
            await message.answer(billboard_text, reply_markup=keyboard, parse_mode="HTML")
        
        @self.router.callback_query(F.data.startswith("subscribe_"))
        async def handle_subscribe_callback(callback: CallbackQuery) -> None:
            """Обработчик выбора тарифа для подписки"""
            if not callback.from_user or not callback.data:
                return
            
            subscription_type = callback.data.replace("subscribe_", "")
            
            if subscription_type not in ("standard", "pro", "premium"):
                await callback.answer("Неверный тариф", show_alert=True)
                return
            
            # Проверяем текущую подписку
            if self.supabase:
                limits = await self.supabase.get_or_create_user_limits(callback.from_user.id)
                current_subscription = limits.get("subscription_type", "trial")
                expires_at = limits.get("expires_at")
                
                # Если уже есть активная подписка того же или более высокого уровня
                subscription_levels = {"trial": 0, "standard": 1, "pro": 2, "premium": 3}
                current_level = subscription_levels.get(current_subscription, 0)
                selected_level = subscription_levels.get(subscription_type, 0)
                
                if expires_at:
                    try:
                        if isinstance(expires_at, str):
                            expires_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                        else:
                            expires_dt = expires_at
                        if expires_dt > datetime.utcnow().replace(tzinfo=expires_dt.tzinfo):
                            if current_level >= selected_level:
                                expires_str = expires_dt.strftime("%d.%m.%Y")
                                subscription_names = {
                                    "standard": "Standard",
                                    "pro": "Pro",
                                    "premium": "Premium"
                                }
                                await callback.answer(
                                    f"У вас уже активна {subscription_names.get(current_subscription, '')} подписка до {expires_str}",
                                    show_alert=True
                                )
                                return
                    except:
                        pass
            
            # Определяем параметры подписки
            if subscription_type == "standard":
                price_amount = 100
                title = "Standard подписка ExpenseCatBot"
                description = (
                    "Получите Standard подписку и увеличьте лимит до 50 чеков в месяц!\n\n"
                    "✨ Что входит:\n"
                    "• 50 чеков в месяц (вместо 10)\n"
                    "• Все функции бота без ограничений"
                )
                limit_receipts = 50
            elif subscription_type == "pro":
                price_amount = 200
                title = "Pro подписка ExpenseCatBot"
                description = (
                    "Получите Pro подписку и увеличьте лимит до 100 чеков в месяц!\n\n"
                    "✨ Что входит:\n"
                    "• 100 чеков в месяц (вместо 10)\n"
                    "• Все функции бота без ограничений"
                )
                limit_receipts = 100
            elif subscription_type == "premium":
                price_amount = 500
                title = "Premium подписка ExpenseCatBot"
                description = (
                    "Получите Premium подписку с безлимитным количеством чеков!\n\n"
                    "✨ Что входит:\n"
                    "• Безлимит чеков\n"
                    "• Все функции бота без ограничений"
                )
                limit_receipts = None  # Безлимит
            
            price = LabeledPrice(label=f"{subscription_type.capitalize()} подписка (1 месяц)", amount=price_amount)
            
            # Для Telegram Stars provider_token не требуется
            invoice_kwargs = {
                "chat_id": callback.from_user.id,
                "title": title,
                "description": description,
                "payload": f"subscription_{subscription_type}_{callback.from_user.id}_{int(datetime.utcnow().timestamp())}",
                "currency": "XTR",  # XTR - валюта Telegram Stars
                "prices": [price],
                "start_parameter": f"{subscription_type}_subscription",
                "need_name": False,
                "need_phone_number": False,
                "need_email": False,
                "need_shipping_address": False,
                "send_phone_number_to_provider": False,
                "send_email_to_provider": False,
                "is_flexible": False,
            }
            
            # Для Telegram Stars provider_token не нужен, но aiogram может требовать его
            try:
                await self.bot.send_invoice(**invoice_kwargs)
                await callback.answer()
            except TypeError:
                # Если aiogram требует provider_token, передаем пустую строку
                invoice_kwargs["provider_token"] = ""
                await self.bot.send_invoice(**invoice_kwargs)
                await callback.answer()
            except Exception as exc:
                logging.exception(f"Error sending invoice: {exc}")
                # Отправляем техническую ошибку в технический чат
                if callback.from_user:
                    await self._send_tech_error_to_channel(
                        user_id=callback.from_user.id,
                        username=callback.from_user.username,
                        first_name=callback.from_user.first_name,
                        error=exc,
                        context="Отправка счета на оплату"
                    )
                # Пользователю отправляем дружелюбное сообщение
                await callback.answer("Ошибка при создании счета. Попробуйте позже или обратитесь в поддержку через /feedback.", show_alert=True)
        
        @self.router.pre_checkout_query()
        async def handle_pre_checkout_query(pre_checkout_query: PreCheckoutQuery) -> None:
            """Обработчик предварительной проверки платежа"""
            await pre_checkout_query.answer(ok=True)
        
        @self.router.message(F.successful_payment)
        async def handle_successful_payment(message: Message) -> None:
            """Обработчик успешной оплаты подписки"""
            if not message.from_user or not message.successful_payment:
                return
            
            payment = message.successful_payment
            user_id = message.from_user.id
            
            # Проверяем, что это оплата подписки
            if payment.invoice_payload.startswith("subscription_"):
                try:
                    # Извлекаем тип подписки из payload: subscription_{type}_{user_id}_{timestamp}
                    payload_parts = payment.invoice_payload.split("_")
                    if len(payload_parts) >= 2:
                        subscription_type = payload_parts[1]  # pro или premium
                    else:
                        subscription_type = "pro"  # По умолчанию pro для старых payload
                    
                    # Активируем подписку на 1 месяц
                    if self.supabase:
                        subscription = await self.supabase.activate_subscription(
                            user_id=user_id,
                            subscription_type=subscription_type,
                            months=1
                        )
                        expires_at = subscription.get("expires_at")
                        if expires_at:
                            try:
                                if isinstance(expires_at, str):
                                    expires_dt = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
                                else:
                                    expires_dt = expires_at
                                expires_str = expires_dt.strftime("%d.%m.%Y")
                            except:
                                expires_str = "через месяц"
                        else:
                            expires_str = "через месяц"
                        
                        # Формируем сообщение в зависимости от типа подписки
                        if subscription_type == "premium":
                            subscription_name = "Premium"
                            limit_text = "Безлимит чеков"
                            emoji = "👑"
                        elif subscription_type == "pro":
                            subscription_name = "Pro"
                            limit_text = "100 чеков в месяц"
                            emoji = "⭐"
                        elif subscription_type == "standard":
                            subscription_name = "Standard"
                            limit_text = "50 чеков в месяц"
                            emoji = "📦"
                        else:
                            subscription_name = "Unknown"
                            limit_text = "Неизвестный лимит"
                            emoji = "❓"
                        
                        await message.answer(
                            f"✅ Спасибо за покупку!\n\n"
                            f"{emoji} {subscription_name} подписка активирована до {expires_str}\n\n"
                            f"Теперь у вас:\n"
                            f"• {limit_text}\n"
                            f"• Все функции бота доступны без ограничений\n\n"
                            f"Приятного использования! 🚀"
                        )
                        logging.info(f"{subscription_name} subscription activated for user {user_id}")
                    else:
                        await message.answer(
                            "⚠️ Оплата получена, но не удалось активировать подписку. "
                            "Обратитесь в поддержку."
                        )
                except Exception as exc:
                    logging.exception(f"Error activating subscription after payment: {exc}")
                    await message.answer(
                        "⚠️ Произошла ошибка при активации подписки. "
                        "Обратитесь в поддержку через /feedback"
                    )

        @self.router.message(Command("receipt"))
        async def handle_receipt_entry(message: Message, state: FSMContext) -> None:
            await state.set_state(ReceiptStates.waiting_for_photo)
            instructions = (
                "📸 Пришлите фото/скан чека (jpg/png/pdf).\n\n"
                "💡 Советы для лучшего распознавания:\n"
                "• Чек должен занимать всё пространство на фото\n"
                "• Если чек длинный, сделайте панораму (горизонтальное фото)\n"
                "• Убедитесь, что текст четкий и читаемый\n"
                "• Если на чеке есть QR-код, можно сфотографировать только его - это ускорит обработку"
            )
            await message.answer(instructions)

        @self.router.message(ReceiptStates.waiting_for_photo, F.photo | F.document)
        async def handle_receipt_upload(message: Message, state: FSMContext) -> None:
            if message.media_group_id:
                await self._collect_media_group(message)
                return
            await self._process_receipt_message(message, state)

        @self.router.message(Command("statement"))
        async def handle_statement_entry(message: Message, state: FSMContext) -> None:
            await state.set_state(StatementStates.waiting_for_statement)
            await message.answer("Загрузите CSV/XLSX/PDF банковской выписки.")

        @self.router.message(StatementStates.waiting_for_statement, F.document)
        async def handle_statement_upload(message: Message, state: FSMContext) -> None:
            if message.media_group_id:
                await self._collect_media_group(message)
                return
            await self._process_statement_message(message)
            await state.clear()

        @self.router.message(Command("report"))
        async def handle_report(message: Message, state: FSMContext) -> None:
            try:
                logging.info(f"📊 [REPORT] Command /report received from user {message.from_user.id if message.from_user else 'unknown'}")
                logging.info(f"📊 [REPORT] Message object: {message}")
                logging.info(f"📊 [REPORT] State before clear: {await state.get_state()}")
                
                await state.clear()
                logging.info(f"📊 [REPORT] State after clear: {await state.get_state()}")
                
                if not self.supabase:
                    logging.warning("📊 [REPORT] Supabase not available for report command")
                    await message.answer(
                        "Отчёты по расходам появятся после подключения базы (Supabase)."
                    )
                    return
                
                if not message.from_user:
                    logging.warning("📊 [REPORT] No user in /report command")
                    await message.answer("❌ Ошибка: не удалось определить пользователя.")
                    return
                
                logging.info(f"📊 [REPORT] Creating keyboard for user {message.from_user.id}")
                # Показываем меню выбора периода
                keyboard = InlineKeyboardMarkup(inline_keyboard=[
                    [
                        InlineKeyboardButton(text="📅 Текущий месяц", callback_data="report_current_month"),
                        InlineKeyboardButton(text="📅 Прошлый месяц", callback_data="report_last_month"),
                    ],
                    [
                        InlineKeyboardButton(text="📅 Текущая неделя", callback_data="report_current_week"),
                        InlineKeyboardButton(text="📅 Прошлая неделя", callback_data="report_last_week"),
                    ],
                    [
                        InlineKeyboardButton(text="📅 Текущий год", callback_data="report_current_year"),
                        InlineKeyboardButton(text="📅 Произвольный период", callback_data="report_custom"),
                    ],
                    [
                        InlineKeyboardButton(text="📅 За один день", callback_data="report_single_day"),
                    ],
                ])
                logging.info(f"📊 [REPORT] Keyboard created, sending message to user {message.from_user.id}")
                result = await message.answer(
                    "📊 Выберите период для отчета:",
                    reply_markup=keyboard
                )
                logging.info(f"📊 [REPORT] ✅ Report menu sent successfully to user {message.from_user.id}, message_id={result.message_id if result else 'None'}")
            except Exception as exc:
                logging.exception(f"📊 [REPORT] ❌ Error in handle_report: {exc}")
                logging.error(f"📊 [REPORT] Error type: {type(exc).__name__}")
                logging.error(f"📊 [REPORT] Error args: {exc.args}")
                try:
                    # Отправляем техническую ошибку в технический чат
                    if message.from_user:
                        await self._send_tech_error_to_channel(
                            user_id=message.from_user.id,
                            username=message.from_user.username,
                            first_name=message.from_user.first_name,
                            error=exc,
                            context="Обработка команды /report"
                        )
                    # Пользователю отправляем дружелюбное сообщение
                    await message.answer("❌ Ошибка при обработке команды /report. Пожалуйста, попробуйте позже или обратитесь в поддержку через /feedback")
                except Exception as send_exc:
                    logging.error(f"📊 [REPORT] Failed to send error message to user: {send_exc}")

        @self.router.message(Command("export"))
        async def handle_export(message: Message, state: FSMContext) -> None:
            await state.clear()
            if not self.supabase:
                await message.answer("Экспорт доступен после подключения Supabase.")
                return
            
            # Проверяем, есть ли у пользователя настройки (валюта по умолчанию)
            if message.from_user:
                settings = await self.supabase.get_user_settings(message.from_user.id)
                if not settings:
                    # Если настроек нет, предлагаем выбрать валюту
                    keyboard = self._create_currency_keyboard()
                    await message.answer(
                        "📤 Экспортирую данные в CSV\n\n"
                        "💰 Для начала выберите валюту по умолчанию:",
                        reply_markup=keyboard
                    )
                    await state.set_state(SetupStates.waiting_for_currency)
                    return
            
            # Показываем меню выбора периода
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="📅 Текущий месяц", callback_data="export_current_month"),
                    InlineKeyboardButton(text="📅 Прошлый месяц", callback_data="export_last_month"),
                ],
                [
                    InlineKeyboardButton(text="📅 Текущая неделя", callback_data="export_current_week"),
                    InlineKeyboardButton(text="📅 Прошлая неделя", callback_data="export_last_week"),
                ],
                [
                    InlineKeyboardButton(text="📅 Текущий год", callback_data="export_current_year"),
                    InlineKeyboardButton(text="📅 Произвольный период", callback_data="export_custom"),
                ],
                [
                    InlineKeyboardButton(text="📅 Все данные", callback_data="export_all"),
                ],
            ])
            await message.answer(
                "📤 Выберите период для экспорта:",
                reply_markup=keyboard
            )

        @self.router.message(Command("expense"))
        async def handle_expense_entry(message: Message, state: FSMContext) -> None:
            await state.clear()
            
            # Получаем валюту по умолчанию пользователя
            default_currency = "RUB"
            if self.supabase and message.from_user:
                settings = await self.supabase.get_user_settings(message.from_user.id)
                if settings and settings.get("default_currency"):
                    default_currency = settings.get("default_currency")
            
            instructions = (
                "📝 Введите расход в формате:\n\n"
                "Примеры:\n"
                "• кофе 1300\n"
                "• автосервис 10к\n"
                "• такси 1200 KZT\n"
                "• продукты 5000 15.12.2025\n\n"
                "Вы можете указать:\n"
                "• Описание расхода\n"
                "• Сумму (можно использовать сокращения: к, тыс, т.р., тр и т.д.)\n"
                "• Валюту (₽, руб, ₸, тг, $, usd, €, eur и др.)\n"
                "• Дату (ДД.ММ.ГГГГ, опционально)"
            )
            
            await message.answer(instructions)
            await state.set_state(ExpenseStates.waiting_for_expense_text)
            logging.info(f"📝 [EXPENSE] Set state to waiting_for_expense_text for user {message.from_user.id if message.from_user else 'unknown'}")

        @self.router.message(Command("qr"))
        async def handle_qr_command(message: Message, state: FSMContext) -> None:
            await state.clear()
            if not message.photo and not message.document:
                await message.answer("Отправьте фото или документ с QR-кодом.")
                return
            file = await self._resolve_file(message)
            if file is None:
                await message.answer("Не удалось прочитать файл.")
                return
            file_bytes = await self._download_file(file.file_path)
            qr_codes = read_qr_codes(file_bytes)
            if not qr_codes:
                await message.answer("QR-коды не найдены на изображении.")
                return
            result_text = "📱 Найденные QR-коды:\n\n"
            for i, qr in enumerate(qr_codes, 1):
                result_text += f"{i}. Тип: {qr['type']}\n"
                result_text += f"   Данные: {qr['data']}\n\n"
            truncated_result = truncate_message_for_telegram(result_text)
            await message.answer(truncated_result)

        @self.router.message(FeedbackStates.waiting_for_feedback_text, F.photo | F.document)
        async def handle_feedback_photo(message: Message, state: FSMContext) -> None:
            """Обработчик фото/документов в контексте обратной связи"""
            # Сохраняем фото как часть обратной связи, не обрабатываем как чек
            data = await state.get_data()
            feedback_type = data.get("feedback_type", "unknown")
            
            user_id = message.from_user.id if message.from_user else None
            username = message.from_user.username if message.from_user else "unknown"
            first_name = message.from_user.first_name if message.from_user else "unknown"
            
            type_names = {
                "bug": "ошибка",
                "suggestion": "предложение",
                "complaint": "жалоба"
            }
            type_emojis = {
                "bug": "🐛",
                "suggestion": "💡",
                "complaint": "😞"
            }
            
            type_name = type_names.get(feedback_type, feedback_type)
            emoji = type_emojis.get(feedback_type, "📝")
            
            # Получаем текст обратной связи: сначала из подписи к фото, потом из состояния, потом заглушка
            if message.caption:
                feedback_text = message.caption.strip()
            else:
                feedback_text = data.get("feedback_text", "Фото приложено к обратной связи")
            
            # Если фото есть, добавляем информацию о нем
            if message.photo or message.document:
                file_type = "фото" if message.photo else "документ"
                if not message.caption:  # Добавляем информацию о файле только если не было подписи
                    feedback_text += f"\n\n📎 Приложен {file_type}"
            
            # Логируем обратную связь с фото
            logging.info(
                f"Feedback [{feedback_type}] with photo from user_id={user_id} "
                f"(@{username}, {first_name}): {feedback_text}"
            )
            
            # Сохраняем в базу данных, если Supabase подключен
            if self.supabase:
                try:
                    await self.supabase.save_feedback(
                        user_id=user_id,
                        username=username,
                        first_name=first_name,
                        feedback_type=feedback_type,
                        feedback_text=feedback_text
                    )
                except Exception as exc:
                    logging.exception(f"Ошибка при сохранении отзыва с фото в базу данных: {exc}")
            
            # Отправляем в канал обратной связи, если настроен
            if self.feedback_chat_id:
                logging.info(f"📤 Sending feedback with photo to channel: {self.feedback_chat_id}")
                try:
                    # Отправляем фото в канал, если есть
                    photo_bytes = None
                    if message.photo:
                        # Берем самое большое фото
                        photo = message.photo[-1]
                        file = await self.bot.get_file(photo.file_id)
                        photo_bytes = await self._download_file(file.file_path)
                        logging.info(f"📷 Downloaded photo for feedback, size: {len(photo_bytes) if photo_bytes else 0} bytes")
                    elif message.document:
                        file = await self.bot.get_file(message.document.file_id)
                        photo_bytes = await self._download_file(file.file_path)
                        logging.info(f"📄 Downloaded document for feedback, size: {len(photo_bytes) if photo_bytes else 0} bytes")
                    
                    await self._send_feedback_to_channel(
                        user_id=user_id,
                        username=username,
                        first_name=first_name,
                        feedback_type=feedback_type,
                        type_name=type_name,
                        emoji=emoji,
                        feedback_text=feedback_text,
                        photo_bytes=photo_bytes
                    )
                    logging.info(f"✅ Feedback with photo successfully sent to channel")
                except Exception as exc:
                    logging.exception(f"❌ Ошибка при отправке отзыва с фото в канал: {exc}")
            else:
                logging.warning("⚠️ Feedback channel not configured (FEEDBACK_CHAT_ID not set), skipping channel send")
            
            await message.answer(
                f"✅ Спасибо за вашу обратную связь!\n\n"
                f"{emoji} Тип: {type_name}\n"
                f"📝 Сообщение: {feedback_text}\n\n"
                f"Ваше сообщение с фото сохранено и будет рассмотрено."
            )
            await state.clear()

        @self.router.message(F.photo | F.document)
        async def handle_smart_upload(message: Message, state: FSMContext) -> None:
            """Автоматическая обработка фото и документов: фото → чек, документ → выписка"""
            # Проверяем, не находимся ли мы в специфическом состоянии (feedback и т.д.)
            current_state = await state.get_state()
            if current_state:
                state_name = str(current_state)
                # Если в состоянии feedback, не обрабатываем как чек/выписку
                if "FeedbackStates" in state_name:
                    return  # Пусть специфический обработчик обработает
            
            if message.media_group_id:
                await self._collect_media_group(message)
                return
            
            # Определяем тип: фото → чек, документ → выписка
            if message.photo:
                # Фото всегда считаем чеком
                await state.clear()
                await self._process_receipt_message(message, state)
                return
            
            # Для документов определяем по расширению и MIME типу
            classification = classify_upload_kind(message)
            if classification == "receipt":
                await state.clear()
                await self._process_receipt_message(message, state)
                return
            if classification == "statement":
                await state.clear()
                await self._process_statement_message(message)
                return
            
            # Если не удалось определить, считаем документ выпиской
            if message.document:
                await state.clear()
                await self._process_statement_message(message)
                return
            
            instructions = (
                "Не удалось определить тип файла. Используйте:\n"
                "• /receipt — для чеков\n"
                "• /statement — для выписок\n\n"
                "💡 Если это чек:\n"
                "• Чек должен занимать всё пространство на фото\n"
                "• Если чек длинный, сделайте панораму\n"
                "• Если есть QR-код, можно сфотографировать только его"
            )
            await message.answer(instructions)

        # Обработчик с состоянием должен быть ПЕРЕД общим F.text обработчиком
        @self.router.message(ExpenseStates.waiting_for_expense_text)
        async def handle_expense_text(message: Message, state: FSMContext) -> None:
            """Обработчик ввода текста расхода"""
            if not message.text or not message.from_user:
                await message.answer("❌ Пожалуйста, отправьте текст расхода.")
                return
            
            # Получаем валюту по умолчанию пользователя
            default_currency = "RUB"
            if self.supabase:
                settings = await self.supabase.get_user_settings(message.from_user.id)
                if settings and settings.get("default_currency"):
                    default_currency = settings.get("default_currency")
            
            # Парсим расход
            parsed = parse_manual_expense(message.text, default_currency)
            if not parsed:
                await message.answer(
                    "❌ Не удалось распознать расход. Пожалуйста, укажите сумму.\n\n"
                    "Примеры:\n"
                    "• кофе 1300\n"
                    "• автосервис 10к\n"
                    "• такси 1200 KZT"
                )
                return
            
            # Сохраняем распарсенные данные в состояние
            await state.update_data(
                parsed_expense=parsed,
                expense_text=message.text
            )
            
            # Показываем подтверждение распознавания (товар, цена, валюта)
            currency_symbols = {
                "RUB": "₽",
                "KZT": "₸",
                "USD": "$",
                "EUR": "€",
                "GBP": "£",
                "GEL": "₾",
            }
            currency_symbol = currency_symbols.get(parsed.currency, parsed.currency)
            
            confirmation_text = (
                f"📝 <b>Расход:</b> {parsed.description}\n"
                f"💰 <b>Сумма:</b> {parsed.amount:.2f} {currency_symbol}\n"
                f"📅 <b>Дата:</b> {parsed.occurred_at.strftime('%d.%m.%Y')}\n\n"
                f"Всё верно?"
            )
            
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="✅ Да, верно", callback_data="expense_confirm_parsed"),
                    InlineKeyboardButton(text="❌ Нет, исправить", callback_data="expense_cancel")
                ]
            ])
            
            await message.answer(confirmation_text, reply_markup=keyboard, parse_mode="HTML")
            await state.set_state(ExpenseStates.waiting_for_confirmation)

        # Обработчики состояний для отчетов - должны быть ПЕРЕД общим F.text обработчиком
        @self.router.message(ReportStates.waiting_for_start_date)
        async def handle_report_start_date(message: Message, state: FSMContext) -> None:
            """Обработчик ввода даты начала периода"""
            try:
                # Парсим дату в формате ДД.ММ.ГГГГ
                date_obj = datetime.strptime(message.text.strip(), "%d.%m.%Y")
                start_date = date_obj.strftime("%Y-%m-%d")
                await state.update_data(start_date=start_date)
                await message.answer(
                    "📅 Введите дату окончания периода в формате ДД.ММ.ГГГГ (например, 31.12.2025):"
                )
                await state.set_state(ReportStates.waiting_for_end_date)
            except ValueError:
                await message.answer(
                    "❌ Неверный формат даты. Используйте формат ДД.ММ.ГГГГ (например, 01.12.2025):"
                )
        
        @self.router.message(ReportStates.waiting_for_end_date)
        async def handle_report_end_date(message: Message, state: FSMContext) -> None:
            """Обработчик ввода даты окончания периода"""
            try:
                # Парсим дату в формате ДД.ММ.ГГГГ
                date_obj = datetime.strptime(message.text.strip(), "%d.%m.%Y")
                end_date = date_obj.strftime("%Y-%m-%d")
                data = await state.get_data()
                start_date = data.get("start_date")
                
                if not start_date:
                    await message.answer("❌ Ошибка: не найдена дата начала. Начните заново с /report")
                    await state.clear()
                    return
            
                # Проверяем, что дата окончания не раньше даты начала
                if end_date < start_date:
                    await message.answer("❌ Дата окончания не может быть раньше даты начала. Введите корректную дату:")
                    return
                
                # Используем единый метод для обработки отчета
                await self._process_report_request(
                    message.from_user.id,
                    message,
                    state,
                    start_date=start_date,
                    end_date=end_date
                )
            except ValueError:
                await message.answer(
                    "❌ Неверный формат даты. Используйте формат ДД.ММ.ГГГГ (например, 31.12.2025):"
                )

        @self.router.message(ReportStates.waiting_for_single_date)
        async def handle_report_single_date(message: Message, state: FSMContext) -> None:
            """Обработчик ввода даты для отчета за один день"""
            try:
                # Парсим дату в формате ДД.ММ.ГГГГ
                date_obj = datetime.strptime(message.text.strip(), "%d.%m.%Y")
                single_date = date_obj.strftime("%Y-%m-%d")
                
                # Используем одну и ту же дату как начало и конец периода
                start_date = single_date
                end_date = single_date

                # Используем единый метод для обработки отчета
                await self._process_report_request(
                    message.from_user.id,
                    message,
                    state,
                    start_date=start_date,
                    end_date=end_date
                )
            except ValueError:
                await message.answer(
                    "❌ Неверный формат даты. Используйте формат ДД.ММ.ГГГГ (например, 15.12.2025):"
                )
        
        # Обработчики состояний для экспорта - должны быть ПЕРЕД общим F.text обработчиком
        @self.router.message(ExportStates.waiting_for_start_date)
        async def handle_export_start_date(message: Message, state: FSMContext) -> None:
            """Обработчик ввода даты начала периода для экспорта"""
            try:
                # Парсим дату в формате ДД.ММ.ГГГГ
                date_obj = datetime.strptime(message.text.strip(), "%d.%m.%Y")
                start_date = date_obj.strftime("%Y-%m-%d")
                await state.update_data(start_date=start_date)
                await message.answer(
                    "📅 Введите дату окончания периода в формате ДД.ММ.ГГГГ (например, 31.12.2025):"
                )
                await state.set_state(ExportStates.waiting_for_end_date)
            except ValueError:
                await message.answer(
                    "❌ Неверный формат даты. Используйте формат ДД.ММ.ГГГГ (например, 01.12.2025):"
                )
        
        @self.router.message(ExportStates.waiting_for_end_date)
        async def handle_export_end_date(message: Message, state: FSMContext) -> None:
            """Обработчик ввода даты окончания периода для экспорта"""
            try:
                # Парсим дату в формате ДД.ММ.ГГГГ
                date_obj = datetime.strptime(message.text.strip(), "%d.%m.%Y")
                end_date = date_obj.strftime("%Y-%m-%d")
                data = await state.get_data()
                start_date = data.get("start_date")
                
                if not start_date:
                    await message.answer("❌ Ошибка: не найдена дата начала. Начните заново с /export")
                    await state.clear()
                    return

                # Проверяем, что дата окончания не раньше даты начала
                if end_date < start_date:
                    await message.answer("❌ Дата окончания не может быть раньше даты начала. Введите корректную дату:")
                    return

                # Выполняем экспорт
                await message.answer("📤 Формирую выгрузку, это может занять пару секунд…")
                csv_blob = await self.supabase.export_expenses_csv(
                    message.from_user.id,
                    start_date=start_date,
                    end_date=end_date
                )
                
                filename = f"expensecat_export_{start_date}_{end_date}.csv"
                # Форматируем даты для отображения
                try:
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                    period_text = f"{start_dt.strftime('%d.%m.%Y')} - {end_dt.strftime('%d.%m.%Y')}"
                except:
                    period_text = f"{start_date} - {end_date}"
                
                file = BufferedInputFile(csv_blob.encode("utf-8"), filename=filename)
                await message.answer_document(
                    document=file,
                    caption=f"✅ Готово. Период: {period_text}\n\nИспользуй CSV в Excel/Sheets или импортируй обратно.",
                )
                await state.clear()
            except ValueError:
                await message.answer(
                    "❌ Неверный формат даты. Используйте формат ДД.ММ.ГГГГ (например, 31.12.2025):"
                )
        
        # Обработчики команд должны быть ПЕРЕД общим F.text обработчиком
        @self.router.message(Command("feedback"))
        async def handle_feedback(message: Message, state: FSMContext) -> None:
            """Обработчик команды для обратной связи"""
            await state.set_state(FeedbackStates.waiting_for_feedback_type)
            
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="🐛 Сообщить об ошибке", callback_data="feedback_bug"),
                ],
                [
                    InlineKeyboardButton(text="💡 Предложить функцию", callback_data="feedback_suggestion"),
                    InlineKeyboardButton(text="😞 Пожаловаться", callback_data="feedback_complaint"),
                ],
            ])
            
            await message.answer(
                "📢 Выберите тип обратной связи:",
                reply_markup=keyboard
            )

        @self.router.message(Command("delete_all"))
        async def handle_delete_all(message: Message, state: FSMContext) -> None:
            if not self.supabase:
                await message.answer(
                    "Удаление данных доступно только при подключенной базе (Supabase)."
                )
                return
            
            if not message.from_user:
                await message.answer("Не удалось определить пользователя.")
                return
            
            # Сохраняем user_id в state для подтверждения
            await state.update_data(user_id=message.from_user.id)
            await state.set_state(DeleteStates.waiting_for_confirmation)
            
            # Создаем кнопки для подтверждения
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="⚠️ Да, удалить все данные", callback_data="delete_confirm"),
                ],
                [
                    InlineKeyboardButton(text="❌ Отменить", callback_data="delete_cancel"),
                ]
            ])
            
            await message.answer(
                "⚠️ ВНИМАНИЕ!\n\n"
                "Вы собираетесь удалить ВСЕ ваши данные:\n"
                "• Все чеки\n"
                "• Все расходы\n"
                "• Все банковские транзакции\n\n"
                "Это действие НЕОБРАТИМО!\n\n"
                "Вы уверены?",
                reply_markup=keyboard
            )

        @self.router.message(Command("delete_expense"))
        async def handle_delete_expense(message: Message, state: FSMContext) -> None:
            """Обработчик команды удаления расхода"""
            logging.info(f"Command /delete_expense received from user {message.from_user.id if message.from_user else 'unknown'}")
            try:
                if not self.supabase or not message.from_user:
                    await message.answer("❌ Удаление расходов доступно только при подключенной базе данных.")
                    return
                
                await state.clear()
                
                # Получаем список расходов (последние 3 месяца, максимум 20 записей)
                logging.info(f"Fetching expenses list for user {message.from_user.id}")
                expenses = await self.supabase.fetch_expenses_list(message.from_user.id, limit=20, months_back=3)
                logging.info(f"Found {len(expenses) if expenses else 0} expenses")
                
                if not expenses:
                    await message.answer("📭 У вас нет сохраненных расходов за последние 3 месяца.")
                    return
                
                # Формируем сообщение со списком расходов
                text_lines = [f"🗑️ Выберите расход для удаления:\n(показаны последние 3 месяца, {len(expenses)} записей)\n"]
                keyboard_buttons = []
                
                for expense in expenses:  # Показываем все полученные записи
                    expense_id = expense.get("id")
                    amount = expense.get("amount", 0)
                    currency = expense.get("currency", "")
                    date = expense.get("date", "")
                    
                    # Получаем название продукта из note
                    note = expense.get("note") or expense.get("description") or "Без названия"
                    
                    # Пытаемся получить время из связанного чека (receipts)
                    purchased_at = None
                    receipts_data = expense.get("receipts")
                    if receipts_data and isinstance(receipts_data, list) and len(receipts_data) > 0:
                        purchased_at = receipts_data[0].get("purchased_at")
                    elif receipts_data and isinstance(receipts_data, dict):
                        purchased_at = receipts_data.get("purchased_at")
                    
                    # Форматируем полную дату и время
                    try:
                        if purchased_at:
                            # Используем время из чека
                            if isinstance(purchased_at, str):
                                date_obj = datetime.fromisoformat(purchased_at.replace('Z', '+00:00'))
                            else:
                                date_obj = purchased_at
                            date_str = date_obj.strftime("%d.%m.%Y %H:%M")
                        elif isinstance(date, str):
                            # Если дата в формате YYYY-MM-DD, добавляем время 00:00
                            if len(date) == 10:
                                date_obj = datetime.strptime(date, "%Y-%m-%d")
                            else:
                                date_obj = datetime.fromisoformat(date.replace('Z', '+00:00'))
                            date_str = date_obj.strftime("%d.%m.%Y %H:%M")
                        else:
                            date_obj = date
                            date_str = date_obj.strftime("%d.%m.%Y %H:%M")
                    except Exception as e:
                        logging.warning(f"Ошибка форматирования даты для expense {expense_id}: {e}")
                        date_str = str(date)[:10] + " 00:00"
                    
                    # Преобразуем код валюты в символ
                    currency_symbols = {
                        "RUB": "₽",
                        "KZT": "₸",
                        "USD": "$",
                        "EUR": "€",
                        "GBP": "£",
                        "CNY": "¥",
                        "JPY": "¥",
                    }
                    currency_symbol = currency_symbols.get(currency.upper(), currency.upper()[:3])
                    
                    # Формируем компактный текст кнопки
                    # Формат: название продукта сумма символ_валюты дата_время
                    note_short = note[:20] if len(note) > 20 else note
                    # Убираем категорию и иконку из кнопки для экономии места
                    button_text = f"{note_short} {amount:.0f}{currency_symbol} {date_str}"
                    
                    keyboard_buttons.append([
                        InlineKeyboardButton(
                            text=button_text,
                            callback_data=f"delete_expense_{expense_id}"
                        )
                    ])
                
                keyboard = InlineKeyboardMarkup(inline_keyboard=keyboard_buttons)
                await message.answer(text_lines[0], reply_markup=keyboard)
                logging.info(f"Sent expenses list with {len(keyboard_buttons)} buttons")
            except Exception as exc:
                logging.exception(f"Error in handle_delete_expense: {exc}")
                # Отправляем техническую ошибку в технический чат
                if message.from_user:
                    await self._send_tech_error_to_channel(
                        user_id=message.from_user.id,
                        username=message.from_user.username,
                        first_name=message.from_user.first_name,
                        error=exc,
                        context="Получение списка расходов для удаления"
                    )
                # Пользователю отправляем дружелюбное сообщение
                await message.answer("❌ Ошибка при получении списка расходов. Пожалуйста, попробуйте позже или обратитесь в поддержку через /feedback")

        @self.router.message(Command("settings"))
        async def handle_settings(message: Message, state: FSMContext) -> None:
            """Обработчик команды настроек"""
            await state.clear()
            
            if not self.supabase or not message.from_user:
                await message.answer("❌ Настройки доступны только при подключенной базе данных.")
                return
            
            # Получаем текущие настройки
            settings = await self.supabase.get_user_settings(message.from_user.id)
            current_currency = settings.get("default_currency", "RUB") if settings else "RUB"
            
            currency_symbols = {
                "RUB": "₽",
                "KZT": "₸",
                "USD": "$",
                "EUR": "€",
                "GBP": "£",
                "GEL": "₾",
            }
            current_symbol = currency_symbols.get(current_currency, current_currency)
            
            keyboard = self._create_currency_keyboard()
            await message.answer(
                f"⚙️ Настройки\n\n"
                f"💰 Текущая валюта по умолчанию: {current_symbol} {current_currency}\n\n"
                f"Выберите новую валюту:",
                reply_markup=keyboard
            )

        # Обработчики состояний для feedback - должны быть ПЕРЕД общим F.text обработчиком
        @self.router.message(FeedbackStates.waiting_for_feedback_text)
        async def handle_feedback_text(message: Message, state: FSMContext) -> None:
            """Обработчик ввода текста обратной связи"""
            if not message.text or message.text.startswith("/"):
                await message.answer(
                    "❌ Пожалуйста, отправьте текстовое сообщение. "
                    "Используйте /cancel для отмены."
                )
                return
            
            feedback_text = message.text.strip()
            data = await state.get_data()
            feedback_type = data.get("feedback_type", "unknown")
            
            user_id = message.from_user.id if message.from_user else None
            username = message.from_user.username if message.from_user else "unknown"
            first_name = message.from_user.first_name if message.from_user else "unknown"
            
            type_names = {
                "bug": "ошибка",
                "suggestion": "предложение",
                "complaint": "жалоба"
            }
            type_emojis = {
                "bug": "🐛",
                "suggestion": "💡",
                "complaint": "😞"
            }
            
            type_name = type_names.get(feedback_type, feedback_type)
            emoji = type_emojis.get(feedback_type, "📝")
            
            # Логируем обратную связь
            logging.info(
                f"Feedback [{feedback_type}] from user_id={user_id} "
                f"(@{username}, {first_name}): {feedback_text}"
            )
            
            # Сохраняем в базу данных, если Supabase подключен
            if self.supabase:
                try:
                    await self.supabase.save_feedback(
                        user_id=user_id,
                        username=username,
                        first_name=first_name,
                        feedback_type=feedback_type,
                        feedback_text=feedback_text
                    )
                except Exception as exc:
                    logging.exception(f"Ошибка при сохранении отзыва в базу данных: {exc}")
            
            # Отправляем в канал обратной связи, если настроен
            if self.feedback_chat_id:
                logging.info(f"📤 Sending feedback to channel: {self.feedback_chat_id}")
                try:
                    await self._send_feedback_to_channel(
                        user_id=user_id,
                        username=username,
                        first_name=first_name,
                        feedback_type=feedback_type,
                        type_name=type_name,
                        emoji=emoji,
                        feedback_text=feedback_text
                    )
                    logging.info(f"✅ Feedback successfully sent to channel")
                except Exception as exc:
                    logging.exception(f"❌ Ошибка при отправке отзыва в канал: {exc}")
            else:
                logging.warning("⚠️ Feedback channel not configured (FEEDBACK_CHAT_ID not set), skipping channel send")
            
            await message.answer(
                f"✅ Спасибо за вашу обратную связь!\n\n"
                f"{emoji} Тип: {type_name}\n"
                f"📝 Ваше сообщение:\n{feedback_text}\n\n"
                "Мы обязательно рассмотрим ваше сообщение и учтём его при улучшении бота."
            )
            
            await state.clear()

        @self.router.message(F.text)
        async def handle_text_message(message: Message, state: FSMContext) -> None:
            """Автоматическая обработка текстовых сообщений как ручного ввода расхода"""
            # ВАЖНО: Команды должны обрабатываться отдельными обработчиками Command
            # Проверяем команды в самом начале, чтобы не перехватывать их
            if message.text and message.text.startswith("/"):
                logging.info(f"📝 [TEXT_HANDLER] Skipping command: {message.text}")
                return  # Команды обрабатываются отдельными обработчиками Command, не трогаем их
            
            logging.debug(f"📝 [TEXT_HANDLER] Processing text message: {message.text[:50] if message.text else 'None'}")
            
            # Проверяем, не находимся ли мы в специфическом состоянии (feedback, export, statement и т.д.)
            current_state = await state.get_state()
            
            # Если находимся в состоянии ожидания текста расхода, используем существующий обработчик
            if current_state == ExpenseStates.waiting_for_expense_text:
                return  # Пусть существующий обработчик обработает
            
            # Если находимся в других специфических состояниях, не обрабатываем как расход
            # НО: если пользователь в состоянии подтверждения или выбора категории и вводит новый текст,
            # это должен быть новый расход (пользователь хочет начать заново)
            if current_state:
                state_name = str(current_state)
                # Проверяем все возможные состояния, которые должны обрабатываться отдельно
                # Исключаем ExpenseStates.waiting_for_confirmation и waiting_for_category - 
                # если пользователь вводит новый текст в этих состояниях, это новый расход
                if any(s in state_name for s in [
                    "FeedbackStates", "ExportStates", "StatementStates", 
                    "ReceiptStates", "SetupStates", "DeleteStates", 
                    "DeleteExpenseStates", "ReportStates"
                ]):
                    return  # Не обрабатываем, пусть специфический обработчик обработает
                
                # Если пользователь в состоянии подтверждения или выбора категории, 
                # и вводит новый текст - это новый расход, очищаем состояние
                if current_state in [ExpenseStates.waiting_for_confirmation, ExpenseStates.waiting_for_category]:
                    await state.clear()
                    # Продолжаем обработку как новый расход
            
            # Если это обычный текст и мы не в специфическом состоянии, обрабатываем как ручной ввод расхода
            if not message.text or not message.from_user:
                return
            
            # Получаем валюту по умолчанию пользователя
            default_currency = "RUB"
            if self.supabase:
                settings = await self.supabase.get_user_settings(message.from_user.id)
                if settings and settings.get("default_currency"):
                    default_currency = settings.get("default_currency")
            
            # Парсим расход
            logging.info(f"📝 [TEXT_HANDLER] Parsing expense: text='{message.text}', default_currency={default_currency}")
            parsed = parse_manual_expense(message.text, default_currency)
            if not parsed:
                # Если не удалось распознать, не отвечаем (может быть это просто сообщение)
                logging.info(f"📝 [TEXT_HANDLER] Failed to parse expense from text: '{message.text}'")
                return
            
            logging.info(f"📝 [TEXT_HANDLER] Successfully parsed expense: description='{parsed.description}', amount={parsed.amount}, currency={parsed.currency}")
            
            # Сохраняем распарсенные данные в состояние
            await state.update_data(
                parsed_expense=parsed,
                expense_text=message.text
            )
            
            # Показываем подтверждение распознавания (товар, цена, валюта)
            currency_symbols = {
                "RUB": "₽",
                "KZT": "₸",
                "USD": "$",
                "EUR": "€",
                "GBP": "£",
                "GEL": "₾",
            }
            currency_symbol = currency_symbols.get(parsed.currency, parsed.currency)
            
            confirmation_text = (
                f"📝 <b>Расход:</b> {parsed.description}\n"
                f"💰 <b>Сумма:</b> {parsed.amount:.2f} {currency_symbol}\n"
                f"📅 <b>Дата:</b> {parsed.occurred_at.strftime('%d.%m.%Y')}\n\n"
                f"Всё верно?"
            )
            
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="✅ Да, верно", callback_data="expense_confirm_parsed"),
                    InlineKeyboardButton(text="❌ Нет, исправить", callback_data="expense_cancel")
                ]
            ])
            
            await message.answer(confirmation_text, reply_markup=keyboard, parse_mode="HTML")
            await state.set_state(ExpenseStates.waiting_for_confirmation)

        @self.router.callback_query(F.data.startswith("feedback_"))
        async def handle_feedback_type(callback: CallbackQuery, state: FSMContext) -> None:
            """Обработчик выбора типа обратной связи"""
            await callback.answer()
            
            feedback_type = callback.data.replace("feedback_", "")
            type_names = {
                "bug": "ошибка",
                "suggestion": "предложение",
                "complaint": "жалоба"
            }
            type_emojis = {
                "bug": "🐛",
                "suggestion": "💡",
                "complaint": "😞"
            }
            type_instructions = {
                "bug": "Опишите ошибку подробно. Что вы делали, когда это произошло? Что должно было произойти и что произошло на самом деле?",
                "suggestion": "Опишите вашу идею подробно. Какую функцию вы хотели бы видеть в боте?",
                "complaint": "Опишите проблему, с которой вы столкнулись. Что вас не устраивает?"
            }
            
            type_name = type_names.get(feedback_type, feedback_type)
            emoji = type_emojis.get(feedback_type, "📝")
            instruction = type_instructions.get(feedback_type, "Опишите проблему подробно.")
            
            await state.update_data(feedback_type=feedback_type)
            await state.set_state(FeedbackStates.waiting_for_feedback_text)
            
            await callback.message.answer(
                f"{emoji} Вы выбрали: {type_name}\n\n{instruction}"
            )

        @self.router.message(Command("import"))
        async def handle_import(message: Message, state: FSMContext) -> None:
            await state.clear()
            await message.answer(
                "Для импорта пришли CSV/XLSX/PDF выписку или отчёт из другого сервиса. "
                "Мы автоматически распознаем формат и подскажем, что делать дальше."
            )

        # Обработчик выбора валюты должен быть ПЕРЕД общим обработчиком report_,
        # чтобы перехватывать callback'и report_currency_* первым
        @self.router.callback_query(F.data.startswith("report_currency_"))
        async def handle_report_currency(callback: CallbackQuery, state: FSMContext) -> None:
            """Обработчик выбора валюты для отчета"""
            try:
                await callback.answer()
                logging.info(f"📊 [REPORT_CURRENCY] Callback received: {callback.data} from user {callback.from_user.id if callback.from_user else 'unknown'}")
                
                if not callback.from_user:
                    await callback.message.answer("❌ Ошибка: не удалось определить пользователя.")
                    return
                
                # Получаем сохраненный отчет из состояния
                state_data = await state.get_data()
                report = state_data.get("report_data")
                
                if not report:
                    await callback.message.answer("❌ Ошибка: данные отчета не найдены. Попробуйте выбрать период заново.")
                    await state.clear()
                    return
                
                selected_currency = callback.data.replace("report_currency_", "")
                
                # Форматируем отчет с учетом выбранной валюты
                if selected_currency == "all":
                    # Общий отчет - показываем все валюты
                    report_text = format_report(report)
                else:
                    # Отчет для конкретной валюты - фильтруем данные
                    filtered_report = {
                        "period": report.get("period", ""),
                        "currencies_data": {
                            selected_currency: report.get("currencies_data", {}).get(selected_currency, {})
                        },
                        "most_expensive_by_currency": {
                            selected_currency: report.get("most_expensive_by_currency", {}).get(selected_currency, {})
                        }
                    }
                    report_text = format_report(filtered_report)
                
                if not report_text or not report_text.strip():
                    await callback.message.answer("📊 Нет данных за выбранный период.")
                    await state.clear()
                    return
                
                # Обрезаем если слишком длинный
                truncated_report = truncate_message_for_telegram(report_text)
                await callback.message.answer(truncated_report)
                await state.clear()
                logging.info(f"✅ Report sent successfully to user {callback.from_user.id} for currency {selected_currency}")
            except Exception as exc:
                logging.exception(f"Error in handle_report_currency: {exc}")
                try:
                    # Отправляем техническую ошибку в технический чат
                    if callback.from_user:
                        await self._send_tech_error_to_channel(
                            user_id=callback.from_user.id,
                            username=callback.from_user.username,
                            first_name=callback.from_user.first_name,
                            error=exc,
                            context="Получение отчета по валюте",
                            additional_info=f"selected_currency={selected_currency if 'selected_currency' in locals() else 'N/A'}"
                        )
                    # Пользователю отправляем дружелюбное сообщение
                    await callback.message.answer("❌ Ошибка при получении отчета. Пожалуйста, попробуйте позже или обратитесь в поддержку через /feedback")
                except Exception as send_exc:
                    logging.error(f"Failed to send error message to user: {send_exc}")
        
        @self.router.callback_query(F.data.startswith("report_"))
        async def handle_report_period(callback: CallbackQuery, state: FSMContext) -> None:
            try:
                logging.info(f"📊 [REPORT_CALLBACK] Callback received: {callback.data} from user {callback.from_user.id if callback.from_user else 'unknown'}")
                logging.info(f"📊 [REPORT_CALLBACK] Callback object: {callback}")
                await callback.answer()
                logging.info(f"📊 [REPORT_CALLBACK] Callback answered")
                
                if not self.supabase:
                    logging.warning("📊 [REPORT_CALLBACK] Supabase not available for report")
                    await callback.message.answer("Отчёты по расходам появятся после подключения базы (Supabase).")
                    return
                
                if not callback.from_user:
                    logging.warning("📊 [REPORT_CALLBACK] No user in callback for report")
                    await callback.message.answer("❌ Ошибка: не удалось определить пользователя.")
                    return
                
                now = datetime.utcnow()
                period = None
                start_date = None
                end_date = None
                
                if callback.data == "report_current_month":
                    period = now.strftime("%Y-%m")
                    logging.info(f"Report period: current month = {period}")
                elif callback.data == "report_last_month":
                    # Прошлый месяц
                    last_month = (now.replace(day=1) - timedelta(days=1))
                    period = last_month.strftime("%Y-%m")
                    logging.info(f"Report period: last month = {period}")
                elif callback.data == "report_current_week":
                    # Текущая неделя (понедельник - воскресенье)
                    days_since_monday = now.weekday()
                    start_date = (now - timedelta(days=days_since_monday)).strftime("%Y-%m-%d")
                    # Вычисляем воскресенье текущей недели (6 - weekday() дней до воскресенья)
                    days_until_sunday = 6 - now.weekday()
                    end_date = (now + timedelta(days=days_until_sunday)).strftime("%Y-%m-%d")
                    logging.info(f"Report period: current week = {start_date} to {end_date}")
                elif callback.data == "report_last_week":
                    # Прошлая неделя
                    days_since_monday = now.weekday()
                    week_start = now - timedelta(days=days_since_monday + 7)
                    week_end = now - timedelta(days=days_since_monday + 1)
                    start_date = week_start.strftime("%Y-%m-%d")
                    end_date = week_end.strftime("%Y-%m-%d")
                    logging.info(f"Report period: last week = {start_date} to {end_date}")
                elif callback.data == "report_current_year":
                    # Текущий год
                    start_date = now.replace(month=1, day=1).strftime("%Y-%m-%d")
                    end_date = now.strftime("%Y-%m-%d")
                    logging.info(f"Report period: current year = {start_date} to {end_date}")
                elif callback.data == "report_custom":
                    # Произвольный период - запрашиваем даты
                    logging.info("Report period: custom (requesting dates)")
                    await callback.message.answer(
                        "📅 Введите дату начала периода в формате ДД.ММ.ГГГГ (например, 01.12.2025):"
                    )
                    await state.set_state(ReportStates.waiting_for_start_date)
                    return
                elif callback.data == "report_single_day":
                    # Отчет за один день - запрашиваем одну дату
                    logging.info("Report period: single day (requesting date)")
                    await callback.message.answer(
                        "📅 Введите дату в формате ДД.ММ.ГГГГ (например, 15.12.2025):"
                    )
                    await state.set_state(ReportStates.waiting_for_single_date)
                    return
                else:
                    # Проверяем, может быть это callback для выбора валюты
                    if callback.data.startswith("report_currency_"):
                        logging.info(f"📊 [REPORT_CALLBACK] Currency selection callback detected: {callback.data}, but handler may not be registered correctly")
                        # Пробуем обработать через обработчик валют
                        await callback.answer("Обрабатываю выбор валюты...")
                        # Передаем управление обработчику валют
                        return
                    logging.warning(f"Unknown report period: {callback.data}")
                    logging.warning(f"📊 [REPORT_CALLBACK] Full callback data: {callback.data}, callback type: {type(callback.data)}")
                    await callback.message.answer("❌ Неизвестный период для отчета.")
                    return
                
                # Используем единый метод для обработки отчета
                await self._process_report_request(
                    callback.from_user.id,
                    callback,
                    state,
                    start_date=start_date,
                    end_date=end_date,
                    period=period
                )
            except Exception as exc:
                logging.exception(f"Error in handle_report_period: {exc}")
                try:
                    # Отправляем техническую ошибку в технический чат
                    if callback.from_user:
                        await self._send_tech_error_to_channel(
                            user_id=callback.from_user.id,
                            username=callback.from_user.username,
                            first_name=callback.from_user.first_name,
                            error=exc,
                            context="Получение отчета по периоду",
                            additional_info=f"period={period if 'period' in locals() else 'N/A'}, start_date={start_date if 'start_date' in locals() else 'N/A'}, end_date={end_date if 'end_date' in locals() else 'N/A'}"
                        )
                    # Пользователю отправляем дружелюбное сообщение
                    await callback.message.answer("❌ Ошибка при получении отчета. Пожалуйста, попробуйте позже или обратитесь в поддержку через /feedback")
                except Exception as send_exc:
                    logging.error(f"Failed to send error message to user: {send_exc}")
                await state.clear()
        
        @self.router.callback_query(F.data.startswith("export_"))
        async def handle_export_period(callback: CallbackQuery, state: FSMContext) -> None:
            await callback.answer()
            
            if not self.supabase:
                await callback.message.answer("Экспорт доступен после подключения Supabase.")
                return
            
            now = datetime.utcnow()
            period = None
            start_date = None
            end_date = None
            
            if callback.data == "export_current_month":
                period = now.strftime("%Y-%m")
            elif callback.data == "export_last_month":
                # Прошлый месяц
                last_month = (now.replace(day=1) - timedelta(days=1))
                period = last_month.strftime("%Y-%m")
            elif callback.data == "export_current_week":
                # Текущая неделя (понедельник - воскресенье)
                days_since_monday = now.weekday()
                start_date = (now - timedelta(days=days_since_monday)).strftime("%Y-%m-%d")
                # Вычисляем воскресенье текущей недели (6 - weekday() дней до воскресенья)
                days_until_sunday = 6 - now.weekday()
                end_date = (now + timedelta(days=days_until_sunday)).strftime("%Y-%m-%d")
            elif callback.data == "export_last_week":
                # Прошлая неделя
                days_since_monday = now.weekday()
                week_start = now - timedelta(days=days_since_monday + 7)
                week_end = now - timedelta(days=days_since_monday + 1)
                start_date = week_start.strftime("%Y-%m-%d")
                end_date = week_end.strftime("%Y-%m-%d")
            elif callback.data == "export_current_year":
                # Текущий год
                start_date = now.replace(month=1, day=1).strftime("%Y-%m-%d")
                end_date = now.strftime("%Y-%m-%d")
            elif callback.data == "export_custom":
                # Произвольный период - запрашиваем даты
                await callback.message.answer(
                    "📅 Введите дату начала периода в формате ДД.ММ.ГГГГ (например, 01.12.2025):"
                )
                await state.set_state(ExportStates.waiting_for_start_date)
                return
            elif callback.data == "export_all":
                period = None
                start_date = None
                end_date = None
            
            # Выполняем экспорт
            await callback.message.answer("📤 Формирую выгрузку, это может занять пару секунд…")
            csv_blob = await self.supabase.export_expenses_csv(
                callback.from_user.id, 
                period=period,
                start_date=start_date,
                end_date=end_date
            )
            
            # Формируем имя файла и описание периода
            if period:
                filename = f"expensecat_export_{period}.csv"
                # Форматируем период для отображения
                try:
                    date_obj = datetime.strptime(period, "%Y-%m")
                    months = ["январь", "февраль", "март", "апрель", "май", "июнь",
                             "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]
                    month_name = months[date_obj.month - 1]
                    period_text = f"{month_name} {date_obj.year}"
                except:
                    period_text = period
            elif start_date and end_date:
                filename = f"expensecat_export_{start_date}_{end_date}.csv"
                # Форматируем даты для отображения
                try:
                    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
                    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
                    period_text = f"{start_dt.strftime('%d.%m.%Y')} - {end_dt.strftime('%d.%m.%Y')}"
                except:
                    period_text = f"{start_date} - {end_date}"
            else:
                filename = "expensecat_export_all.csv"
                period_text = "все данные"
            
            file = BufferedInputFile(csv_blob.encode("utf-8"), filename=filename)
            await callback.message.answer_document(
                document=file,
                caption=f"✅ Готово. Период: {period_text}\n\nИспользуй CSV в Excel/Sheets или импортируй обратно.",
            )
            await state.clear()
        
        @self.router.callback_query(F.data == "expense_confirm_parsed", ExpenseStates.waiting_for_confirmation)
        async def handle_expense_confirm_parsed(callback: CallbackQuery, state: FSMContext) -> None:
            """Обработчик подтверждения распознавания расхода"""
            await callback.answer()
            
            # Получаем распарсенные данные из состояния
            data = await state.get_data()
            parsed = data.get("parsed_expense")
                
            if not parsed:
                await callback.message.answer("❌ Ошибка: данные расхода не найдены. Начните заново с /expense")
                await state.clear()
                return

            # Показываем клавиатуру с категориями
            currency_symbols = {
                "RUB": "₽",
                "KZT": "₸",
                "USD": "$",
                "EUR": "€",
                "GBP": "£",
                "GEL": "₾",
            }
            currency_symbol = currency_symbols.get(parsed.currency, parsed.currency)
            
            summary_text = (
                f"📝 Расход: {parsed.description}\n"
                f"💰 Сумма: {parsed.amount:.2f} {currency_symbol}\n"
                f"📅 Дата: {parsed.occurred_at.strftime('%d.%m.%Y')}\n\n"
                f"Выберите категорию расхода:"
            )
            
            keyboard = self._create_category_keyboard()
            await callback.message.edit_text(summary_text, reply_markup=keyboard)
            await state.set_state(ExpenseStates.waiting_for_category)
        
        @self.router.callback_query(F.data.startswith("expense_category_"), ExpenseStates.waiting_for_category)
        async def handle_expense_category(callback: CallbackQuery, state: FSMContext) -> None:
            """Обработчик выбора категории расхода"""
            await callback.answer()
            
            if not callback.from_user or not self.supabase:
                await callback.message.answer("❌ Ошибка: база данных недоступна.")
                await state.clear()
                return
            
            # Извлекаем категорию из callback_data
            category = callback.data.replace("expense_category_", "")
            
            # Получаем распарсенные данные из состояния
            data = await state.get_data()
            parsed = data.get("parsed_expense")
            
            if not parsed:
                await callback.message.answer("❌ Ошибка: данные расхода не найдены. Начните заново с /expense")
                await state.clear()
                return
            
            # Устанавливаем категорию
            parsed.category = category
            
            # Сохраняем расход в базу сразу после выбора категории
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
                
                await callback.message.edit_text(
                    f"✅ <b>Расход добавлен!</b>\n\n"
                    f"📝 {parsed.description}\n"
                    f"💰 {parsed.amount:.2f} {currency_symbol}\n"
                    f"📂 {category}\n"
                    f"📅 {parsed.occurred_at.strftime('%d.%m.%Y')}",
                    parse_mode="HTML"
                )
                logging.info(f"Manual expense saved: user={callback.from_user.id}, amount={parsed.amount}, category={category}")
            except Exception as exc:
                logging.exception(f"Error saving manual expense: {exc}")
                # Отправляем техническую ошибку в технический чат
                if callback.from_user:
                    await self._send_tech_error_to_channel(
                        user_id=callback.from_user.id,
                        username=callback.from_user.username,
                        first_name=callback.from_user.first_name,
                        error=exc,
                        context="Сохранение мануального расхода",
                        additional_info=f"parsed: {json.dumps(parsed.__dict__ if hasattr(parsed, '__dict__') else str(parsed), ensure_ascii=False, default=str)[:500]}"
                    )
                # Пользователю отправляем дружелюбное сообщение
                await callback.message.edit_text("❌ Ошибка при сохранении расхода. Пожалуйста, попробуйте позже или обратитесь в поддержку через /feedback")
            
            await state.clear()
        
        @self.router.callback_query(F.data == "expense_cancel")
        async def handle_expense_cancel(callback: CallbackQuery, state: FSMContext) -> None:
            """Обработчик отмены добавления расхода"""
            await callback.answer("Отменено")
            current_state = await state.get_state()
            if current_state == ExpenseStates.waiting_for_confirmation:
                # Если отменяем на этапе подтверждения распознавания, предлагаем ввести заново
                await callback.message.edit_text(
                    "❌ Распознавание отменено.\n\n"
                    "Введите расход заново в формате:\n"
                    "• кофе 1300\n"
                    "• автосервис 10к\n"
                    "• такси 1200 KZT"
                )
                await state.set_state(ExpenseStates.waiting_for_expense_text)
            else:
                # Если отменяем на этапе выбора категории, просто отменяем
                await callback.message.edit_text("❌ Добавление расхода отменено.")
                await state.clear()


        @self.router.callback_query(F.data == "receipt_confirm")
        async def handle_receipt_confirm(callback: CallbackQuery, state: FSMContext) -> None:
            """Обработчик подтверждения чека"""
            try:
                await callback.answer()
                logging.info(f"✅ Получено подтверждение чека от user_id={callback.from_user.id if callback.from_user else 'unknown'}")
                data = await state.get_data()
                parsed_receipt = data.get("parsed_receipt")
                receipt_payload = data.get("receipt_payload")
                
                logging.info(f"Данные из FSM state: parsed_receipt={bool(parsed_receipt)}, receipt_payload={bool(receipt_payload)}, keys={list(data.keys())}")
                
                if not receipt_payload or not callback.from_user:
                    logging.warning(f"❌ Ошибка: данные чека не найдены в FSM state для user_id={callback.from_user.id if callback.from_user else 'unknown'}")
                    await callback.message.answer("Ошибка: данные чека не найдены.")
                    await state.clear()
                    return
                
                if not callback.from_user:
                    logging.warning(f"❌ Ошибка: не удалось определить пользователя")
                    await callback.message.answer("Ошибка: не удалось определить пользователя.")
                    await state.clear()
                    return
                
                logging.info(f"✅ Все проверки пройдены, переходим к сохранению")
                # Лимит уже был проверен при отправке фото, счетчик уже был увеличен после распознавания
                # Не проверяем лимит повторно при подтверждении, так как это одна операция
                
                if not self.supabase:
                    logging.error(f"❌ self.supabase is None, невозможно сохранить чек")
                    await callback.message.answer("⚠️ Ошибка: база данных недоступна")
                    await state.clear()
                    return
                
                logging.info(f"🔍 Начинаем проверку валидации суммы перед сохранением")
                # Проверяем несоответствие суммы перед сохранением
                if parsed_receipt:
                    logging.info(f"parsed_receipt найден, проверяем сумму")
                    items_sum = sum(item.price for item in parsed_receipt.items)
                    total = parsed_receipt.total or 0.0
                    difference = abs(items_sum - total)
                    tolerance = max(total * 0.01, 1.0)
                    
                    if difference > tolerance:
                        # Несоответствие суммы - не сохраняем, отправляем фото в rejected receipts
                        file_bytes = data.get("receipt_photo_bytes")
                        mime_type = data.get("receipt_photo_mime_type", "image/jpeg")
                        
                        # Формируем таблицу результатов калькуляции
                        receipt_table = None
                        if parsed_receipt:
                            receipt_table = format_receipt_table(parsed_receipt)
                        
                        if file_bytes and self.supabase:
                            try:
                                file_path = await self.supabase.save_rejected_receipt_photo(
                                    user_id=callback.from_user.id,
                                    file_bytes=file_bytes,
                                    reason="validation_error",
                                    mime_type=mime_type
                                )
                                # Отправляем информацию о фейле в канал
                                if file_path and self.failed_receipts_chat_id:
                                    await self._send_failed_receipt_to_channel(
                                        user_id=callback.from_user.id,
                                        username=callback.from_user.username,
                                        first_name=callback.from_user.first_name,
                                        reason="validation_error",
                                        file_path=file_path,
                                        file_bytes=file_bytes if file_bytes and isinstance(file_bytes, bytes) and len(file_bytes) > 0 else None,  # Отправляем фото напрямую
                                        mime_type=mime_type,
                                        error_message=f"Несоответствие суммы: товаров={len(parsed_receipt.items)}, позиции={items_sum:.2f}, итого={total:.2f}, разница={difference:.2f}",
                                        receipt_table=receipt_table  # Добавляем результаты калькуляции
                                    )
                            except Exception as exc:
                                logging.warning(f"Failed to save rejected receipt photo: {exc}")
                        
                        items_count = len(parsed_receipt.items)
                        await callback.message.answer(
                            f"❌ Чек не сохранен из-за несоответствия суммы:\n"
                            f"Товаров в чеке: {items_count}\n"
                            f"Сумма всех позиций: {items_sum:.2f} {parsed_receipt.currency}\n"
                            f"Итого по чеку: {total:.2f} {parsed_receipt.currency}\n"
                            f"Разница: {difference:.2f} {parsed_receipt.currency}\n\n"
                            f"Пожалуйста, отправьте фото чека заново."
                        )
                        await state.clear()
                        return
                
                # Сохраняем чек в базу
                logging.info(f"💾 Начинаем сохранение чека в базу данных (user_id={callback.from_user.id}, store={receipt_payload.get('store')}, total={receipt_payload.get('total')})")
                if not self.supabase:
                    logging.error(f"❌ self.supabase is None, невозможно сохранить чек")
                    await callback.message.answer("⚠️ Ошибка: база данных недоступна")
                    await state.clear()
                    return
                save_start = time.perf_counter()
                stored_receipt, is_duplicate = await self.supabase.upsert_receipt(receipt_payload)
                save_time = time.perf_counter() - save_start
                logging.info(f"⏱️ [PERF] Сохранение чека (handle_receipt_confirm): {save_time*1000:.1f}ms")
                logging.info(f"Результат сохранения: stored_receipt={bool(stored_receipt)}, is_duplicate={is_duplicate}, receipt_id={stored_receipt.get('id') if stored_receipt else None}")
                
                # Проверяем, что получили реальную запись с id
                if not stored_receipt or not stored_receipt.get("id"):
                    # Отправляем техническую ошибку в технический чат
                    if callback.from_user:
                        await self._send_tech_error_to_channel(
                            user_id=callback.from_user.id,
                            username=callback.from_user.username,
                            first_name=callback.from_user.first_name,
                            error=Exception("stored_receipt is None or missing id"),
                            context="Сохранение чека - отсутствует id",
                            additional_info=f"stored_receipt: {json.dumps(stored_receipt, ensure_ascii=False, default=str)[:500] if stored_receipt else 'None'}"
                        )
                    # Пользователю отправляем дружелюбное сообщение
                    await callback.message.answer("⚠️ Ошибка: не удалось сохранить чек в базу данных. Пожалуйста, попробуйте позже или обратитесь в поддержку через /feedback")
                    await state.clear()
                    return
                
                if is_duplicate:
                    await callback.message.answer("⚠️ Этот чек уже был сохранен ранее (дубликат)")
                else:
                    # Создаем expense записи для каждой позиции из receipt только если это новый чек
                    expense_start = time.perf_counter()
                    expense_payloads = build_expenses_from_receipt_items(stored_receipt)
                    
                    # Сохраняем все expenses
                    saved_count = 0
                    duplicate_count = 0
                    for expense_payload in expense_payloads:
                        expense_result = await self.supabase.record_expense(expense_payload, check_duplicates=False)
                    if expense_result.get("duplicate"):
                            duplicate_count += 1
                    else:
                            saved_count += 1
                    
                    expense_time = time.perf_counter() - expense_start
                    logging.info(f"⏱️ [PERF] Создание {len(expense_payloads)} expense записей: {expense_time*1000:.1f}ms (сохранено: {saved_count}, дубликатов: {duplicate_count})")
                    
                    if saved_count > 0:
                        await callback.message.answer(f"✅ Чек сохранен в базу данных\n📝 Создано {saved_count} записей расходов")
                    else:
                        await callback.message.answer("✅ Чек сохранен в базу данных\n⚠️ Расходы не созданы: все записи являются дубликатами")
                await state.clear()
            except Exception as exc:
                logging.exception(f"Ошибка при сохранении чека: {exc}")
                # Отправляем техническую ошибку в технический чат
                if callback.from_user:
                    await self._send_tech_error_to_channel(
                        user_id=callback.from_user.id,
                        username=callback.from_user.username,
                        first_name=callback.from_user.first_name,
                        error=exc,
                        context="Сохранение чека в базу данных",
                        additional_info=f"receipt_payload: {json.dumps(receipt_payload, ensure_ascii=False, default=str)[:500] if 'receipt_payload' in locals() else 'N/A'}"
                    )
                # Пользователю отправляем дружелюбное сообщение
                await callback.message.answer("⚠️ Не удалось сохранить в базу данных. Пожалуйста, попробуйте позже или обратитесь в поддержку через /feedback")
                await state.clear()

        @self.router.callback_query(F.data == "receipt_reject")
        async def handle_receipt_reject(callback: CallbackQuery, state: FSMContext) -> None:
            """Обработчик отклонения чека"""
            await callback.answer()
            
            # Сохраняем фото в Storage для дальнейшего изучения
            if self.supabase and callback.from_user:
                data = await state.get_data()
                file_bytes = data.get("receipt_photo_bytes")
                mime_type = data.get("receipt_photo_mime_type", "image/jpeg")
                parsed_receipt = data.get("parsed_receipt")
                
                # Формируем таблицу результатов калькуляции
                receipt_table = None
                if parsed_receipt:
                    receipt_table = format_receipt_table(parsed_receipt)
                
                if file_bytes:
                    try:
                        file_path = await self.supabase.save_rejected_receipt_photo(
                            user_id=callback.from_user.id,
                            file_bytes=file_bytes,
                            reason="rejected_by_user",
                            mime_type=mime_type
                        )
                        if file_path:
                            logging.info(f"Saved rejected receipt photo: {file_path}")
                            # Отправляем информацию о фейле в канал
                            if self.failed_receipts_chat_id:
                                await self._send_failed_receipt_to_channel(
                                    user_id=callback.from_user.id,
                                    username=callback.from_user.username,
                                    first_name=callback.from_user.first_name,
                                    reason="rejected_by_user",
                                    file_path=file_path,
                                    file_bytes=file_bytes if file_bytes and isinstance(file_bytes, bytes) and len(file_bytes) > 0 else None,  # Отправляем фото напрямую
                                    mime_type=mime_type,
                                    receipt_table=receipt_table  # Добавляем результаты калькуляции
                                )
                    except Exception as exc:
                        logging.warning(f"Failed to save rejected receipt photo: {exc}")
            
            await callback.message.answer("Понял, отправьте фото чека заново.")
            await state.clear()

        @self.router.callback_query(F.data == "delete_confirm")
        async def handle_delete_confirm(callback: CallbackQuery, state: FSMContext) -> None:
            """Обработчик подтверждения удаления всех данных"""
            await callback.answer()
            
            if not self.supabase or not callback.from_user:
                await callback.message.answer("Ошибка: база данных не подключена или не удалось определить пользователя.")
                await state.clear()
                return
            
            # Проверяем, что user_id из state совпадает с user_id из callback
            data = await state.get_data()
            stored_user_id = data.get("user_id")
            
            if stored_user_id != callback.from_user.id:
                await callback.message.answer("Ошибка: несоответствие пользователя.")
                await state.clear()
                return
            
            try:
                # Удаляем все данные пользователя
                await callback.message.answer("Удаляю все данные...")
                result = await self.supabase.delete_all_user_data(callback.from_user.id)
                
                total_deleted = sum(result.values())
                message_text = (
                    f"✅ Все данные удалены!\n\n"
                    f"Удалено записей:\n"
                    f"• Чеков: {result.get('receipts', 0)}\n"
                    f"• Расходов: {result.get('expenses', 0)}\n"
                    f"• Банковских транзакций: {result.get('bank_transactions', 0)}\n\n"
                    f"Всего: {total_deleted}"
                )
                await callback.message.answer(message_text)
            except Exception as exc:
                logging.exception(f"Ошибка при удалении данных: {exc}")
                # Отправляем техническую ошибку в технический чат
                if callback.from_user:
                    await self._send_tech_error_to_channel(
                        user_id=callback.from_user.id,
                        username=callback.from_user.username,
                        first_name=callback.from_user.first_name,
                        error=exc,
                        context="Удаление всех данных пользователя"
                    )
                # Пользователю отправляем дружелюбное сообщение
                await callback.message.answer("⚠️ Ошибка при удалении данных. Пожалуйста, попробуйте позже или обратитесь в поддержку через /feedback")
            finally:
                await state.clear()

        @self.router.callback_query(F.data == "delete_cancel")
        async def handle_delete_cancel(callback: CallbackQuery, state: FSMContext) -> None:
            """Обработчик отмены удаления данных"""
            await callback.answer()
            await callback.message.answer("Удаление отменено. Ваши данные сохранены.")
            await state.clear()

        @self.router.callback_query(F.data.startswith("delete_expense_"))
        async def handle_delete_expense_confirm(callback: CallbackQuery, state: FSMContext) -> None:
            """Обработчик подтверждения удаления расхода"""
            await callback.answer()
            
            if not self.supabase or not callback.from_user:
                await callback.message.answer("❌ Ошибка: база данных не подключена.")
                return
            
            # Извлекаем ID расхода из callback_data
            expense_id_str = callback.data.replace("delete_expense_", "")
            try:
                expense_id = int(expense_id_str)
            except ValueError:
                await callback.message.answer("❌ Ошибка: неверный ID расхода.")
                return
            
            # Получаем информацию о расходе для подтверждения (без ограничения по периоду для поиска)
            expenses = await self.supabase.fetch_expenses_list(callback.from_user.id, limit=1000, months_back=0)
            expense = next((e for e in expenses if e.get("id") == expense_id), None)
            
            if not expense:
                await callback.message.answer("❌ Расход не найден.")
                await state.clear()
                return
            
            note = expense.get("note") or expense.get("description") or "Без названия"
            amount = expense.get("amount", 0)
            currency = expense.get("currency", "")
            date = expense.get("date", "")
            source = expense.get("source", "")
            category = expense.get("category", "")
            
            # Сохраняем ID и source в state для подтверждения
            await state.update_data(expense_id=expense_id, user_id=callback.from_user.id, source=source)
            await state.set_state(DeleteExpenseStates.waiting_for_confirmation)
            
            try:
                if isinstance(date, str):
                    date_obj = datetime.strptime(date, "%Y-%m-%d")
                else:
                    date_obj = date
                date_str = date_obj.strftime("%d.%m.%Y")
            except:
                date_str = str(date)[:10]
            
            source_names = {"receipt": "Чек", "bank": "Банк", "manual": "Вручную"}
            source_name = source_names.get(source, source)
            source_icon = {"receipt": "🧾", "bank": "🏦", "manual": "✍️"}.get(source, "💰")
            
            # Кнопки подтверждения
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="✅ Да, удалить", callback_data="confirm_delete_expense"),
                    InlineKeyboardButton(text="❌ Отмена", callback_data="cancel_delete_expense")
                ]
            ])
            
            category_text = f"\n📂 Категория: {category}" if category else ""
            await callback.message.answer(
                f"⚠️ Вы уверены, что хотите удалить расход?\n\n"
                f"{source_icon} Источник: {source_name}\n"
                f"📝 Название: {note}\n"
                f"💰 Сумма: {amount:.2f} {currency}\n"
                f"📅 Дата: {date_str}{category_text}",
                reply_markup=keyboard
            )

        @self.router.callback_query(F.data == "confirm_delete_expense")
        async def handle_confirm_delete_expense(callback: CallbackQuery, state: FSMContext) -> None:
            """Обработчик финального подтверждения удаления расхода"""
            await callback.answer()
            
            if not self.supabase or not callback.from_user:
                await callback.message.answer("❌ Ошибка: база данных не подключена.")
                await state.clear()
                return
            
            data = await state.get_data()
            expense_id = data.get("expense_id")
            user_id = data.get("user_id")
            
            if not expense_id or user_id != callback.from_user.id:
                await callback.message.answer("❌ Ошибка: несоответствие данных.")
                await state.clear()
                return
            
            try:
                # Получаем source из state для сообщения
                source = data.get("source", "")
                success = await self.supabase.delete_expense(user_id, expense_id)
                if success:
                    # Формируем сообщение о каскадном удалении
                    source_text = ""
                    if source == "receipt":
                        source_text = "\n🧾 Чек также удален."
                    elif source == "bank":
                        source_text = "\n🏦 Банковская транзакция также удалена."
                    
                    await callback.message.answer(f"✅ Расход успешно удален.{source_text}")
                else:
                    await callback.message.answer("❌ Не удалось удалить расход. Возможно, он уже был удален.")
            except Exception as exc:
                logging.exception(f"Ошибка при удалении расхода: {exc}")
                # Отправляем техническую ошибку в технический чат
                if callback.from_user:
                    await self._send_tech_error_to_channel(
                        user_id=callback.from_user.id,
                        username=callback.from_user.username,
                        first_name=callback.from_user.first_name,
                        error=exc,
                        context="Удаление расхода",
                        additional_info=f"expense_id={expense_id if 'expense_id' in locals() else 'N/A'}"
                    )
                # Пользователю отправляем дружелюбное сообщение
                await callback.message.answer("⚠️ Ошибка при удалении расхода. Пожалуйста, попробуйте позже или обратитесь в поддержку через /feedback")
            finally:
                await state.clear()

        @self.router.callback_query(F.data == "cancel_delete_expense")
        async def handle_cancel_delete_expense(callback: CallbackQuery, state: FSMContext) -> None:
            """Обработчик отмены удаления расхода"""
            await callback.answer()
            await callback.message.answer("❌ Удаление расхода отменено.")
            await state.clear()

        @self.router.callback_query(F.data.startswith("setup_currency_"))
        async def handle_setup_currency(callback: CallbackQuery, state: FSMContext) -> None:
            """Обработчик выбора валюты при первом запуске или изменении настроек"""
            await callback.answer()
            
            if not self.supabase or not callback.from_user:
                await callback.message.answer("❌ Ошибка: база данных не подключена.")
                await state.clear()
                return
            
            # Извлекаем валюту из callback_data
            currency = callback.data.replace("setup_currency_", "").upper()
            
            # Проверяем, что валюта валидна
            valid_currencies = ["RUB", "KZT", "USD", "EUR", "GBP", "GEL", "BYN", "KGS", "CNY", "CHF", "AED", "CAD", "AUD"]
            if currency not in valid_currencies:
                await callback.message.answer("❌ Неверная валюта. Попробуйте еще раз.")
                return
            
            # Сохраняем валюту
            success = await self.supabase.set_user_default_currency(callback.from_user.id, currency)
            
            if success:
                currency_symbols = {
                    "RUB": "₽",
                    "KZT": "₸",
                    "USD": "$",
                    "EUR": "€",
                    "GBP": "£",
                    "GEL": "₾",
                }
                symbol = currency_symbols.get(currency, currency)
                
                # Проверяем, это первый запуск или изменение настроек
                settings = await self.supabase.get_user_settings(callback.from_user.id)
                if settings and settings.get("default_currency"):
                    # Изменение настроек
                    await callback.message.answer(
                        f"✅ Валюта по умолчанию изменена на: {symbol} {currency}\n\n"
                        f"Теперь при добавлении расходов без указания валюты будет использоваться {symbol} {currency}."
                    )
                else:
                    # Первый запуск
                    await callback.message.answer(
                        f"✅ Готово! Валюта по умолчанию: {symbol} {currency}\n\n"
                        "✨ Теперь вы можете:\n"
                        "📸 Отправлять фото чеков — распознаю автоматически\n"
                        "📝 Добавлять расходы вручную текстом\n"
                        "🏦 Импортировать банковские выписки\n"
                        "📊 Получать отчёты по категориям и магазинам\n"
                        "💰 Поддерживаю несколько валют\n"
                        "📤 Экспортировать данные в CSV\n\n"
                        "📋 Основные команды:\n"
                        "/expense — добавить расход вручную\n"
                        "/report — получить отчёт\n"
                        "/statement — импортировать выписку\n"
                        "/settings — изменить настройки"
                    )
            else:
                await callback.message.answer("❌ Ошибка при сохранении настроек. Попробуйте еще раз.")
            
            await state.clear()

    async def _process_receipt_message(self, message: Message, state: FSMContext) -> None:
        start_time = time.perf_counter()
        logging.info(f"⏱️ [PERF] Начало обработки чека для user_id={message.from_user.id if message.from_user else 'unknown'}")
        
        # Проверяем лимит чеков перед обработкой
        limit_check_start = time.perf_counter()
        if self.supabase and message.from_user:
            can_save, limits = await self.supabase.check_receipt_limit(message.from_user.id)
            limit_check_time = time.perf_counter() - limit_check_start
            logging.info(f"⏱️ [PERF] Проверка лимита: {limit_check_time*1000:.1f}ms")
            if not can_save:
                receipts_count = limits.get("receipts_count", 0)
                limit_receipts = limits.get("limit_receipts", 20)
                await message.answer(
                    f"⚠️ Достигнут лимит запросов\n\n"
                    f"📊 Использовано чеков: {receipts_count}/{limit_receipts}\n\n"
                    f"Для продолжения распознавания чеков оформите подписку:\n"
                    f"• ⭐ Pro — 100 чеков/месяц за 200 ⭐\n"
                    f"• 👑 Premium — безлимит за 500 ⭐\n\n"
                    f"Используйте команду /subscribe для выбора тарифа.\n\n"
                    f"💡 Вы все еще можете использовать функции, которые не требуют распознавания:\n"
                    f"• 📊 Получение отчетов (/report)\n"
                    f"• 📥 Выгрузка данных в CSV (/export)\n"
                    f"• ✏️ Добавление расходов вручную (/expense)"
                )
                total_time = time.perf_counter() - start_time
                logging.info(f"⏱️ [PERF] Обработка чека завершена (лимит): {total_time*1000:.1f}ms")
                return
        
        await message.answer("🔍 Распознаю...")
        
        receipt_processing_start = time.perf_counter()
        result = await self._handle_receipt_from_message(message)
        
        # Используем file_bytes из результата, чтобы не загружать файл дважды
        file_bytes_for_storage = result.file_bytes
        mime_type_for_storage = result.mime_type
        receipt_processing_time = time.perf_counter() - receipt_processing_start
        logging.info(f"⏱️ [PERF] Обработка чека (_handle_receipt_from_message): {receipt_processing_time*1000:.1f}ms")
        logging.info(f"Receipt processing result: success={result.success}, has_summary={bool(result.summary)}, has_error={bool(result.error)}")
        
        # Если была ошибка валидации, сохраняем фото
        if not result.success and result.error and file_bytes_for_storage and self.supabase and message.from_user:
            try:
                file_path = await self.supabase.save_rejected_receipt_photo(
                    user_id=message.from_user.id,
                    file_bytes=file_bytes_for_storage,
                    reason="validation_error",
                    mime_type=mime_type_for_storage or "image/jpeg"
                )
                # Формируем таблицу результатов калькуляции, если есть распознанные данные
                receipt_table = None
                if result.parsed_receipt:
                    receipt_table = format_receipt_table(result.parsed_receipt)
                
                # Отправляем информацию о фейле в канал
                if file_path and self.failed_receipts_chat_id:
                    await self._send_failed_receipt_to_channel(
                        user_id=message.from_user.id,
                        username=message.from_user.username,
                        first_name=message.from_user.first_name,
                        reason="validation_error",
                        file_path=file_path,
                        file_bytes=file_bytes_for_storage if file_bytes_for_storage and isinstance(file_bytes_for_storage, bytes) and len(file_bytes_for_storage) > 0 else None,  # Отправляем фото напрямую
                        mime_type=mime_type_for_storage or "image/jpeg",
                        error_message=result.error,
                        receipt_table=receipt_table  # Добавляем результаты калькуляции
                    )
            except Exception as exc:
                logging.warning(f"Failed to save rejected receipt photo: {exc}")
        
        if result.success and result.summary:
            # Проверяем валидацию суммы перед сохранением данных в FSM
            validation_passed = True
            if result.parsed_receipt:
                items_sum = sum(item.price for item in result.parsed_receipt.items)
                total = result.parsed_receipt.total or 0.0
                difference = abs(items_sum - total)
                tolerance = max(total * 0.01, 1.0)
                validation_passed = difference <= tolerance
                logging.info(f"🔍 Валидация суммы: разница={difference:.2f}, tolerance={tolerance:.2f}, пройдена={validation_passed}")
            
            if not validation_passed:
                logging.warning(f"❌ Валидация не пройдена, чек не будет сохранен")
                # Если валидация не пройдена, сохраняем фото и отправляем уведомление в канал
                if file_bytes_for_storage and self.supabase and message.from_user:
                    try:
                        file_path = await self.supabase.save_rejected_receipt_photo(
                            user_id=message.from_user.id,
                            file_bytes=file_bytes_for_storage,
                            reason="validation_error",
                            mime_type=mime_type_for_storage or "image/jpeg"
                        )
                        # Формируем таблицу результатов калькуляции
                        receipt_table = None
                        if result.parsed_receipt:
                            receipt_table = format_receipt_table(result.parsed_receipt)
                        
                        # Отправляем информацию о фейле в канал
                        if file_path:
                            logging.info(f"Rejected receipt photo saved: {file_path}")
                            if self.failed_receipts_chat_id:
                                items_sum = sum(item.price for item in result.parsed_receipt.items)
                                total = result.parsed_receipt.total or 0.0
                                difference = abs(items_sum - total)
                                items_count = len(result.parsed_receipt.items)
                                error_text = f"Несоответствие суммы: товаров={items_count}, позиции={items_sum:.2f}, итого={total:.2f}, разница={difference:.2f}"
                                if result.qr_url_found_but_failed:
                                    error_text += " | Не удалось получить данные по QR-коду"
                                logging.info(f"Sending failed receipt notification to channel: {self.failed_receipts_chat_id}")
                                await self._send_failed_receipt_to_channel(
                                    user_id=message.from_user.id,
                                    username=message.from_user.username,
                                    first_name=message.from_user.first_name,
                                    reason="validation_error",
                                    file_path=file_path,
                                    file_bytes=file_bytes_for_storage if file_bytes_for_storage and isinstance(file_bytes_for_storage, bytes) and len(file_bytes_for_storage) > 0 else None,  # Отправляем фото напрямую
                                    mime_type=mime_type_for_storage or "image/jpeg",
                                    error_message=error_text,
                                    receipt_table=receipt_table  # Добавляем результаты калькуляции
                                )
                            else:
                                logging.warning("failed_receipts_chat_id not configured, skipping channel notification")
                        else:
                            logging.warning("file_path is None, cannot send notification to channel")
                    except Exception as exc:
                        logging.exception(f"Failed to save rejected receipt photo or send notification: {exc}")
                
                # Формируем сообщение об ошибке
                error_message = "❌ Не удалось корректно распознать чек.\n\n"
                
                # Добавляем информацию о QR-коде, если он был найден, но данные не получены
                if result.qr_url_found_but_failed:
                    error_message += (
                        f"⚠️ Не удалось получить данные по QR-коду.\n"
                        f"Пожалуйста, отправьте полное фото чека со всеми позициями.\n"
                    )
                else:
                    error_message += "Пожалуйста, отправьте фото чека заново.\n"
                
                # Добавляем детальную информацию о QR-кодах для отладки
                if result.qr_codes:
                    error_message += "\n📱 Найденные QR-коды:\n"
                    # Определяем финальный метод распознавания
                    final_method = result.recognition_method or "unknown"
                    if final_method == "openai_qr_data":
                        final_method_desc = "✅ Данные из QR-кода (структурирование через OpenAI)"
                    else:
                        final_method_desc = "📷 Распознавание по фото (OpenAI Vision)"
                    
                    error_message += f"Метод распознавания: {final_method_desc}\n"
                    if result.qr_time:
                        error_message += f"Время чтения QR-кодов: {result.qr_time*1000:.1f}мс\n"
                    if result.openai_time:
                        error_message += f"Время OpenAI запроса: {result.openai_time*1000:.1f}мс ({result.openai_time:.2f}с)\n"
                    error_message += "\n"
                    
                    for i, qr in enumerate(result.qr_codes, 1):
                        error_message += f"{i}. Тип: {qr['type']}\n"
                        qr_data = qr['data']
                        qr_data_preview = qr_data[:100] + "..." if len(qr_data) > 100 else qr_data
                        error_message += f"   Данные: {qr_data_preview}\n"
                        
                        # Добавляем информацию о парсинге, если есть
                        if result.qr_parsing_info and qr_data in result.qr_parsing_info:
                            info = result.qr_parsing_info[qr_data]
                            if info.get("status") == "success":
                                error_message += f"   ✅ Успешно получены данные за {info.get('fetch_time_ms', 0):.1f}мс"
                                if info.get("items_count") is not None:
                                    error_message += f" ({info['items_count']} позиций)"
                                error_message += "\n"
                            elif info.get("status") == "failed":
                                error_message += f"   ❌ Не удалось получить данные за {info.get('fetch_time_ms', 0):.1f}мс"
                                if info.get("reason"):
                                    error_message += f" ({info['reason']})"
                                error_message += "\n"
                            elif info.get("status") == "ignored":
                                error_message += f"   ⏭️ Игнорирован: {info.get('reason', '')}\n"
                            elif info.get("status") == "not_url":
                                error_message += f"   ℹ️ {info.get('reason', '')}\n"
                        error_message += "\n"
                
                await message.answer(error_message)
                await state.clear()  # Очищаем состояние FSM
                return
            
            # Сохраняем данные чека в FSM для подтверждения (только если валидация пройдена)
            if result.parsed_receipt and result.receipt_payload:
                logging.info(f"💾 Сохранение данных чека в FSM state (user_id={message.from_user.id if message.from_user else 'unknown'}, store={result.parsed_receipt.store}, total={result.parsed_receipt.total})")
                await state.update_data(
                    parsed_receipt=result.parsed_receipt,
                    receipt_payload=result.receipt_payload,
                    receipt_photo_bytes=file_bytes_for_storage,  # Сохраняем фото для возможного отклонения
                    receipt_photo_mime_type=mime_type_for_storage or "image/jpeg",
                )
                await state.set_state(ReceiptStates.waiting_for_confirmation)
                logging.info(f"✅ Данные чека сохранены в FSM, состояние установлено: ReceiptStates.waiting_for_confirmation")
                
                # Создаем кнопки для подтверждения только если данные успешно сохранены в FSM
                keyboard = InlineKeyboardMarkup(inline_keyboard=[
                    [
                        InlineKeyboardButton(text="✅ Все верно", callback_data="receipt_confirm"),
                    ],
                    [
                        InlineKeyboardButton(text="❌ Есть ошибка (переснять)", callback_data="receipt_reject"),
                    ]
                ])
            else:
                logging.warning(f"⚠️ Не удалось сохранить данные в FSM: parsed_receipt={bool(result.parsed_receipt)}, receipt_payload={bool(result.receipt_payload)}")
                # Не создаем кнопки, если данные не сохранены
                keyboard = None
            
            # Генерируем и отправляем изображение (валидация уже пройдена)
            if result.parsed_receipt:
                img_start = time.perf_counter()
                img_bytes = generate_receipt_image(result.parsed_receipt)
                img_time = time.perf_counter() - img_start
                if img_time > 0.1:
                    logging.info(f"⏱️ [PERF] Генерация изображения чека: {img_time*1000:.1f}ms")
                
                if img_bytes:
                    # Отправляем изображение
                    logging.info(f"Отправка изображения чека с кнопками подтверждения (user_id={message.from_user.id if message.from_user else 'unknown'})")
                    photo = BufferedInputFile(img_bytes, filename="receipt.png")
                    await message.answer_photo(photo, reply_markup=keyboard if keyboard else None)
                    logging.info(f"✅ Изображение чека отправлено, ожидаем подтверждения от пользователя")
                    
                    # Отправляем результат валидации отдельным сообщением
                    items_sum = sum(item.price for item in result.parsed_receipt.items)
                    total = result.parsed_receipt.total or 0.0
                    items_count = len(result.parsed_receipt.items)
                    # Форматируем дату для отображения
                    date_str = ""
                    if result.parsed_receipt.purchased_at:
                        date_str = result.parsed_receipt.purchased_at.strftime("%d.%m.%Y")
                    validation_text = (
                        f"✅ Валидация пройдена:\n"
                        f"Товаров в чеке: {items_count}\n"
                        f"Сумма всех позиций: {items_sum:.2f} {result.parsed_receipt.currency}\n"
                        f"Итого по чеку: {total:.2f} {result.parsed_receipt.currency}"
                    )
                    if date_str:
                        validation_text += f"\n📅 Дата: {date_str}"
                    await message.answer(validation_text)
                    
                    # Добавляем дополнительную информацию текстом, если есть
                    additional_info = ""
                    if "QR-кода" in result.summary or "QR" in result.summary:
                        # Извлекаем информацию о QR-коде из summary
                        qr_part = result.summary.split("📱")[-1] if "📱" in result.summary else ""
                        if qr_part:
                            additional_info = f"\n\n📱{qr_part}"
                    
                    if additional_info:
                        await message.answer(additional_info.strip())
                    
                    total_time = time.perf_counter() - start_time
                    logging.info(f"⏱️ [PERF] _process_receipt_message всего: {total_time*1000:.1f}ms ({total_time:.2f}s)")
                    return
            
            # Fallback: отправляем текстом если не удалось сгенерировать изображение
            # Кнопки показываем только если они были созданы (данные успешно сохранены в FSM)
            if keyboard:
                logging.info(f"Fallback: отправка текстового сообщения с кнопками (user_id={message.from_user.id if message.from_user else 'unknown'})")
                truncated_summary = truncate_message_for_telegram(result.summary)
                logging.info(f"Sending receipt summary to user (text fallback): {len(result.summary)} chars")
                await message.answer(truncated_summary, reply_markup=keyboard)
                logging.info(f"✅ Текстовое сообщение с кнопками отправлено, ожидаем подтверждения от пользователя")
            else:
                # Если кнопки не созданы, отправляем только текст без кнопок
                logging.info(f"Fallback: отправка текстового сообщения без кнопок (данные не сохранены в FSM)")
                truncated_summary = truncate_message_for_telegram(result.summary)
                await message.answer(truncated_summary)
            
            total_time = time.perf_counter() - start_time
            logging.info(f"⏱️ [PERF] _process_receipt_message всего: {total_time*1000:.1f}ms ({total_time:.2f}s)")
            return
        error_msg = result.error or "Не удалось обработать чек."
        logging.warning(f"Sending error to user: {error_msg}")
        await message.answer(error_msg)

    async def _process_statement_message(self, message: Message) -> None:
        await message.answer("Выписка принята, распознаю…")
        result = await self._handle_statement_from_message(message)
        if result.success and result.summary:
            truncated_summary = truncate_message_for_telegram(result.summary)
            await message.answer(truncated_summary)
            return
        await message.answer(result.error or "Не удалось обработать выписку.")

    async def _handle_receipt_from_message(self, message: Message) -> ProcessingResult:
        start_time = time.perf_counter()
        
        file_load_start = time.perf_counter()
        file = await self._resolve_file(message)
        if file is None:
            # Сохраняем статистику неуспешного распознавания (файл не прочитан)
            if self.supabase and message.from_user:
                try:
                    await self.supabase.save_receipt_recognition_stat(
                        user_id=message.from_user.id,
                        recognition_method="unknown",  # Файл не прочитан, метод не определен
                        success=False,
                        error_message="Не удалось прочитать файл"
                    )
                except Exception as stat_exc:
                    logging.warning(f"Failed to save recognition stat: {stat_exc}")
            return ProcessingResult(success=False, error="Не удалось прочитать файл.", recognition_method="unknown")
        mime_type = detect_mime_type(message, file.file_path)
        file_bytes = await self._download_file(file.file_path)
        
        # Сохраняем оригинальные байты для возможного сохранения при отклонении
        file_bytes_for_storage = file_bytes
        mime_type_for_storage = mime_type
        
        file_load_time = time.perf_counter() - file_load_start
        logging.info(f"⏱️ [PERF] Загрузка файла в _handle_receipt_from_message: {file_load_time*1000:.1f}ms")
        
        try:
            convert_start = time.perf_counter()
            file_bytes, mime_type = convert_heic_if_needed(file_bytes, mime_type)
            # Обновляем file_bytes_for_storage после конвертации
            file_bytes_for_storage = file_bytes
            mime_type_for_storage = mime_type
            convert_time = time.perf_counter() - convert_start
            if convert_time > 0.1:
                logging.info(f"⏱️ [PERF] Конвертация HEIC в _handle_receipt_from_message: {convert_time*1000:.1f}ms")
        except ReceiptParsingError as exc:
            return ProcessingResult(success=False, error=str(exc), file_bytes=file_bytes_for_storage, mime_type=mime_type_for_storage)
        
        # Сначала проверяем QR-коды
        qr_start = time.perf_counter()
        qr_codes = read_qr_codes(file_bytes)
        qr_time = time.perf_counter() - qr_start
        logging.info(f"⏱️ [PERF] Чтение QR-кодов: {qr_time*1000:.1f}ms, найдено: {len(qr_codes)}")
        
        # Переменная для хранения данных из QR-кода для отправки в OpenAI
        qr_data_from_url = None
        qr_url_found_but_failed = None  # URL QR-кода, если он был найден, но данные не получены
        qr_parsing_info = {}  # Словарь: ключ - данные QR-кода, значение - информация о парсинге
        
        # Если есть QR-код или штрих-код с URL, пытаемся получить данные оттуда
        if qr_codes:
            # Сначала ищем QR-коды с URL (игнорируем CODE39 и другие не-URL коды)
            for qr in qr_codes:
                qr_data = qr.get("data", "")
                qr_type = qr.get("type", "")
                
                # Игнорируем CODE39 и другие штрих-коды, которые не являются URL
                if qr_type == "CODE39" or (not is_url(qr_data) and qr_type != "QRCODE"):
                    logging.info(f"Игнорируем код типа {qr_type}: {qr_data[:50]}... (не URL и не QR-код)")
                    qr_parsing_info[qr_data] = {
                        "type": qr_type,
                        "data_preview": qr_data[:50] + "..." if len(qr_data) > 50 else qr_data,
                        "status": "ignored",
                        "reason": "не URL и не QR-код"
                    }
                    continue
                
                logging.info(f"Проверяем код: {qr_data[:100]}... (тип: {qr_type})")
                
                # Проверяем, является ли это URL
                if is_url(qr_data):
                    logging.info(f"✅ Найден код с URL (тип: {qr_type}): {qr_data}")
                    # Пытаемся получить данные с URL
                    qr_fetch_start = time.perf_counter()
                    qr_data_from_url = await fetch_receipt_from_qr_url(qr_data)
                    qr_fetch_time = time.perf_counter() - qr_fetch_start
                    logging.info(f"⏱️ [PERF] Получение данных с QR URL: {qr_fetch_time*1000:.1f}ms")
                    
                    if qr_data_from_url:
                        items_count = len(qr_data_from_url.get("items", []))
                        logging.info(f"✅ Получены данные с URL, отправляем их в OpenAI для структурирования")
                        qr_parsing_info[qr_data] = {
                            "type": qr_type,
                            "data_preview": qr_data[:80] + "..." if len(qr_data) > 80 else qr_data,
                            "status": "success",
                            "method": "qr_url",
                            "fetch_time_ms": round(qr_fetch_time * 1000, 1),
                            "items_count": items_count
                        }
                    else:
                        logging.warning(f"⚠️ Не удалось получить данные с URL, используем изображение")
                        qr_data_from_url = None  # Сброс, чтобы использовать изображение
                        # Сохраняем информацию о том, что QR был найден, но данные не получены
                        qr_url_found_but_failed = qr_data
                        
                        # Пытаемся определить причину ошибки, проверяя URL напрямую
                        failure_reason = "не удалось получить данные"
                        try:
                            check_response = requests.get(
                                qr_data,
                                timeout=5,
                                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                                verify=False,
                                allow_redirects=True
                            )
                            if check_response.status_code != 200:
                                failure_reason = f"HTTP статус {check_response.status_code}"
                            else:
                                # Проверяем на наличие капчи в HTML
                                html_content = check_response.text.lower()
                                captcha_indicators = [
                                    "captcha", "recaptcha", "hcaptcha", "cloudflare", 
                                    "проверка на робота", "подтвердите что вы не робот",
                                    "verify you are human", "challenge"
                                ]
                                if any(indicator in html_content for indicator in captcha_indicators):
                                    failure_reason = "обнаружена капча (проверка на робота)"
                                elif "text/html" in check_response.headers.get("Content-Type", "").lower():
                                    # Проверяем, есть ли данные в HTML
                                    if len(html_content) < 1000:  # Очень короткий ответ
                                        failure_reason = "получен пустой или короткий HTML ответ"
                                    else:
                                        failure_reason = "не удалось извлечь данные из HTML (возможно требуется JavaScript)"
                        except Exception as check_exc:
                            if "timeout" in str(check_exc).lower():
                                failure_reason = "таймаут при запросе"
                            elif "connection" in str(check_exc).lower():
                                failure_reason = "ошибка подключения"
                            else:
                                failure_reason = f"ошибка: {type(check_exc).__name__}"
                        
                        qr_parsing_info[qr_data] = {
                            "type": qr_type,
                            "data_preview": qr_data[:80] + "..." if len(qr_data) > 80 else qr_data,
                            "status": "failed",
                            "method": "qr_url",
                            "fetch_time_ms": round(qr_fetch_time * 1000, 1),
                            "reason": failure_reason
                        }
                    # Если найден QR-код с URL, игнорируем все остальные коды
                    break
                else:
                    logging.info(f"QR-код не является URL: {qr_data[:50]}... (тип: {qr_type})")
                    qr_parsing_info[qr_data] = {
                        "type": qr_type,
                        "data_preview": qr_data[:50] + "..." if len(qr_data) > 50 else qr_data,
                        "status": "not_url",
                        "reason": "QR-код не содержит URL"
                    }
        
        # Если QR-кода нет или не удалось получить данные, используем OpenAI
        try:
            logging.info(f"Using original image: {len(file_bytes)} bytes ({len(file_bytes) / 1024:.1f} КБ)")
            
            # Определяем mime_type
            if Image is not None:
                try:
                    image = Image.open(io.BytesIO(file_bytes))
                    mime_type = f"image/{image.format.lower()}" if image.format else "image/jpeg"
                except Exception:
                    mime_type = "image/jpeg"
            else:
                mime_type = "image/jpeg"
            
            # Определяем способ распознавания для статистики
            # Проверяем, содержат ли данные из QR-кода позиции товаров
            use_qr_data = False
            if qr_data_from_url:
                items_in_qr = qr_data_from_url.get("items") or []
                # Используем данные из QR-кода только если там есть реальные позиции товаров
                if items_in_qr and not (len(items_in_qr) == 1 and items_in_qr[0].get("name") in ["Без позиций", "Покупка"]):
                    use_qr_data = True
                    recognition_method = "openai_qr_data"
                else:
                    logging.warning(f"⚠️ Данные из QR-кода не содержат позиций товаров, используем изображение для распознавания")
                    recognition_method = "openai_photo"
            else:
                recognition_method = "openai_photo"
            
            # Отправляем в OpenAI (с данными из QR-кода, если есть реальные позиции, иначе с изображением)
            openai_start = time.perf_counter()
            if use_qr_data:
                logging.info(f"Отправляем данные из QR-кода в OpenAI для структурирования (найдено {len(items_in_qr)} позиций)")
                response_json = await parse_receipt_with_ai(file_bytes, mime_type, qr_data=qr_data_from_url)
            else:
                logging.info("Starting OpenAI receipt parsing from image...")
                response_json = await parse_receipt_with_ai(file_bytes, mime_type)
            openai_time = time.perf_counter() - openai_start
            logging.info(f"⏱️ [PERF] OpenAI запрос: {openai_time*1000:.1f}ms ({openai_time:.2f}s)")
            
            # Увеличиваем счетчик чеков после успешного запроса в OpenAI
            # (независимо от того, дубликат это или нет, так как запрос в OpenAI уже оплачен)
            if self.supabase and message.from_user:
                increment_start = time.perf_counter()
                await self.supabase.increment_receipt_count(message.from_user.id)
                increment_time = time.perf_counter() - increment_start
                if increment_time > 0.1:
                    logging.info(f"⏱️ [PERF] Увеличение счетчика чеков: {increment_time*1000:.1f}ms")
            
            # Извлекаем только content из message и информацию о токенах
            try:
                choices = response_json.get("choices", [])
                if not choices:
                    raise ReceiptParsingError("OpenAI response не содержит choices")
                
                ai_message = choices[0].get("message", {})
                content = ai_message.get("content")
                refusal = ai_message.get("refusal")
                
                # Проверяем, не отказалась ли модель обрабатывать запрос
                if refusal:
                    refusal_msg = f"OpenAI отказался обрабатывать запрос: {refusal}"
                    logging.warning(refusal_msg)
                    # Используем специальный код ошибки для refusal, чтобы показать пользователю дружелюбное сообщение
                    raise ReceiptParsingError("REFUSAL: Не удалось распознать чек. Пожалуйста, попробуйте отправить фото чека еще раз или убедитесь, что на фото четко виден кассовый чек.")
                
                if not content:
                    raise ReceiptParsingError("OpenAI response не содержит content")
                
                # Парсим JSON из content
                parse_start = time.perf_counter()
                content_json = None
                try:
                    # Парсим JSON из content
                    content_json = json.loads(content)
                except json.JSONDecodeError:
                    # Если не JSON, пытаемся декодировать escape-последовательности
                    try:
                        content_decoded = content.encode().decode('unicode_escape')
                        content_json = json.loads(content_decoded)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        logging.warning("Не удалось распарсить JSON из content")
                        raise ReceiptParsingError("Не удалось распарсить ответ от OpenAI")
                parse_time = time.perf_counter() - parse_start
                if parse_time > 0.01:
                    logging.info(f"⏱️ [PERF] Парсинг JSON: {parse_time*1000:.1f}ms")
                
                # Преобразуем в ParsedReceipt для форматирования
                build_start = time.perf_counter()
                parsed_receipt = build_parsed_receipt(content_json)
                build_time = time.perf_counter() - build_start
                if build_time > 0.01:
                    logging.info(f"⏱️ [PERF] Построение ParsedReceipt: {build_time*1000:.1f}ms")
                
                # Логируем категории из OpenAI ответа
                items_from_ai = content_json.get("items", [])
                categories_from_ai = {}
                items_with_cat = 0
                items_without_cat = 0
                # Проверяем наличие дубликатов в ответе OpenAI
                item_names = [item.get("name", "") for item in items_from_ai if isinstance(item, dict)]
                duplicates = {}
                for i, name in enumerate(item_names):
                    if name in duplicates:
                        duplicates[name].append(i)
                    else:
                        duplicates[name] = [i]
                duplicate_items = {name: indices for name, indices in duplicates.items() if len(indices) > 1}
                if duplicate_items:
                    logging.warning(f"⚠️ Найдены дубликаты позиций в ответе OpenAI: {duplicate_items}")
                
                for item in items_from_ai:
                    if isinstance(item, dict):
                        cat = item.get("category")
                        if cat:
                            categories_from_ai[cat] = categories_from_ai.get(cat, 0) + 1
                            items_with_cat += 1
                        else:
                            items_without_cat += 1
                logging.info(f"Категории из OpenAI ответа: всего items={len(items_from_ai)}, с категорией={items_with_cat}, без категории={items_without_cat}, категории={categories_from_ai}")
                
                # Проверяем правильность распознавания: сумма позиций должна совпадать с тоталом
                # ВАЖНО: используем все позиции из parsed_receipt.items, включая дубликаты
                # ВАЖНО: цены НЕ подгоняются под общую сумму - мы только проверяем соответствие
                items_sum = sum(item.price for item in parsed_receipt.items)
                total = parsed_receipt.total or 0.0
                difference = abs(items_sum - total)
                
                # Логируем информацию о позициях для диагностики
                logging.info(f"📊 Валидация суммы: items_count={len(parsed_receipt.items)}, items_from_ai_count={len(items_from_ai)}, items_sum={items_sum:.2f}, total={total:.2f}, difference={difference:.2f}")
                if len(parsed_receipt.items) != len(items_from_ai):
                    logging.warning(f"⚠️ Количество позиций изменилось: было {len(items_from_ai)}, стало {len(parsed_receipt.items)}")
                    # Логируем все позиции для сравнения
                    logging.info(f"Позиции из OpenAI: {[item.get('name', '') for item in items_from_ai if isinstance(item, dict)]}")
                    logging.info(f"Позиции в parsed_receipt: {[item.name for item in parsed_receipt.items]}")
                
                # Допускаем погрешность до 1% или 1 единицу валюты (что больше)
                tolerance = max(total * 0.01, 1.0)
                
                # Проверяем валидацию суммы
                validation_passed = difference <= tolerance
                
                validation_message = ""
                if not validation_passed:
                    validation_message = (
                        f"\n\n⚠️ Несоответствие суммы:\n"
                        f"Сумма позиций: {items_sum:.2f} {parsed_receipt.currency}\n"
                        f"Итого: {total:.2f} {parsed_receipt.currency}\n"
                        f"Разница: {difference:.2f} {parsed_receipt.currency}"
                    )
                    logging.warning(
                        f"⚠️ Несоответствие суммы: сумма позиций={items_sum:.2f}, "
                        f"итого={total:.2f}, разница={difference:.2f} (допустимо {tolerance:.2f})"
                    )
                else:
                    validation_message = (
                        f"\n\n✅ Валидация пройдена:\n"
                        f"Сумма позиций: {items_sum:.2f} {parsed_receipt.currency}\n"
                        f"Итого: {total:.2f} {parsed_receipt.currency}"
                    )
                    logging.info(f"✅ Сумма позиций совпадает с итого: {items_sum:.2f} = {total:.2f}")
                
                # Форматируем чек в виде таблицы только если валидация пройдена
                receipt_table = ""
                if validation_passed:
                    receipt_table = format_receipt_table(parsed_receipt)
                else:
                    # Если валидация не пройдена, показываем только основную информацию без деталей товаров
                    receipt_table = (
                        f"🏪 Магазин: {parsed_receipt.store}\n"
                        f"💰 Итого: {total:.2f} {parsed_receipt.currency}\n"
                        f"📅 Дата: {parsed_receipt.purchased_at.strftime('%d.%m.%Y %H:%M') if parsed_receipt.purchased_at else 'Не указана'}"
                    )
                
                # Извлекаем информацию о токенах
                usage = response_json.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                
                # Проверяем кэшированные токены
                prompt_tokens_details = usage.get("prompt_tokens_details", {})
                cached_tokens = prompt_tokens_details.get("cached_tokens", 0)
                
                # Формируем итоговый ответ
                summary = receipt_table + validation_message
                
                # Добавляем уведомление, если QR-код был найден, но данные не получены
                if qr_url_found_but_failed:
                    qr_notification = (
                        f"\n\n⚠️ Найден QR-код, но не удалось получить данные по URL.\n"
                        f"Пожалуйста, перефотографируйте чек целиком (не только QR-код)."
                    )
                    summary += qr_notification
                
                # Добавляем информацию о QR-кодах, если они были найдены
                if qr_codes:
                    qr_info = "\n\n📱 Найденные QR-коды:\n"
                    # Определяем финальный метод распознавания
                    final_method = recognition_method
                    if final_method == "openai_qr_data":
                        final_method_desc = "✅ Данные из QR-кода (структурирование через OpenAI)"
                    else:
                        final_method_desc = "📷 Распознавание по фото (OpenAI Vision)"
                    
                    qr_info += f"Метод распознавания: {final_method_desc}\n"
                    qr_info += f"Время чтения QR-кодов: {qr_time*1000:.1f}мс\n"
                    qr_info += f"Время OpenAI запроса: {openai_time*1000:.1f}мс ({openai_time:.2f}с)\n\n"
                    
                    for i, qr in enumerate(qr_codes, 1):
                        qr_info += f"{i}. Тип: {qr['type']}\n"
                        qr_data = qr['data']
                        qr_data_preview = qr_data[:100] + "..." if len(qr_data) > 100 else qr_data
                        qr_info += f"   Данные: {qr_data_preview}\n"
                        
                        # Добавляем информацию о парсинге, если есть
                        if qr_data in qr_parsing_info:
                            info = qr_parsing_info[qr_data]
                            if info.get("status") == "success":
                                qr_info += f"   ✅ Успешно получены данные за {info.get('fetch_time_ms', 0):.1f}мс"
                                if info.get("items_count") is not None:
                                    qr_info += f" ({info['items_count']} позиций)"
                                qr_info += "\n"
                            elif info.get("status") == "failed":
                                qr_info += f"   ❌ Не удалось получить данные за {info.get('fetch_time_ms', 0):.1f}мс"
                                if info.get("reason"):
                                    qr_info += f" ({info['reason']})"
                                qr_info += "\n"
                            elif info.get("status") == "ignored":
                                qr_info += f"   ⏭️ Игнорирован: {info.get('reason', '')}\n"
                            elif info.get("status") == "not_url":
                                qr_info += f"   ℹ️ {info.get('reason', '')}\n"
                        qr_info += "\n"
                    summary += qr_info
                
                # Логируем информацию о кэшировании
                if cached_tokens > 0:
                    logging.info(f"✅ Кэширование работает! Кэшировано токенов: {cached_tokens} из {prompt_tokens}")
                else:
                    logging.warning(f"⚠️ Кэширование не работает. Prompt tokens: {prompt_tokens}. Возможно, промпт слишком короткий (< 1024 токенов)")
                
                logging.info(f"Parsed receipt: store={parsed_receipt.store}, total={parsed_receipt.total}, items={len(parsed_receipt.items)}, tokens: {total_tokens}")
                
                # Подготавливаем данные для сохранения (но не сохраняем сразу)
                receipt_payload = None
                if self.supabase and message.from_user:
                    try:
                        # Создаем payload для базы
                        payload_start = time.perf_counter()
                        receipt_payload = build_receipt_payload(message.from_user.id, parsed_receipt)
                        payload_time = time.perf_counter() - payload_start
                        if payload_time > 0.01:
                            logging.info(f"⏱️ [PERF] Создание payload: {payload_time*1000:.1f}ms")
                        logging.info(f"Создан payload для сохранения: store={receipt_payload.get('store')}, total={receipt_payload.get('total')}")
                    except Exception as db_exc:
                        logging.exception(f"Ошибка при создании payload: {db_exc}")
                
                total_time = time.perf_counter() - start_time
                logging.info(f"⏱️ [PERF] _handle_receipt_from_message всего: {total_time*1000:.1f}ms ({total_time:.2f}s)")
                
                # Сохраняем статистику успешного распознавания
                if self.supabase and message.from_user:
                    try:
                        await self.supabase.save_receipt_recognition_stat(
                            user_id=message.from_user.id,
                            recognition_method=recognition_method,
                            success=True
                        )
                    except Exception as stat_exc:
                        logging.warning(f"Failed to save recognition stat: {stat_exc}")
                
                return ProcessingResult(
                    success=True,
                    summary=summary,
                    parsed_receipt=parsed_receipt,
                    receipt_payload=receipt_payload,
                    qr_url_found_but_failed=qr_url_found_but_failed,
                    qr_codes=qr_codes if qr_codes else None,
                    qr_parsing_info=qr_parsing_info if qr_parsing_info else None,
                    qr_time=qr_time,
                    openai_time=openai_time,
                    file_bytes=file_bytes_for_storage,
                    mime_type=mime_type_for_storage,
                    recognition_method=recognition_method,
                )
            except Exception as exc:
                logging.error(f"Error extracting content: {exc}", exc_info=True)
                # Fallback: пытаемся извлечь данные и показать таблицу или JSON
                response_str = ""
                
                # Пытаемся сохранить в базу даже при ошибке парсинга
                if self.supabase and message.from_user:
                    try:
                        # Пытаемся извлечь данные из response_json напрямую
                        choices = response_json.get("choices", [])
                        if choices:
                            ai_message = choices[0].get("message", {})
                            content = ai_message.get("content", "")
                            if content:
                                try:
                                    fallback_data = json.loads(content)
                                    parsed_receipt = build_parsed_receipt(fallback_data)
                                    
                                    # Проверяем правильность распознавания: сумма позиций должна совпадать с тоталом
                                    items_sum = sum(item.price for item in parsed_receipt.items)
                                    total = parsed_receipt.total or 0.0
                                    difference = abs(items_sum - total)
                                    tolerance = max(total * 0.01, 1.0)
                                    
                                    items_count = len(parsed_receipt.items)
                                    validation_message = ""
                                    if difference > tolerance:
                                        validation_message = (
                                            f"\n\n⚠️ Несоответствие суммы:\n"
                                            f"Товаров в чеке: {items_count}\n"
                                            f"Сумма всех позиций: {items_sum:.2f} {parsed_receipt.currency}\n"
                                            f"Итого по чеку: {total:.2f} {parsed_receipt.currency}\n"
                                            f"Разница: {difference:.2f} {parsed_receipt.currency}"
                                        )
                                        logging.warning(
                                            f"⚠️ Несоответствие суммы (fallback): товаров={items_count}, сумма позиций={items_sum:.2f}, "
                                            f"итого={total:.2f}, разница={difference:.2f}"
                                        )
                                    else:
                                        validation_message = (
                                            f"\n\n✅ Валидация пройдена:\n"
                                            f"Товаров в чеке: {items_count}\n"
                                            f"Сумма всех позиций: {items_sum:.2f} {parsed_receipt.currency}\n"
                                            f"Итого по чеку: {total:.2f} {parsed_receipt.currency}"
                                        )
                                    
                                    # Форматируем в таблицу
                                    response_str = format_receipt_table(parsed_receipt) + validation_message
                                    
                                    # Подготавливаем данные для сохранения (но не сохраняем сразу)
                                    receipt_payload = None
                                    if self.supabase and message.from_user:
                                        try:
                                            receipt_payload = build_receipt_payload(message.from_user.id, parsed_receipt)
                                            logging.info(f"Создан payload для сохранения (fallback): store={receipt_payload.get('store')}, total={receipt_payload.get('total')}")
                                        except Exception as payload_exc:
                                            logging.exception(f"Ошибка при создании payload (fallback): {payload_exc}")
                                    
                                    # Сохраняем статистику успешного распознавания (fallback)
                                    if self.supabase and message.from_user:
                                        try:
                                            await self.supabase.save_receipt_recognition_stat(
                                                user_id=message.from_user.id,
                                                recognition_method=recognition_method,
                                                success=True
                                            )
                                        except Exception as stat_exc:
                                            logging.warning(f"Failed to save recognition stat: {stat_exc}")
                                    
                                    return ProcessingResult(
                                        success=True,
                                        summary=response_str,
                                        parsed_receipt=parsed_receipt,
                                        receipt_payload=receipt_payload,
                                        qr_url_found_but_failed=qr_url_found_but_failed,
                                        file_bytes=file_bytes_for_storage,
                                        mime_type=mime_type_for_storage,
                                        recognition_method=recognition_method,
                                    )
                                except Exception as fallback_exc:
                                    logging.exception(f"Ошибка при парсинге в fallback: {fallback_exc}")
                                    # Если не удалось распарсить, проверяем наличие refusal
                                    choices = response_json.get("choices", [])
                                    has_refusal = False
                                    if choices:
                                        ai_message = choices[0].get("message", {})
                                        if ai_message.get("refusal"):
                                            has_refusal = True
                                    
                                    if has_refusal:
                                        response_str = "Не удалось распознать чек. Пожалуйста, попробуйте отправить фото чека еще раз или убедитесь, что на фото четко виден кассовый чек."
                                    else:
                                        response_str = "Не удалось распознать чек. Пожалуйста, попробуйте отправить фото чека еще раз."
                    except Exception as db_exc:
                        logging.exception(f"Ошибка при сохранении в базу (fallback): {db_exc}")
                        if not response_str:
                            # Проверяем наличие refusal
                            choices = response_json.get("choices", [])
                            has_refusal = False
                            if choices:
                                ai_message = choices[0].get("message", {})
                                if ai_message.get("refusal"):
                                    has_refusal = True
                            
                            if has_refusal:
                                response_str = "Не удалось распознать чек. Пожалуйста, попробуйте отправить фото чека еще раз или убедитесь, что на фото четко виден кассовый чек."
                            else:
                                response_str = "Не удалось распознать чек. Пожалуйста, попробуйте отправить фото чека еще раз."
                
                if not response_str:
                    # Проверяем, есть ли refusal в ответе OpenAI
                    choices = response_json.get("choices", [])
                    has_refusal = False
                    if choices:
                        ai_message = choices[0].get("message", {})
                        if ai_message.get("refusal"):
                            has_refusal = True
                    
                    if has_refusal:
                        # Показываем дружелюбное сообщение без технических деталей
                        response_str = "Не удалось распознать чек. Пожалуйста, попробуйте отправить фото чека еще раз или убедитесь, что на фото четко виден кассовый чек."
                    else:
                        # Для других ошибок показываем общее сообщение без технических деталей
                        response_str = "Не удалось распознать чек. Пожалуйста, попробуйте отправить фото чека еще раз."
                
                # Сохраняем статистику успешного распознавания (с ошибкой парсинга, но OpenAI ответил)
                if self.supabase and message.from_user:
                    try:
                        await self.supabase.save_receipt_recognition_stat(
                            user_id=message.from_user.id,
                            recognition_method=recognition_method,
                            success=True  # OpenAI ответил, даже если парсинг не удался
                        )
                    except Exception as stat_exc:
                        logging.warning(f"Failed to save recognition stat: {stat_exc}")
                
                return ProcessingResult(
                    success=True,
                    summary=response_str,
                    qr_url_found_but_failed=qr_url_found_but_failed,
                    file_bytes=file_bytes_for_storage,
                    mime_type=mime_type_for_storage,
                    recognition_method=recognition_method,
                )
        except ReceiptParsingError as exc:
            logging.exception("Receipt parsing failed")
            error_msg = str(exc)
            
            # Преобразуем технические ошибки в понятные сообщения для пользователя
            user_error = format_user_friendly_error(exc)
            
            # Если это специальное сообщение для пользователя (начинается с "REFUSAL:"), показываем его без префикса
            if error_msg.startswith("REFUSAL: "):
                user_error = error_msg.replace("REFUSAL: ", "")
            
            # Сохраняем статистику неуспешного распознавания
            if self.supabase and message.from_user:
                try:
                    await self.supabase.save_receipt_recognition_stat(
                        user_id=message.from_user.id,
                        recognition_method=recognition_method if 'recognition_method' in locals() else "openai_photo",
                        success=False,
                        error_message=error_msg[:500]  # Сохраняем оригинальное техническое сообщение для логов
                    )
                except Exception as stat_exc:
                    logging.warning(f"Failed to save recognition stat: {stat_exc}")
            
            # Получаем значения переменных, если они были определены
            qr_codes_val = qr_codes if 'qr_codes' in locals() and qr_codes else None
            qr_parsing_info_val = qr_parsing_info if 'qr_parsing_info' in locals() and qr_parsing_info else None
            qr_time_val = qr_time if 'qr_time' in locals() else None
            openai_time_val = openai_time if 'openai_time' in locals() else None
            qr_url_found_but_failed_val = qr_url_found_but_failed if 'qr_url_found_but_failed' in locals() else None
            
            return ProcessingResult(
                success=False, 
                error=user_error, 
                file_bytes=file_bytes_for_storage, 
                mime_type=mime_type_for_storage, 
                recognition_method=recognition_method if 'recognition_method' in locals() else "openai_photo",
                qr_codes=qr_codes_val,
                qr_parsing_info=qr_parsing_info_val,
                qr_time=qr_time_val,
                openai_time=openai_time_val,
                qr_url_found_but_failed=qr_url_found_but_failed_val
            )
        except Exception as exc:
            logging.exception("Image preprocessing or parsing failed")
            error_msg = str(exc)
            
            # Преобразуем технические ошибки в понятные сообщения для пользователя
            user_error = format_user_friendly_error(exc)
            
            # Сохраняем статистику неуспешного распознавания
            if self.supabase and message.from_user:
                try:
                    await self.supabase.save_receipt_recognition_stat(
                        user_id=message.from_user.id,
                        recognition_method=recognition_method if 'recognition_method' in locals() else "openai_photo",
                        success=False,
                        error_message=error_msg[:500]  # Сохраняем оригинальное техническое сообщение для логов
                    )
                except Exception as stat_exc:
                    logging.warning(f"Failed to save recognition stat: {stat_exc}")
            
            # Получаем значения переменных, если они были определены
            qr_codes_val = qr_codes if 'qr_codes' in locals() and qr_codes else None
            qr_parsing_info_val = qr_parsing_info if 'qr_parsing_info' in locals() and qr_parsing_info else None
            qr_time_val = qr_time if 'qr_time' in locals() else None
            openai_time_val = openai_time if 'openai_time' in locals() else None
            qr_url_found_but_failed_val = qr_url_found_but_failed if 'qr_url_found_but_failed' in locals() else None
            
            return ProcessingResult(
                success=False, 
                error=user_error, 
                file_bytes=file_bytes_for_storage, 
                mime_type=mime_type_for_storage, 
                recognition_method=recognition_method if 'recognition_method' in locals() else "openai_photo",
                qr_codes=qr_codes_val,
                qr_parsing_info=qr_parsing_info_val,
                qr_time=qr_time_val,
                openai_time=openai_time_val,
                qr_url_found_but_failed=qr_url_found_but_failed_val
            )

    async def _handle_statement_from_message(self, message: Message) -> ProcessingResult:
        file = await self._resolve_file(message)
        if file is None:
            return ProcessingResult(success=False, error="Не удалось прочитать файл выписки.")
        file_bytes = await self._download_file(file.file_path)
        try:
            transactions = await parse_bank_statement(file_bytes)
        except StatementParsingError as exc:
            logging.exception("Bank statement parsing failed")
            return ProcessingResult(success=False, error=f"Не удалось распознать выписку: {exc}")

        if not transactions:
            return ProcessingResult(success=False, error="Не удалось найти операции в файле.")

        summary = format_statement_summary(transactions)
        if not self.supabase or not message.from_user:
            return ProcessingResult(success=True, summary=summary)

        payloads = [build_bank_payload(message.from_user.id, txn) for txn in transactions]
        stored = await self.supabase.upsert_bank_transactions(payloads)
        await reconcile_transactions(self.supabase, message.from_user.id, stored)
        return ProcessingResult(
            success=True,
            summary=f"{summary}\n\nИмпортировано операций: {len(stored)}",
        )

    async def _collect_media_group(self, message: Message) -> None:
        if not message.media_group_id:
            return
        group_id = str(message.media_group_id)
        # SECURITY: добавить rate-limiting и проверку квот перед массовой обработкой альбомов.
        bucket = self._media_group_cache.setdefault(group_id, [])
        bucket.append(message)
        if group_id not in self._media_group_tasks:
            self._media_group_tasks[group_id] = asyncio.create_task(
                self._finalize_media_group(group_id)
            )

    async def _finalize_media_group(self, group_id: str) -> None:
        await asyncio.sleep(1.2)
        messages = self._media_group_cache.pop(group_id, [])
        self._media_group_tasks.pop(group_id, None)
        if not messages:
            return
        chat_id = messages[0].chat.id
        await self.bot.send_message(chat_id, f"Получено {len(messages)} файлов, распознаю батч…")
        summaries: List[str] = []
        for idx, message in enumerate(messages, start=1):
            classification = classify_upload_kind(message)
            if classification == "receipt":
                result = await self._handle_receipt_from_message(message)
            elif classification == "statement":
                result = await self._handle_statement_from_message(message)
            else:
                result = ProcessingResult(success=False, error="Не удалось определить тип файла.")
            prefix = f"[{idx}]"
            if result.success and result.summary:
                # Обрезаем каждый summary, чтобы избежать проблем
                truncated_summary = truncate_message_for_telegram(result.summary, max_length=3000)
                summaries.append(f"{prefix}\n{truncated_summary}")
            else:
                summaries.append(f"{prefix} Ошибка: {result.error}")
        
        # Формируем итоговое сообщение
        full_message = "Готово. Результаты по файлам:\n\n" + "\n\n".join(summaries)
        
        # Обрезаем итоговое сообщение
        truncated_message = truncate_message_for_telegram(full_message)
        
        # Если сообщение было обрезано, отправляем его частями
        if len(full_message) > 4000:
            # Отправляем заголовок
            await self.bot.send_message(chat_id, "Готово. Результаты по файлам:")
            # Отправляем каждый summary отдельно
            for summary in summaries:
                await self.bot.send_message(chat_id, summary)
        else:
            await self.bot.send_message(chat_id, truncated_message)

    async def _resolve_file(self, message: Message) -> Optional[Any]:
        if message.photo:
            return await self.bot.get_file(message.photo[-1].file_id)
        if message.document:
            return await self.bot.get_file(message.document.file_id)
        return None

    async def _download_file(self, file_path: str) -> bytes:
        stream = await self.bot.download_file(file_path)
        buffer = io.BytesIO()
        buffer.write(stream.read())
        return buffer.getvalue()

    async def _send_snapshot(self, chat_id: int, snapshot: Snapshot) -> None:
        try:
            print(f"[DEBUG] _send_snapshot called for step: {snapshot.step}")
            if snapshot.pil_image is None:
                logging.error(f"Snapshot {snapshot.step} has None pil_image!")
                print(f"[ERROR] Snapshot {snapshot.step} has None pil_image!")
                return
            png_bytes = image_to_png_bytes(snapshot.pil_image)
            if not png_bytes:
                logging.warning(f"Failed to convert snapshot {snapshot.step} to PNG bytes")
                print(f"[WARNING] Failed to convert snapshot {snapshot.step} to PNG bytes")
                return
            caption = f"{snapshot.step}: {snapshot.description}"
            print(f"[DEBUG] Sending photo to Telegram: {snapshot.step} ({len(png_bytes)} bytes)")
            logging.info(f"Sending snapshot {snapshot.step} ({len(png_bytes)} bytes)")
            await self.bot.send_photo(
                chat_id=chat_id,
                photo=BufferedInputFile(png_bytes, filename=f"{snapshot.step}.png"),
                caption=caption,
            )
            print(f"[DEBUG] Successfully sent snapshot {snapshot.step} to Telegram")
            logging.info(f"Successfully sent snapshot {snapshot.step}")
        except Exception as exc:
            print(f"[ERROR] Exception sending snapshot {snapshot.step}: {exc}")
            logging.exception(f"Error sending snapshot {snapshot.step}: {exc}")


def detect_mime_type(message: Message, file_path: str) -> str:
    if message.document and message.document.mime_type:
        return message.document.mime_type
    guessed, _ = mimetypes.guess_type(file_path)
    return guessed or "image/jpeg"


STATEMENT_EXTENSIONS = (".csv", ".tsv", ".xls", ".xlsx", ".pdf")
STATEMENT_MIME_PREFIXES = (
    "text/csv",
    "text/tab-separated-values",
    "application/pdf",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)


def classify_upload_kind(message: Message) -> Optional[str]:
    if message.photo:
        return "receipt"
    document = message.document
    if not document:
        return None
    mime_type = (document.mime_type or "").lower()
    file_name = (document.file_name or "").lower()
    if mime_type.startswith("image/"):
        return "receipt"
    if any(file_name.endswith(ext) for ext in STATEMENT_EXTENSIONS):
        return "statement"
    if mime_type in STATEMENT_MIME_PREFIXES:
        return "statement"
    caption = (message.caption or "").lower()
    if "выпис" in caption or "statement" in caption:
        return "statement"
    if "чек" in caption or "receipt" in caption:
        return "receipt"
    return None


def format_receipt_summary(parsed: ParsedReceipt) -> str:
    lines = [
        "Вот что удалось извлечь:",
        f"• Магазин: {parsed.store}",
        f"• Дата: {parsed.purchased_at.strftime('%Y-%m-%d %H:%M')}",
        f"• Сумма: {parsed.total:.2f} {parsed.currency}",
    ]
    if parsed.tax_amount is not None:
        lines.append(f"• Налог: {parsed.tax_amount:.2f} {parsed.currency}")
    if parsed.merchant_address:
        lines.append(f"• Адрес: {parsed.merchant_address}")
    if parsed.items:
        lines.append("• Позиции:")
        for item in parsed.items[:10]:
            lines.append(
                f"   - {item.name} ×{item.quantity:g} = {item.price:.2f} {parsed.currency}"
            )
        if len(parsed.items) > 10:
            lines.append(f"   … и ещё {len(parsed.items) - 10} позиций")
    return "\n".join(lines)


def format_receipt_table(parsed: ParsedReceipt) -> str:
    """
    Форматирует чек в виде таблицы: название, количество, общая цена позиции, итог.
    """
    lines = []
    
    # Заголовок с магазином и датой
    if parsed.store:
        lines.append(f"🏪 {parsed.store}")
    if parsed.purchased_at:
        lines.append(f"📅 {parsed.purchased_at.strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    
    # Фиксированная компактная ширина колонок
    name_width = 25  # Название товара (обрезаем до 25 символов)
    qty_width = 6    # Количество
    price_width = 12 # Сумма
    
    # Заголовок таблицы
    lines.append(f"{'Товар':<{name_width}} {'Кол-во':>{qty_width}} {'Сумма':>{price_width}}")
    lines.append("-" * (name_width + qty_width + price_width + 4))
    
    # Позиции
    for item in parsed.items:
        # Обрезаем название если слишком длинное
        name = item.name[:25] if len(item.name) > 25 else item.name
        quantity = item.quantity
        total_price = item.price
        
        # Исправляем перепутанные данные: если quantity выглядит как цена за единицу
        # Если quantity близко к total_price (в пределах 5%), то количество = 1
        if quantity > 0 and total_price > 0:
            if abs(quantity - total_price) / max(quantity, total_price) < 0.05:
                # quantity и price почти равны, значит количество = 1
                quantity = 1.0
            else:
                # Пробуем вычислить количество: price / quantity
                calculated_qty = total_price / quantity
                # Если получилось разумное количество (от 0.5 до 100) и близко к целому
                if 0.5 <= calculated_qty <= 100:
                    rounded_qty = round(calculated_qty)
                    # Если округленное значение близко к вычисленному, используем его
                    if abs(calculated_qty - rounded_qty) < 0.1:
                        quantity = float(rounded_qty)
                    else:
                        quantity = calculated_qty
                # Если quantity больше total_price, точно перепутано
                elif quantity > total_price and calculated_qty >= 0.5:
                    quantity = round(calculated_qty) if calculated_qty <= 100 else calculated_qty
        
        # Форматируем строку таблицы
        # Используем целое число для quantity, если оно целое
        if quantity == int(quantity):
            qty_str = f"{int(quantity)}"
        else:
            qty_str = f"{quantity:g}"
        price_str = f"{total_price:.2f}"
        
        # Выравниваем с фиксированными позициями
        lines.append(f"{name:<{name_width}} {qty_str:>{qty_width}} {price_str:>{price_width}}")
    
    # Итоговая строка
    lines.append("-" * (name_width + qty_width + price_width + 4))
    total_str = f"{parsed.total:.2f} {parsed.currency}"
    lines.append(f"{'ИТОГО':<{name_width}} {'':>{qty_width}} {total_str:>{price_width}}")
    
    # Добавляем информацию о количестве товаров
    items_count = len(parsed.items)
    if items_count > 0:
        items_sum = sum(item.price for item in parsed.items)
        lines.append("")
        lines.append(f"Всего товаров: {items_count}")
        lines.append(f"Сумма всех позиций: {items_sum:.2f} {parsed.currency}")
    
    # Оборачиваем в код-блок для моноширинного шрифта
    table_text = "\n".join(lines)
    return f"```\n{table_text}\n```"


def generate_receipt_image(parsed: ParsedReceipt) -> Optional[bytes]:
    """
    Генерирует изображение с таблицей чека.
    Возвращает bytes изображения в формате PNG или None если PIL недоступен.
    """
    if Image is None or ImageDraw is None:
        return None
    
    try:
        # Параметры изображения
        padding = 40
        line_height = 35
        font_size = 24
        header_font_size = 28
        title_font_size = 32
        
        # Пытаемся загрузить шрифт с поддержкой казахских символов
        # Приоритет: DejaVu Sans (лучшая поддержка Unicode) > Liberation Sans > системные шрифты > стандартный
        font_paths = [
            # Linux шрифты (для Railway/Docker)
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            # macOS шрифты
            "/System/Library/Fonts/Helvetica.ttc",
            "/System/Library/Fonts/Arial.ttf",
            # Windows шрифты (если используется)
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
        ]
        
        title_font = None
        header_font = None
        font = None
        
        # Пытаемся найти шрифт с поддержкой Unicode
        for font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    title_font = ImageFont.truetype(font_path, title_font_size)
                    header_font = ImageFont.truetype(font_path, header_font_size)
                    font = ImageFont.truetype(font_path, font_size)
                    logging.info(f"Using font: {font_path}")
                    break
            except Exception as e:
                logging.debug(f"Failed to load font {font_path}: {e}")
                continue
        
        # Если не нашли подходящий шрифт, используем стандартный
        if title_font is None:
            logging.warning("No suitable font found, using default font (may not support Kazakh characters)")
            title_font = ImageFont.load_default()
            header_font = ImageFont.load_default()
            font = ImageFont.load_default()
        
        # Подготавливаем данные для таблицы (без эмодзи для изображения)
        store_text = parsed.store if parsed.store else ""
        date_text = parsed.purchased_at.strftime('%Y-%m-%d %H:%M') if parsed.purchased_at else ""
        
        # Обрабатываем товары
        items_data = []
        for item in parsed.items:
            name = item.name[:30] if len(item.name) > 30 else item.name
            quantity = item.quantity
            total_price = item.price
            
            # Исправляем перепутанные данные (та же логика что и в format_receipt_table)
            if quantity > 0 and total_price > 0:
                if abs(quantity - total_price) / max(quantity, total_price) < 0.05:
                    quantity = 1.0
                else:
                    calculated_qty = total_price / quantity
                    if 0.5 <= calculated_qty <= 100:
                        rounded_qty = round(calculated_qty)
                        if abs(calculated_qty - rounded_qty) < 0.1:
                            quantity = float(rounded_qty)
                        else:
                            quantity = calculated_qty
                    elif quantity > total_price and calculated_qty >= 0.5:
                        quantity = round(calculated_qty) if calculated_qty <= 100 else calculated_qty
            
            qty_str = f"{int(quantity)}" if quantity == int(quantity) else f"{quantity:g}"
            price_str = f"{total_price:.2f}"
            
            items_data.append({
                "name": name,
                "qty": qty_str,
                "price": price_str
            })
        
        # Создаем временное изображение для вычисления размеров текста
        temp_img = Image.new('RGB', (2000, 100), color='white')
        temp_draw = ImageDraw.Draw(temp_img)
        
        # Вычисляем максимальную ширину названий товаров
        max_name_width_px = 0
        for item in items_data:
            bbox = temp_draw.textbbox((0, 0), item["name"], font=font)
            name_width_px = bbox[2] - bbox[0]
            max_name_width_px = max(max_name_width_px, name_width_px)
        
        # Ширина колонок в пикселях (ориентируемся на таблицу)
        name_col_width = max(max_name_width_px, 300)  # Минимум 300px для названия
        qty_col_width = 100  # Увеличено для количества
        price_col_width = 150
        
        # Ширина таблицы определяет ширину всего изображения
        table_width = name_col_width + qty_col_width + price_col_width + 80  # 80px для отступов между колонками
        total_width = table_width + padding * 2
        
        # Разбиваем название магазина на строки, если оно не помещается
        store_lines = []
        if store_text:
            max_store_width = total_width - padding * 2  # Доступная ширина для текста
            words = store_text.split()
            current_line = ""
            
            for word in words:
                test_line = current_line + (" " if current_line else "") + word
                bbox = temp_draw.textbbox((0, 0), test_line, font=title_font)
                test_width = bbox[2] - bbox[0]
                
                if test_width <= max_store_width:
                    current_line = test_line
                else:
                    if current_line:
                        store_lines.append(current_line)
                    current_line = word
            
            if current_line:
                store_lines.append(current_line)
        else:
            store_lines = []
        
        # Вычисляем высоту
        header_lines = len(store_lines)  # Название магазина может быть в несколько строк
        if date_text:
            header_lines += 1
        header_lines += 1  # пустая строка
        
        table_lines = 2 + len(items_data) + 1  # заголовок + разделитель + строки + итог
        total_height = padding * 2 + header_lines * line_height + table_lines * line_height
        
        # Создаем реальное изображение
        img = Image.new('RGB', (total_width, total_height), color='white')
        draw = ImageDraw.Draw(img)
        
        y = padding
        x = padding
        
        # Рисуем заголовок магазина (может быть в несколько строк)
        for store_line in store_lines:
            draw.text((x, y), store_line, fill='black', font=title_font)
            y += line_height
        
        # Рисуем дату
        if date_text:
            draw.text((x, y), date_text, fill='black', font=header_font)
            y += line_height
        
        y += line_height // 2  # пустая строка
        
        # Вычисляем позиции колонок
        name_col_x = x
        qty_col_x = x + name_col_width + 40  # 40px отступ между колонками
        price_col_x = qty_col_x + qty_col_width + 40
        
        # Рисуем заголовок таблицы
        draw.text((name_col_x, y), "Товар", fill='black', font=header_font)
        draw.text((qty_col_x, y), "Кол-во", fill='black', font=header_font)
        draw.text((price_col_x, y), "Сумма", fill='black', font=header_font)
        y += line_height
        
        # Разделитель
        draw.line([(x, y), (total_width - padding, y)], fill='gray', width=2)
        y += line_height
        
        # Рисуем товары с правильным выравниванием
        for item in items_data:
            # Название товара (слева)
            draw.text((name_col_x, y), item["name"], fill='black', font=font)
            # Количество (по центру своей колонки)
            qty_bbox = draw.textbbox((0, 0), item["qty"], font=font)
            qty_text_width = qty_bbox[2] - qty_bbox[0]
            qty_x = qty_col_x + (qty_col_width - qty_text_width) // 2  # Центрируем
            draw.text((qty_x, y), item["qty"], fill='black', font=font)
            # Цена (справа в своей колонке)
            price_bbox = draw.textbbox((0, 0), item["price"], font=font)
            price_text_width = price_bbox[2] - price_bbox[0]
            price_x = price_col_x + (price_col_width - price_text_width)  # Выравниваем справа
            draw.text((price_x, y), item["price"], fill='black', font=font)
            y += line_height
        
        # Разделитель перед итогом
        draw.line([(x, y), (total_width - padding, y)], fill='gray', width=2)
        y += line_height
        
        # Итог
        total_str = f"{parsed.total:.2f} {parsed.currency}"
        draw.text((name_col_x, y), "ИТОГО", fill='black', font=header_font)
        # Цена итога выравниваем справа
        total_bbox = draw.textbbox((0, 0), total_str, font=header_font)
        total_text_width = total_bbox[2] - total_bbox[0]
        total_x = price_col_x + (price_col_width - total_text_width)  # Выравниваем справа
        draw.text((total_x, y), total_str, fill='black', font=header_font)
        
        # Сохраняем в bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes.getvalue()
        
    except Exception as exc:
        logging.exception(f"Ошибка при генерации изображения чека: {exc}")
        return None


def format_report(report: Dict[str, Any], currency: Optional[str] = None) -> str:
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
            # Если валюта не найдена, возвращаем пустой отчет
            logging.info(f"format_report: currency {currency} not found in report")
            return f"📊 Отчёт за {period}\n\n💰 Нет данных для валюты {currency} за выбранный период."
    
    if not currencies_data:
        logging.info(f"format_report: no currencies data for period {period}")
        return f"📊 Отчёт за {period}\n\n💰 Всего расходов: 0.00\n\nНет данных за выбранный период."
    
    # Форматируем период для отображения
    display_period = period
    if " - " in period:
        # Произвольный период (YYYY-MM-DD - YYYY-MM-DD)
        try:
            start_str, end_str = period.split(" - ")
            start_date = datetime.strptime(start_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_str, "%Y-%m-%d")
            display_period = f"{start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}"
        except:
            pass
    elif len(period) == 7 and period[4] == "-":
        # Месяц (YYYY-MM)
        try:
            date_obj = datetime.strptime(period, "%Y-%m")
            # Получаем название месяца на русском
            months = ["январь", "февраль", "март", "апрель", "май", "июнь",
                     "июль", "август", "сентябрь", "октябрь", "ноябрь", "декабрь"]
            month_name = months[date_obj.month - 1]
            display_period = f"{month_name} {date_obj.year}"
        except:
            pass
    
    lines = [f"📊 Отчёт за {display_period}"]
    
    # Символы валют
    currency_symbols = {
        "RUB": "₽",
        "KZT": "₸",
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "GEL": "₾",
    }
    
    # Всего расходов - всегда показываем по каждой валюте отдельно
    lines.append("💰 Всего расходов:")
    if currencies_data:
        for currency in sorted(currencies_data.keys()):
            currency_info = currencies_data[currency]
            total = currency_info.get("total", 0.0)
            symbol = currency_symbols.get(currency, currency)
            lines.append(f"  {symbol} {total:.2f}")
    else:
        lines.append("  0.00")
    lines.append("")
    
    # Топы по каждой валюте отдельно
    # Итерируемся по всем валютам из топов (может быть больше чем в currencies_data)
    most_expensive_by_currency = report.get("most_expensive_by_currency", {})
    
    # Объединяем валюты из currencies_data и most_expensive_by_currency
    all_currencies = set(currencies_data.keys()) | set(most_expensive_by_currency.keys())
    
    for currency in sorted(all_currencies):
        currency_tops = most_expensive_by_currency.get(currency, {})
        symbol = currency_symbols.get(currency, currency)
        
        # Самая дорогая покупка для этой валюты
        item_info = currency_tops.get("item", {})
        item_date = ""  # Инициализируем переменную заранее
        if item_info.get("name") and item_info.get("price", 0) > 0:
            item_name = item_info.get("name", "Неизвестно")
            item_price = item_info.get("price", 0.0)
            item_store = item_info.get("store") or "Без названия"
            item_date = item_info.get("date", "")
            
            # Форматируем дату
            date_str = ""
            if item_date:
                try:
                    if "T" in item_date:
                        date_obj = datetime.fromisoformat(item_date.replace("Z", "+00:00"))
                    else:
                        date_obj = datetime.strptime(item_date[:10], "%Y-%m-%d")
                    date_str = date_obj.strftime("%d.%m.%Y")
                except:
                    date_str = item_date[:10] if item_date and len(item_date) >= 10 else item_date
            
            store_name = item_store[:30] if item_store and len(item_store) > 30 else (item_store or "Без названия")
            
            # Показываем заголовок только если несколько валют
            if len(currencies_data) > 1:
                lines.append(f"💎 Самая дорогая покупка ({symbol}):")
            else:
                lines.append("💎 Самая дорогая покупка:")
            
            if date_str:
                lines.append(f"  {item_name} - {item_price:.2f} {symbol} ({store_name}, {date_str})")
            else:
                lines.append(f"  {item_name} - {item_price:.2f} {symbol} ({store_name})")
            lines.append("")
    
        # Самый дорогой расход для этой валюты
        expense_info = currency_tops.get("expense", {})
        exp_date = ""  # Инициализируем переменную заранее
        if expense_info.get("amount", 0) > 0:
            exp_amount = expense_info.get("amount", 0.0)
            exp_store = expense_info.get("store") or "Без названия"
            exp_date = expense_info.get("date", "")
            
            # Форматируем дату
            date_str = ""
            if exp_date:
                try:
                    if "T" in exp_date:
                        date_obj = datetime.fromisoformat(exp_date.replace("Z", "+00:00"))
                    else:
                        date_obj = datetime.strptime(exp_date[:10], "%Y-%m-%d")
                    date_str = date_obj.strftime("%d.%m.%Y")
                except:
                    date_str = exp_date[:10] if exp_date and len(exp_date) >= 10 else exp_date
            
            store_name = exp_store[:30] if exp_store and len(exp_store) > 30 else (exp_store or "Без названия")
            
            # Показываем заголовок только если несколько валют
            if len(currencies_data) > 1:
                lines.append(f"💸 Самый дорогой расход ({symbol}):")
            else:
                lines.append("💸 Самый дорогой расход:")
            
            if date_str:
                lines.append(f"  {exp_amount:.2f} {symbol} - {store_name} ({date_str})")
            else:
                lines.append(f"  {exp_amount:.2f} {symbol} - {store_name}")
            lines.append("")
    
    # Формируем отчет по каждой валюте отдельно
    for currency in sorted(currencies_data.keys()):
        currency_info = currencies_data[currency]
        total = currency_info.get("total", 0.0)
        by_category = currency_info.get("by_category", {})
        by_store = currency_info.get("by_store", {})
        symbol = currency_symbols.get(currency, currency)
        
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


def format_statement_summary(transactions: List[ParsedBankTransaction]) -> str:
    totals: Dict[str, float] = {}
    for txn in transactions:
        totals[txn.currency] = totals.get(txn.currency, 0.0) + txn.amount
    lines = [
        "Выписка обработана.",
        f"Операций найдено: {len(transactions)}",
        "Итоги по валютам:",
    ]
    for currency, total in totals.items():
        lines.append(f"• {currency}: {total:.2f}")
    lines.append("Операции:")
    for txn in transactions:
        lines.append(
            f"   - {txn.booked_at.strftime('%Y-%m-%d')} {txn.merchant} "
            f"{txn.amount:.2f} {txn.currency}"
        )
    return "\n".join(lines)


def convert_heic_if_needed(file_bytes: bytes, mime_type: str) -> tuple[bytes, str]:
    normalized = (mime_type or "").lower()
    if normalized not in ("image/heic", "image/heif"):
        return file_bytes, mime_type
    if not HEIF_SUPPORT or read_heif is None or Image is None:
        raise ReceiptParsingError(
            "Формат HEIC не поддерживается: установите pillow-heif или отправьте фото как JPG."
        )
    heif_file = read_heif(file_bytes)
    image = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue(), "image/jpeg"


def preprocess_image_for_openai(file_bytes: bytes) -> tuple[bytes, str]:
    """
    Возвращает оригинальное изображение без обработки.
    Возвращает: (оригинальные байты, mime_type)
    """
    # Определяем mime_type по содержимому файла
    if Image is not None:
        try:
            image = Image.open(io.BytesIO(file_bytes))
            mime_type = f"image/{image.format.lower()}" if image.format else "image/jpeg"
            logging.info(f"Original image: {len(file_bytes)} bytes, format: {mime_type}")
            return file_bytes, mime_type
        except Exception:
            pass
    
    # Если не удалось определить формат, возвращаем как JPEG
    logging.info(f"Original image: {len(file_bytes)} bytes, format: image/jpeg (default)")
    return file_bytes, "image/jpeg"


def _find_corners_by_brightness(image: np.ndarray, gray: np.ndarray, image_bgr: np.ndarray) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Метод 1: Поиск углов через анализ яркости и контраста.
    Ищем переходы от темного фона к светлому чеку.
    """
    h, w = gray.shape[:2]
    
    # Вычисляем среднюю яркость по строкам и столбцам
    mean_h = np.mean(gray, axis=1)  # Средняя яркость по строкам
    mean_w = np.mean(gray, axis=0)  # Средняя яркость по столбцам
    
    # Вычисляем градиенты (изменение яркости) - ищем резкие переходы
    grad_h = np.abs(np.diff(mean_h))  # Изменение яркости между соседними строками
    grad_w = np.abs(np.diff(mean_w))  # Изменение яркости между соседними столбцами
    
    # Находим пороги для градиентов (резкие переходы)
    grad_h_threshold = np.percentile(grad_h, 90)  # Верхние 10% самых резких переходов
    grad_w_threshold = np.percentile(grad_w, 90)
    
    logging.info(f"[BRIGHTNESS] Gradient thresholds: h={grad_h_threshold:.1f}, w={grad_w_threshold:.1f}")
    logging.info(f"[BRIGHTNESS] Brightness range: h=[{np.min(mean_h):.1f}, {np.max(mean_h):.1f}], w=[{np.min(mean_w):.1f}, {np.max(mean_w):.1f}]")
    
    # Находим индексы, где есть резкие переходы (края чека)
    # Ищем переходы от темного к светлому (положительный градиент)
    grad_h_pos = np.diff(mean_h)  # Положительный = переход к светлому
    grad_w_pos = np.diff(mean_w)
    
    # Находим области с высокой яркостью (чек может быть белым или желтоватым)
    # Используем более низкий порог для учета разных цветов чеков
    brightness_threshold_h = np.percentile(mean_h, 55)  # 55-й перцентиль - более низкий порог
    brightness_threshold_w = np.percentile(mean_w, 55)
    
    # Также вычисляем медиану для более устойчивого определения ярких областей
    median_h = np.median(mean_h)
    median_w = np.median(mean_w)
    
    # Используем максимум из перцентиля и медианы для более надежного определения
    brightness_threshold_h = max(brightness_threshold_h, median_h * 0.8)
    brightness_threshold_w = max(brightness_threshold_w, median_w * 0.8)
    
    # ШАГ 1: Сначала находим вертикальные границы (top/bottom) по всей ширине
    # Верхняя граница: первая строка, где яркость резко увеличивается И остается высокой
    # Или где яркость выше порога и предыдущая строка была темнее
    top = None
    for i in range(h - 1):
        # Проверяем резкий переход к светлому
        if grad_h_pos[i] > grad_h_threshold and mean_h[i+1] > brightness_threshold_h:
            top = i
            break
        # Или проверяем переход от темного к светлому
        elif i > 0 and mean_h[i] > brightness_threshold_h and mean_h[i-1] < brightness_threshold_h * 0.7:
            top = i
            break
    
    # Нижняя граница: ищем переход от светлого чека к темному фону
    # Ищем от низа к верху, находим последнюю строку с высокой яркостью,
    # затем ищем первую строку, где яркость резко падает (конец чека)
    bottom = None
    # Сначала находим последнюю яркую строку
    last_bright_row = None
    for i in range(h - 1, top if top is not None else 0, -1):
        if mean_h[i] > brightness_threshold_h:
            last_bright_row = i
            break
    
    # Если нашли яркую строку, ищем переход к темному (конец чека)
    if last_bright_row is not None:
        # Ищем от последней яркой строки дальше вниз, где яркость резко падает
        for i in range(last_bright_row, min(h - 1, last_bright_row + 50), 1):
            if i < h - 1:
                # Если яркость резко упала (переход к темному фону)
                if mean_h[i] < brightness_threshold_h * 0.7:
                    bottom = i - 1  # Берем предыдущую строку (последняя строка чека)
                    break
        # Если не нашли переход, используем последнюю яркую строку
        if bottom is None:
            bottom = last_bright_row
    
    logging.info(f"[BRIGHTNESS] Step 1 - Vertical boundaries: top={top}, bottom={bottom}")
    
    # ШАГ 2: В найденной вертикальной области ищем горизонтальные границы (left/right)
    # Это более точно, так как мы ищем только в области, где есть чек
    if top is not None and bottom is not None and bottom > top:
        # Вычисляем среднюю яркость по столбцам ТОЛЬКО в вертикальной области чека
        receipt_region = gray[top:bottom+1, :]
        mean_w_receipt = np.mean(receipt_region, axis=0)  # Средняя яркость по столбцам в области чека
        
        # Вычисляем градиенты для этой области
        grad_w_receipt = np.abs(np.diff(mean_w_receipt))
        grad_w_threshold_receipt = np.percentile(grad_w_receipt, 90)
        grad_w_pos_receipt = np.diff(mean_w_receipt)
        
        # Используем медиану как порог - более устойчиво к выбросам
        # Чек должен занимать большую часть, поэтому ищем столбцы ярче медианы
        median_w_receipt = np.median(mean_w_receipt)
        brightness_threshold_w_receipt = median_w_receipt * 0.9  # 90% от медианы
        
        # Также используем более строгий порог для градиентов
        grad_w_threshold_receipt = np.percentile(grad_w_receipt, 85)  # 85-й перцентиль
        
        logging.info(f"[BRIGHTNESS] Receipt region brightness range: [{np.min(mean_w_receipt):.1f}, {np.max(mean_w_receipt):.1f}], median={median_w_receipt:.1f}")
        logging.info(f"[BRIGHTNESS] Receipt region threshold: {brightness_threshold_w_receipt:.1f} (90% of median)")
        
        # Левая граница: ищем первый столбец, где начинается чек
        # Ищем от самого левого края (0) к правому, чтобы найти первый яркий столбец
        left = None
        
        # Ищем от левого края (0) к правому
        # Используем более строгий подход: ищем переход от темного фона к светлому чеку
        for i in range(w - 1):
            # Проверяем резкий переход к светлому (градиент)
            if grad_w_pos_receipt[i] > grad_w_threshold_receipt:
                # Если следующий столбец яркий, а предыдущий темный - это начало чека
                if i > 0 and mean_w_receipt[i+1] > brightness_threshold_w_receipt and mean_w_receipt[i-1] < brightness_threshold_w_receipt * 0.7:
                    left = i
                    logging.info(f"[BRIGHTNESS] Found left boundary at {i} (gradient transition)")
                    break
                # Или если текущий столбец уже яркий, а предыдущий был темным
                elif i > 0 and mean_w_receipt[i] > brightness_threshold_w_receipt and mean_w_receipt[i-1] < brightness_threshold_w_receipt * 0.7:
                    left = i
                    logging.info(f"[BRIGHTNESS] Found left boundary at {i} (brightness transition)")
                    break
        
        # Если не нашли через градиенты, ищем просто первый яркий столбец от левого края
        if left is None:
            for i in range(w):
                if mean_w_receipt[i] > brightness_threshold_w_receipt:
                    left = i
                    logging.info(f"[BRIGHTNESS] Found left boundary at {i} (first bright column)")
                    break
        
        # Правая граница: ищем переход от светлого чека к темному фону
        # Ищем от правого края к левому, находим последний яркий столбец
        right = None
        last_bright_col = None
        
        # Ищем от правого края к левому, находим последний яркий столбец
        for i in range(w - 1, left if left is not None else 0, -1):
            if mean_w_receipt[i] > brightness_threshold_w_receipt:
                last_bright_col = i
                break
        
        # Если нашли яркий столбец, ищем переход к темному (правый край чека)
        if last_bright_col is not None:
            # Ищем от последнего яркого столбца дальше вправо, где яркость резко падает
            for i in range(last_bright_col, min(w - 1, last_bright_col + 50), 1):
                if i < w - 1:
                    # Если яркость резко упала (переход к темному фону)
                    if mean_w_receipt[i] < brightness_threshold_w_receipt * 0.7:
                        right = i - 1  # Берем предыдущий столбец (последний столбец чека)
                        logging.info(f"[BRIGHTNESS] Found right boundary at {right} (transition to dark)")
                        break
            # Если не нашли переход, используем последний яркий столбец
            if right is None:
                right = last_bright_col
                logging.info(f"[BRIGHTNESS] Found right boundary at {right} (last bright column)")
        
        # Если все еще не нашли, используем исходный метод
        if right is None:
            for i in range(w - 1, left if left is not None else 0, -1):
                if mean_w_receipt[i] > brightness_threshold_w_receipt:
                    right = i
                    logging.info(f"[BRIGHTNESS] Found right boundary at {right} (fallback)")
                    break
        
        logging.info(f"[BRIGHTNESS] Step 2 - Horizontal boundaries in receipt region: left={left}, right={right}")
    else:
        # Fallback: если не нашли вертикальные границы, используем старый метод
        logging.warning("[BRIGHTNESS] Could not find vertical boundaries, using full-width search")
        left = None
        for i in range(w - 1):
            if grad_w_pos[i] > grad_w_threshold and mean_w[i+1] > brightness_threshold_w:
                left = i
                break
        
        right = None
        for i in range(w - 1, 0, -1):
            if mean_w[i] > brightness_threshold_w:
                right = i
                break
    
    logging.info(f"[BRIGHTNESS] Found boundaries: top={top}, bottom={bottom}, left={left}, right={right}")
    
    if top is not None and bottom is not None and left is not None and right is not None:
        # Проверяем валидность границ
        if bottom > top and right > left:
                area_h = bottom - top
                area_w = right - left
                area_percent = (area_h * area_w) / (h * w) * 100
                
                logging.info(f"[BRIGHTNESS] Area size: {area_w}x{area_h} ({area_percent:.1f}% of image)")
                
                # Чек должен занимать минимум 45% изображения (немного снизили для учета погрешностей)
                min_area_percent = 45
                max_area_percent = 95
                
                # Сохраняем исходные значения
                left_original = left
                right_original = right
                area_w_original = area_w
                area_percent_original = area_percent
                
                # Если область слишком мала, пробуем более мягкие пороги
                if area_percent < min_area_percent:
                    logging.warning(f"[BRIGHTNESS] Area too small ({area_percent:.1f}%), trying softer thresholds")
                    
                    # Используем более мягкие пороги для поиска границ
                    brightness_threshold_w_receipt_soft = np.percentile(mean_w_receipt, 50)  # 50-й перцентиль
                    grad_w_threshold_receipt_soft = np.percentile(grad_w_receipt, 80)  # 80-й перцентиль
                    
                    # Пересчитываем левую границу
                    left_soft = None
                    for i in range(w - 1):
                        if grad_w_pos_receipt[i] > grad_w_threshold_receipt_soft and mean_w_receipt[i+1] > brightness_threshold_w_receipt_soft:
                            left_soft = i
                            break
                    
                    # Пересчитываем правую границу
                    right_soft = None
                    for i in range(w - 1, 0, -1):
                        if mean_w_receipt[i] > brightness_threshold_w_receipt_soft:
                            right_soft = i
                            break
                    
                    if left_soft is not None and right_soft is not None and right_soft > left_soft:
                        area_w_soft = right_soft - left_soft
                        area_percent_soft = (area_h * area_w_soft) / (h * w) * 100
                        logging.info(f"[BRIGHTNESS] Soft thresholds: left={left_soft}, right={right_soft}, area={area_percent_soft:.1f}%")
                        
                        # Используем soft thresholds, если они лучше исходных или в допустимых пределах
                        logging.info(f"[BRIGHTNESS] Comparing: soft={area_percent_soft:.1f}% vs original={area_percent_original:.1f}%, min={min_area_percent}%, max={max_area_percent}%")
                        if area_percent_soft > area_percent_original and area_percent_soft <= max_area_percent:
                            left = left_soft
                            right = right_soft
                            area_w = area_w_soft
                            area_percent = area_percent_soft
                            logging.info(f"[BRIGHTNESS] ✓ Using soft thresholds (better): area={area_percent:.1f}% vs original {area_percent_original:.1f}%")
                        elif min_area_percent <= area_percent_soft <= max_area_percent:
                            left = left_soft
                            right = right_soft
                            area_w = area_w_soft
                            area_percent = area_percent_soft
                            logging.info(f"[BRIGHTNESS] ✓ Using soft thresholds (in range): area={area_percent:.1f}%")
                        else:
                            logging.warning(f"[BRIGHTNESS] ✗ Soft thresholds rejected: {area_percent_soft:.1f}% (better check: {area_percent_soft > area_percent_original}, in range: {min_area_percent <= area_percent_soft <= max_area_percent})")
                
                # Пересчитываем area_percent на случай, если значения изменились
                area_w = right - left
                area_percent = (area_h * area_w) / (h * w) * 100
                
                # Проверяем, что область в допустимых пределах
                logging.info(f"[BRIGHTNESS] Final check: area={area_w}x{area_h} ({area_percent:.1f}%), min={min_area_percent}%, max={max_area_percent}%")
                if min_area_percent <= area_percent <= max_area_percent and area_h > 50 and area_w > 50:
                    # Создаем углы
                    box = np.array([
                        [left, top],
                        [left, bottom],
                        [right, bottom],
                        [right, top]
                    ], dtype=np.float32)
                    logging.info(f"[BRIGHTNESS] Found corners: {box}")
                    
                    # Создаем визуализацию
                    result_image = image_bgr.copy()
                    
                    # 1. Рисуем графики яркости по краям
                    max_h = np.max(mean_h) if np.max(mean_h) > 0 else 1
                    max_w = np.max(mean_w) if np.max(mean_w) > 0 else 1
                    
                    # График яркости слева (вертикальный)
                    graph_width = min(100, w // 10)
                    for y_idx in range(h):
                        bar_height = int((mean_h[y_idx] / max_h) * graph_width)
                        color = (0, 255, 0) if mean_h[y_idx] > brightness_threshold_h else (0, 0, 255)
                        cv2.line(result_image, (0, y_idx), (bar_height, y_idx), color, 1)
                    
                    # График яркости сверху (горизонтальный)
                    graph_height = min(100, h // 10)
                    for x_idx in range(w):
                        bar_width = int((mean_w[x_idx] / max_w) * graph_height)
                        color = (0, 255, 0) if mean_w[x_idx] > brightness_threshold_w else (0, 0, 255)
                        cv2.line(result_image, (x_idx, 0), (x_idx, bar_width), color, 1)
                    
                    # 2. Рисуем найденные границы
                    # Верхняя граница
                    if top is not None:
                        cv2.line(result_image, (0, top), (w, top), (255, 0, 255), 2)  # Пурпурный
                        cv2.putText(result_image, f"TOP:{top}", (10, top - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    
                    # Нижняя граница
                    if bottom is not None:
                        cv2.line(result_image, (0, bottom), (w, bottom), (255, 0, 255), 2)
                        cv2.putText(result_image, f"BOTTOM:{bottom}", (10, bottom + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    
                    # Левая граница
                    if left is not None:
                        cv2.line(result_image, (left, 0), (left, h), (255, 0, 255), 2)
                        cv2.putText(result_image, f"LEFT:{left}", (left + 5, 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    
                    # Правая граница
                    if right is not None:
                        cv2.line(result_image, (right, 0), (right, h), (255, 0, 255), 2)
                        cv2.putText(result_image, f"RIGHT:{right}", (right - 80, 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    
                    # 3. Выделяем найденную область
                    overlay = result_image.copy()
                    cv2.rectangle(overlay, (left, top), (right, bottom), (0, 255, 255), -1)  # Желтый
                    cv2.addWeighted(overlay, 0.3, result_image, 0.7, 0, result_image)
                    
                    # 4. Рисуем рамку
                    cv2.rectangle(result_image, (left, top), (right, bottom), (0, 255, 255), 3)
                    
                    # 5. Информация
                    info_text = f"BRIGHTNESS: {area_w}x{area_h} ({area_percent:.1f}%)"
                    cv2.putText(result_image, info_text, (left, top - 15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # 6. Добавляем информацию о порогах
                    threshold_text = f"h_th={brightness_threshold_h:.0f}, w_th={brightness_threshold_w_receipt:.0f}"
                    cv2.putText(result_image, threshold_text, (left, bottom + 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    return box, result_image
                else:
                    logging.warning(f"[BRIGHTNESS] Area too large or too small: {area_percent:.1f}%")
                    # Все равно создаем визуализацию
                    result_image = image_bgr.copy()
                    if top is not None and bottom is not None and left is not None and right is not None:
                        # Рисуем найденные границы
                        if top is not None:
                            cv2.line(result_image, (0, top), (w, top), (255, 0, 255), 2)
                            cv2.putText(result_image, f"TOP:{top}", (10, top - 5), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                        if bottom is not None:
                            cv2.line(result_image, (0, bottom), (w, bottom), (255, 0, 255), 2)
                            cv2.putText(result_image, f"BOTTOM:{bottom}", (10, bottom + 20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                        if left is not None:
                            cv2.line(result_image, (left, 0), (left, h), (255, 0, 255), 2)
                            cv2.putText(result_image, f"LEFT:{left}", (left + 5, 20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                        if right is not None:
                            cv2.line(result_image, (right, 0), (right, h), (255, 0, 255), 2)
                            cv2.putText(result_image, f"RIGHT:{right}", (right - 80, 20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                        info_text = f"BRIGHTNESS: Area {area_percent:.1f}% (invalid)"
                        cv2.putText(result_image, info_text, (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    return None, result_image
        else:
            logging.warning(f"[BRIGHTNESS] Invalid boundaries: top={top}, bottom={bottom}, left={left}, right={right}")
    else:
        logging.warning(f"[BRIGHTNESS] Missing boundaries: top={top}, bottom={bottom}, left={left}, right={right}")
    
    logging.warning("[BRIGHTNESS] Failed to find corners")
    # Все равно возвращаем изображение с визуализацией
    result_image = image_bgr.copy()
    info_text = "BRIGHTNESS: No corners found"
    cv2.putText(result_image, info_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return None, result_image


def _find_corners_by_contour(image: np.ndarray, gray: np.ndarray, image_bgr: np.ndarray) -> tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Метод 2: Поиск углов через контуры.
    Возвращает (box, largest_contour, result_image) или (None, None, None) если не найдено.
    """
    h, w = gray.shape[:2]
    
    # Пробуем несколько методов поиска контуров
    all_contours = []
    
    # Метод 1: Адаптивная бинаризация для светлых областей (чеки обычно светлые)
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY_INV, 11, 2)
    
    # Метод 2: Пороговая бинаризация с несколькими порогами
    _, binary_white1 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    _, binary_white2 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    _, binary_white3 = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)
    
    # Метод 3: Canny для поиска краев
    edges = cv2.Canny(gray, 50, 150)
    
    # Метод 4: Градиенты для поиска краев чека
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_mag = np.uint8(255 * grad_mag / (np.max(grad_mag) + 1e-5))
    _, grad_binary = cv2.threshold(grad_mag, 30, 255, cv2.THRESH_BINARY)  # Снижен порог
    
    # Комбинируем все методы
    combined = cv2.bitwise_or(adaptive_thresh, binary_white1)
    combined = cv2.bitwise_or(combined, binary_white2)
    combined = cv2.bitwise_or(combined, binary_white3)
    combined = cv2.bitwise_or(combined, edges)
    combined = cv2.bitwise_or(combined, grad_binary)
    
    # Морфология - менее агрессивная
    kernel_small = np.ones((5, 5), np.uint8)
    kernel_medium = np.ones((10, 10), np.uint8)
    kernel_large = np.ones((15, 15), np.uint8)
    
    # Пробуем разные варианты морфологии
    variants = []
    
    # Вариант 1: Мягкая морфология
    closed1 = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
    opened1 = cv2.morphologyEx(closed1, cv2.MORPH_OPEN, kernel_small, iterations=1)
    variants.append(opened1)
    
    # Вариант 2: Более агрессивная морфология
    closed2 = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    opened2 = cv2.morphologyEx(closed2, cv2.MORPH_OPEN, kernel_medium, iterations=1)
    variants.append(opened2)
    
    # Вариант 3: Дилатация для соединения разрывов
    dilated = cv2.dilate(combined, kernel_medium, iterations=2)
    closed3 = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_large, iterations=1)
    variants.append(closed3)
    
    # Ищем контуры во всех вариантах
    all_contours = []
    for variant in variants:
        contours, _ = cv2.findContours(variant, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.extend(contours)
        logging.info(f"[CONTOUR] Found {len(contours)} contours in variant")
    
    # Удаляем дубликаты (контуры с очень похожими площадями и центрами)
    unique_contours = []
    seen_areas = set()
    for contour in all_contours:
        area = cv2.contourArea(contour)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            # Группируем по площади и центру
            area_key = (area // 1000, cx // 50, cy // 50)
            if area_key not in seen_areas:
                seen_areas.add(area_key)
                unique_contours.append(contour)
    
    contours = unique_contours
    logging.info(f"[CONTOUR] Total unique contours: {len(contours)}")
    
    # Создаем визуализацию промежуточных шагов
    debug_image = image_bgr.copy()
    
    # Показываем комбинированное бинарное изображение
    debug_overlay = debug_image.copy()
    combined_colored = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
    cv2.addWeighted(debug_overlay, 0.3, combined_colored, 0.7, 0, debug_image)
    
    if not contours:
        logging.warning("[CONTOUR] No contours found")
        info_text = "CONTOUR: No contours found"
        cv2.putText(debug_image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return None, None, debug_image
    
    # Сортируем контуры по площади
    contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
    
    # Рисуем все найденные контуры разными цветами (больше контуров для отладки)
    for idx, contour in enumerate(contours_sorted[:15]):  # Показываем первые 15 контуров
        color = (
            (idx * 17) % 255,
            (idx * 37) % 255,
            (idx * 53) % 255
        )
        cv2.drawContours(debug_image, [contour], -1, color, 2)
        area = cv2.contourArea(contour)
        area_percent = area / (h * w) * 100
        # Находим центр контура для подписи
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(debug_image, f"C{idx}:{area_percent:.1f}%", (cx, cy), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    largest_contour = None
    box = None
    
    # Пробуем найти контур, который не является углами кадра
    # Снижаем минимальную площадь до 5% (было 10%)
    min_area_percent = 5.0
    min_area = min_area_percent / 100.0 * h * w
    
    for contour in contours_sorted:
        area = cv2.contourArea(contour)
        area_percent = area / (h * w) * 100
        
        if area < min_area:
            logging.debug(f"[CONTOUR] Skipping contour: area {area_percent:.1f}% < {min_area_percent}%")
            continue
        
        # Аппроксимируем контур до 4 точек
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(approx) < 4:
            # Если не 4 точки, используем минимальный прямоугольник
            rect = cv2.minAreaRect(contour)
            approx_box = cv2.boxPoints(rect)
        else:
            if len(approx) == 4:
                approx_box = approx.reshape(4, 2)
            else:
                # Берем 4 крайние точки
                x_coords = approx[:, 0, 0]
                y_coords = approx[:, 0, 1]
                approx_box = np.array([
                    [np.min(x_coords), np.min(y_coords)],  # top-left
                    [np.min(x_coords), np.max(y_coords)],  # bottom-left
                    [np.max(x_coords), np.max(y_coords)],  # bottom-right
                    [np.max(x_coords), np.min(y_coords)]   # top-right
                ], dtype=np.float32)
        
        # Проверяем, не являются ли углы углами кадра
        # Используем более мягкую проверку - контур считается углами кадра только если
        # все 4 угла находятся очень близко к краям (в пределах 20px)
        margin = 20
        corners_near_edge = sum(
            1 for pt in approx_box
            if (pt[0] < margin or pt[0] > w - margin) and (pt[1] < margin or pt[1] > h - margin)
        )
        
        # Также проверяем, что контур не покрывает слишком большую область (не весь кадр)
        x_coords = approx_box[:, 0]
        y_coords = approx_box[:, 1]
        width = np.max(x_coords) - np.min(x_coords)
        height = np.max(y_coords) - np.min(y_coords)
        width_percent = width / w * 100
        height_percent = height / h * 100
        
        # Контур считается валидным, если:
        # 1. Не все углы находятся у краев кадра (меньше 3 из 4) - это главный критерий
        # 2. Площадь контура разумная (не слишком мала и не покрывает весь кадр)
        # 3. Размеры bounding box используются только как дополнительная проверка
        
        # Главный критерий: если не все углы у краев, контур скорее всего валидный
        # Даже если размеры большие, но углы не у краев - это может быть большой чек
        is_valid_by_corners = corners_near_edge < 3
        
        # Дополнительная проверка: площадь контура не должна быть слишком близка к площади всего изображения
        # (если контур покрывает >95% площади изображения, это скорее всего рамка)
        area_ratio = area / (h * w)
        is_valid_by_area = area_ratio < 0.95
        
        # Размеры используются только для логирования, не для отклонения
        is_valid = is_valid_by_corners and is_valid_by_area
        
        if is_valid:
            largest_contour = contour
            box = approx_box
            logging.info(f"[CONTOUR] Found valid contour: area = {area_percent:.1f}%, "
                        f"size = {width_percent:.1f}% x {height_percent:.1f}%, "
                        f"corners_near_edge = {corners_near_edge}, corners = {box}")
            break
        elif is_valid_by_corners and not is_valid_by_area:
            # Если углы не у краев, но площадь большая - возможно это большой чек, используем его
            logging.info(f"[CONTOUR] Using contour with large area but valid corners: "
                        f"area = {area_percent:.1f}%, corners_near_edge = {corners_near_edge}")
            largest_contour = contour
            box = approx_box
            break
        else:
            logging.debug(f"[CONTOUR] Rejected contour: corners_near_edge={corners_near_edge}, "
                         f"area={area_percent:.1f}%, size={width_percent:.1f}% x {height_percent:.1f}%")
    
    # Если не нашли подходящий контур, используем самый большой
    if largest_contour is None:
        largest_contour = contours_sorted[0]
        area = cv2.contourArea(largest_contour)
        area_percent = area / (h * w) * 100
        logging.info(f"[CONTOUR] Using largest contour: area = {area_percent:.1f}%")
        
        if area < min_area:
            logging.warning(f"[CONTOUR] Contour too small ({area_percent:.1f}% < {min_area_percent}%)")
            # Все равно создаем визуализацию
            result_image = debug_image.copy()
            cv2.drawContours(result_image, [largest_contour], -1, (0, 255, 0), 5)
            info_text = f"CONTOUR: Contour too small ({area_percent:.1f}% < {min_area_percent}%)"
            cv2.putText(result_image, info_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return None, None, result_image
        
        # Аппроксимируем контур до 4 точек
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        logging.info(f"[CONTOUR] Approximated to {len(approx)} points")
        
        if len(approx) >= 4:
            if len(approx) == 4:
                box = approx.reshape(4, 2)
            else:
                rect = cv2.minAreaRect(largest_contour)
                box = cv2.boxPoints(rect)
        else:
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
        
        # Проверяем, не являются ли углы углами кадра (более мягкая проверка)
        margin = 20
        corners_near_edge = sum(
            1 for pt in box
            if (pt[0] < margin or pt[0] > w - margin) and (pt[1] < margin or pt[1] > h - margin)
        )
        
        # Проверяем размеры
        x_coords = box[:, 0]
        y_coords = box[:, 1]
        width = np.max(x_coords) - np.min(x_coords)
        height = np.max(y_coords) - np.min(y_coords)
        width_percent = width / w * 100
        height_percent = height / h * 100
        
        # Проверяем, являются ли углы углами кадра
        # Если углы не у краев (corners_near_edge < 3), используем контур напрямую, даже если размеры большие
        if corners_near_edge < 3:
            # Контур валидный - углы не у краев, значит это не рамка изображения
            logging.info(f"[CONTOUR] Using contour directly: corners_near_edge={corners_near_edge}, "
                        f"size={width_percent:.1f}% x {height_percent:.1f}%, box={box}")
            # box уже установлен выше, просто продолжаем
        else:
            # Углы у краев - возможно это рамка, но проверим площадь контура
            area_ratio = area / (h * w)
            if area_ratio < 0.95:
                # Площадь разумная, используем контур
                logging.info(f"[CONTOUR] Using contour despite corners near edge: "
                            f"area={area_percent:.1f}%, corners_near_edge={corners_near_edge}")
            else:
                # Площадь слишком большая и углы у краев - скорее всего рамка
                logging.warning(f"[CONTOUR] Contour likely is frame (area={area_percent:.1f}%, "
                              f"corners_near_edge={corners_near_edge}), trying bounding box")
                # Используем bounding box самого большого контура
                x, y, w_bb, h_bb = cv2.boundingRect(largest_contour)
                
                # Проверяем, что это не весь кадр
                if w_bb < w * 0.98 and h_bb < h * 0.98:
                    box = np.array([
                        [x, y],
                        [x, y + h_bb],
                        [x + w_bb, y + h_bb],
                        [x + w_bb, y]
                    ], dtype=np.float32)
                    logging.info(f"[CONTOUR] Using bounding box corners: {box}")
                else:
                    # Даже bounding box покрывает почти весь кадр - используем сам контур
                    logging.warning(f"[CONTOUR] Bounding box also covers too much ({w_bb/w*100:.1f}% x {h_bb/h*100:.1f}%), "
                                  f"using contour directly")
                    # box уже установлен из контура выше, используем его
    
    # Создаем финальное изображение с визуализацией
    result_image = debug_image.copy()
    
    # Рисуем выбранный контур зеленым
    if largest_contour is not None:
        cv2.drawContours(result_image, [largest_contour], -1, (0, 255, 0), 5)
    
    # Добавляем информацию о методе
    if box is not None:
        area = (box[2][0] - box[0][0]) * (box[2][1] - box[0][1])
        area_percent = area / (h * w) * 100
        info_text = f"CONTOUR: {len(contours)} contours, area={area_percent:.1f}%"
        cv2.putText(result_image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return box, largest_contour, result_image


def _find_corners_by_text(image: np.ndarray, gray: np.ndarray, image_bgr: np.ndarray) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Метод 3: Поиск углов через текстовые проекции.
    Возвращает (box, result_image) или (None, None) если не найдено.
    """
    h, w = gray.shape[:2]
    
    # Ищем внутренний прямоугольник через текстовые области
    _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text_mask = 255 - binary_otsu
    
    # Создаем визуализацию
    result_image = image_bgr.copy()
    
    # Показываем текстовую маску (полупрозрачно)
    text_mask_colored = cv2.cvtColor(text_mask, cv2.COLOR_GRAY2BGR)
    overlay = result_image.copy()
    cv2.addWeighted(overlay, 0.7, text_mask_colored, 0.3, 0, result_image)
    
    # Ищем края текстовой области
    horizontal_proj = np.sum(text_mask, axis=1)
    vertical_proj = np.sum(text_mask, axis=0)
    
    h_threshold = np.max(horizontal_proj) * 0.10
    v_threshold = np.max(vertical_proj) * 0.10
    
    # Рисуем графики проекций
    max_h_proj = np.max(horizontal_proj) if np.max(horizontal_proj) > 0 else 1
    max_v_proj = np.max(vertical_proj) if np.max(vertical_proj) > 0 else 1
    
    # Горизонтальная проекция слева (показывает где есть текст по строкам)
    graph_width = min(150, w // 8)
    for y_idx in range(h):
        bar_width = int((horizontal_proj[y_idx] / max_h_proj) * graph_width)
        color = (0, 255, 0) if horizontal_proj[y_idx] > h_threshold else (0, 0, 255)
        cv2.line(result_image, (0, y_idx), (bar_width, y_idx), color, 1)
    
    # Вертикальная проекция сверху (показывает где есть текст по столбцам)
    graph_height = min(150, h // 8)
    for x_idx in range(w):
        bar_height = int((vertical_proj[x_idx] / max_v_proj) * graph_height)
        color = (0, 255, 0) if vertical_proj[x_idx] > v_threshold else (0, 0, 255)
        cv2.line(result_image, (x_idx, 0), (x_idx, bar_height), color, 1)
    
    h_indices = np.where(horizontal_proj > h_threshold)[0]
    v_indices = np.where(vertical_proj > v_threshold)[0]
    
    if len(h_indices) > 0 and len(v_indices) > 0:
        top = h_indices[0]
        bottom = h_indices[-1]
        left = v_indices[0]
        right = v_indices[-1]
        
        # Проверяем, что это не весь кадр
        if (bottom - top) < h * 0.85 and (right - left) < w * 0.85:
            # Создаем углы из текстовых границ
            box = np.array([
                [left, top],
                [left, bottom],
                [right, bottom],
                [right, top]
            ], dtype=np.float32)
            logging.info(f"[TEXT] Found corners: {box}")
            
            # Рисуем найденные границы пурпурным
            cv2.line(result_image, (0, top), (w, top), (255, 0, 255), 3)
            cv2.line(result_image, (0, bottom), (w, bottom), (255, 0, 255), 3)
            cv2.line(result_image, (left, 0), (left, h), (255, 0, 255), 3)
            cv2.line(result_image, (right, 0), (right, h), (255, 0, 255), 3)
            
            # Информация
            area = (right - left) * (bottom - top)
            area_percent = area / (h * w) * 100
            info_text = f"TEXT: h_th={h_threshold:.0f}, v_th={v_threshold:.0f}, area={area_percent:.1f}%"
            cv2.putText(result_image, info_text, (left, top - 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            return box, result_image
    
    logging.warning("[TEXT] Failed to find corners")
    # Все равно возвращаем изображение с визуализацией
    info_text = f"TEXT: No corners found, h_th={h_threshold:.0f}, v_th={v_threshold:.0f}"
    cv2.putText(result_image, info_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return None, result_image


def _find_and_align_white_receipt_color_with_corners(image: np.ndarray) -> tuple[np.ndarray, bool, list]:
    """
    Находит углы белого чека и рисует их на изображении для визуализации.
    Использует метод, указанный в CORNER_DETECTION_METHOD.
    Возвращает:
    - изображение с отмеченными углами (лучший результат или выбранный метод)
    - флаг, были ли найдены углы
    - список всех результатов от разных методов [(method_name, image, area_percent), ...]
    """
    try:
        h, w = image.shape[:2]
        
        # Конвертируем в grayscale для поиска белого чека
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            gray = image
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        method = CORNER_DETECTION_METHOD
        logging.info(f"Using corner detection method: {method} (will run all methods for visualization)")
        
        box = None
        largest_contour = None
        result_image = None
        all_method_results = []  # Список для хранения всех результатов
        
        # ВСЕГДА прогоняем все методы для визуализации, но выбираем основной результат по методу
        # Сохраняем результаты всех методов для выбора основного
        method_results = {}  # {method_name: (box, image, area_percent, contour)}
        
        # Метод 1: Brightness
        try:
            box_b, img_b = _find_corners_by_brightness(image, gray, image_bgr)
            if img_b is not None:
                if box_b is not None:
                    area_b = (box_b[2][0] - box_b[0][0]) * (box_b[2][1] - box_b[0][1])
                    area_percent_b = area_b / (h * w) * 100
                else:
                    area_percent_b = 0.0
                method_results["brightness"] = (box_b, img_b, area_percent_b, None)
                all_method_results.append(("brightness", img_b.copy(), area_percent_b))
                logging.info(f"[VISUALIZATION] Brightness method: area {area_percent_b:.1f}%")
        except Exception as exc:
            logging.warning(f"[VISUALIZATION] Brightness method failed: {exc}")
        
        # Метод 2: Contour
        try:
            box_c, contour_c, img_c = _find_corners_by_contour(image, gray, image_bgr)
            if img_c is not None:
                if box_c is not None:
                    area_c = (box_c[2][0] - box_c[0][0]) * (box_c[2][1] - box_c[0][1])
                    area_percent_c = area_c / (h * w) * 100
                    if largest_contour is None:
                        largest_contour = contour_c
                else:
                    area_percent_c = 0.0
                method_results["contour"] = (box_c, img_c, area_percent_c, contour_c)
                all_method_results.append(("contour", img_c.copy(), area_percent_c))
                logging.info(f"[VISUALIZATION] Contour method: area {area_percent_c:.1f}%")
        except Exception as exc:
            logging.warning(f"[VISUALIZATION] Contour method failed: {exc}")
        
        # Метод 3: Text
        try:
            box_t, img_t = _find_corners_by_text(image, gray, image_bgr)
            if img_t is not None:
                if box_t is not None:
                    area_t = (box_t[2][0] - box_t[0][0]) * (box_t[2][1] - box_t[0][1])
                    area_percent_t = area_t / (h * w) * 100
                else:
                    area_percent_t = 0.0
                method_results["text"] = (box_t, img_t, area_percent_t, None)
                all_method_results.append(("text", img_t.copy(), area_percent_t))
                logging.info(f"[VISUALIZATION] Text method: area {area_percent_t:.1f}%")
        except Exception as exc:
            logging.warning(f"[VISUALIZATION] Text method failed: {exc}")
        
        # Теперь выбираем основной результат в зависимости от метода
        if method == "all":
            logging.info("Selecting best result from all methods...")
            # Собираем все результаты с найденными углами
            results = []
            for method_name, (method_box, method_img, method_area, method_contour) in method_results.items():
                if method_box is not None:
                    results.append((method_name, method_box, method_img, method_area))
                    if method_contour is not None and largest_contour is None:
                        largest_contour = method_contour
            
            # Выбираем лучший результат
            if results:
                # Сортируем по близости к идеальному размеру (50-70% изображения)
                ideal_min, ideal_max = 50, 70
                results.sort(key=lambda x: abs(x[3] - (ideal_min + ideal_max) / 2))
                
                best_method, box, result_image, area_percent = results[0]
                logging.info(f"[ALL] Selected best method: {best_method} with area {area_percent:.1f}%")
                
                # Если лучший результат слишком мал или велик, пробуем другие
                if area_percent < 40 or area_percent > 90:
                    for method_name, method_box, method_img, method_area in results[1:]:
                        if 40 <= method_area <= 90:
                            best_method, box, result_image, area_percent = method_name, method_box, method_img, method_area
                            logging.info(f"[ALL] Switched to {best_method} with better area {method_area:.1f}%")
                            break
            else:
                logging.warning("[ALL] All methods failed to find corners")
                # Используем первое доступное изображение или исходное
                if all_method_results:
                    result_image = all_method_results[0][1].copy()
                else:
                    result_image = image_bgr.copy()
                return result_image, False, all_method_results
        else:
            # Используем результат выбранного метода
            if method in method_results:
                box, result_image, area_percent, method_contour = method_results[method]
                if method_contour is not None:
                    largest_contour = method_contour
                logging.info(f"[{method.upper()}] Using {method} method result: area {area_percent:.1f}%")
            else:
                logging.warning(f"[{method.upper()}] Method {method} result not found, using brightness")
                if "brightness" in method_results:
                    box, result_image, area_percent, _ = method_results["brightness"]
                else:
                    # Если ничего не найдено, используем исходное изображение
                    result_image = image_bgr.copy()
                    box = None
            
            if box is None:
                logging.warning(f"[{method.upper()}] Failed to find corners")
                if result_image is None:
                    result_image = image_bgr.copy()
                return result_image, False, all_method_results
        
        # Рисуем контур (если есть) и углы
        # Для метода contour и режима all рисуем найденный контур
        if (method == "contour" or method == "all") and largest_contour is not None:
            cv2.drawContours(result_image, [largest_contour], -1, (0, 255, 0), 5)
        
        logging.info(f"Drawing corners: final box points = {box}")
        
        # Рисуем линии между углами (синие, толстые)
        for i in range(4):
            pt1 = tuple(map(int, box[i]))
            pt2 = tuple(map(int, box[(i + 1) % 4]))
            cv2.line(result_image, pt1, pt2, (255, 0, 0), 5)
            logging.info(f"Drawing line from {pt1} to {pt2}")
        
        # Рисуем углы большими красными кружками
        for i, point in enumerate(box):
            pt = tuple(map(int, point))
            # Большой красный кружок
            cv2.circle(result_image, pt, 20, (0, 0, 255), -1)
            # Белая обводка для контраста
            cv2.circle(result_image, pt, 20, (255, 255, 255), 2)
            # Номер угла
            cv2.putText(result_image, f"{i+1}", (pt[0] + 25, pt[1] + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            logging.info(f"Drawing corner {i+1} at {pt}")
        
        logging.info("Drawing corners: success, returning image with corners")
        return result_image, True, all_method_results
    except Exception as exc:
        logging.debug(f"Drawing corners failed: {exc}")
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), False, []
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), False, []


def _find_and_align_white_receipt_color(image: np.ndarray) -> np.ndarray:
    """
    Находит белый прямоугольный чек на цветном изображении и выравнивает его перспективой.
    Работает с цветным изображением, конвертирует в grayscale для обработки.
    """
    try:
        h, w = image.shape[:2]
        
        # Конвертируем в grayscale для поиска белого чека
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Пробуем несколько методов для выделения белого чека
        # Метод 1: Простой порог для белого (чек обычно очень светлый)
        _, binary_white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        
        # Метод 2: Адаптивный порог как fallback
        binary_adaptive = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Используем комбинацию: берем белые области из обоих методов
        binary = cv2.bitwise_or(binary_white, binary_adaptive)
        
        # Инвертируем: белый чек становится черным для поиска контуров
        inverted = 255 - binary
        
        # Морфология для соединения разрозненных частей чека
        kernel = np.ones((20, 20), np.uint8)
        closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel, iterations=3)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Находим контуры
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logging.debug("No contours found for receipt alignment")
            return image
        
        # Находим самый большой контур (предположительно чек)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Проверяем, что это достаточно большая область (минимум 15% изображения)
        if area < 0.15 * h * w:
            logging.debug(f"Contour area too small: {area / (h * w) * 100:.1f}%")
            return image
        
        # Аппроксимируем контур до 4 точек (углы прямоугольника)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) >= 4:
            # Если нашли 4+ точки, берем 4 угла
            if len(approx) == 4:
                box = approx.reshape(4, 2)
            else:
                # Если больше 4 точек, используем минимальный ограничивающий прямоугольник
                rect = cv2.minAreaRect(largest_contour)
                box = cv2.boxPoints(rect)
            
            # Применяем перспективное преобразование для выравнивания (на цветном изображении)
            warped = _four_point_transform_color(image, box)
            logging.info(f"Receipt aligned (color): found {len(approx)} points, area: {area / (h * w) * 100:.1f}%")
            return warped
        else:
            # Если не нашли 4 угла, используем минимальный ограничивающий прямоугольник
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            warped = _four_point_transform_color(image, box)
            logging.info(f"Receipt aligned (color, using minAreaRect): area: {area / (h * w) * 100:.1f}%")
            return warped
    except Exception as exc:
        logging.debug(f"Receipt alignment (color) failed: {exc}")
        return image


def _crop_white_receipt_color(image: np.ndarray) -> np.ndarray:
    """
    Обрезает цветное изображение по границам чека.
    Ищет текстовые области (темные на светлом) для определения границ чека.
    """
    try:
        h, w = image.shape[:2]
        original_size = w * h
        
        # Конвертируем в grayscale для поиска границ
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Метод: Ищем текстовые области (темные на светлом фоне)
        # Используем Otsu для автоматического определения порога
        _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Инвертируем: текст становится белым для поиска контуров
        text_mask = 255 - binary_otsu
        
        # Морфология для соединения текстовых областей
        kernel = np.ones((10, 10), np.uint8)
        closed = cv2.morphologyEx(text_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Находим контуры
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logging.info(f"Found {len(contours)} contours for receipt cropping")
        if not contours:
            logging.warning("No contours found for receipt cropping, trying white areas method")
            # Fallback: ищем белые области
            _, binary_white = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            inverted = 255 - binary_white
            kernel = np.ones((20, 20), np.uint8)
            closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel, iterations=3)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return image
        
        # Находим самый большой контур (чек)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w_cont, h_cont = cv2.boundingRect(largest_contour)
        area = cv2.contourArea(largest_contour)
        area_percent = area / original_size * 100
        logging.info(f"Largest contour: x={x}, y={y}, w={w_cont}, h={h_cont}, area={area_percent:.1f}%")
        
        # Если контур занимает почти весь кадр (>90%), значит обрезка не нужна или алгоритм не работает
        # Пробуем другой метод - ищем текстовые области
        if area_percent > 90:
            logging.warning(f"Contour covers {area_percent:.1f}% of image, trying text-based cropping")
            # Метод 2: Ищем текстовые области через проекции
            _, binary_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            text_mask = 255 - binary_otsu
            
            horizontal_proj = np.sum(text_mask, axis=1)
            vertical_proj = np.sum(text_mask, axis=0)
            
            h_threshold = max(np.max(horizontal_proj) * 0.01, np.mean(horizontal_proj) * 0.5)
            v_threshold = max(np.max(vertical_proj) * 0.01, np.mean(vertical_proj) * 0.5)
            
            h_indices = np.where(horizontal_proj > h_threshold)[0]
            v_indices = np.where(vertical_proj > v_threshold)[0]
            
            if len(h_indices) > 0 and len(v_indices) > 0:
                padding = 10
                top = max(0, h_indices[0] - padding)
                bottom = min(h, h_indices[-1] + 1 + padding)
                left = max(0, v_indices[0] - padding)
                right = min(w, v_indices[-1] + 1 + padding)
                
                crop_h = bottom - top
                crop_w = right - left
                crop_size = crop_w * crop_h
                crop_percent = crop_size / original_size * 100
                
                if crop_percent < 95 and crop_h > 50 and crop_w > 50:
                    cropped = image[top:bottom, left:right]
                    logging.info(f"Cropped receipt (text projections): {w}x{h} -> {crop_w}x{crop_h} ({crop_percent:.1f}%)")
                    return cropped
            
            logging.warning(f"Text-based cropping also failed, contour covers {area_percent:.1f}%")
            return image
        
        # Добавляем небольшие отступы (2% или минимум 10px)
        padding_x = max(10, int(w_cont * 0.02))
        padding_y = max(10, int(h_cont * 0.02))
        x = max(0, x - padding_x)
        y = max(0, y - padding_y)
        w_cont = min(w - x, w_cont + 2 * padding_x)
        h_cont = min(h - y, h_cont + 2 * padding_y)
        
        crop_size = w_cont * h_cont
        crop_percent = crop_size / original_size * 100
        logging.info(f"Crop check: crop_size={crop_size}, original_size={original_size}, percent={crop_percent:.1f}%, h_cont={h_cont}, w_cont={w_cont}")
        if crop_size > original_size * 0.1 and h_cont > 50 and w_cont > 50:
            cropped = image[y : y + h_cont, x : x + w_cont]
            logging.info(f"Cropped receipt (color): {w}x{h} -> {w_cont}x{h_cont} ({crop_percent:.1f}%)")
            return cropped
        else:
            logging.warning(f"Crop conditions not met: crop_size={crop_size}, threshold={original_size * 0.1}, h_cont={h_cont}, w_cont={w_cont}")
        
        return image
    except Exception as exc:
        logging.debug(f"Receipt cropping (color) failed: {exc}")
        return image


def _four_point_transform_color(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Перспективное преобразование для цветного изображения.
    """
    rect = _order_points(np.array(pts, dtype="float32"))
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))
    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, destination)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def _find_and_align_white_receipt(image: np.ndarray) -> np.ndarray:
    """
    Находит белый прямоугольный чек на изображении и выравнивает его перспективой.
    Чеки всегда прямоугольные и белые, поэтому ищем большой белый прямоугольник.
    Работает с grayscale изображением.
    """
    try:
        h, w = image.shape[:2]
        
        # Убеждаемся, что изображение в grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Шаг 1: Адаптивный порог для выделения белых областей
        # Используем адаптивный порог, чтобы учесть разные условия освещения
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Инвертируем: белый чек становится черным для поиска контуров
        inverted = 255 - binary
        
        # Морфология для соединения разрозненных частей чека
        kernel = np.ones((20, 20), np.uint8)
        # Закрытие для заполнения пробелов
        closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel, iterations=3)
        # Открытие для удаления шума
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Находим контуры
        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            logging.debug("No contours found for receipt alignment")
            return image
        
        # Находим самый большой контур (предположительно чек)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Проверяем, что это достаточно большая область (минимум 15% изображения)
        if area < 0.15 * h * w:
            logging.debug(f"Contour area too small: {area / (h * w) * 100:.1f}%")
            return image
        
        # Шаг 2: Аппроксимируем контур до 4 точек (углы прямоугольника)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) >= 4:
            # Если нашли 4+ точки, берем 4 угла
            if len(approx) == 4:
                box = approx.reshape(4, 2)
            else:
                # Если больше 4 точек, используем минимальный ограничивающий прямоугольник
                rect = cv2.minAreaRect(largest_contour)
                box = cv2.boxPoints(rect)
            
            # Применяем перспективное преобразование для выравнивания
            warped = _four_point_transform(image, box)
            logging.info(f"Receipt aligned: found {len(approx)} points, area: {area / (h * w) * 100:.1f}%")
            return warped
        else:
            # Если не нашли 4 угла, используем минимальный ограничивающий прямоугольник
            rect = cv2.minAreaRect(largest_contour)
            box = cv2.boxPoints(rect)
            warped = _four_point_transform(image, box)
            logging.info(f"Receipt aligned (using minAreaRect): area: {area / (h * w) * 100:.1f}%")
            return warped
    except Exception as exc:
        logging.debug(f"Receipt alignment failed: {exc}")
        return image


def _crop_white_receipt(image: np.ndarray) -> np.ndarray:
    """
    Обрезает изображение по границам чека.
    Использует адаптивный порог для работы с разными условиями освещения.
    """
    try:
        h, w = image.shape[:2]
        original_size = w * h
        
        # Убеждаемся, что изображение в grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Используем адаптивный порог для выделения чека
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Инвертируем для поиска контуров
        inverted = 255 - binary
        
        # Морфология для соединения частей
        kernel = np.ones((15, 15), np.uint8)
        closed = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Находим контуры
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image
        
        # Находим самый большой контур (чек)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w_cont, h_cont = cv2.boundingRect(largest_contour)
        
        # Добавляем небольшие отступы (2% или минимум 10px)
        padding_x = max(10, int(w_cont * 0.02))
        padding_y = max(10, int(h_cont * 0.02))
        x = max(0, x - padding_x)
        y = max(0, y - padding_y)
        w_cont = min(w - x, w_cont + 2 * padding_x)
        h_cont = min(h - y, h_cont + 2 * padding_y)
        
        crop_size = w_cont * h_cont
        if crop_size > original_size * 0.1 and h_cont > 50 and w_cont > 50:
            cropped = image[y : y + h_cont, x : x + w_cont]
            logging.info(f"Cropped receipt: {w}x{h} -> {w_cont}x{h_cont} ({crop_size/original_size*100:.1f}%)")
            return cropped
        
        return image
    except Exception as exc:
        logging.debug(f"Receipt cropping failed: {exc}")
        return image


def _perspective_correct(image: np.ndarray) -> np.ndarray:
    """
    Корректирует перспективу чека для выравнивания краев.
    Использует детекцию углов чека и перспективное преобразование.
    """
    try:
        h, w = image.shape[:2]
        
        # Находим края чека через детекцию контуров
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        # Расширяем края для лучшего соединения
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image
        
        # Находим самый большой контур (предположительно чек)
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        
        # Проверяем, что контур достаточно большой (минимум 30% от изображения)
        if area < 0.3 * h * w:
            return image
        
        # Аппроксимируем контур до 4 точек (углы чека)
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Если нашли 4 угла, применяем перспективное преобразование
        if len(approx) == 4:
            box = approx.reshape(4, 2)
            warped = _four_point_transform(image, box)
            logging.info("Perspective correction applied")
            return warped
        else:
            # Если не нашли 4 угла, используем минимальный ограничивающий прямоугольник
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            warped = _four_point_transform(image, box)
            logging.info("Perspective correction applied (using minAreaRect)")
            return warped
    except Exception as exc:
        logging.debug(f"Perspective correction failed: {exc}")
        return image


def _auto_crop_aggressive(image: np.ndarray) -> np.ndarray:
    """
    Максимально агрессивный автокроп: обрезает по самому краю контента, без отступов.
    Использует несколько методов для максимальной точности.
    """
    try:
        h, w = image.shape[:2]
        original_size = w * h
        
        # Метод 1: Поиск белых областей (чек обычно белый на темном фоне)
        # Используем высокий порог для выделения белого чека
        _, white_mask = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)
        
        # Находим контуры белых областей
        contours_white, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_white:
            # Находим самый большой белый контур (предположительно чек)
            largest_contour = max(contours_white, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            if area > original_size * 0.1:  # Чек должен занимать минимум 10% изображения
                x, y, w_cont, h_cont = cv2.boundingRect(largest_contour)
                # Добавляем отступы 2% чтобы не обрезать сам чек
                padding_x = max(10, int(w_cont * 0.02))
                padding_y = max(10, int(h_cont * 0.02))
                x = max(0, x - padding_x)
                y = max(0, y - padding_y)
                w_cont = min(w - x, w_cont + 2 * padding_x)
                h_cont = min(h - y, h_cont + 2 * padding_y)
                cropped = image[y : y + h_cont, x : x + w_cont]
                crop_size = w_cont * h_cont
                logging.info(f"Aggressive crop (white areas, 2% padding): {w}x{h} -> {w_cont}x{h_cont} ({crop_size/original_size*100:.1f}%)")
                return cropped
        
        # Метод 2: Проекции с очень низким порогом (самый точный для текста)
        _, binary_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_mask = 255 - binary_otsu
        
        horizontal_proj = np.sum(text_mask, axis=1)
        vertical_proj = np.sum(text_mask, axis=0)
        
        # Экстремально низкий порог - обрезаем по самому краю текста
        h_max = np.max(horizontal_proj)
        v_max = np.max(vertical_proj)
        h_mean = np.mean(horizontal_proj)
        v_mean = np.mean(vertical_proj)
        
        # Используем минимальный порог из нескольких вариантов
        h_threshold = min(h_max * 0.0005, h_mean * 0.3, np.percentile(horizontal_proj, 10))
        v_threshold = min(v_max * 0.0005, v_mean * 0.3, np.percentile(vertical_proj, 10))
        
        h_indices = np.where(horizontal_proj > h_threshold)[0]
        v_indices = np.where(vertical_proj > v_threshold)[0]
        
        if len(h_indices) > 0 and len(v_indices) > 0:
            # Добавляем небольшие отступы (1% или минимум 5px) чтобы не обрезать сам чек
            padding_h = max(5, int((h_indices[-1] - h_indices[0]) * 0.01))
            padding_w = max(5, int((v_indices[-1] - v_indices[0]) * 0.01))
            top = max(0, h_indices[0] - padding_h)
            bottom = min(h, h_indices[-1] + 1 + padding_h)
            left = max(0, v_indices[0] - padding_w)
            right = min(w, v_indices[-1] + 1 + padding_w)
            
            crop_h = bottom - top
            crop_w = right - left
            crop_size = crop_w * crop_h
            
            if crop_size > original_size * 0.05 and crop_h > 30 and crop_w > 30:
                cropped = image[top:bottom, left:right]
                logging.info(f"Aggressive crop (projections, 1% padding): {w}x{h} -> {crop_w}x{crop_h} ({crop_size/original_size*100:.1f}%)")
                return cropped
        
        # Метод 2: Адаптивный порог + контуры (для случаев, когда проекции не работают)
        binary = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        inverted = 255 - binary
        
        # Более агрессивная морфология для соединения всех частей
        kernel = np.ones((20, 20), np.uint8)
        dilated = cv2.dilate(inverted, kernel, iterations=3)
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=3)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            all_points = np.concatenate(contours)
            x, y, w_cont, h_cont = cv2.boundingRect(all_points)
            
            # Минимальный отступ - только 5 пикселей или 1%
            padding_x = max(5, int(w_cont * 0.01))
            padding_y = max(5, int(h_cont * 0.01))
            x = max(0, x - padding_x)
            y = max(0, y - padding_y)
            w_cont = min(w - x, w_cont + 2 * padding_x)
            h_cont = min(h - y, h_cont + 2 * padding_y)
            
            crop_size = w_cont * h_cont
            if crop_size > original_size * 0.05 and h_cont > 30 and w_cont > 30:
                cropped = image[y : y + h_cont, x : x + w_cont]
                logging.info(f"Aggressive crop (contours, minimal padding): {w}x{h} -> {w_cont}x{h_cont} ({crop_size/original_size*100:.1f}%)")
                return cropped
        
        # Метод 3: Поиск краев через градиенты (fallback)
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_magnitude = np.uint8(gradient_magnitude / gradient_magnitude.max() * 255)
        
        _, grad_binary = cv2.threshold(gradient_magnitude, 20, 255, cv2.THRESH_BINARY)
        
        h_grad_proj = np.sum(grad_binary, axis=1)
        v_grad_proj = np.sum(grad_binary, axis=0)
        
        h_grad_threshold = np.max(h_grad_proj) * 0.01
        v_grad_threshold = np.max(v_grad_proj) * 0.01
        
        h_grad_indices = np.where(h_grad_proj > h_grad_threshold)[0]
        v_grad_indices = np.where(v_grad_proj > v_grad_threshold)[0]
        
        if len(h_grad_indices) > 0 and len(v_grad_indices) > 0:
            top = h_grad_indices[0]
            bottom = h_grad_indices[-1] + 1
            left = v_grad_indices[0]
            right = v_grad_indices[-1] + 1
            
            crop_h = bottom - top
            crop_w = right - left
            crop_size = crop_w * crop_h
            
            if crop_size > original_size * 0.05 and crop_h > 30 and crop_w > 30:
                cropped = image[top:bottom, left:right]
                logging.info(f"Aggressive crop (gradients, no padding): {w}x{h} -> {crop_w}x{crop_h} ({crop_size/original_size*100:.1f}%)")
                return cropped
        
        logging.debug(f"Aggressive crop: no cropping applied (image {w}x{h})")
        return image
    except Exception as exc:
        logging.debug(f"Aggressive crop failed: {exc}")
        return image


def extract_receipt_text(file_bytes: bytes, force_engine: Optional[str] = None) -> tuple[str, Optional[bytes], List[Snapshot]]:
    """
    Извлекает текст из изображения чека используя локальный OCR.
    force_engine: "tesseract" или "paddleocr" для принудительного выбора движка, None - использовать OCR_ENGINE
    """
    engine = force_engine or OCR_ENGINE
    
    # Если выбран "both", по умолчанию используем tesseract
    if engine == "both":
        engine = "tesseract"
    
    # Проверяем доступность выбранного OCR движка
    if engine == "paddleocr":
        if not PADDLEOCR_AVAILABLE or PaddleOCR is None:
            logging.warning("PaddleOCR не установлен, переключаемся на Tesseract")
            engine = "tesseract"
    else:
            # Проверяем, может ли PaddleOCR инициализироваться
            ocr_instance = get_paddleocr_instance()
            if ocr_instance is None:
                logging.warning("PaddleOCR не может подключиться к хостам моделей, переключаемся на Tesseract")
                engine = "tesseract"
    
    # Проверяем доступность Tesseract (после возможного переключения)
    if engine == "tesseract":
        if not TESSERACT_AVAILABLE or Image is None or pytesseract is None:
            raise ReceiptParsingError(
                "Tesseract не настроен: установите Tesseract, pytesseract и Pillow."
            )
    
    if Image is None:
        raise ReceiptParsingError("Pillow не установлен")
    snapshots: List[Snapshot] = []
    try:
        # Открываем изображение и сразу конвертируем в RGB, чтобы не зависеть от контекстного менеджера
        image = Image.open(io.BytesIO(file_bytes))
        image = image.convert("RGB")
        snapshots_id_before = id(snapshots)
        print(f"[DEBUG] Starting prepare_image_for_ocr, snapshots count: {len(snapshots)}, id: {snapshots_id_before}")
        processed, preview_image = prepare_image_for_ocr(image, snapshots)
        snapshots_id_after = id(snapshots)
        print(f"[DEBUG] After prepare_image_for_ocr, snapshots count: {len(snapshots)}, id: {snapshots_id_after}, same object: {snapshots_id_before == snapshots_id_after}")
        logging.info(f"Created {len(snapshots)} snapshots during preprocessing")
        logging.debug(
            "OCR pipeline: processed size=%s preview=%s",
            processed.size,
            bool(preview_image),
        )
        # Используем выбранный OCR движок
        if engine == "paddleocr":
            # Для PaddleOCR используем оригинальное RGB изображение (не обработанное grayscale)
            # PaddleOCR лучше работает с цветными изображениями
            text, _ = run_paddleocr(image)
        else:
            text, _ = run_multi_pass_ocr(processed)
        
        # Используем preview_image (который теперь такой же как ocr_ready) для лучшего качества
        final_preview = preview_image or processed
        if final_preview.height < final_preview.width:
            final_preview = final_preview.rotate(90, expand=True)
        preview_bytes = image_to_png_bytes(final_preview)
    except Exception as exc:  # pragma: no cover - depends on external binary
        logging.exception("Error in extract_receipt_text")
        engine_name = engine.capitalize()
        raise ReceiptParsingError(f"{engine_name} не смог прочитать изображение.") from exc
    cleaned = text.strip()
    if not cleaned:
        engine_name = engine.capitalize()
        raise ReceiptParsingError(f"{engine_name} вернул пустой текст.")
    
    print(f"[DEBUG] extract_receipt_text returning {len(snapshots)} snapshots")
    logging.info(f"Returning {len(snapshots)} snapshots")
    return cleaned, preview_bytes, snapshots


def prepare_image_for_ocr(
    image: "Image.Image", snapshots: Optional[List[Snapshot]] = None
) -> tuple["Image.Image", Optional["Image.Image"]]:
    if not TESSERACT_AVAILABLE or Image is None:
        return image, image
    snapshots_id_incoming = id(snapshots) if snapshots is not None else None
    if snapshots is None:
        snapshots = []
        print(f"[DEBUG] prepare_image_for_ocr: snapshots was None, created new list")
    else:
        print(f"[DEBUG] prepare_image_for_ocr: received snapshots list, id: {id(snapshots)}, count: {len(snapshots)}")
    print(f"[DEBUG] prepare_image_for_ocr: initial snapshots count: {len(snapshots)}, id: {id(snapshots)}")
    logging.debug(f"Starting prepare_image_for_ocr, initial snapshots count: {len(snapshots)}")
    raw = image.convert("RGB")
    snapshots.append(Snapshot("raw", raw.copy(), "Исходное изображение"))
    print(f"[DEBUG] Added raw snapshot, total: {len(snapshots)}")
    logging.debug(f"Added raw snapshot, total: {len(snapshots)}")
    pil_image = raw.convert("L")
    if ImageOps is not None:
        pil_image = ImageOps.autocontrast(pil_image)
    cv_image = np.array(pil_image)
    snapshots.append(
        Snapshot("grayscale", Image.fromarray(cv_image.copy()), "Градации серого после автоконтраста")
    )
    # Убрали denoising и CLAHE - они портили изображение
    # Убрали auto_rotate - он портил ориентацию
    # Оставляем только deskew для исправления небольшого наклона
    cv_image = _deskew_image(cv_image)
    snapshots.append(
        Snapshot("deskew", Image.fromarray(cv_image.copy()), "После выравнивания горизонта")
    )
    # Теперь обрезаем уже выровненное изображение
    cv_image = _auto_crop(cv_image)
    snapshots.append(
        Snapshot("autocrop", Image.fromarray(cv_image.copy()), "После автокропа")
    )
    # Убрали perspective_refine и ensure_portrait - они портили ориентацию
    # Оставляем только финальный поворот в _final_orientation_fix
    # Убрали пороговую фильтрацию - она портила изображение
    # Используем изображение как есть для OCR
    ocr_ready = cv_image.copy()
    # Preview будет таким же как ocr_ready (лучшее качество)
    preview = ocr_ready.copy()
    ocr_ready, preview = _final_orientation_fix(ocr_ready, preview)
    snapshots.append(
        Snapshot("ocr_ready", Image.fromarray(ocr_ready.copy()), "Изображение, поданное в Tesseract")
    )
    print(f"[DEBUG] Finished prepare_image_for_ocr, total snapshots: {len(snapshots)}")
    logging.debug(f"Finished prepare_image_for_ocr, total snapshots: {len(snapshots)}")
    max_side = 1800
    h, w = ocr_ready.shape[:2]
    if max(h, w) > max_side:
        ratio = max_side / max(h, w)
        ocr_ready = cv2.resize(
            ocr_ready,
            (int(w * ratio), int(h * ratio)),
            interpolation=cv2.INTER_CUBIC,
        )
    processed = Image.fromarray(ocr_ready)
    preview_image = Image.fromarray(preview) if preview is not None else None
    return processed, preview_image


def _auto_crop(image: np.ndarray) -> np.ndarray:
    """
    Агрессивный автокроп: использует несколько методов для определения границ чека.
    Метод 1: Проекции с адаптивным порогом
    Метод 2: Морфология + контуры
    Метод 3: Анализ градиентов
    """
    try:
        h, w = image.shape[:2]
        original_size = w * h
        
        # Метод 1: Проекции с более агрессивным порогом
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_mask = 255 - binary
        
        # Горизонтальные и вертикальные проекции
        horizontal_proj = np.sum(text_mask, axis=1)
        vertical_proj = np.sum(text_mask, axis=0)
        
        # Используем более низкий порог: 1% от максимума или абсолютное значение
        h_max = np.max(horizontal_proj)
        v_max = np.max(vertical_proj)
        h_threshold = max(h_max * 0.01, np.mean(horizontal_proj) * 2)
        v_threshold = max(v_max * 0.01, np.mean(vertical_proj) * 2)
        
        h_indices = np.where(horizontal_proj > h_threshold)[0]
        v_indices = np.where(vertical_proj > v_threshold)[0]
        
        if len(h_indices) > 0 and len(v_indices) > 0:
            top = max(0, h_indices[0] - 5)
            bottom = min(h, h_indices[-1] + 5)
            left = max(0, v_indices[0] - 5)
            right = min(w, v_indices[-1] + 5)
            
            crop_h = bottom - top
            crop_w = right - left
            crop_size = crop_w * crop_h
            
            # Проверяем, что обрезали хотя бы 5% от оригинала
            if crop_size > original_size * 0.05 and crop_h > 50 and crop_w > 50:
                cropped = image[top:bottom, left:right]
                logging.debug(f"Auto-crop (projections): {w}x{h} -> {crop_w}x{crop_h} ({crop_size/original_size*100:.1f}%)")
                return cropped
        
        # Метод 2: Морфология для соединения текста, затем контуры
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(text_mask, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Находим самый большой контур или объединяем все
            if len(contours) == 1:
                contour = contours[0]
            else:
                # Объединяем все контуры
                all_points = np.concatenate(contours)
                contour = all_points
            
            x, y, w_cont, h_cont = cv2.boundingRect(contour)
            
            # Отступы
            padding = 5
            x = max(0, x - padding)
            y = max(0, y - padding)
            w_cont = min(w - x, w_cont + 2 * padding)
            h_cont = min(h - y, h_cont + 2 * padding)
            
            crop_size = w_cont * h_cont
            if crop_size > original_size * 0.05 and h_cont > 50 and w_cont > 50:
                cropped = image[y : y + h_cont, x : x + w_cont]
                logging.debug(f"Auto-crop (contours): {w}x{h} -> {w_cont}x{h_cont} ({crop_size/original_size*100:.1f}%)")
                return cropped
        
        # Метод 3: Анализ градиентов для определения краев
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_magnitude = np.uint8(gradient_magnitude / gradient_magnitude.max() * 255)
        
        # Порог для градиентов
        _, grad_binary = cv2.threshold(gradient_magnitude, 30, 255, cv2.THRESH_BINARY)
        
        # Проекции градиентов
        h_grad_proj = np.sum(grad_binary, axis=1)
        v_grad_proj = np.sum(grad_binary, axis=0)
        
        h_grad_threshold = np.max(h_grad_proj) * 0.02
        v_grad_threshold = np.max(v_grad_proj) * 0.02
        
        h_grad_indices = np.where(h_grad_proj > h_grad_threshold)[0]
        v_grad_indices = np.where(v_grad_proj > v_grad_threshold)[0]
        
        if len(h_grad_indices) > 0 and len(v_grad_indices) > 0:
            top = max(0, h_grad_indices[0] - 5)
            bottom = min(h, h_grad_indices[-1] + 5)
            left = max(0, v_grad_indices[0] - 5)
            right = min(w, v_grad_indices[-1] + 5)
            
            crop_h = bottom - top
            crop_w = right - left
            crop_size = crop_w * crop_h
            
            if crop_size > original_size * 0.05 and crop_h > 50 and crop_w > 50:
                cropped = image[top:bottom, left:right]
                logging.debug(f"Auto-crop (gradients): {w}x{h} -> {crop_w}x{crop_h} ({crop_size/original_size*100:.1f}%)")
                return cropped
        
        logging.debug(f"Auto-crop: no cropping applied (image {w}x{h})")
        return image
    except Exception as exc:
        logging.debug(f"Auto-crop failed: {exc}")
        return image


def _auto_rotate(image: np.ndarray) -> np.ndarray:
    try:
        angle = _detect_rotation_angle(image)
        if angle is not None and abs(angle) > 0.3:
            logging.debug("Auto-rotate: rotating by %s degrees", angle)
            return _rotate_image(image, angle)
        logging.debug("Auto-rotate: no rotation applied (angle=%s)", angle)
    except Exception:
        pass
    return image


def _detect_rotation_angle(image: np.ndarray) -> Optional[float]:
    angle = _osd_angle(image)
    if angle is not None and abs(angle) > 0.1:
        logging.debug("Detect rotation: OSD angle=%s", angle)
        return angle
    hough_angle = _hough_angle(image)
    logging.debug("Detect rotation: Hough angle=%s", hough_angle)
    return hough_angle


def _osd_angle(image: np.ndarray) -> Optional[float]:
    if pytesseract is None:
        return None
    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
    except pytesseract.pytesseract.TesseractError:
        return None
    angle = osd.get("rotate", 0)
    if angle is None:
        return None
    if angle > 180:
        angle -= 360
    return float(angle)


def _hough_angle(image: np.ndarray) -> Optional[float]:
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=120, minLineLength=100, maxLineGap=10)
    if lines is None or len(lines) == 0:
        return None
    angles: List[float] = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)
    if not angles:
        return None
    avg_angle = np.median(angles)
    # Snap to closest multiple of 90 to avoid upside-down outputs
    for snap in (0, 90, -90, 180, -180):
        if abs(avg_angle - snap) < 10:
            return avg_angle - snap
    return avg_angle


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated = cv2.warpAffine(
        image,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def _deskew_image(image: np.ndarray) -> np.ndarray:
    """
    Исправляет небольшой наклон изображения (skew).
    Ограничивает угол поворота до ±5 градусов, чтобы не портить уже правильно ориентированные изображения.
    """
    try:
        # Используем более консервативный порог для определения текста
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(binary < 250))
        if coords.size == 0:
            return image
        
        # Вычисляем минимальный ограничивающий прямоугольник
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        
        # Нормализуем угол
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        # Ограничиваем угол поворота: исправляем только небольшие наклоны (±5 градусов)
        # Если угол больше, значит изображение уже правильно ориентировано или нужен другой подход
        max_skew_angle = 5.0
        if abs(angle) < 0.5:
            # Угол слишком мал, не поворачиваем
            return image
        if abs(angle) > max_skew_angle:
            # Угол слишком большой - вероятно, это не наклон, а неправильная ориентация
            # Не исправляем здесь, пусть другие функции разбираются
            logging.debug(f"Deskew: angle {angle:.2f}° too large, skipping")
            return image
        
        logging.debug(f"Deskew: correcting skew by {angle:.2f}°")
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return rotated
    except Exception as exc:
        logging.debug(f"Deskew failed: {exc}")
        return image


def _adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """Старая версия с комбинацией двух порогов - может создавать артефакты"""
    try:
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        thresh_gauss = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            35,
            11,
        )
        thresh_mean = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            41,
            8,
        )
        combined = cv2.bitwise_and(thresh_gauss, thresh_mean)
        return combined
    except Exception:
        return image


def _simple_adaptive_threshold(image: np.ndarray) -> np.ndarray:
    """
    Упрощенная пороговая фильтрация - только один адаптивный порог.
    Более мягкая, не создает артефактов от комбинации двух порогов.
    """
    try:
        # Небольшое размытие для сглаживания
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        # Один адаптивный порог с более мягкими параметрами
        thresh = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            15,  # Блок меньшего размера для лучшей адаптации
            10,  # Константа вычитания
        )
        return thresh
    except Exception as exc:
        logging.debug(f"Simple adaptive threshold failed: {exc}")
        return image


def _perspective_refine(image: np.ndarray) -> np.ndarray:
    try:
        edges = cv2.Canny(image, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return image
        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < 0.3 * image.shape[0] * image.shape[1]:
            return image
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        warped = _four_point_transform(image, box)
        return warped
    except Exception:
        return image


def _ensure_portrait(image: np.ndarray) -> np.ndarray:
    """
    Упрощенная версия: не поворачиваем изображение здесь.
    Поворот будет сделан только в _force_portrait в конце, если нужно.
    Это предотвращает множественные повороты, которые портят ориентацию.
    """
    # Просто возвращаем изображение как есть - ориентацию исправим в конце
    return image


def _should_rotate_90(image: np.ndarray) -> bool:
    try:
        _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_mask = 255 - binary
        horizontal_proj = np.sum(text_mask, axis=1)
        vertical_proj = np.sum(text_mask, axis=0)
        horizontal_variation = float(np.std(horizontal_proj))
        vertical_variation = float(np.std(vertical_proj))
        decision = vertical_variation > horizontal_variation
        logging.debug(
            "Should rotate 90 decision=%s (vertical_var=%.2f horizontal_var=%.2f)",
            decision,
            vertical_variation,
            horizontal_variation,
        )
        return decision
    except Exception:
        return image.shape[1] > image.shape[0]


def _align_text_structure(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if image is None:
        return None
    try:
        gray = image
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text_mask = 255 - binary
        horizontal_proj = np.sum(text_mask, axis=1)
        vertical_proj = np.sum(text_mask, axis=0)
        horizontal_variation = float(np.std(horizontal_proj))
        vertical_variation = float(np.std(vertical_proj))
        logging.debug(
            "Align text: vertical_var=%.2f horizontal_var=%.2f",
            vertical_variation,
            horizontal_variation,
        )
        if vertical_variation > horizontal_variation * 1.2:
            logging.debug("Align text: rotating 90 degrees (vertical dominates)")
            return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        if horizontal_variation > vertical_variation * 1.2 and horizontal_variation > 50000:
            logging.debug("Align text: rotating -90 degrees (horizontal dominates)")
            return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception as exc:
        logging.debug("Align text skipped: %s", exc)
    return image


def _force_portrait(image: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if image is None:
        return None
    h, w = image.shape[:2]
    if h >= w:
        return image
    logging.debug("Force portrait: rotating image to ensure height>=width")
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)


def _final_orientation_fix(
    image: np.ndarray, preview: Optional[np.ndarray]
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    # Упрощенная логика: только финальный поворот для портретной ориентации
    # Убираем множественные повороты, которые приводят к зеркальному отражению
    rotated_image = _force_portrait(image)
    rotated_preview = (
        _force_portrait(preview) if preview is not None else None
    )
    logging.debug("Final orientation fix: applied force_portrait only")
    return rotated_image, rotated_preview


def _determine_final_rotation(image: np.ndarray) -> int:
    angle = _osd_angle(image)
    rotation = _normalize_to_quadrant(angle) if angle is not None else 0
    if rotation == 0 and _should_rotate_90(image):
        rotation = 90
    logging.debug("Final rotation decision angle=%s rotation=%s", angle, rotation)
    return rotation


def _normalize_to_quadrant(angle: Optional[float]) -> int:
    if angle is None:
        return 0
    normalized = int(round(angle / 90.0)) * 90
    normalized %= 360
    if normalized == 270:
        normalized = -90
    if normalized == 180:
        normalized = -180
    return normalized


def _rotate_numpy_array(image: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 0:
        return image
    if rotation % 360 == 90 or rotation % 360 == -270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if rotation % 360 == 180 or rotation % 360 == -180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if rotation % 360 == 270 or rotation % 360 == -90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image


def _morphology_cleanup(image: np.ndarray) -> np.ndarray:
    try:
        kernel = np.ones((3, 3), np.uint8)
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel, iterations=1)
        return closed
    except Exception:
        return image


def _build_preview(image: np.ndarray) -> Optional[np.ndarray]:
    try:
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        blurred = cv2.GaussianBlur(normalized, (5, 5), 0)
        preview = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        return preview
    except Exception:
        return None


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(np.array(pts, dtype="float32"))
    (tl, tr, br, bl) = rect
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = int(max(width_a, width_b))
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = int(max(height_a, height_b))
    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )
    matrix = cv2.getPerspectiveTransform(rect, destination)
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped


def translate_text_if_needed(text: str) -> str:
    """
    Переводит текст с казахского на русский, если обнаружен казахский текст.
    Если перевод недоступен или текст уже на русском, возвращает оригинал.
    """
    if not TRANSLATION_AVAILABLE or GoogleTranslator is None:
        logging.debug("Translation not available, returning original text")
        return text
    
    try:
        # Проверяем, есть ли казахские символы (кириллица + специфичные казахские буквы)
        kazakh_chars = set("әіңғүұқөһӘІҢҒҮҰҚӨҺ")
        has_kazakh = any(char in text for char in kazakh_chars)
        
        # Также проверяем наличие казахских слов
        kazakh_words = ["тауар", "төлем", "салық", "қарта", "чек", "қасыр", "қосымша"]
        has_kazakh_words = any(word.lower() in text.lower() for word in kazakh_words)
        
        if not (has_kazakh or has_kazakh_words):
            logging.debug("No Kazakh text detected, returning original")
            return text
        
        logging.info("Detected Kazakh text, translating to Russian...")
        # Переводим с казахского на русский
        translator = GoogleTranslator(source='kk', target='ru')
        translated = translator.translate(text)
        
        if translated and translated != text:
            logging.info(f"Translated text: {len(text)} -> {len(translated)} chars")
            return translated
        else:
            logging.debug("Translation returned same text, returning original")
            return text
    except Exception as exc:
        logging.warning(f"Translation failed: {exc}, returning original text")
        return text


def format_receipt_text(text: str) -> str:
    return "Текст чека (локальный OCR):\n\n" + text


def build_text_from_data(ocr_data: Dict[str, List[Any]]) -> str:
    lines: Dict[tuple, List[tuple]] = {}
    keys = ("page_num", "block_num", "par_num", "line_num")
    for idx, text in enumerate(ocr_data.get("text", [])):
        if not text or not text.strip():
            continue
        key = tuple(ocr_data[k][idx] for k in keys)
        left = ocr_data.get("left", [0])[idx]
        conf = ocr_data.get("conf", [0])[idx]
        lines.setdefault(key, []).append((left, text.strip(), conf))
    sorted_keys = sorted(lines.keys())
    collected: List[str] = []
    for key in sorted_keys:
        words = sorted(lines[key], key=lambda item: item[0])
        collected.append(" ".join(word for _, word, _ in words))
    return "\n".join(collected)


def run_paddleocr(image: "Image.Image") -> tuple[str, float]:
    """Запускает PaddleOCR на изображении и возвращает текст и оценку уверенности."""
    if not PADDLEOCR_AVAILABLE or PaddleOCR is None:
        raise ReceiptParsingError("PaddleOCR не доступен")
    
    ocr_instance = get_paddleocr_instance()
    if ocr_instance is None:
        raise ReceiptParsingError("Не удалось инициализировать PaddleOCR")
    
    try:
        # Убеждаемся, что изображение в RGB формате
        if image.mode != "RGB":
            logging.info(f"Converting image from {image.mode} to RGB for PaddleOCR")
            image = image.convert("RGB")
        
        # Конвертируем PIL Image в numpy array для PaddleOCR
        img_array = np.array(image)
        logging.info(f"PaddleOCR input image shape: {img_array.shape}, dtype: {img_array.dtype}")
        
        # PaddleOCR ожидает BGR формат для OpenCV, но PIL использует RGB
        # Конвертируем RGB в BGR
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            logging.info("Converted RGB to BGR for PaddleOCR")
        elif len(img_array.shape) == 2:
            # Если grayscale, конвертируем в BGR (3 канала)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            logging.info("Converted grayscale to BGR for PaddleOCR")
        
        # Запускаем OCR
        logging.info("Calling PaddleOCR.ocr()...")
        try:
            result = ocr_instance.ocr(img_array, cls=True)
        except Exception as ocr_exc:
            logging.warning(f"PaddleOCR with cls=True failed: {ocr_exc}, trying without cls...")
            # Пробуем без cls (угол поворота)
            result = ocr_instance.ocr(img_array, cls=False)
        
        logging.info(f"PaddleOCR returned result: {type(result)}, length: {len(result) if result else 0}")
        
        # PaddleOCR возвращает список результатов для каждой строки
        # Формат: [[[координаты], (текст, уверенность)], ...]
        if not result:
            logging.warning("PaddleOCR returned empty result")
            return "", 0.0
        if not result[0]:
            logging.warning("PaddleOCR returned result with empty first element")
            return "", 0.0
        
        lines = []
        total_confidence = 0.0
        count = 0
        
        logging.info(f"Processing {len(result[0])} line results from PaddleOCR")
        for idx, line_result in enumerate(result[0]):
            try:
                if not line_result or len(line_result) < 2:
                    logging.debug(f"Skipping line_result {idx}: invalid format")
                    continue
                
                # Извлекаем текст и уверенность
                if isinstance(line_result[1], tuple):
                    text = line_result[1][0] if len(line_result[1]) > 0 else ""
                    confidence = line_result[1][1] if len(line_result[1]) > 1 else 1.0
                else:
                    text = str(line_result[1])
                    confidence = 1.0
                
                if text and text.strip():
                    lines.append(text.strip())
                    total_confidence += confidence
                    count += 1
            except Exception as line_exc:
                logging.warning(f"Error processing line_result {idx}: {line_exc}")
                continue
        
        if not lines:
            logging.warning("PaddleOCR returned no text lines")
            return "", 0.0
        
        text = "\n".join(lines)
        avg_confidence = total_confidence / count if count > 0 else 0.0
        
        logging.info(f"PaddleOCR recognized {count} lines, avg confidence: {avg_confidence:.2f}")
        return text, avg_confidence
        
    except Exception as exc:
        error_msg = str(exc)
        logging.error(f"PaddleOCR error: {error_msg}", exc_info=True)
        # Пробуем более простой вариант без cls
        try:
            logging.info("Retrying PaddleOCR without angle classification...")
            # Убеждаемся, что изображение в RGB
            if image.mode != "RGB":
                image = image.convert("RGB")
            img_array_simple = np.array(image)
            if len(img_array_simple.shape) == 3 and img_array_simple.shape[2] == 3:
                img_array_simple = cv2.cvtColor(img_array_simple, cv2.COLOR_RGB2BGR)
            result = ocr_instance.ocr(img_array_simple, cls=False)
            if result and result[0]:
                lines = []
                for line_result in result[0]:
                    if line_result and len(line_result) >= 2:
                        if isinstance(line_result[1], tuple):
                            text = line_result[1][0] if len(line_result[1]) > 0 else ""
                        else:
                            text = str(line_result[1])
                        if text and text.strip():
                            lines.append(text.strip())
                if lines:
                    text = "\n".join(lines)
                    logging.info(f"PaddleOCR retry successful: {len(text)} chars")
                    return text, 0.5
        except Exception as retry_exc:
            logging.error(f"PaddleOCR retry also failed: {retry_exc}", exc_info=True)
        raise ReceiptParsingError(f"PaddleOCR не смог обработать изображение: {error_msg}") from exc


def run_multi_pass_ocr(image: "Image.Image") -> tuple[str, int]:
    """Запускает OCR с несколькими вариантами изображения и выбирает лучший результат (Tesseract)."""
    # Используем Tesseract
    if not TESSERACT_AVAILABLE or pytesseract is None:
        raise ReceiptParsingError("Ни Tesseract, ни PaddleOCR не доступны")
    
    variants = build_ocr_variants(image)
    best_text = ""
    best_score = float("-inf")
    for variant in variants:
        text, score = run_ocr_pass(variant)
        if score > best_score and text.strip():
            best_score = score
            best_text = text
    logging.debug("OCR multipass: single orientation score=%.2f", best_score)
    if best_text.strip():
        return best_text, 0
    fallback_text = pytesseract.image_to_string(
        image,
        lang=RECEIPT_FALLBACK_LANG,
        config=TESSERACT_CONFIG,
    )
    return fallback_text, 0


def build_ocr_variants(image: "Image.Image") -> List["Image.Image"]:
    base = image
    variants = [base]
    if ImageOps is not None:
        variants.append(ImageOps.invert(base))
    if ImageFilter is not None:
        variants.append(base.filter(ImageFilter.UnsharpMask(radius=2, percent=150)))
    return variants


def run_ocr_pass(image: "Image.Image") -> tuple[str, float]:
    if Output is None:
        text = pytesseract.image_to_string(
            image,
            lang=RECEIPT_FALLBACK_LANG,
            config=TESSERACT_CONFIG,
        )
        return text, float(len(text))
    data = pytesseract.image_to_data(
        image,
        lang=RECEIPT_FALLBACK_LANG,
        config=TESSERACT_CONFIG,
        output_type=Output.DICT,
    )
    text = build_text_from_data(data)
    score = _score_ocr_data(data)
    return text, score


def _score_ocr_data(data: Dict[str, List[Any]]) -> float:
    numeric: List[float] = []
    for conf in data.get("conf", []):
        try:
            numeric.append(float(conf))
        except (TypeError, ValueError):
            continue
    if not numeric:
        return -float("inf")
    return sum(numeric) / len(numeric)


def image_to_png_bytes(image: "Image.Image") -> Optional[bytes]:
    try:
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    except Exception:
        return None


ENABLE_LOCAL_OCR = os.getenv("ENABLE_LOCAL_OCR", "1").lower() not in ("0", "false")


async def parse_receipt_pipeline(file_bytes: bytes, mime_type: str) -> ParsedReceipt:
    try:
        return await parse_receipt_with_ai(file_bytes, mime_type)
    except ReceiptParsingError as exc:
        logging.warning("Primary OCR failed, trying fallback: %s", exc)
        if not ENABLE_LOCAL_OCR or not TESSERACT_AVAILABLE or not mime_type.startswith("image/"):
            raise
        return parse_receipt_with_tesseract(file_bytes)


def parse_receipt_with_tesseract(file_bytes: bytes) -> ParsedReceipt:
    if not TESSERACT_AVAILABLE or Image is None or pytesseract is None:
        raise ReceiptParsingError("Локальный OCR недоступен (нет pytesseract/Pillow).")
    try:
        with Image.open(io.BytesIO(file_bytes)) as image:
            image = image.convert("RGB")
            text = pytesseract.image_to_string(image, lang=RECEIPT_FALLBACK_LANG)
    except Exception as exc:  # pragma: no cover - depends on external binary
        raise ReceiptParsingError("Локальный OCR не смог прочитать изображение.") from exc
    if not text.strip():
        raise ReceiptParsingError("Локальный OCR вернул пустой результат.")
    return build_parsed_receipt_from_text(text)


def build_parsed_receipt_from_text(text: str) -> ParsedReceipt:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    store = lines[0] if lines else "Unknown store"
    total = _extract_total_from_text(lines)
    currency = _currency_from_value(text) or "RUB"
    purchased_at = parse_datetime_flexible(_find_first_date(text))
    items = [
        ParsedReceiptItem(
            name="Покупка",
            quantity=1.0,
            price=total if total is not None else 0.0,
        )
    ]
    return ParsedReceipt(
        store=store,
        total=total if total is not None else 0.0,
        currency=currency,
        purchased_at=purchased_at,
        tax_amount=None,
        items=items,
    )


def _extract_total_from_text(lines: List[str]) -> Optional[float]:
    for line in reversed(lines):
        if re.search(r"(итог|итого|total|amount|к оплате)", line, re.IGNORECASE):
            numbers = re.findall(r"(-?\d+[.,]?\d*)", line)
            if numbers:
                return safe_float(numbers[-1])
    candidates: List[float] = []
    for line in lines[-10:]:
        candidates.extend(
            safe_float(value)
            for value in re.findall(r"(-?\d+[.,]?\d*)", line)
        )
    return max(candidates) if candidates else None


def _find_first_date(text: str) -> Optional[str]:
    match = re.search(r"(\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?)", text)
    if match:
        token = match.group(1)
        parts = token.replace("-", ".").replace("/", ".").split(".")
        if len(parts) == 2:
            # Нет года, добавляем текущий год
            day, month = parts[0], parts[1]
            current_year = datetime.utcnow().year
            # Пробуем распарсить дату с текущим годом
            try:
                test_date = datetime.strptime(f"{day}.{month}.{current_year}", "%d.%m.%Y")
                # Если дата в будущем, используем предыдущий год
                if test_date > datetime.utcnow():
                    current_year -= 1
            except ValueError:
                pass
            return f"{day}.{month}.{current_year}"
        elif len(parts) == 3:
            # Есть год, проверяем, не в будущем ли дата
            day, month, year_str = parts[0], parts[1], parts[2]
            try:
                # Если год двухзначный (например, 23), интерпретируем как 20XX
                if len(year_str) == 2:
                    year_short = int(year_str)
                    # Интерпретируем как текущий век (20XX)
                    year = 2000 + year_short
                    # Проверяем, не в будущем ли дата
                    test_date = datetime.strptime(f"{day}.{month}.{year}", "%d.%m.%Y")
                    if test_date > datetime.utcnow():
                        # Если дата в будущем, уменьшаем год на 1
                        year -= 1
                else:
                    # Год четырехзначный
                    year = int(year_str)
                    # Проверяем, не в будущем ли дата
                    test_date = datetime.strptime(f"{day}.{month}.{year}", "%d.%m.%Y")
                    if test_date > datetime.utcnow():
                        # Если дата в будущем, уменьшаем год на 1
                        year -= 1
                return f"{day}.{month}.{year}"
            except (ValueError, TypeError):
                # Если не удалось распарсить, возвращаем как есть
                return token
        return token
    return None


def get_receipt_parser() -> "ReceiptParserAI":
    global _receipt_parser
    if _receipt_parser is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ReceiptParsingError("OPENAI_API_KEY не задан, парсинг чеков невозможен.")
        _receipt_parser = ReceiptParserAI(api_key=api_key)
    return _receipt_parser


async def parse_receipt_with_ai(
    file_bytes: bytes, 
    mime_type: str, 
    qr_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Отправляет изображение в OpenAI и возвращает полный JSON response."""
    parser = get_receipt_parser()
    return await parser.parse(file_bytes, mime_type, qr_data=qr_data)


async def improve_receipt_data_with_ai(receipt_data: Dict[str, Any]) -> Dict[str, Any]:
    """Улучшает данные чека через OpenAI без отправки изображения."""
    parser = get_receipt_parser()
    return await parser.improve_receipt_data(receipt_data)


def build_data_url(file_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(file_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def read_qr_codes(file_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Читает QR-коды и штрих-коды из изображения.
    Возвращает список словарей с данными о найденных кодах.
    Пробует несколько вариантов обработки изображения для лучшего распознавания.
    Использует pyzbar, OpenCV QRCodeDetector и OCR как резервные методы.
    """
    all_results = []
    seen_data = set()  # Чтобы избежать дубликатов
    
    if Image is None:
        logging.warning("Pillow не установлен, чтение QR-кодов недоступно")
        return []
    
    try:
        # Открываем изображение
        image = Image.open(io.BytesIO(file_bytes))
        
        # Конвертируем в RGB если нужно
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Метод 1: pyzbar (если доступен)
        if QR_READER_AVAILABLE and pyzbar_decode is not None:
            # Пробуем несколько вариантов обработки изображения
            # Увеличиваем изображение для лучшего распознавания QR-кодов
            image_large = image.resize((image.width * 3, image.height * 3), Image.Resampling.LANCZOS)
            image_very_large = image.resize((image.width * 4, image.height * 4), Image.Resampling.LANCZOS)
            
            variants = [
                ("original", image),
                ("grayscale", image.convert("L").convert("RGB")),
                ("enhanced", ImageOps.autocontrast(image)),
                ("enhanced_large", ImageOps.autocontrast(image_large)),
                ("large", image_large),
                ("very_large", image_very_large),
                ("sharp", image.filter(ImageFilter.SHARPEN) if ImageFilter is not None else image),
            ]
            
            for variant_name, processed_image in variants:
                try:
                    # Читаем коды из обработанного изображения
                    codes = pyzbar_decode(processed_image)
                    
                    for code in codes:
                        try:
                            data = code.data.decode("utf-8")
                            code_type = code.type
                            
                            # Игнорируем CODE39 и другие штрих-коды, которые не являются URL
                            # Они не несут полезной информации для нас
                            if code_type == "CODE39" and not is_url(data):
                                logging.debug(f"Игнорируем CODE39 (не URL): {data[:50]}...")
                                continue
                            
                            # Пропускаем дубликаты
                            if data in seen_data:
                                continue
                            seen_data.add(data)
                            
                            rect = code.rect
                            result = {
                                "data": data,
                                "type": code_type,
                                "rect": {
                                    "left": rect.left,
                                    "top": rect.top,
                                    "width": rect.width,
                                    "height": rect.height,
                                },
                            }
                            all_results.append(result)
                            logging.info(f"Найден код типа {code_type} (pyzbar, вариант {variant_name}): {data[:100]}...")
                        except UnicodeDecodeError:
                            # Пробуем другие кодировки
                            try:
                                data = code.data.decode("latin-1")
                                if data not in seen_data:
                                    seen_data.add(data)
                                    code_type = code.type
                                    rect = code.rect
                                    result = {
                                        "data": data,
                                        "type": code_type,
                                        "rect": {
                                            "left": rect.left,
                                            "top": rect.top,
                                            "width": rect.width,
                                            "height": rect.height,
                                        },
                                    }
                                    all_results.append(result)
                                    logging.info(f"Найден код типа {code_type} (latin-1, вариант {variant_name}): {data[:100]}...")
                            except Exception:
                                continue
                except Exception as exc:
                    logging.debug(f"Ошибка при обработке варианта {variant_name}: {exc}")
                    continue
            
            # Если ничего не найдено, пробуем с OpenCV для улучшения контраста
            if not all_results and cv2 is not None:
                try:
                    import numpy as np
                    # Конвертируем PIL в numpy array
                    img_array = np.array(image)
                    # Конвертируем RGB в BGR для OpenCV
                    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                    
                    # Пробуем улучшить контраст
                    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                    # Адаптивная бинаризация
                    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                    # Конвертируем обратно в RGB для pyzbar
                    adaptive_rgb = cv2.cvtColor(adaptive, cv2.COLOR_GRAY2RGB)
                    adaptive_pil = Image.fromarray(adaptive_rgb)
                    
                    codes = pyzbar_decode(adaptive_pil)
                    for code in codes:
                        try:
                            data = code.data.decode("utf-8")
                            if data not in seen_data:
                                seen_data.add(data)
                                code_type = code.type
                                rect = code.rect
                                result = {
                                    "data": data,
                                    "type": code_type,
                                    "rect": {
                                        "left": rect.left,
                                        "top": rect.top,
                                        "width": rect.width,
                                        "height": rect.height,
                                    },
                                }
                                all_results.append(result)
                                logging.info(f"Найден код типа {code_type} (OpenCV+pyzbar): {data[:100]}...")
                        except Exception:
                            continue
                except Exception as exc:
                    logging.debug(f"Ошибка при OpenCV обработке для pyzbar: {exc}")
        
        # Метод 2: OpenCV QRCodeDetector (более надежный для QR-кодов)
        # Вызываем ВСЕГДА, даже если pyzbar что-то нашел, так как он может найти правильный QR-код
        if cv2 is not None:
            logging.info("Пробуем OpenCV QRCodeDetector...")
            try:
                import numpy as np
                img_array = np.array(image)
                img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
                
                # Используем QRCodeDetector из OpenCV
                qr_detector = cv2.QRCodeDetector()
                
                # Пробуем также обрезать нижнюю часть изображения (где обычно находится QR-код на чеках)
                # Берем нижние 40% изображения
                h, w = gray.shape
                bottom_crop = gray[int(h * 0.6):, :]  # Нижние 40%
                bottom_half = gray[int(h * 0.5):, :]  # Нижняя половина
                
                # Улучшаем контраст перед обработкой
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                gray_clahe = clahe.apply(gray)
                bottom_crop_clahe = clahe.apply(bottom_crop)
                bottom_half_clahe = clahe.apply(bottom_half)
                
                # Увеличиваем изображение для лучшего распознавания (QR-коды должны быть достаточно большими)
                scale_factor = 3
                scale_factor_5 = 5  # Еще большее увеличение для маленьких QR-кодов
                gray_large = cv2.resize(gray, (gray.shape[1] * scale_factor, gray.shape[0] * scale_factor), interpolation=cv2.INTER_CUBIC)
                gray_very_large = cv2.resize(gray, (gray.shape[1] * scale_factor_5, gray.shape[0] * scale_factor_5), interpolation=cv2.INTER_CUBIC)
                gray_clahe_large = cv2.resize(gray_clahe, (gray_clahe.shape[1] * scale_factor, gray_clahe.shape[0] * scale_factor), interpolation=cv2.INTER_CUBIC)
                bottom_crop_large = cv2.resize(bottom_crop, (bottom_crop.shape[1] * scale_factor, bottom_crop.shape[0] * scale_factor), interpolation=cv2.INTER_CUBIC)
                bottom_half_large = cv2.resize(bottom_half, (bottom_half.shape[1] * scale_factor, bottom_half.shape[0] * scale_factor), interpolation=cv2.INTER_CUBIC)
                
                # Пробуем несколько вариантов обработки
                gray_variants = [
                    ("original", gray),
                    ("original_large", gray_large),
                    ("original_very_large", gray_very_large),
                    ("bottom_crop", bottom_crop),
                    ("bottom_crop_large", bottom_crop_large),
                    ("bottom_half", bottom_half),
                    ("bottom_half_large", bottom_half_large),
                    ("clahe", gray_clahe),
                    ("clahe_large", gray_clahe_large),
                    ("bottom_crop_clahe", bottom_crop_clahe),
                    ("bottom_half_clahe", bottom_half_clahe),
                    ("adaptive", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
                    ("adaptive_large", cv2.adaptiveThreshold(gray_large, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
                    ("adaptive_bottom_crop", cv2.adaptiveThreshold(bottom_crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
                    ("adaptive_bottom_crop_large", cv2.adaptiveThreshold(bottom_crop_large, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
                    ("adaptive_inv", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)),
                    ("otsu", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
                    ("otsu_large", cv2.threshold(gray_large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
                    ("otsu_bottom_crop", cv2.threshold(bottom_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
                    ("otsu_bottom_crop_large", cv2.threshold(bottom_crop_large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
                    ("otsu_inv", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]),
                ]
                
                for variant_name, processed_gray in gray_variants:
                    try:
                        # Пробуем detectAndDecodeMulti (для нескольких QR-кодов)
                        retval, decoded_info, points, straight_qrcode = qr_detector.detectAndDecodeMulti(processed_gray)
                        logging.info(f"OpenCV QRCodeDetector ({variant_name}): retval={retval}, decoded_info={decoded_info}")
                        
                        if retval and decoded_info:
                            for i, data in enumerate(decoded_info):
                                if data and data not in seen_data:
                                    seen_data.add(data)
                                    result = {
                                        "data": data,
                                        "type": "QRCODE",
                                        "rect": {
                                            "left": 0,
                                            "top": 0,
                                            "width": processed_gray.shape[1],
                                            "height": processed_gray.shape[0],
                                        },
                                    }
                                    all_results.append(result)
                                    logging.info(f"✅ Найден QR-код (OpenCV QRCodeDetector, {variant_name}): {data[:100]}...")
                                    # Если нашли QR-код с URL, можно остановиться
                                    if is_url(data):
                                        logging.info(f"✅ Найден QR-код с URL, останавливаем поиск")
                                        break
                        else:
                            # Если detectAndDecodeMulti не сработал, пробуем detectAndDecode (для одного QR-кода)
                            # В старых версиях OpenCV может возвращать только 3 значения
                            try:
                                result_single = qr_detector.detectAndDecode(processed_gray)
                                if isinstance(result_single, tuple):
                                    if len(result_single) >= 2:
                                        retval_single, decoded_info_single = result_single[0], result_single[1]
                                    else:
                                        retval_single, decoded_info_single = False, ""
                                else:
                                    # Если вернулась строка напрямую
                                    retval_single, decoded_info_single = bool(result_single), result_single if result_single else ""
                                
                                logging.info(f"OpenCV QRCodeDetector single ({variant_name}): retval={retval_single}, decoded_info={decoded_info_single}")
                                
                                if retval_single and decoded_info_single and decoded_info_single not in seen_data:
                                    seen_data.add(decoded_info_single)
                                    result = {
                                        "data": decoded_info_single,
                                        "type": "QRCODE",
                                        "rect": {
                                            "left": 0,
                                            "top": 0,
                                            "width": processed_gray.shape[1],
                                            "height": processed_gray.shape[0],
                                        },
                                    }
                                    all_results.append(result)
                                    logging.info(f"✅ Найден QR-код (OpenCV QRCodeDetector single, {variant_name}): {decoded_info_single[:100]}...")
                                    # Если нашли QR-код с URL, можно остановиться
                                    if is_url(decoded_info_single):
                                        logging.info(f"✅ Найден QR-код с URL, останавливаем поиск")
                                        break
                            except ValueError as ve:
                                # Если не удалось распаковать, пробуем другой способ
                                logging.debug(f"Не удалось распаковать результат detectAndDecode: {ve}")
                                continue
                        
                        # Если нашли QR-код с URL, прекращаем перебор вариантов
                        if any(is_url(r.get("data", "")) for r in all_results):
                            logging.info(f"✅ Найден QR-код с URL, прекращаем перебор вариантов обработки")
                            break
                    except Exception as exc:
                        logging.warning(f"Ошибка при OpenCV QRCodeDetector ({variant_name}): {exc}", exc_info=True)
                        continue
            except Exception as exc:
                logging.warning(f"Ошибка при использовании OpenCV QRCodeDetector: {exc}")
        else:
            logging.info("OpenCV недоступен, пропускаем QRCodeDetector")
        
        # Метод 3: qreader (современная библиотека с YOLO для детекции QR-кодов)
        if QREADER_AVAILABLE and qreader_instance is not None:
            logging.info("Пробуем qreader...")
            try:
                import numpy as np
                img_array = np.array(image)
                
                # qreader работает с numpy массивами
                try:
                    decoded_result = qreader_instance.detect_and_decode(image=img_array)
                    # qreader может вернуть кортеж или строку
                    if isinstance(decoded_result, tuple):
                        decoded_text = decoded_result[0] if decoded_result else None
                    else:
                        decoded_text = decoded_result
                    
                    if decoded_text and decoded_text not in seen_data:
                        seen_data.add(decoded_text)
                        result = {
                            "data": decoded_text,
                            "type": "QRCODE",
                            "rect": {
                                "left": 0,
                                "top": 0,
                                "width": image.width,
                                "height": image.height,
                            },
                        }
                        all_results.append(result)
                        logging.info(f"✅ Найден QR-код (qreader): {decoded_text[:100]}...")
                        # Если нашли QR-код с URL, можно остановиться
                        if is_url(decoded_text):
                            logging.info(f"✅ Найден QR-код с URL через qreader, останавливаем поиск")
                            return all_results
                except Exception as exc:
                    logging.debug(f"qreader не смог распознать QR-код: {exc}")
                
                # Пробуем также на увеличенном изображении (только если не нашли URL)
                if not any(is_url(r.get("data", "")) for r in all_results):
                    try:
                        image_large = image.resize((image.width * 3, image.height * 3), Image.Resampling.LANCZOS)
                        img_array_large = np.array(image_large)
                        decoded_result_large = qreader_instance.detect_and_decode(image=img_array_large)
                        # qreader может вернуть кортеж или строку
                        if isinstance(decoded_result_large, tuple):
                            decoded_text_large = decoded_result_large[0] if decoded_result_large else None
                        else:
                            decoded_text_large = decoded_result_large
                        
                        if decoded_text_large and decoded_text_large not in seen_data:
                            seen_data.add(decoded_text_large)
                            result = {
                                "data": decoded_text_large,
                                "type": "QRCODE",
                                "rect": {
                                    "left": 0,
                                    "top": 0,
                                    "width": image_large.width,
                                    "height": image_large.height,
                                },
                            }
                            all_results.append(result)
                            logging.info(f"✅ Найден QR-код (qreader, увеличенное): {decoded_text_large[:100]}...")
                            # Если нашли QR-код с URL, можно остановиться
                            if is_url(decoded_text_large):
                                logging.info(f"✅ Найден QR-код с URL через qreader (увеличенное), останавливаем поиск")
                                return all_results
                    except Exception as exc:
                        logging.debug(f"qreader не смог распознать QR-код на увеличенном изображении: {exc}")
            except Exception as exc:
                logging.warning(f"Ошибка при использовании qreader: {exc}")
        else:
            logging.info("qreader недоступен, пропускаем")
        
        return all_results
    except Exception as exc:
        logging.exception(f"Ошибка при чтении QR-кодов: {exc}")
        return []


def is_url(text: str | tuple) -> bool:
    """Проверяет, является ли текст URL."""
    if not text:
        return False
    # Обрабатываем кортежи (qreader может вернуть кортеж)
    if isinstance(text, tuple):
        text = text[0] if text else ""
    if not isinstance(text, str):
        return False
    text = text.strip()
    return text.startswith(("http://", "https://"))


def parse_ofd_kz_html(html_content: str) -> Optional[Dict[str, Any]]:
    """
    Парсит HTML страницу ofd1.kz/ofd.kz и извлекает данные чека.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        logging.warning("BeautifulSoup4 не установлен, HTML парсинг недоступен")
        return None
    
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Извлекаем основные данные
        store_elem = soup.select_one('.ticket_header span, .ticket_header div span')
        store_raw = store_elem.get_text(strip=True) if store_elem else "Неизвестно"
        # Нормализуем название магазина сразу при парсинге
        store = normalize_store_name(store_raw) if store_raw != "Неизвестно" else store_raw
        
        # Дата и время
        date_elem = soup.select_one('.ticket_date_time')
        date_str = date_elem.get_text(strip=True) if date_elem else None
        
        # Адрес
        address_elem = soup.select_one('.wb-all')
        address = address_elem.get_text(strip=True) if address_elem else None
        
        # Товары
        items = []
        items_list = soup.select('.ready_ticket__items_list li')
        
        for item_elem in items_list:
            # Название товара
            name_elem = item_elem.select_one('.wb-all')
            if not name_elem:
                continue
            name = name_elem.get_text(strip=True)
            if not name or len(name) < 3:
                continue
            
            # Цена и количество в следующем элементе
            price_elem = item_elem.select_one('.ready_ticket__item')
            if not price_elem:
                continue
            
            # Удаляем теги <b>, <small> и другие, которые могут содержать лишние числа (например, скидки, номера)
            # Копируем элемент и удаляем ненужные теги
            from bs4 import BeautifulSoup
            price_elem_str = str(price_elem)
            price_elem_clean = BeautifulSoup(price_elem_str, 'html.parser')
            for tag in price_elem_clean.find_all(['b', 'small', 'span']):
                # Удаляем только если содержимое - это маленькое число (вероятно скидка или номер)
                tag_text = tag.get_text(strip=True)
                if tag_text.isdigit() and int(tag_text) < 10:
                    tag.decompose()
            
            # Получаем весь текст элемента, включая пробелы
            price_text = price_elem_clean.get_text(separator=' ', strip=False)
            # Убираем лишние пробелы, но сохраняем структуру
            price_text = ' '.join(price_text.split())
            # Удаляем ведущие нули перед числами (например, "0 15.00" -> "15.00")
            # Но только если это не часть числа (например, "0.15" остается как есть)
            price_text = re.sub(r'\b0\s+(\d+\.?\d*)', r'\1', price_text)
            
            logging.info(f"🔍 Parsing price text: '{price_text}'")
            
            # Ищем итоговую цену после знака "=" (формат: "= 13 299.00" или "=13299.00")
            # Это самый надежный способ - итоговая цена всегда после "="
            total_price_match = re.search(r'=\s*([\d\s]+\.?\d*)', price_text)
            if total_price_match:
                total_price_str = total_price_match.group(1).replace(' ', '')
                try:
                    total_price = float(total_price_str)
                    logging.info(f"✅ Найдена итоговая цена: {total_price}")
                except ValueError:
                    logging.warning(f"Could not parse total price from: '{total_price_str}'")
                    continue
                
                # Ищем количество и цену за единицу (формат: "415.00 x 1" или "13 299.00 x 1")
                # Ищем паттерн: число с пробелами (может быть с точкой), затем x, затем число
                # Важно: ищем паттерн ДО знака "=", чтобы не перепутать с итоговой суммой
                price_before_equals = price_text.split('=')[0] if '=' in price_text else price_text
                logging.info(f"🔍 Текст до знака '=': '{price_before_equals}'")
                # Улучшенный паттерн: ищем "цена x количество", где количество - целое число без точки
                # Игнорируем числа после количества (например, "15.00 x 2 1" -> берем "2", игнорируем "1")
                # Паттерн: число с точкой (может быть с пробелами для тысяч), затем x, затем целое число
                # Сначала пробуем строгий паттерн: цена с точкой, количество до пробела или конца строки
                qty_price_match = re.search(r'([\d\s]+\.\d+)\s*[xX×]\s*(\d+)(?:\s|$)', price_before_equals)
                if not qty_price_match:
                    # Пробуем более гибкий паттерн без обязательного пробела после количества
                    qty_price_match = re.search(r'([\d\s]+\.\d+)\s*[xX×]\s*(\d+)', price_before_equals)
                
                if qty_price_match:
                    unit_price_str = qty_price_match.group(1).replace(' ', '')
                    quantity_str = qty_price_match.group(2)  # Берем только первое число после "x"
                    logging.info(f"🔍 Найден паттерн 'цена x количество': unit_price_str='{unit_price_str}', quantity_str='{quantity_str}'")
                    try:
                        unit_price = float(unit_price_str)
                        quantity = float(quantity_str)
                        logging.info(f"✅ Распарсено: unit_price={unit_price}, quantity={quantity}")
                        # Проверяем, что total_price соответствует unit_price * quantity
                        expected_total = unit_price * quantity
                        if abs(total_price - expected_total) > 0.01:
                            # Используем вычисленное значение, если оно отличается
                            logging.warning(f"⚠️ Price mismatch: unit_price={unit_price}, quantity={quantity}, expected={expected_total}, got={total_price}")
                            total_price = expected_total
                        logging.info(f"✅ Финальные значения: unit_price={unit_price}, quantity={quantity}, total={total_price}")
                    except ValueError:
                        logging.warning(f"Could not parse unit price or quantity from: '{unit_price_str}', '{quantity_str}'")
                        quantity = 1.0
                        unit_price = total_price
                else:
                    # Если не нашли количество, предполагаем quantity = 1
                    logging.warning(f"⚠️ Не найден паттерн 'цена x количество' в тексте: '{price_before_equals}'")
                    quantity = 1.0
                    unit_price = total_price
                    logging.info(f"✅ Используем значения по умолчанию: quantity={quantity}, unit_price={unit_price}, total={total_price}")
            else:
                # Если не нашли "=", пытаемся найти цену другим способом
                # Ищем паттерн с "x" для количества
                # Улучшенный паттерн: игнорируем числа после количества (например, "15.00 x 2 1" -> берем "2")
                qty_price_match = re.search(r'([\d\s]+\.?\d*)\s*[xX×]\s*(\d+)(?:\s+\d+)*', price_text)
                if qty_price_match:
                    unit_price_str = qty_price_match.group(1).replace(' ', '')
                    quantity_str = qty_price_match.group(2)  # Берем только первое число после "x"
                    try:
                        unit_price = float(unit_price_str)
                        quantity = float(quantity_str)
                        total_price = unit_price * quantity
                        logging.debug(f"Parsed (no =): unit_price={unit_price}, quantity={quantity}, total={total_price}")
                    except ValueError:
                        logging.warning(f"Could not parse price from text: '{price_text}'")
                        continue
                else:
                    # Ищем просто большое число (цена с пробелами)
                    # Ищем числа вида "13 299.00" или "13299.00"
                    numbers = re.findall(r'[\d\s]+\.?\d*', price_text)
                    # Фильтруем маленькие числа (меньше 10, вероятно это не цена товара)
                    # и числа, которые выглядят как количество (целые числа < 100)
                    valid_prices = []
                    for num_str in numbers:
                        num_clean = num_str.replace(' ', '')
                        try:
                            num_val = float(num_clean)
                            # Игнорируем маленькие числа (вероятно это количество или скидка)
                            # и числа без десятичной части, которые меньше 100 (вероятно количество)
                            if num_val >= 10 and ('.' in num_clean or num_val >= 100):
                                valid_prices.append((num_val, num_str))
                        except:
                            continue
                    
                    if valid_prices:
                        # Берем самое большое число как итоговую цену
                        total_price, _ = max(valid_prices, key=lambda x: x[0])
                        quantity = 1.0
                        unit_price = total_price
                    else:
                        logging.warning(f"Could not parse price from text: '{price_text}'")
                        continue
            
            logging.info(f"💾 Сохраняем товар в items: name='{name}', quantity={quantity}, price={total_price}")
            items.append({
                "name": name,
                "quantity": quantity,
                "price": total_price,
                "category": None
            })
        
        # Итоговая сумма
        total_elem = soup.select_one('.ticket_total')
        if not total_elem:
            total_elem = soup.select_one('.total_sum, [class*="total"]')
        total_text = total_elem.get_text(strip=True) if total_elem else ""
        # Убираем все пробелы и переносы строк
        total_text = re.sub(r'\s+', '', total_text)
        total_match = re.search(r'(\d+\.?\d*)', total_text)
        total = float(total_match.group(1)) if total_match else 0.0
        
        logging.info(f"📦 Всего распарсено товаров: {len(items)}")
        for idx, item in enumerate(items):
            logging.info(f"  Товар {idx+1}: name='{item['name']}', quantity={item['quantity']}, price={item['price']}")
        
        # Если не нашли total, суммируем товары
        if total == 0.0 and items:
            total = sum(item["price"] for item in items)
        
        # Если не нашли товары, но есть общая сумма, создаем один товар
        if not items and total > 0:
            items.append({
                "name": "Покупка",
                "quantity": 1.0,
                "price": total,
                "category": None
            })
        
        # Валюта (обычно KZT для Казахстана)
        currency = "KZT"
        
        # Парсим дату
        purchased_at = None
        if date_str:
            try:
                # Формат: "2025-12-03 19:08:23.000000" или "3 дек 2025, 19:08"
                if '.' in date_str:
                    purchased_at = datetime.strptime(date_str.split('.')[0], "%Y-%m-%d %H:%M:%S").isoformat()
                else:
                    # Пытаемся распарсить другой формат
                    purchased_at = datetime.utcnow().isoformat()
            except:
                purchased_at = datetime.utcnow().isoformat()
        
        result = {
            "store": store,
            "merchant_address": address,
            "purchased_at": purchased_at or datetime.utcnow().isoformat(),
            "currency": currency,
            "total": total,
            "tax_amount": None,
            "items": items
        }
        
        logging.info(f"Распарсили ofd.kz HTML: store={store}, items={len(items)}, total={total}")
        return result
        
    except Exception as exc:
        logging.exception(f"Ошибка при парсинге HTML ofd.kz: {exc}")
        return None


async def fetch_receipt_from_qr_url(qr_url: str) -> Optional[Dict[str, Any]]:
    """
    Получает данные чека по URL из QR-кода.
    Пытается получить данные через API endpoint или парсинг HTML.
    """
    from urllib.parse import urlparse, parse_qs
    
    try:
        logging.info(f"Попытка получить данные чека по URL: {qr_url}")
        parsed_url = urlparse(qr_url)
        query_params = parse_qs(parsed_url.query)
        
        # Для consumer.oofd.kz сначала пробуем известный API endpoint
        if "consumer.oofd.kz" in qr_url and all(key in query_params for key in ['i', 'f', 's', 't']):
            api_url = f"{parsed_url.scheme}://{parsed_url.netloc}/api/tickets/get-by-url"
            api_params = {
                't': query_params['t'][0],
                'i': query_params['i'][0],
                'f': query_params['f'][0],
                's': query_params['s'][0],
            }
            
            logging.info(f"Пробуем API endpoint: {api_url}")
            try:
                api_response = requests.get(
                    api_url,
                    params=api_params,
                    timeout=10,
                    verify=False,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Referer": qr_url
                    }
                )
                
                if api_response.status_code == 200:
                    try:
                        api_data = api_response.json()
                        if api_data:
                            # Нормализуем название магазина из API данных
                            if isinstance(api_data, dict) and "store" in api_data:
                                api_data["store"] = normalize_store_name(api_data["store"])
                            logging.info(f"✅ Получены данные через API endpoint: {list(api_data.keys()) if isinstance(api_data, dict) else 'list'}")
                            return api_data
                    except json.JSONDecodeError:
                        logging.warning("API вернул не JSON")
            except Exception as api_exc:
                logging.debug(f"Ошибка при запросе к API endpoint: {api_exc}")
        
        # Если API не сработал, делаем обычный запрос к URL
        response = requests.get(
            qr_url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            verify=False,
            allow_redirects=True
        )
        
        if response.status_code != 200:
            logging.warning(f"Не удалось получить данные: статус {response.status_code}")
            return None
        
        content_type = response.headers.get("Content-Type", "").lower()
        
        # Если это JSON
        if "application/json" in content_type or response.text.strip().startswith("{"):
            try:
                data = response.json()
                # Нормализуем название магазина из JSON данных
                if isinstance(data, dict) and "store" in data:
                    data["store"] = normalize_store_name(data["store"])
                logging.info(f"✅ Получены JSON данные из QR-кода: {list(data.keys())}")
                return data
            except json.JSONDecodeError:
                logging.warning("Ответ не является валидным JSON")
        
        # Если это HTML, пытаемся найти JSON данные в странице
        if "text/html" in content_type:
            html_content = response.text
            
            # Ищем JSON в script тегах
            json_patterns = [
                r'window\.receiptData\s*=\s*({.+?});',
                r'var\s+receipt\s*=\s*({.+?});',
                r'const\s+receipt\s*=\s*({.+?});',
                r'let\s+receipt\s*=\s*({.+?});',
                r'__INITIAL_STATE__\s*=\s*({.+?});',
                r'window\.__data__\s*=\s*({.+?});',
            ]
            
            for pattern in json_patterns:
                match = re.search(pattern, html_content, re.DOTALL | re.IGNORECASE)
                if match:
                    try:
                        json_str = match.group(1)
                        # Пытаемся найти полный JSON объект
                        brace_count = json_str.count('{') - json_str.count('}')
                        if brace_count > 0:
                            remaining = html_content[match.end():]
                            for i, char in enumerate(remaining):
                                if char == '}':
                                    brace_count -= 1
                                    if brace_count == 0:
                                        json_str = json_str + remaining[:i+1]
                                        break
                        
                        data = json.loads(json_str)
                        # Нормализуем название магазина из JSON в HTML
                        if isinstance(data, dict) and "store" in data:
                            data["store"] = normalize_store_name(data["store"])
                        logging.info(f"✅ Найден JSON в HTML: {list(data.keys()) if isinstance(data, dict) else 'list'}")
                        return data
                    except (json.JSONDecodeError, IndexError):
                        continue
            
            # Пробуем парсить HTML напрямую для ofd.kz
            if "ofd1.kz" in qr_url or "oofd.kz" in qr_url or "ofd.kz" in qr_url:
                parsed_data = parse_ofd_kz_html(html_content)
                if parsed_data:
                    items = parsed_data.get("items", [])
                    total = parsed_data.get("total", 0.0)
                    store = parsed_data.get("store", "")
                    
                    if items and total > 0 and store and store != "Неизвестно":
                        logging.info(f"✅ Успешно распарсили HTML: {list(parsed_data.keys())}")
                        return parsed_data
                    else:
                        logging.warning(f"⚠️ HTML парсинг вернул пустые данные: store={store}, items={len(items)}, total={total}")
                
                logging.warning("HTML парсинг не дал результатов. Это может быть SPA приложение.")
                return None
            
            logging.warning("Не удалось найти JSON данные в HTML")
            return None
        
        logging.warning(f"Неизвестный тип контента: {content_type}")
        return None
        
    except requests.exceptions.RequestException as exc:
        logging.exception(f"Ошибка при запросе данных по QR-коду: {exc}")
        return None
    except Exception as exc:
        logging.exception(f"Неожиданная ошибка при получении данных из QR-кода: {exc}")
        return None


def extract_choice_text(response_json: Dict[str, Any]) -> str:
    choices = response_json.get("choices")
    if not choices:
        raise ReceiptParsingError("OpenAI ответ не содержит choices.")
    message = choices[0].get("message", {})
    content = message.get("content")
    if isinstance(content, list):
        content = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        )
    if not isinstance(content, str):
        raise ReceiptParsingError("Не удалось извлечь текст из ответа модели.")
    stripped = content.strip()
    if not stripped:
        raise ReceiptParsingError("Модель вернула пустой ответ.")
    return stripped


def build_parsed_receipt(data: Dict[str, Any]) -> ParsedReceipt:
    items_payload = data.get("items") or []
    items: List[ParsedReceiptItem] = []
    for item in items_payload:
        if not isinstance(item, dict):
            continue
        items.append(
            ParsedReceiptItem(
                name=str(item.get("name") or "Без названия"),
                quantity=safe_float(item.get("quantity"), default=1.0),
                price=safe_float(item.get("price")),
                category=item.get("category"),
            )
        )
    if not items:
        items = [
            ParsedReceiptItem(
                name="Без позиций",
                quantity=1.0,
                price=safe_float(data.get("total")),
            )
        ]
    tax_value = data.get("tax_amount")
    tax_amount = (
        safe_float(tax_value)
        if tax_value not in (None, "")
        else None
    )
    return ParsedReceipt(
        store=str(data.get("store") or "Unknown store"),
        total=safe_float(data.get("total")),
        currency=str(data.get("currency") or "RUB"),
        purchased_at=parse_datetime_flexible(data.get("purchased_at")),
        tax_amount=tax_amount,
        items=items,
        merchant_address=data.get("merchant_address"),
        external_id=data.get("external_id"),
    )


def parse_datetime_flexible(raw_value: Optional[str]) -> datetime:
    if not raw_value:
        return datetime.utcnow()
    value = raw_value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    
    current_year = datetime.utcnow().year
    current_date = datetime.utcnow()
    
    for candidate in (value, f"{value}T00:00:00"):
        try:
            dt = datetime.fromisoformat(candidate)
            # Проверяем, aware или naive datetime
            if dt.tzinfo is not None:
                # Aware datetime - сравниваем с aware
                from datetime import timezone
                now = datetime.now(timezone.utc)
                # Если дата в будущем, уменьшаем год на 1
                if dt > now:
                    dt = dt.replace(year=dt.year - 1)
                # Возвращаем naive datetime для совместимости
                dt = dt.replace(tzinfo=None)
            else:
                # Naive datetime - сравниваем с naive
                now = datetime.utcnow()
                # Если дата в будущем, уменьшаем год на 1
                if dt > now:
                    dt = dt.replace(year=dt.year - 1)
            
            # Если год слишком старый (более чем на 1 год назад), корректируем
            # Например, если сейчас 2026, а дата 2023, то это скорее всего ошибка парсинга
            # Используем текущий год или предыдущий, в зависимости от того, прошла ли эта дата
            if dt.year < current_year - 1:
                # Год слишком старый, заменяем на текущий или предыдущий
                test_date_current = dt.replace(year=current_year)
                test_date_previous = dt.replace(year=current_year - 1)
                # Выбираем год так, чтобы дата не была в будущем
                if test_date_current > current_date:
                    dt = test_date_previous
                else:
                    dt = test_date_current
            
            return dt
        except ValueError:
            continue
    try:
        dt = datetime.strptime(value, "%Y-%m-%d")
        # Если дата в будущем, уменьшаем год на 1
        if dt > datetime.utcnow():
            dt = dt.replace(year=dt.year - 1)
        # Если год слишком старый, корректируем
        if dt.year < current_year - 1:
            test_date_current = dt.replace(year=current_year)
            test_date_previous = dt.replace(year=current_year - 1)
            if test_date_current > current_date:
                dt = test_date_previous
            else:
                dt = test_date_current
        return dt
    except ValueError:
        # Пробуем другие форматы даты
        try:
            # Формат DD.MM.YYYY или DD.MM.YY
            dt = datetime.strptime(value, "%d.%m.%Y")
            if dt > datetime.utcnow():
                dt = dt.replace(year=dt.year - 1)
            # Если год слишком старый, корректируем
            if dt.year < current_year - 1:
                test_date_current = dt.replace(year=current_year)
                test_date_previous = dt.replace(year=current_year - 1)
                if test_date_current > current_date:
                    dt = test_date_previous
                else:
                    dt = test_date_current
            return dt
        except ValueError:
            try:
                dt = datetime.strptime(value, "%d.%m.%y")
                # Если год двухзначный, интерпретируем правильно
                if dt.year > datetime.utcnow().year:
                    dt = dt.replace(year=dt.year - 100)
                if dt > datetime.utcnow():
                    dt = dt.replace(year=dt.year - 1)
                # Если год слишком старый, корректируем
                if dt.year < current_year - 1:
                    test_date_current = dt.replace(year=current_year)
                    test_date_previous = dt.replace(year=current_year - 1)
                    if test_date_current > current_date:
                        dt = test_date_previous
                    else:
                        dt = test_date_current
                return dt
            except ValueError:
                pass
    return datetime.utcnow()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_store_name(store_name: str) -> str:
    """
    Нормализует название магазина, сокращая длинные юридические формы и убирая лишние префиксы.
    Например: 
    - "ТОВАРИЩЕСТВО С ОГРАНИЧЕННОЙ ОТВЕТСТВЕННОСТЬЮ "АЛМАСТОР"" -> "ТОО "АЛМАСТОР""
    - "Филиал №81 ТОВАРИЩЕСТВО С ОГРАНИЧЕННОЙ ОТВЕТСТВЕННОСТЬЮ "АЛМАСТОР"" -> "ТОО "АЛМАСТОР""
    """
    if not store_name:
        return store_name
    
    normalized = store_name
    
    # Убираем префиксы типа "Филиал №81", "Магазин №", "Точка продаж" и т.д.
    prefix_patterns = [
        r'^ФИЛИАЛ\s+№?\s*\d+\s+',
        r'^ФИЛИАЛ\s+№?\s*\d+\.\s*',
        r'^МАГАЗИН\s+№?\s*\d+\s+',
        r'^ТОЧКА\s+ПРОДАЖ\s+№?\s*\d+\s+',
        r'^ТП\s+№?\s*\d+\s+',
        r'^ОТДЕЛ\s+№?\s*\d+\s+',
        r'^ПОДРАЗДЕЛЕНИЕ\s+№?\s*\d+\s+',
    ]
    for pattern in prefix_patterns:
        normalized = re.sub(pattern, '', normalized, flags=re.IGNORECASE)
    
    # Обрабатываем обрезанные названия типа "Товарищество с ограниченной о"
    # Заменяем на полную форму для последующей нормализации
    if re.search(r'\bТОВАРИЩЕСТВО\s+С\s+ОГРАНИЧЕННОЙ\s+[ОО]\s*$', normalized, re.IGNORECASE):
        normalized = re.sub(r'\bТОВАРИЩЕСТВО\s+С\s+ОГРАНИЧЕННОЙ\s+[ОО]\s*$', 
                           'ТОВАРИЩЕСТВО С ОГРАНИЧЕННОЙ ОТВЕТСТВЕННОСТЬЮ', 
                           normalized, flags=re.IGNORECASE)
    
    # Словарь замен для организационно-правовых форм (регистронезависимо)
    replacements = {
        r'\bТОВАРИЩЕСТВО\s+С\s+ОГРАНИЧЕННОЙ\s+ОТВЕТСТВЕННОСТЬЮ\b': 'ТОО',
        r'\bТОО\s+С\s+ОГРАНИЧЕННОЙ\s+ОТВЕТСТВЕННОСТЬЮ\b': 'ТОО',
        r'\bЗАКРЫТОЕ\s+АКЦИОНЕРНОЕ\s+ОБЩЕСТВО\b': 'ЗАО',
        r'\bОТКРЫТОЕ\s+АКЦИОНЕРНОЕ\s+ОБЩЕСТВО\b': 'ОАО',
        r'\bПУБЛИЧНОЕ\s+АКЦИОНЕРНОЕ\s+ОБЩЕСТВО\b': 'ПАО',
        r'\bОБЩЕСТВО\s+С\s+ОГРАНИЧЕННОЙ\s+ОТВЕТСТВЕННОСТЬЮ\b': 'ООО',
        r'\bИНДИВИДУАЛЬНЫЙ\s+ПРЕДПРИНИМАТЕЛЬ\b': 'ИП',
        r'\bИНДИВИДУАЛЬНЫЙ\s+ПРЕДПРИНИМАТЕЛЬ\s+ИП\b': 'ИП',
        r'\bАКЦИОНЕРНОЕ\s+ОБЩЕСТВО\b': 'АО',
        r'\bПРОИЗВОДСТВЕННЫЙ\s+КООПЕРАТИВ\b': 'ПК',
        r'\bПОТРЕБИТЕЛЬСКИЙ\s+КООПЕРАТИВ\b': 'ПК',
        r'\bПОЛНОЕ\s+ТОВАРИЩЕСТВО\b': 'ПТ',
        r'\bКОММАНДИТНОЕ\s+ТОВАРИЩЕСТВО\b': 'КТ',
        r'\bОБЩЕСТВО\s+С\s+ДОПОЛНИТЕЛЬНОЙ\s+ОТВЕТСТВЕННОСТЬЮ\b': 'ОДО',
    }
    
    for pattern, replacement in replacements.items():
        normalized = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
    
    # Убираем лишние пробелы и обрезаем слишком длинные названия
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    # Если название все еще слишком длинное (более 50 символов), обрезаем
    if len(normalized) > 50:
        # Пытаемся найти кавычки с названием компании
        quoted_match = re.search(r'["""]([^"""]+)["""]', normalized)
        if quoted_match:
            company_name = quoted_match.group(1)
            # Берем аббревиатуру + название в кавычках
            abbrev_match = re.search(r'^([А-Я]{2,4})\s*', normalized)
            if abbrev_match:
                abbrev = abbrev_match.group(1)
                normalized = f'{abbrev} "{company_name}"'
            else:
                normalized = f'"{company_name}"'
        else:
            # Просто обрезаем до 50 символов
            normalized = normalized[:47] + "..."
    
    return normalized


def build_receipt_payload(user_id: int, parsed: ParsedReceipt) -> Dict[str, Any]:
    # Формируем хеш из user_id, даты/времени оплаты и суммы
    receipt_hash = calculate_hash(
        f"{user_id}|{parsed.purchased_at.isoformat()}|{parsed.total}"
    )
    
    # Нормализуем название магазина
    normalized_store = normalize_store_name(parsed.store)
    
    # ВАЖНО: сохраняем все позиции, включая дубликаты
    # Не удаляем дубликаты, так как они могут быть реальными повторяющимися позициями в чеке
    items_list = [asdict(item) for item in parsed.items]
    
    # Логируем информацию о позициях перед сохранением
    logging.info(f"💾 Сохранение чека: items_count={len(items_list)}, total={parsed.total}")
    if len(items_list) > 0:
        # Проверяем наличие дубликатов по названию
        item_names = [item.get("name", "") for item in items_list]
        name_counts = {}
        for name in item_names:
            name_counts[name] = name_counts.get(name, 0) + 1
        duplicates = {name: count for name, count in name_counts.items() if count > 1}
        if duplicates:
            logging.info(f"📋 Дубликаты позиций в чеке (это нормально): {duplicates}")
    
    return {
        "user_id": user_id,
        "store": normalized_store,
        "total": parsed.total,
        "currency": parsed.currency,
        "purchased_at": parsed.purchased_at.isoformat(),
        "tax_amount": parsed.tax_amount,
        "items": items_list,  # Сохраняем все позиции, включая дубликаты
        "receipt_hash": receipt_hash,
        "external_id": parsed.external_id,
        "merchant_address": parsed.merchant_address,
    }


def build_expenses_from_receipt_items(receipt_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Создает список expenses для каждой позиции (item) из чека.
    Каждая строка чека = отдельная запись в expenses.
    """
    items = receipt_record.get("items", [])
    if not items or not isinstance(items, list) or len(items) == 0:
        # Если нет items, создаем один expense на весь чек (для обратной совместимости)
        return [build_expense_payload_from_receipt(receipt_record)]
    
    user_id = receipt_record.get("user_id")
    receipt_id = receipt_record.get("id")
    receipt_hash = receipt_record.get("receipt_hash", "")
    purchased_at = receipt_record.get("purchased_at", "")
    currency = receipt_record.get("currency", "RUB")
    store_name = receipt_record.get("store", "")
    normalized_store = normalize_store_name(store_name) if store_name else ""
    period = (purchased_at or "")[:7]
    
    expenses = []
    for idx, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        
        item_name = item.get("name", "Без названия")
        item_price = float(item.get("price", 0.0))
        item_quantity = float(item.get("quantity", 1.0))
        item_category = item.get("category")
        
        # Формируем уникальный хеш для каждой позиции
        # Используем receipt_hash, индекс позиции и название для уникальности
        expense_hash = calculate_hash(
            f"{user_id}|receipt|{receipt_hash}|item_{idx}|{item_name}|{item_price}"
        )
        
        # Формируем note с информацией о количестве, если quantity > 1
        note_parts = [item_name]
        if item_quantity > 1.0:
            note_parts.append(f"×{item_quantity}")
        note = " ".join(note_parts)
        
        payload = {
            "user_id": user_id,
            "source": "receipt",
            "store": normalized_store,
            "amount": item_price,  # Сумма за позицию (уже с учетом quantity)
            "currency": currency,
            "date": purchased_at,
            "receipt_id": receipt_id,
            "expense_hash": expense_hash,
            "status": "pending_review",
            "period": period,
            "note": note,
        }
        
        # Добавляем категорию из item, если есть
        if item_category:
            payload["category"] = item_category
        else:
            payload["category"] = "Другое"
        
        expenses.append(payload)
        logging.info(f"📝 Создан expense для позиции: {item_name}, цена={item_price}, количество={item_quantity}, категория={item_category}")
    
    logging.info(f"✅ Создано {len(expenses)} expenses из {len(items)} позиций чека")
    return expenses


def build_expense_payload_from_receipt(receipt_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Устаревшая функция - создает один expense на весь чек.
    Используется только для обратной совместимости, когда нет items.
    """
    expense_hash = calculate_hash(
        f"{receipt_record.get('user_id')}|receipt|{receipt_record.get('receipt_hash')}"
    )
    
    # Определяем категорию из items чека (берем самую частую категорию)
    category = None
    items = receipt_record.get("items", [])
    if items and isinstance(items, list):
        category_counts = {}
        items_with_category = 0
        items_without_category = 0
        for item in items:
            if isinstance(item, dict):
                item_category = item.get("category")
                if item_category:
                    category_counts[item_category] = category_counts.get(item_category, 0) + 1
                    items_with_category += 1
                else:
                    items_without_category += 1
        
        logging.info(f"Извлечение категории из чека: всего items={len(items)}, с категорией={items_with_category}, без категории={items_without_category}, категории={category_counts}")
        
        if category_counts:
            # Берем самую частую категорию
            category = max(category_counts.items(), key=lambda x: x[1])[0]
            logging.info(f"✅ Определена категория расхода: {category} (встречается {category_counts[category]} раз(а))")
        else:
            logging.warning(f"⚠️ Не найдено ни одной категории в items чека (store={receipt_record.get('store')}, items={len(items)})")
    else:
        logging.warning(f"⚠️ Нет items в чеке для определения категории (store={receipt_record.get('store')})")
    
    # Нормализуем название магазина
    store_name = receipt_record.get("store", "")
    normalized_store = normalize_store_name(store_name) if store_name else ""
    
    payload = {
        "user_id": receipt_record.get("user_id"),
        "source": "receipt",
        "store": normalized_store,
        "amount": receipt_record.get("total"),
        "currency": receipt_record.get("currency"),
        "date": receipt_record.get("purchased_at"),
        "receipt_id": receipt_record.get("id"),
        "expense_hash": expense_hash,
        "status": "pending_review",
        "period": (receipt_record.get("purchased_at") or "")[:7],
    }
    
    # Добавляем категорию если определили
    if category:
        payload["category"] = category
    else:
        logging.warning(f"⚠️ Расход будет сохранен без категории (store={receipt_record.get('store')}, amount={receipt_record.get('total')})")
    
    return payload


def build_bank_payload(user_id: int, txn: ParsedBankTransaction) -> Dict[str, Any]:
    txn_hash = calculate_hash(
        f"{user_id}|{txn.source_id}|{txn.amount}|{txn.booked_at.isoformat()}"
    )
    return {
        "user_id": user_id,
        "amount": txn.amount,
        "currency": txn.currency,
        "merchant": txn.merchant,
        "booked_at": txn.booked_at.isoformat(),
        "description": txn.description,
        "source_id": txn.source_id,
        "transaction_hash": txn_hash,
    }


async def reconcile_transactions(
    gateway: SupabaseGateway,
    user_id: int,
    bank_records: List[Dict[str, Any]],
) -> None:
    """Simplified reconciliation placeholder."""

    if not bank_records:
        return

    for record in bank_records:
        expense_payload = {
            "user_id": user_id,
            "source": "bank",
            "store": record.get("merchant"),
            "amount": record.get("amount"),
            "currency": record.get("currency"),
            "date": record.get("booked_at"),
            "bank_transaction_id": record.get("id"),
            "expense_hash": calculate_hash(
                f"{user_id}|bank|{record.get('transaction_hash')}"
            ),
            "status": "pending_review",
            "period": (record.get("booked_at") or "")[:7],
        }
        expense_result = await gateway.record_expense(expense_payload)
        if expense_result.get("duplicate"):
            logging.info(f"Skipped duplicate expense from bank transaction: {record.get('transaction_hash')}")


def calculate_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


async def parse_bank_statement(file_bytes: bytes) -> List[ParsedBankTransaction]:
    """Parse CSV/PDF bank statements into normalized transactions."""

    return await asyncio.to_thread(_parse_bank_statement_sync, file_bytes)


def _parse_bank_statement_sync(file_bytes: bytes) -> List[ParsedBankTransaction]:
    rows = _load_statement_rows(file_bytes)
    transactions: List[ParsedBankTransaction] = []
    for idx, row in enumerate(rows):
        normalized_row = _normalize_statement_row(row)
        txn = _build_transaction_from_row(normalized_row, idx)
        if txn:
            transactions.append(txn)

    if not transactions:
        raise StatementParsingError("Выписка не содержит распознанных операций.")
    return transactions


def _load_statement_rows(file_bytes: bytes) -> List[Dict[str, Any]]:
    header = file_bytes[:4]
    if header.startswith(b"%PDF"):
        return _extract_pdf_rows(file_bytes)
    if header.startswith(b"PK\x03\x04"):
        raise StatementParsingError("XLSX пока не поддерживается — экспортируйте CSV.")

    text = _decode_statement_text(file_bytes)
    delimiter = _detect_delimiter(text)
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if not reader.fieldnames:
        raise StatementParsingError("Не удалось определить структуру файла.")
    rows = list(reader)
    if not rows:
        raise StatementParsingError("Файл не содержит строк операций.")
    return rows


def _extract_pdf_rows(file_bytes: bytes) -> List[Dict[str, Any]]:
    if not PDF_SUPPORT or pdfplumber is None:
        raise StatementParsingError(
            "Для разбора PDF требуется пакет pdfplumber. Установите зависимости из requirements."
        )

    rows: List[Dict[str, Any]] = []
    text_lines: List[str] = []
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:  # type: ignore[arg-type]
        for page in pdf.pages:
            text = page.extract_text() or ""
            if text:
                text_lines.extend(text.splitlines())
            for table in page.extract_tables() or []:
                if not table or len(table) < 2:
                    continue
                headers = [_normalize_pdf_header(cell) for cell in table[0]]
                if not any(headers):
                    continue
                for raw_row in table[1:]:
                    row_dict: Dict[str, Any] = {}
                    for header, cell in zip(headers, raw_row):
                        key = header.strip().lower()
                        if not key:
                            continue
                        value = cell.strip() if isinstance(cell, str) else cell
                        row_dict[key] = value
                    if row_dict:
                        rows.append(row_dict)

    if rows:
        return rows

    fallback_rows = _parse_pdf_text_lines(text_lines)
    if not fallback_rows:
        raise StatementParsingError(
            "Не удалось извлечь табличные данные из PDF. Попробуйте экспортировать CSV."
        )
    return fallback_rows


def _parse_pdf_text_lines(lines: List[str]) -> List[Dict[str, Any]]:
    structured: List[Dict[str, Any]] = []
    headers: Optional[List[str]] = None
    for line in lines:
        normalized_line = " ".join(line.strip().split())
        if not normalized_line:
            continue
        segments = [segment.strip() for segment in _split_text_row(normalized_line)]
        if not segments:
            continue
        if headers is None:
            headers = [_normalize_pdf_header(cell) for cell in segments]
            continue
        if len(segments) != len(headers):
            continue
        row_dict = {
            header: value
            for header, value in zip(headers, segments)
            if header
        }
        if row_dict:
            structured.append(row_dict)
    return structured


def _split_text_row(line: str) -> List[str]:
    return [part for part in re.split(r"\s{2,}", line) if part]


def _normalize_pdf_header(cell: Optional[str]) -> str:
    if cell is None:
        return ""
    return " ".join(str(cell).strip().lower().split())


def _decode_statement_text(file_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1251", "windows-1251", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise StatementParsingError("Не удалось декодировать файл выписки.")


def _detect_delimiter(text: str) -> str:
    sample = text[:2000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        for delimiter in (",", ";", "\t", "|"):
            if delimiter in sample:
                return delimiter
    return ","


def _normalize_statement_row(row: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in row.items():
        if key is None:
            continue
        normalized_key = key.strip().lower()
        if isinstance(value, str):
            normalized[normalized_key] = value.strip()
        else:
            normalized[normalized_key] = value
    return normalized


def _build_transaction_from_row(row: Dict[str, Any], idx: int) -> Optional[ParsedBankTransaction]:
    amount = _extract_amount(row)
    if amount is None:
        return None

    currency = _first_value(row, ("currency", "валюта", "код валюты"))
    if currency:
        currency = currency.upper()
    else:
        currency = _infer_currency(row) or "RUB"
    merchant = (
        _first_value(
            row,
            (
                "merchant",
                "store",
                "контрагент",
                "получатель",
                "отправитель",
                "инфо",
            ),
        )
        or _first_value(row, _DESCRIPTION_KEYS)
        or "Unknown merchant"
    )
    description = _first_value(row, _DESCRIPTION_KEYS) or merchant
    booked_raw = _first_value(
        row,
        (
            "date",
            "booked_at",
            "дата",
            "дата операции",
            "posting date",
            "operation date",
            "valuedate",
        ),
    )
    booked_at = parse_datetime_flexible(booked_raw) if booked_raw else datetime.utcnow()
    source_id = _first_value(
        row,
        (
            "id",
            "transaction id",
            "номер документа",
            "doc number",
            "reference",
        ),
    ) or calculate_hash(f"{idx}|{merchant}|{booked_at.isoformat()}|{amount}")

    return ParsedBankTransaction(
        amount=amount,
        currency=currency,
        merchant=merchant,
        booked_at=booked_at,
        description=description,
        source_id=str(source_id),
    )


def _extract_amount(row: Dict[str, Any]) -> Optional[float]:
    amount_raw = _first_value(
        row,
        (
            "amount",
            "sum",
            "сумма",
            "итого",
            "сумма операции",
            "сумма операции (р.)",
        ),
    )
    if amount_raw:
        return safe_float(_normalize_number(amount_raw))

    debit_raw = _first_value(row, ("debit", "расход", "списание"))
    credit_raw = _first_value(row, ("credit", "приход", "зачисление"))
    debit = safe_float(_normalize_number(debit_raw), default=0.0)
    credit = safe_float(_normalize_number(credit_raw), default=0.0)
    if debit and not credit:
        return -abs(debit)
    if credit and not debit:
        return abs(credit)
    if debit or credit:
        return credit - debit
    return None


def _normalize_number(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, float)):
        return str(value)
    cleaned = str(value)
    for char in ("\xa0", " ", "\t"):
        cleaned = cleaned.replace(char, "")
    cleaned = cleaned.replace(",", ".")
    allowed = set("0123456789.-")
    cleaned = "".join(ch for ch in cleaned if ch in allowed)
    return cleaned


_DESCRIPTION_KEYS = (
    "description",
    "назначение платежа",
    "назначение",
    "детали",
    "comment",
    "описание",
    "details",
)


def _first_value(row: Dict[str, Any], keys: tuple) -> Optional[str]:
    for key in keys:
        if key in row and row[key] not in (None, ""):
            value = row[key]
            return str(value)
    return None


# Словарь вариантов написания тысяч для разных валют
_THOUSAND_VARIANTS = {
    "RUB": ["т.р", "т р", "тр", "тыс руб", "тыс.руб", "тысяч руб", "тысяч рублей", "тыс. рублей", "тыс р", "тыс.р"],
    "KZT": ["т.тг", "т тг", "ттг", "тыс тг", "тыс.тг", "тысяч тг", "тысяч тенге", "тыс. тенге", "тыс т", "тыс.т"],
    "USD": ["тыс долл", "тыс.долл", "тысяч долл", "тысяч долларов", "тыс. долларов", "тыс usd", "тыс.usd", "тысяч usd", "тыс $", "тыс.$", "т.долл", "т долл", "тдолл", "thousand usd", "k usd", "k$"],
    "EUR": ["тыс евро", "тыс.евро", "тысяч евро", "тысяч евр", "тыс. евро", "тыс eur", "тыс.eur", "тысяч eur", "тыс €", "тыс.€", "т.евро", "т евро", "тевро", "thousand eur", "k eur", "k€"],
    "GBP": ["тыс фунт", "тыс.фунт", "тысяч фунт", "тысяч фунтов", "тыс. фунтов", "тыс gbp", "тыс.gbp", "тысяч gbp", "тыс £", "тыс.£", "т.фунт", "т фунт", "тфунт", "thousand gbp", "k gbp", "k£"],
    "GEL": ["тыс лари", "тыс.лари", "тысяч лари", "тыс gel", "тыс.gel", "тысяч gel", "тыс ₾", "тыс.₾", "т.лари", "т лари", "тлари", "k gel", "k₾"],
    "BYN": ["тыс бел", "тыс.бел", "тысяч бел", "тысяч белорусских", "тыс byn", "тыс.byn", "тысяч byn", "т.бел", "т бел", "тбел", "k byn"],
    "KGS": ["тыс сом", "тыс.сом", "тысяч сом", "тысяч сомов", "тыс kgs", "тыс.kgs", "тысяч kgs", "т.сом", "т сом", "тсом", "k kgs"],
    "CNY": ["тыс юань", "тыс.юань", "тысяч юань", "тысяч юаней", "тыс cny", "тыс.cny", "тысяч cny", "тыс ¥", "тыс.¥", "т.юань", "т юань", "тюань", "k cny", "k¥"],
    "CHF": ["тыс франк", "тыс.франк", "тысяч франк", "тысяч франков", "тыс chf", "тыс.chf", "тысяч chf", "т.франк", "т франк", "тфранк", "k chf"],
    "AED": ["тыс дирх", "тыс.дирх", "тысяч дирх", "тысяч дирхамов", "тыс aed", "тыс.aed", "тысяч aed", "т.дирх", "т дирх", "тдирх", "k aed"],
    "CAD": ["тыс канад", "тыс.канад", "тысяч канад", "тысяч канадских", "тыс cad", "тыс.cad", "тысяч cad", "т.канад", "т канад", "тканад", "k cad"],
    "AUD": ["тыс австрал", "тыс.австрал", "тысяч австрал", "тысяч австралийских", "тыс aud", "тыс.aud", "тысяч aud", "т.австрал", "т австрал", "тавстрал", "k aud"],
}

# Создаем общий паттерн для всех вариантов тысяч
_thousand_pattern = "|".join([
    "к", "k", "тыс", "тысяч", "thousand",
    *[re.escape(v) for variants in _THOUSAND_VARIANTS.values() for v in variants]
])

MANUAL_AMOUNT_PATTERN = re.compile(
    r"(?P<amount>-?\d+[.,]?\d{0,2})\s*(?P<multiplier>" + _thousand_pattern + r")?\s*(?P<currency>₽|рублей|рубля|рубль|руб|р\.?|rub|р\b|₸|тг|т\b|kzt|\$|долларов|доллара|доллар|долл|дол|usd|д\b|евро|eur|е\b|€|byn|сом|kgs|фунтов|фунта|фунт|gbp|£|лари|gel|₾|юаней|юань|cny|¥|франков|франк|chf|дирхамов|дирхам|aed|cad|aud)?",
    re.IGNORECASE,
)
MANUAL_DATE_PATTERN = re.compile(r"(\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?)")


def parse_manual_expense(text: str, default_currency: str = "RUB") -> Optional[ParsedManualExpense]:
    cleaned = " ".join((text or "").strip().split())
    if len(cleaned) < 3:
        return None
    amount_match = MANUAL_AMOUNT_PATTERN.search(cleaned)
    if not amount_match:
        return None
    
    # Логируем для отладки
    logging.debug(f"parse_manual_expense: input='{text}', cleaned='{cleaned}', amount_match='{amount_match.group(0)}'")
    amount = safe_float(amount_match.group("amount"))
    
    # Обработка множителя (тысячи для всех валют)
    multiplier = amount_match.group("multiplier")
    currency_from_multiplier = None
    
    if multiplier:
        multiplier_lower = multiplier.lower().strip()
        # Проверяем общие варианты (к, k, тыс, тысяч, thousand)
        if multiplier_lower in ("к", "k", "тыс", "тысяч", "thousand"):
            amount = amount * 1000
        else:
            # Проверяем варианты для конкретных валют
            for currency_code, variants in _THOUSAND_VARIANTS.items():
                for variant in variants:
                    # Нормализуем вариант (убираем точки и пробелы для сравнения)
                    variant_normalized = re.sub(r'[.\s]+', '', variant.lower())
                    multiplier_normalized = re.sub(r'[.\s]+', '', multiplier_lower)
                    
                    # Проверяем точное совпадение или вхождение
                    if variant_normalized == multiplier_normalized or variant.lower() in multiplier_lower:
                        amount = amount * 1000
                        currency_from_multiplier = currency_code
                        break
                if currency_from_multiplier:
                    break
    
    token_currency = amount_match.group("currency") or ""
    
    detected_currency = (
        currency_from_multiplier
        or _currency_from_value(token_currency)
        or default_currency
    )
    date_match = MANUAL_DATE_PATTERN.search(cleaned)
    occurred_at = (
        _parse_manual_date(date_match.group(1)) if date_match else datetime.utcnow()
    )
    # Извлекаем описание (все, что не сумма, множитель и не дата)
    # Находим позицию суммы в тексте
    amount_start = amount_match.start()
    amount_end = amount_match.end()
    amount_full_match = amount_match.group(0)
    
    # Если сумма в начале текста, описание - это всё после суммы
    if amount_start == 0:
        description_text = cleaned[amount_end:].strip()
    # Если сумма в конце текста, описание - это всё до суммы
    elif amount_end >= len(cleaned):
        description_text = cleaned[:amount_start].strip()
    # Если сумма в середине, берём всё до суммы (описание обычно идёт перед суммой)
    else:
        description_text = cleaned[:amount_start].strip()
    
    # Удаляем дату, если она была найдена
    if date_match:
        date_match_text = date_match.group(0)
        description_text = re.sub(re.escape(date_match_text), "", description_text, count=1)
    
    # Убираем валюту из описания, если она там есть отдельно (на случай, если она была написана отдельно от суммы)
    if token_currency:
        # Удаляем валюту только если она стоит отдельно (не часть другого слова)
        currency_pattern = r'\b' + re.escape(token_currency) + r'\b'
        description_text = re.sub(currency_pattern, "", description_text, count=1, flags=re.IGNORECASE)
    
    # Очищаем описание от лишних пробелов и знаков препинания
    description = re.sub(r'\s+', ' ', description_text).strip(" ,.-")
    # Если описание пустое или слишком короткое, используем значение по умолчанию
    if not description or len(description) < 2:
        description = "Без описания"
    
    # Логируем результат парсинга для отладки
    logging.debug(f"parse_manual_expense: input='{text}', cleaned='{cleaned}', amount_match='{amount_full_match}' (pos {amount_start}-{amount_end}), description='{description}', amount={amount}, currency={detected_currency}")
    
    return ParsedManualExpense(
        description=description,
        amount=amount,
        currency=detected_currency,
        occurred_at=occurred_at,
        store=None,  # Магазин не определяется автоматически
        note=cleaned,
    )


def _parse_manual_date(token: str) -> datetime:
    token = token.replace("-", ".").replace("/", ".")
    parts = token.split(".")
    if len(parts) == 2:
        parts.append(str(datetime.utcnow().year))
    normalized = ".".join(parts)
    for fmt in ("%d.%m.%Y", "%d.%m.%y"):
        try:
            dt = datetime.strptime(normalized, fmt)
            return dt
        except ValueError:
            continue
    return datetime.utcnow()


def format_manual_summary(parsed: ParsedManualExpense) -> str:
    lines = [f"• Описание: {parsed.description}"]
    if parsed.store:
        lines.append(f"• Магазин: {parsed.store}")
    lines.append(f"• Дата: {parsed.occurred_at.strftime('%Y-%m-%d')}")
    lines.append(f"• Сумма: {parsed.amount:.2f} {parsed.currency}")
    if parsed.category:
        lines.append(f"• Категория: {parsed.category}")
    return "\n".join(lines)


def build_manual_expense_payload(user_id: int, parsed: ParsedManualExpense) -> Dict[str, Any]:
    # Используем description для хеша
    expense_hash = calculate_hash(
        f"{user_id}|manual|{parsed.description}|{parsed.occurred_at.isoformat()}|{parsed.amount}"
    )
    payload = {
        "user_id": user_id,
        "source": "manual",
        "store": None,  # Для ручного ввода store не заполняется
        "amount": parsed.amount,
        "currency": parsed.currency,
        "date": parsed.occurred_at.isoformat(),
        "note": parsed.note or parsed.description,
        "expense_hash": expense_hash,
        "status": "pending_review",
        "period": parsed.occurred_at.strftime("%Y-%m"),
    }
    # Добавляем категорию (обязательно должна быть указана)
    if parsed.category:
        payload["category"] = parsed.category
    else:
        payload["category"] = "Другое"  # Категория по умолчанию, если не указана
    return payload


def _infer_currency(row: Dict[str, Any]) -> Optional[str]:
    for value in row.values():
        code = _currency_from_value(value)
        if code:
            return code
    return None


def _currency_from_value(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return None
    text = str(value).strip()
    if not text:
        return None
    
    lower_text = text.lower()
    
    # Сначала проверяем токены (более длинные варианты должны проверяться первыми)
    # Сортируем токены по длине в убывающем порядке, чтобы более длинные совпадали первыми
    sorted_tokens = sorted(_CURRENCY_TOKENS.items(), key=lambda x: len(x[0]), reverse=True)
    for token, code in sorted_tokens:
        # Для одиночных букв используем границы слов, для остальных - просто вхождение
        if len(token) == 1:
            # Одиночная буква - проверяем как отдельное слово
            pattern = r'\b' + re.escape(token) + r'\b'
            if re.search(pattern, lower_text, re.IGNORECASE):
                return code
        else:
            # Многосимвольный токен - проверяем вхождение
            if token in lower_text:
                return code
    
    # Затем проверяем символы
    for symbol, code in _CURRENCY_SYMBOLS.items():
        if symbol in text:
            return code
    
    # И наконец проверяем ISO коды
    iso_match = re.search(r"\b([A-Z]{3})\b", text.upper())
    if iso_match:
        candidate = iso_match.group(1)
        if candidate in _KNOWN_ISO_CODES:
            return "RUB" if candidate == "RUR" else candidate
    return None


_CURRENCY_SYMBOLS = {
    "₽": "RUB",
    "р.": "RUB",
    "руб": "RUB",
    "₸": "KZT",
    "тг": "KZT",
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "₾": "GEL",
    "р": "RUB",  # Одна буква "р" для рублей
    "д": "USD",  # Одна буква "д" для долларов
    "е": "EUR",  # Одна буква "е" для евро
}


_CURRENCY_TOKENS = {
    # Рубли
    "rub": "RUB",
    "rur": "RUB",
    "рос. руб": "RUB",
    "российский рубль": "RUB",
    "рубль": "RUB",
    "рублей": "RUB",
    "рубля": "RUB",
    "рубл": "RUB",
    "р": "RUB",  # Одна буква "р"
    # Тенге
    "тенге": "KZT",
    "казахстан": "KZT",
    "kzt": "KZT",
    "тг": "KZT",
    "т": "KZT",  # Одна буква "т"
    # Белорусские рубли
    "byn": "BYN",
    "бел. руб": "BYN",
    "белорусский рубль": "BYN",
    # Сомы
    "som": "KGS",
    "kgs": "KGS",
    "сом": "KGS",
    # Доллары
    "usd": "USD",
    "доллар": "USD",
    "долларов": "USD",
    "доллара": "USD",
    "долл": "USD",
    "дол": "USD",
    "д": "USD",  # Одна буква "д"
    "$": "USD",
    # Евро
    "eur": "EUR",
    "евро": "EUR",
    "е": "EUR",  # Одна буква "е"
    "€": "EUR",
    # Фунты
    "gbp": "GBP",
    "фунт": "GBP",
    "фунтов": "GBP",
    "фунта": "GBP",
    "£": "GBP",
    # Лари
    "gel": "GEL",
    "лари": "GEL",
    "₾": "GEL",
    # Юани
    "cny": "CNY",
    "юань": "CNY",
    "юаней": "CNY",
    "¥": "CNY",
    # Франки
    "chf": "CHF",
    "франк": "CHF",
    "франков": "CHF",
    # Дирхамы
    "aed": "AED",
    "дирхам": "AED",
    "дирхамов": "AED",
    # Канадские доллары
    "cad": "CAD",
    "канадский доллар": "CAD",
    # Австралийские доллары
    "aud": "AUD",
    "австралийский доллар": "AUD",
}


_KNOWN_ISO_CODES = {
    "RUB",
    "RUR",
    "USD",
    "EUR",
    "KZT",
    "BYN",
    "KGS",
    "GBP",
    "CNY",
    "CHF",
    "AED",
    "CAD",
    "AUD",
    "GEL",
}


async def main() -> None:
    bot = ExpenseCatBot.from_env()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
