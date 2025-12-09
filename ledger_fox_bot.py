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
from aiogram.types import BufferedInputFile, Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton, BotCommand
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
log_level_name = os.getenv("LEDGERFOX_LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_name, logging.INFO)
logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")
log_dir = Path(os.getenv("LEDGERFOX_LOG_DIR", "logs"))
log_dir.mkdir(parents=True, exist_ok=True)
file_handler = logging.FileHandler(log_dir / "ocr_debug.log")
file_handler.setLevel(log_level)
file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logging.getLogger().addHandler(file_handler)
logging.info("LedgerFox logging configured at %s", log_level_name)
logging.info("Active preprocess pipeline hash marker: portrait-fix-v3")


class ReceiptStates(StatesGroup):
    waiting_for_photo = State()
    waiting_for_confirmation = State()


class StatementStates(StatesGroup):
    waiting_for_statement = State()


class DeleteStates(StatesGroup):
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
    store: str
    amount: float
    currency: str
    occurred_at: datetime
    note: Optional[str] = None


@dataclass
class ProcessingResult:
    success: bool
    summary: Optional[str] = None
    error: Optional[str] = None
    parsed_receipt: Optional[ParsedReceipt] = None
    receipt_payload: Optional[Dict[str, Any]] = None


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
    "\n\nВАЖНО: "
    "- \"quantity\" - это количество товара (например, 2, 3, 1.5)"
    "- \"price\" - это ОБЩАЯ сумма за позицию (количество × цена за единицу), а не цена за единицу"
    "- Для каждого товара определи категорию на основе его названия. "
    "Используй категории: \"Продукты\", \"Мясо/Рыба\", \"Молочные продукты\", \"Хлеб/Выпечка\", "
    "\"Овощи/Фрукты\", \"Напитки\", \"Алкоголь\", \"Сладости\", \"Одежда\", \"Обувь\", "
    "\"Бытовая химия\", \"Косметика/Гигиена\", \"Электроника\", \"Техника\", \"Мебель\", "
    "\"Ресторан/Кафе\", \"Доставка еды\", \"Транспорт\", \"Такси\", \"Парковка\", "
    "\"Здоровье\", \"Медицина\", \"Аптека\", \"Образование\", \"Книги\", "
    "\"Развлечения\", \"Кино\", \"Спорт\", \"Фитнес\", \"Путешествия\", "
    "\"Отель\", \"Коммунальные\", \"Интернет/Связь\", \"Подписки\", \"Другое\". "
    "Если категория не очевидна, используй \"Другое\"."
)

RECEIPT_EXTRACTION_PROMPT = os.getenv(
    "RECEIPT_OCR_PROMPT",
    f"Ты извлекаешь данные из фото кассовых чеков. {RECEIPT_BASE_PROMPT}",
).strip()

RECEIPT_DATA_STRUCTURING_PROMPT = os.getenv(
    "RECEIPT_DATA_PROMPT",
    f"Ты получаешь данные чека, которые были извлечены с веб-страницы. Структурируй и улучши эти данные, определи категории товаров. {RECEIPT_BASE_PROMPT}",
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
            try:
                _paddleocr_instance = PaddleOCR(
                    use_angle_cls=True,
                    lang='ch',  # Китайская модель включает русский, английский и другие языки
                    use_gpu=False,  # Используем CPU, можно включить GPU если доступен
                    show_log=False
                )
                logging.info("PaddleOCR initialized with Chinese (multilingual) model")
            except Exception as exc_ch:
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
                        logging.error(f"Failed to initialize PaddleOCR without cls: {exc_no_cls}")
                        raise exc_no_cls
            logging.info("PaddleOCR initialized successfully")
        except Exception as exc:
            logging.error(f"Failed to initialize PaddleOCR: {exc}")
            return None
    return _paddleocr_instance


class ReceiptParsingError(RuntimeError):
    """Raised when AI receipt parsing fails."""


class StatementParsingError(RuntimeError):
    """Raised when bank statement parsing fails."""


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
        # Если есть данные из QR-кода, отправляем их для структурирования
        if qr_data:
            logging.info("Используем данные из QR-кода для структурирования, изображение не отправляется")
            payload = self._build_payload("", qr_data=qr_data)
        else:
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
            "Используй категории: \"Продукты\", \"Мясо/Рыба\", \"Молочные продукты\", \"Хлеб/Выпечка\", "
            "\"Овощи/Фрукты\", \"Напитки\", \"Алкоголь\", \"Сладости\", \"Одежда\", \"Обувь\", "
            "\"Бытовая химия\", \"Косметика/Гигиена\", \"Электроника\", \"Техника\", \"Мебель\", "
            "\"Ресторан/Кафе\", \"Доставка еды\", \"Транспорт\", \"Такси\", \"Парковка\", "
            "\"Здоровье\", \"Медицина\", \"Аптека\", \"Образование\", \"Книги\", "
            "\"Развлечения\", \"Кино\", \"Спорт\", \"Фитнес\", \"Путешествия\", "
            "\"Отель\", \"Коммунальные\", \"Интернет/Связь\", \"Подписки\", \"Другое\". "
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
        
        # Если есть данные из QR-кода, отправляем их для структурирования
        if qr_data is not None:
            user_text = f"Вот данные чека, которые были извлечены с веб-страницы:\n\n{json.dumps(qr_data, ensure_ascii=False, indent=2)}\n\nСтруктурируй эти данные и верни JSON с правильными категориями товаров."
            logging.info(f"Отправляем данные из QR-кода в OpenAI для структурирования")
            
            # Используем промпт для структурирования данных
            system_prompt = self.data_prompt
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
            system_prompt = self.prompt
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
        resp = self._session.post(url, headers=headers, json=payload, timeout=self.timeout)
        if resp.status_code >= 400:
            raise ReceiptParsingError(
                f"OpenAI error {resp.status_code}: {resp.text[:500]}"
            )
        return resp.json()


class SupabaseGateway:
    """Async helper around Supabase client used by LedgerFox."""
    # TODO: добавить таблицы категорий, бюджетов и повторяющихся платежей.

    def __init__(
        self,
        url: str,
        service_key: str,
        receipts_table: str = "receipts",
        bank_table: str = "bank_transactions",
        expenses_table: str = "expenses",
    ) -> None:
        if not SUPABASE_AVAILABLE or create_client is None:
            raise RuntimeError("Supabase client is not installed. Run `pip install supabase`.")
        self._client: Client = create_client(url, service_key)
        self.receipts_table = receipts_table
        self.bank_table = bank_table
        self.expenses_table = expenses_table

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
        receipt_hash = payload.get("receipt_hash")
        logging.info("Upserting receipt %s", receipt_hash)
        
        # Проверяем, существует ли чек
        is_duplicate = await self.check_receipt_exists(receipt_hash)
        
        stored_receipt = await asyncio.to_thread(
            self._table_upsert,
            self.receipts_table,
            payload,
            on_conflict="receipt_hash",
        )
        
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
        """
        user_id = payload.get("user_id")
        date = payload.get("date")
        amount = payload.get("amount")
        currency = payload.get("currency")
        
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

    async def fetch_monthly_report(self, user_id: int, period: str) -> Dict[str, Any]:
        logging.info("Fetching report for user=%s period=%s", user_id, period)
        return await asyncio.to_thread(
            self._fetch_report_sync,
            user_id,
            period,
        )

    async def export_expenses_csv(self, user_id: int, period: Optional[str]) -> str:
        logging.info("Exporting expenses for user=%s period=%s", user_id, period or "all")
        return await asyncio.to_thread(
            self._export_expenses_csv_sync,
            user_id,
            period,
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
                logging.info(f"✅ Успешно сохранено в {table}: {result.data[0].get('id')}")
                return result.data[0]
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

    def _fetch_report_sync(self, user_id: int, period: str) -> Dict[str, Any]:
        data = (
            self._client.table(self.expenses_table)
            .select("*")
            .eq("user_id", user_id)
            .ilike("period", period)
            .execute()
            .data
            or []
        )
        total = sum(entry.get("amount", 0.0) for entry in data)
        
        # Разбивка по категориям
        categories = {}
        for entry in data:
            category = entry.get("category") or "Другое"
            amount = entry.get("amount", 0.0)
            categories[category] = categories.get(category, 0.0) + amount
        
        # Разбивка по магазинам
        stores = {}
        for entry in data:
            store = entry.get("store") or "Без названия"
            amount = entry.get("amount", 0.0)
            stores[store] = stores.get(store, 0.0) + amount
        
        # Разбивка по дням
        daily = {}
        for entry in data:
            date_str = entry.get("date", "")
            if date_str:
                day = date_str[:10]  # YYYY-MM-DD
                amount = entry.get("amount", 0.0)
                daily[day] = daily.get(day, 0.0) + amount
        
        return {
            "period": period,
            "total": total,
            "entries": data,
            "by_category": categories,
            "by_store": stores,
            "by_day": daily,
        }

    def _export_expenses_csv_sync(self, user_id: int, period: Optional[str]) -> str:
        query = self._client.table(self.expenses_table).select("*").eq("user_id", user_id)
        if period:
            query = query.ilike("period", f"{period}%")
        data = query.execute().data or []
        if not data:
            return "user_id,store,amount,currency,date,source,note\n"
        fieldnames = sorted({field for row in data for field in row.keys()})
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow({key: row.get(key) for key in fieldnames})
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


def truncate_message_for_telegram(text: str, max_length: int = 4000) -> str:
    """
    Обрезает сообщение до максимальной длины для Telegram.
    Telegram имеет лимит 4096 символов, оставляем запас.
    """
    if len(text) <= max_length:
        return text
    # Обрезаем и добавляем индикатор обрезки
    truncated = text[:max_length - 50]
    # Пытаемся обрезать по последнему переносу строки
    last_newline = truncated.rfind('\n')
    if last_newline > max_length - 200:
        truncated = truncated[:last_newline]
    return truncated + "\n\n... (сообщение обрезано, слишком длинное)"


class LedgerFoxBot:
    """Telegram bot orchestrating OCR, bank parsing, and Supabase storage."""

    def __init__(self, token: str, supabase_gateway: Optional[SupabaseGateway] = None) -> None:
        self.bot = Bot(token=token)
        self.dp = Dispatcher()
        self.router = Router(name="ledgerfox")
        self.supabase = supabase_gateway
        self._media_group_cache: Dict[str, List[Message]] = {}
        self._media_group_tasks: Dict[str, asyncio.Task] = {}
        self.dp.include_router(self.router)
        self._register_handlers()

    @classmethod
    def from_env(cls) -> "LedgerFoxBot":
        token = os.getenv("LEDGERFOX_BOT_TOKEN")
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        if not token:
            raise RuntimeError("LEDGERFOX_BOT_TOKEN is required to run LedgerFox.")
        gateway = None
        if supabase_url and supabase_key:
            gateway = SupabaseGateway(url=supabase_url, service_key=supabase_key)
        else:
            logging.warning(
                "Supabase credentials not found. Persistence features are disabled until configured."
            )
        return cls(token=token, supabase_gateway=gateway)

    async def run(self) -> None:
        logging.info("Starting LedgerFox bot")
        
        # Настраиваем меню команд
        commands = [
            BotCommand(command="receipt", description="Добавить чек (фото)"),
            BotCommand(command="expense", description="Добавить расход вручную"),
            BotCommand(command="statement", description="Импортировать выписку"),
            BotCommand(command="report", description="Получить отчёт"),
            BotCommand(command="export", description="Экспорт данных в CSV"),
            BotCommand(command="qr", description="Найти QR-коды на фото"),
            BotCommand(command="delete_all", description="Удалить все данные"),
            BotCommand(command="cancel", description="Отменить операцию"),
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
            await message.answer(
                "Привет! Я LedgerFox — распознаю чеки, сверяю с выписками и делаю отчёты.\n\n"
                "Команды:\n"
                "/receipt — загрузить чек\n"
                "/statement — импортировать выписку\n"
                "/report — получить краткий отчёт\n"
                "/cancel — сброс действия"
            )

        @self.router.message(Command("cancel"))
        async def handle_cancel(message: Message, state: FSMContext) -> None:
            await state.clear()
            await message.answer("Ок, отменили.")

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

        @self.router.message(F.photo | F.document)
        async def handle_smart_upload(message: Message, state: FSMContext) -> None:
            if message.media_group_id:
                await self._collect_media_group(message)
                return
            classification = classify_upload_kind(message)
            if classification == "receipt":
                await state.clear()
                await self._process_receipt_message(message, state)
                return
            if classification == "statement":
                await state.clear()
                await self._process_statement_message(message)
                return
            instructions = (
                "Не понял, это чек или выписка. Используйте /receipt или /statement.\n\n"
                "💡 Если это чек:\n"
                "• Чек должен занимать всё пространство на фото\n"
                "• Если чек длинный, сделайте панораму\n"
                "• Если есть QR-код, можно сфотографировать только его"
            )
            await message.answer(instructions)

        @self.router.message(Command("export"))
        async def handle_export(message: Message, state: FSMContext) -> None:
            await state.clear()
            if not self.supabase:
                await message.answer("Экспорт доступен после подключения Supabase.")
                return
            period = None
            if message.text:
                parts = message.text.split(maxsplit=1)
                if len(parts) == 2:
                    period = parts[1].strip()
            await message.answer("Формирую выгрузку, это может занять пару секунд…")
            csv_blob = await self.supabase.export_expenses_csv(message.from_user.id, period)
            filename = f"ledgerfox_export_{period or 'all'}.csv"
            file = BufferedInputFile(csv_blob.encode("utf-8"), filename=filename)
            await message.answer_document(
                document=file,
                caption="Готово. Используй CSV в Excel/Sheets или импортируй обратно.",
            )

        @self.router.message(Command("import"))
        async def handle_import(message: Message, state: FSMContext) -> None:
            await state.clear()
            await message.answer(
                "Для импорта пришли CSV/XLSX/PDF выписку или отчёт из другого сервиса. "
                "Мы автоматически распознаем формат и подскажем, что делать дальше."
            )

        @self.router.message(Command("report"))
        async def handle_report(message: Message, state: FSMContext) -> None:
            await state.clear()
            if not self.supabase:
                await message.answer(
                    "Отчёты по расходам появятся после подключения базы (Supabase)."
                )
                return
            period = datetime.utcnow().strftime("%Y-%m")
            report = await self.supabase.fetch_monthly_report(message.from_user.id, period)
            
            # Форматируем отчет с разбивкой
            report_text = format_report(report)
            
            # Обрезаем если слишком длинный
            truncated_report = truncate_message_for_telegram(report_text)
            await message.answer(truncated_report)

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

        @self.router.message(Command("expense"))
        async def handle_expense_entry(message: Message, state: FSMContext) -> None:
            await state.clear()
            instructions = (
                "💳 Добавление расхода вручную\n\n"
                "Отправьте расход в формате:\n"
                "• Название магазина/места сумма валюта\n"
                "• Название магазина сумма валюта дата\n\n"
                "Примеры:\n"
                "• Кафе 500 руб\n"
                "• Такси 1200 KZT\n"
                "• Продукты 2500 руб 03.12\n"
                "• Ресторан 5000 KZT 2025-12-03\n\n"
                "Валюта определяется автоматически (RUB, KZT, USD и др.)\n"
                "Если дата не указана, используется сегодняшняя."
            )
            await message.answer(instructions)

        @self.router.message(F.text)
        async def handle_text_expense(message: Message, state: FSMContext) -> None:
            if not message.text or message.text.startswith("/"):
                return
            parsed = parse_manual_expense(message.text)
            if not parsed:
                return
            await state.clear()
            await message.answer(format_manual_summary(parsed))
            if not self.supabase or not message.from_user:
                return
            # SECURITY: удостовериться, что пользователь подтверждён, прежде чем писать в базу.
            payload = build_manual_expense_payload(message.from_user.id, parsed)
            expense_result = await self.supabase.record_expense(payload)
            if expense_result.get("duplicate"):
                await message.answer("⚠️ Расход не добавлен: найден дубликат с такой же датой и суммой.")
            else:
                await message.answer("✅ Расход добавлен вручную.")

        @self.router.callback_query(F.data == "receipt_confirm")
        async def handle_receipt_confirm(callback: CallbackQuery, state: FSMContext) -> None:
            """Обработчик подтверждения чека"""
            await callback.answer()
            data = await state.get_data()
            parsed_receipt = data.get("parsed_receipt")
            receipt_payload = data.get("receipt_payload")
            
            if not receipt_payload or not callback.from_user:
                await callback.message.answer("Ошибка: данные чека не найдены.")
                await state.clear()
                return
            
            try:
                # Сохраняем чек в базу
                stored_receipt, is_duplicate = await self.supabase.upsert_receipt(receipt_payload)
                if is_duplicate:
                    await callback.message.answer("⚠️ Этот чек уже был сохранен ранее (дубликат)")
                else:
                    # Создаем expense запись из receipt только если это новый чек
                    expense_payload = build_expense_payload_from_receipt(stored_receipt)
                    expense_result = await self.supabase.record_expense(expense_payload)
                    if expense_result.get("duplicate"):
                        await callback.message.answer("✅ Чек сохранен в базу данных\n⚠️ Расход не создан: найден дубликат (возможно, уже есть в выписке)")
                    else:
                        await callback.message.answer("✅ Чек сохранен в базу данных")
                await state.clear()
            except Exception as exc:
                logging.exception(f"Ошибка при сохранении чека: {exc}")
                await callback.message.answer(f"⚠️ Не удалось сохранить в базу: {str(exc)[:100]}")
                await state.clear()

        @self.router.callback_query(F.data == "receipt_reject")
        async def handle_receipt_reject(callback: CallbackQuery, state: FSMContext) -> None:
            """Обработчик отклонения чека"""
            await callback.answer()
            await callback.message.answer("Понял, отправьте фото чека заново для переснятия.")
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
                await callback.message.answer(f"⚠️ Ошибка при удалении данных: {str(exc)[:200]}")
            finally:
                await state.clear()

        @self.router.callback_query(F.data == "delete_cancel")
        async def handle_delete_cancel(callback: CallbackQuery, state: FSMContext) -> None:
            """Обработчик отмены удаления данных"""
            await callback.answer()
            await callback.message.answer("Удаление отменено. Ваши данные сохранены.")
            await state.clear()

    async def _process_receipt_message(self, message: Message, state: FSMContext) -> None:
        await message.answer("Чек принят, распознаю…")
        result = await self._handle_receipt_from_message(message)
        logging.info(f"Receipt processing result: success={result.success}, has_summary={bool(result.summary)}, has_error={bool(result.error)}")
        if result.success and result.summary:
            # Сохраняем данные чека в FSM для подтверждения
            if result.parsed_receipt and result.receipt_payload:
                await state.update_data(
                    parsed_receipt=result.parsed_receipt,
                    receipt_payload=result.receipt_payload,
                )
                await state.set_state(ReceiptStates.waiting_for_confirmation)
            
            # Создаем кнопки для подтверждения (в разных рядах)
            keyboard = InlineKeyboardMarkup(inline_keyboard=[
                [
                    InlineKeyboardButton(text="✅ Все верно", callback_data="receipt_confirm"),
                ],
                [
                    InlineKeyboardButton(text="❌ Есть ошибка (переснять)", callback_data="receipt_reject"),
                ]
            ])
            
            # Пытаемся сгенерировать изображение
            if result.parsed_receipt:
                img_bytes = generate_receipt_image(result.parsed_receipt)
                if img_bytes:
                    # Отправляем изображение
                    photo = BufferedInputFile(img_bytes, filename="receipt.png")
                    await message.answer_photo(photo, reply_markup=keyboard)
                    
                    # Отправляем результат валидации отдельным сообщением
                    items_sum = sum(item.price for item in result.parsed_receipt.items)
                    total = result.parsed_receipt.total or 0.0
                    difference = abs(items_sum - total)
                    tolerance = max(total * 0.01, 1.0)
                    
                    if difference > tolerance:
                        validation_text = (
                            f"⚠️ Несоответствие суммы:\n"
                            f"Сумма позиций: {items_sum:.2f} {result.parsed_receipt.currency}\n"
                            f"Итого: {total:.2f} {result.parsed_receipt.currency}\n"
                            f"Разница: {difference:.2f} {result.parsed_receipt.currency}"
                        )
                    else:
                        validation_text = (
                            f"✅ Валидация пройдена:\n"
                            f"Сумма позиций: {items_sum:.2f} {result.parsed_receipt.currency}\n"
                            f"Итого: {total:.2f} {result.parsed_receipt.currency}"
                        )
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
                    return
            
            # Fallback: отправляем текстом если не удалось сгенерировать изображение
            truncated_summary = truncate_message_for_telegram(result.summary)
            logging.info(f"Sending receipt summary to user (text fallback): {len(result.summary)} chars")
            await message.answer(truncated_summary, reply_markup=keyboard)
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
        file = await self._resolve_file(message)
        if file is None:
            return ProcessingResult(success=False, error="Не удалось прочитать файл.")
        mime_type = detect_mime_type(message, file.file_path)
        file_bytes = await self._download_file(file.file_path)
        try:
            file_bytes, mime_type = convert_heic_if_needed(file_bytes, mime_type)
        except ReceiptParsingError as exc:
            return ProcessingResult(success=False, error=str(exc))
        # Сначала проверяем QR-коды
        qr_codes = read_qr_codes(file_bytes)
        logging.info(f"Найдено QR-кодов: {len(qr_codes)}")
        
        # Переменная для хранения данных из QR-кода для отправки в OpenAI
        qr_data_from_url = None
        
        # Если есть QR-код или штрих-код с URL, пытаемся получить данные оттуда
        if qr_codes:
            # Сначала ищем QR-коды с URL (игнорируем CODE39 и другие не-URL коды)
            for qr in qr_codes:
                qr_data = qr.get("data", "")
                qr_type = qr.get("type", "")
                
                # Игнорируем CODE39 и другие штрих-коды, которые не являются URL
                if qr_type == "CODE39" or (not is_url(qr_data) and qr_type != "QRCODE"):
                    logging.info(f"Игнорируем код типа {qr_type}: {qr_data[:50]}... (не URL и не QR-код)")
                    continue
                
                logging.info(f"Проверяем код: {qr_data[:100]}... (тип: {qr_type})")
                
                # Проверяем, является ли это URL
                if is_url(qr_data):
                    logging.info(f"✅ Найден код с URL (тип: {qr_type}): {qr_data}")
                    # Пытаемся получить данные с URL
                    qr_data_from_url = await fetch_receipt_from_qr_url(qr_data)
                    if qr_data_from_url:
                        logging.info(f"✅ Получены данные с URL, отправляем их в OpenAI для структурирования")
                    else:
                        logging.warning(f"⚠️ Не удалось получить данные с URL, используем изображение")
                        qr_data_from_url = None  # Сброс, чтобы использовать изображение
                    # Если найден QR-код с URL, игнорируем все остальные коды
                    break
                else:
                    logging.info(f"QR-код не является URL: {qr_data[:50]}... (тип: {qr_type})")
        
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
            
            # Отправляем в OpenAI (с данными из QR-кода, если есть, иначе с изображением)
            if qr_data_from_url:
                logging.info(f"Отправляем данные из QR-кода в OpenAI для структурирования")
                response_json = await parse_receipt_with_ai(file_bytes, mime_type, qr_data=qr_data_from_url)
            else:
                logging.info("Starting OpenAI receipt parsing...")
                response_json = await parse_receipt_with_ai(file_bytes, mime_type)
            
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
                    raise ReceiptParsingError(refusal_msg)
                
                if not content:
                    raise ReceiptParsingError("OpenAI response не содержит content")
                
                # Парсим JSON из content
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
                
                # Преобразуем в ParsedReceipt для форматирования
                parsed_receipt = build_parsed_receipt(content_json)
                
                # Проверяем правильность распознавания: сумма позиций должна совпадать с тоталом
                items_sum = sum(item.price for item in parsed_receipt.items)
                total = parsed_receipt.total or 0.0
                difference = abs(items_sum - total)
                
                # Допускаем погрешность до 1% или 1 единицу валюты (что больше)
                tolerance = max(total * 0.01, 1.0)
                
                validation_message = ""
                if difference > tolerance:
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
                
                # Форматируем чек в виде таблицы
                receipt_table = format_receipt_table(parsed_receipt)
                
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
                
                # Добавляем информацию о QR-кодах, если они были найдены
                if qr_codes:
                    qr_info = "\n\n📱 Найденные QR-коды:\n"
                    for i, qr in enumerate(qr_codes, 1):
                        qr_info += f"{i}. Тип: {qr['type']}\n"
                        qr_info += f"   Данные: {qr['data']}\n"
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
                        receipt_payload = build_receipt_payload(message.from_user.id, parsed_receipt)
                        logging.info(f"Создан payload для сохранения: store={receipt_payload.get('store')}, total={receipt_payload.get('total')}")
                    except Exception as db_exc:
                        logging.exception(f"Ошибка при создании payload: {db_exc}")
                
                return ProcessingResult(
                    success=True,
                    summary=summary,
                    parsed_receipt=parsed_receipt,
                    receipt_payload=receipt_payload,
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
                                    
                                    validation_message = ""
                                    if difference > tolerance:
                                        validation_message = (
                                            f"\n\n⚠️ Несоответствие суммы:\n"
                                            f"Сумма позиций: {items_sum:.2f} {parsed_receipt.currency}\n"
                                            f"Итого: {total:.2f} {parsed_receipt.currency}\n"
                                            f"Разница: {difference:.2f} {parsed_receipt.currency}"
                                        )
                                        logging.warning(
                                            f"⚠️ Несоответствие суммы (fallback): сумма позиций={items_sum:.2f}, "
                                            f"итого={total:.2f}, разница={difference:.2f}"
                                        )
                                    else:
                                        validation_message = (
                                            f"\n\n✅ Валидация пройдена:\n"
                                            f"Сумма позиций: {items_sum:.2f} {parsed_receipt.currency}\n"
                                            f"Итого: {total:.2f} {parsed_receipt.currency}"
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
                                    
                                    return ProcessingResult(
                                        success=True,
                                        summary=response_str,
                                        parsed_receipt=parsed_receipt,
                                        receipt_payload=receipt_payload,
                                    )
                                except Exception as fallback_exc:
                                    logging.exception(f"Ошибка при парсинге в fallback: {fallback_exc}")
                                    # Если не удалось распарсить, показываем JSON
                                    response_str = json.dumps(response_json, ensure_ascii=False, indent=2)
                    except Exception as db_exc:
                        logging.exception(f"Ошибка при сохранении в базу (fallback): {db_exc}")
                        if not response_str:
                            response_str = json.dumps(response_json, ensure_ascii=False, indent=2)
                
                if not response_str:
                    response_str = f"Ошибка обработки: {exc}\n\nПолный ответ:\n{json.dumps(response_json, ensure_ascii=False, indent=2)}"
                
                return ProcessingResult(
                    success=True,
                    summary=response_str,
                )
        except ReceiptParsingError as exc:
            logging.exception("Receipt parsing failed")
            return ProcessingResult(success=False, error=f"Не удалось распознать чек: {exc}")
        except Exception as exc:
            logging.exception("Image preprocessing or parsing failed")
            return ProcessingResult(success=False, error=f"Ошибка обработки изображения: {exc}")

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
        
        # Пытаемся загрузить шрифт, если не получается - используем стандартный
        try:
            title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", title_font_size)
            header_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", header_font_size)
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            try:
                title_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", title_font_size)
                header_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", header_font_size)
                font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
            except:
                # Используем стандартный шрифт
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


def format_report(report: Dict[str, Any]) -> str:
    """
    Форматирует отчет с разбивкой по категориям, топ категорий/магазинов и графиком по дням.
    """
    period = report.get("period", "")
    total = report.get("total", 0.0)
    by_category = report.get("by_category", {})
    by_store = report.get("by_store", {})
    by_day = report.get("by_day", {})
    
    lines = [f"📊 Отчёт за {period}"]
    lines.append(f"💰 Всего расходов: {total:.2f}")
    lines.append("")
    
    # Разбивка по категориям
    if by_category:
        lines.append("📂 По категориям:")
        sorted_categories = sorted(by_category.items(), key=lambda x: x[1], reverse=True)
        for category, amount in sorted_categories[:10]:  # Топ 10
            percentage = (amount / total * 100) if total > 0 else 0
            lines.append(f"  • {category}: {amount:.2f} ({percentage:.1f}%)")
        lines.append("")
    
    # Топ магазинов
    if by_store:
        lines.append("🏪 Топ магазинов:")
        sorted_stores = sorted(by_store.items(), key=lambda x: x[1], reverse=True)
        for store, amount in sorted_stores[:5]:  # Топ 5
            percentage = (amount / total * 100) if total > 0 else 0
            store_name = store[:40] if len(store) > 40 else store
            lines.append(f"  • {store_name}: {amount:.2f} ({percentage:.1f}%)")
        lines.append("")
    
    # График расходов по дням
    if by_day:
        lines.append("📈 Расходы по дням:")
        sorted_days = sorted(by_day.items())
        if sorted_days:
            max_amount = max(by_day.values())
            max_bar_length = 30
            
            for day, amount in sorted_days:
                # Форматируем дату
                try:
                    date_obj = datetime.strptime(day, "%Y-%m-%d")
                    day_str = date_obj.strftime("%d.%m")
                except:
                    day_str = day
                
                # Создаем текстовый график
                bar_length = int((amount / max_amount * max_bar_length)) if max_amount > 0 else 0
                bar = "█" * bar_length
                lines.append(f"  {day_str}: {bar} {amount:.2f}")
    
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
            raise ReceiptParsingError(
                "PaddleOCR не настроен: установите paddlepaddle и paddleocr."
            )
    else:
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
        if len(token.split(".")) == 2:
            current_year = datetime.utcnow().year
            return f"{token}.{current_year}"
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
                
                # Улучшаем контраст перед обработкой
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
                gray_clahe = clahe.apply(gray)
                
                # Увеличиваем изображение для лучшего распознавания (QR-коды должны быть достаточно большими)
                scale_factor = 3
                gray_large = cv2.resize(gray, (gray.shape[1] * scale_factor, gray.shape[0] * scale_factor), interpolation=cv2.INTER_CUBIC)
                gray_clahe_large = cv2.resize(gray_clahe, (gray_clahe.shape[1] * scale_factor, gray_clahe.shape[0] * scale_factor), interpolation=cv2.INTER_CUBIC)
                
                # Пробуем несколько вариантов обработки
                gray_variants = [
                    ("original", gray),
                    ("original_large", gray_large),
                    ("clahe", gray_clahe),
                    ("clahe_large", gray_clahe_large),
                    ("adaptive", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
                    ("adaptive_large", cv2.adaptiveThreshold(gray_large, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)),
                    ("adaptive_inv", cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)),
                    ("otsu", cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
                    ("otsu_large", cv2.threshold(gray_large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]),
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
        store = store_elem.get_text(strip=True) if store_elem else "Неизвестно"
        
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
            price_text = price_elem.get_text(strip=True) if price_elem else ""
            
            # Извлекаем количество и цену за единицу (формат: "3.000 X 649.00" или "=1947.00")
            qty_price_match = re.search(r'(\d+\.?\d*)\s*[xX×]\s*(\d+\.?\d*)', price_text)
            if qty_price_match:
                quantity = float(qty_price_match.group(1))
                unit_price = float(qty_price_match.group(2))
                total_price = quantity * unit_price
            else:
                # Пытаемся найти итоговую цену (формат: "=1947.00")
                total_price_match = re.search(r'[=]\s*(\d+\.?\d*)', price_text)
                if total_price_match:
                    total_price = float(total_price_match.group(1))
                    quantity = 1.0
                    unit_price = total_price
                else:
                    # Просто ищем число
                    price_match = re.search(r'(\d+\.?\d*)', price_text.replace(' ', ''))
                    if price_match:
                        total_price = float(price_match.group(1))
                        quantity = 1.0
                        unit_price = total_price
                    else:
                        continue
            
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
    for candidate in (value, f"{value}T00:00:00"):
        try:
            return datetime.fromisoformat(candidate)
        except ValueError:
            continue
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return datetime.utcnow()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_receipt_payload(user_id: int, parsed: ParsedReceipt) -> Dict[str, Any]:
    # Формируем хеш из user_id, даты/времени оплаты и суммы
    receipt_hash = calculate_hash(
        f"{user_id}|{parsed.purchased_at.isoformat()}|{parsed.total}"
    )
    return {
        "user_id": user_id,
        "store": parsed.store,
        "total": parsed.total,
        "currency": parsed.currency,
        "purchased_at": parsed.purchased_at.isoformat(),
        "tax_amount": parsed.tax_amount,
        "items": [asdict(item) for item in parsed.items],
        "receipt_hash": receipt_hash,
        "external_id": parsed.external_id,
        "merchant_address": parsed.merchant_address,
    }


def build_expense_payload_from_receipt(receipt_record: Dict[str, Any]) -> Dict[str, Any]:
    expense_hash = calculate_hash(
        f"{receipt_record.get('user_id')}|receipt|{receipt_record.get('receipt_hash')}"
    )
    
    # Определяем категорию из items чека (берем самую частую категорию)
    category = None
    items = receipt_record.get("items", [])
    if items and isinstance(items, list):
        category_counts = {}
        for item in items:
            if isinstance(item, dict):
                item_category = item.get("category")
                if item_category:
                    category_counts[item_category] = category_counts.get(item_category, 0) + 1
        
        if category_counts:
            # Берем самую частую категорию
            category = max(category_counts.items(), key=lambda x: x[1])[0]
    
    payload = {
        "user_id": receipt_record.get("user_id"),
        "source": "receipt",
        "store": receipt_record.get("store"),
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


MANUAL_AMOUNT_PATTERN = re.compile(
    r"(?P<amount>-?\d+[.,]?\d{0,2})\s*(?P<currency>₽|р\.?|руб|rub|₸|тг|kzt|\$|usd|eur|€|byn|сом|kgs)?",
    re.IGNORECASE,
)
MANUAL_DATE_PATTERN = re.compile(r"(\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?)")


def parse_manual_expense(text: str) -> Optional[ParsedManualExpense]:
    cleaned = " ".join((text or "").strip().split())
    if len(cleaned) < 3:
        return None
    amount_match = MANUAL_AMOUNT_PATTERN.search(cleaned)
    if not amount_match:
        return None
    amount = safe_float(amount_match.group("amount"))
    token_currency = amount_match.group("currency") or ""
    detected_currency = (
        _currency_from_value(token_currency)
        or _currency_from_value(cleaned)
        or "RUB"
    )
    date_match = MANUAL_DATE_PATTERN.search(cleaned)
    occurred_at = (
        _parse_manual_date(date_match.group(1)) if date_match else datetime.utcnow()
    )
    store_text = cleaned
    store_text = store_text.replace(amount_match.group(0), "", 1)
    if date_match:
        store_text = store_text.replace(date_match.group(0), "", 1)
    store = store_text.strip(" ,.-") or "Без описания"
    return ParsedManualExpense(
        store=store,
        amount=amount,
        currency=detected_currency,
        occurred_at=occurred_at,
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
    return (
        "Записал расход вручную:\n"
        f"• Магазин: {parsed.store}\n"
        f"• Дата: {parsed.occurred_at.strftime('%Y-%m-%d')}\n"
        f"• Сумма: {parsed.amount:.2f} {parsed.currency}"
    )


def build_manual_expense_payload(user_id: int, parsed: ParsedManualExpense) -> Dict[str, Any]:
    expense_hash = calculate_hash(
        f"{user_id}|manual|{parsed.store}|{parsed.occurred_at.isoformat()}|{parsed.amount}"
    )
    return {
        "user_id": user_id,
        "source": "manual",
        "store": parsed.store,
        "amount": parsed.amount,
        "currency": parsed.currency,
        "date": parsed.occurred_at.isoformat(),
        "note": parsed.note,
        "expense_hash": expense_hash,
        "status": "pending_review",
        "period": parsed.occurred_at.strftime("%Y-%m"),
    }


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
    for symbol, code in _CURRENCY_SYMBOLS.items():
        if symbol in text:
            return code
    lower_text = text.lower()
    for token, code in _CURRENCY_TOKENS.items():
        if token in lower_text:
            return code
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
}


_CURRENCY_TOKENS = {
    "rub": "RUB",
    "rur": "RUB",
    "рос. руб": "RUB",
    "российский рубль": "RUB",
    "тенге": "KZT",
    "казахстан": "KZT",
    "kzt": "KZT",
    "byn": "BYN",
    "бел. руб": "BYN",
    "som": "KGS",
    "kgs": "KGS",
    "сом": "KGS",
    "usd": "USD",
    "eur": "EUR",
    "gbp": "GBP",
    "cny": "CNY",
    "chf": "CHF",
    "aed": "AED",
    "cad": "CAD",
    "aud": "AUD",
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
    bot = LedgerFoxBot.from_env()
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
