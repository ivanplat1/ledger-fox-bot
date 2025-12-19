# Обработка банковских выписок

## Обзор

Бот поддерживает импорт банковских выписок в форматах CSV, XLSX и PDF. Выписки автоматически парсятся, транзакции сохраняются в базу данных и создаются соответствующие расходы.

## Процесс обработки

### 1. Загрузка выписки

Пользователь отправляет файл выписки командой `/statement` или просто прикрепляет файл.

**Поддерживаемые форматы:**
- CSV
- XLSX (Excel)
- PDF

### 2. Парсинг выписки

Функция `parse_bank_statement()` обрабатывает файл:

1. **Определение формата** - по расширению файла или MIME-типу
2. **Чтение данных** - извлечение строк из CSV/XLSX или текста из PDF
3. **Нормализация** - приведение названий колонок к стандартным
4. **Парсинг транзакций** - создание объектов `ParsedBankTransaction`

### 3. Извлечение данных из строки

Для каждой строки выписки извлекаются:

- **Сумма** (`amount`) - из колонок: `amount`, `сумма`, `сумма операции`
- **Валюта** (`currency`) - из колонок: `currency`, `валюта`, `код валюты` или определяется автоматически
- **Магазин** (`merchant`) - из колонок: `merchant`, `store`, `контрагент`, `получатель`
- **Описание** (`description`) - из колонок: `description`, `назначение платежа`, `описание`
- **Дата** (`booked_at`) - из колонок: `date`, `booked_at`, `дата`, `дата операции`
- **ID транзакции** (`source_id`) - из колонок: `id`, `transaction id`, `номер документа`

### 4. Сохранение в базу данных

#### Таблица `bank_transactions`

Транзакции сохраняются через `upsert_bank_transactions()`:

```python
payload = {
    "user_id": user_id,
    "amount": txn.amount,
    "currency": txn.currency,
    "merchant": txn.merchant,
    "booked_at": txn.booked_at.isoformat(),  # TIMESTAMPTZ
    "description": txn.description,
    "source_id": txn.source_id,
    "transaction_hash": txn_hash,  # Уникальный хеш для предотвращения дубликатов
}
```

**Ключевые поля:**
- `booked_at` - дата проведения операции (TIMESTAMPTZ)
- `merchant` - название магазина/контрагента
- `source_id` - ID транзакции из выписки банка
- `transaction_hash` - уникальный хеш для предотвращения дубликатов

**Конфликт:** `on_conflict="transaction_hash"` - если транзакция с таким хешем уже существует, она обновляется.

### 5. Создание расходов

После сохранения транзакций функция `reconcile_transactions()` создает расходы:

```python
expense_payload = {
    "user_id": user_id,
    "source": "bank",
    "store": record.get("merchant"),
    "amount": record.get("amount"),
    "currency": record.get("currency"),
    "date": record.get("booked_at"),  # Используется booked_at из транзакции
    "bank_transaction_id": record.get("id"),  # Связь с bank_transactions
    "expense_hash": calculate_hash(f"{user_id}|bank|{transaction_hash}"),
    "status": "pending_review",
    "period": booked_at[:7],  # YYYY-MM
}
```

**Особенности:**
- Каждая транзакция создает один расход
- Расходы имеют `source="bank"`
- Связь с транзакцией через `bank_transaction_id`
- Проверка дубликатов по `expense_hash`

## Схема базы данных

### Таблица `bank_transactions`

```sql
CREATE TABLE bank_transactions (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    amount DECIMAL(10, 2) NOT NULL,
    currency TEXT NOT NULL DEFAULT 'RUB',
    booked_at TIMESTAMPTZ NOT NULL,  -- Дата проведения операции
    description TEXT,
    merchant TEXT,  -- Название магазина/контрагента
    source_id TEXT,  -- ID транзакции из выписки
    transaction_hash TEXT UNIQUE NOT NULL,
    bank_name TEXT,
    account_number TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Индексы:**
- `idx_bank_transactions_user_id` - для фильтрации по пользователю
- `idx_bank_transactions_booked_at` - для фильтрации по дате
- `idx_bank_transactions_hash` - для поиска по хешу (уникальный)

## Миграция схемы

Если у вас старая схема с `transaction_date` вместо `booked_at`, выполните миграцию:

```sql
-- См. migrate_bank_transactions_schema.sql
```

Миграция:
1. Добавляет колонку `booked_at` (TIMESTAMPTZ)
2. Копирует данные из `transaction_date` в `booked_at`
3. Удаляет старую колонку `transaction_date`
4. Добавляет колонки `merchant` и `source_id` (если их нет)
5. Обновляет индексы

## Обработка ошибок

### Ошибка: "Could not find the 'booked_at' column"

**Причина:** Схема базы данных не соответствует коду.

**Решение:**
1. Выполните миграцию `migrate_bank_transactions_schema.sql`
2. Или обновите схему вручную, добавив недостающие колонки

### Ошибка: "Could not find the 'merchant' column"

**Причина:** Колонка `merchant` отсутствует в таблице.

**Решение:** Выполните миграцию или добавьте колонку вручную:
```sql
ALTER TABLE bank_transactions ADD COLUMN merchant TEXT;
```

## Форматы выписок

### CSV

Ожидаемые колонки (в любом порядке):
- Дата: `date`, `booked_at`, `дата`, `дата операции`
- Сумма: `amount`, `сумма`, `сумма операции`
- Валюта: `currency`, `валюта`, `код валюты`
- Контрагент: `merchant`, `store`, `контрагент`, `получатель`
- Описание: `description`, `назначение платежа`, `описание`

### XLSX

Аналогично CSV, но из Excel файла.

### PDF

Текст извлекается из PDF и парсится как CSV.

## Примеры использования

### Команда `/statement`

1. Пользователь выполняет `/statement`
2. Бот запрашивает файл выписки
3. Пользователь отправляет CSV/XLSX/PDF файл
4. Бот парсит файл и показывает сводку
5. Транзакции сохраняются в базу данных
6. Создаются расходы для каждой транзакции

### Автоматическое распознавание

Если пользователь просто отправляет файл (без команды), бот пытается определить тип:
- Если это CSV/XLSX/PDF → обрабатывается как выписка
- Если это изображение → обрабатывается как чек

## Логирование

При обработке выписок логируются:
- Количество найденных транзакций
- Количество сохраненных транзакций
- Ошибки парсинга
- Дубликаты транзакций

Пример лога:
```
INFO Upserting 21 bank transactions
INFO HTTP Request: POST .../bank_transactions
```

## Дополнительные функции

### Проверка дубликатов

Транзакции проверяются на дубликаты по `transaction_hash`. Если транзакция с таким хешем уже существует, она обновляется (upsert).

### Связь с расходами

Каждая транзакция автоматически создает расход в таблице `expenses` с:
- `source="bank"`
- `bank_transaction_id` - ссылка на транзакцию
- `store` - из поля `merchant` транзакции
- `date` - из поля `booked_at` транзакции

