# Настройка Supabase для ExpenseCatBot

## Шаг 1: Создание проекта в Supabase

1. Перейдите на [supabase.com](https://supabase.com)
2. Создайте новый проект
3. Запомните URL проекта и Service Role Key

### Где найти SUPABASE_URL и SUPABASE_SERVICE_ROLE_KEY:

1. **Откройте ваш проект в Supabase Dashboard**
2. **Перейдите в Settings (⚙️) → API**
3. **Найдите секцию "Project API keys"**
4. **Скопируйте:**
   - **Project URL** → это ваш `SUPABASE_URL`
   - **service_role key** (секретный ключ) → это ваш `SUPABASE_SERVICE_ROLE_KEY`
     - ⚠️ **Важно:** Используйте именно `service_role` ключ, а не `anon` или `public` ключ
     - Этот ключ имеет полные права доступа к базе данных
     - Никогда не публикуйте его в публичных репозиториях!

**Пример:**
- `SUPABASE_URL`: `https://abcdefghijklmnop.supabase.co`
- `SUPABASE_SERVICE_ROLE_KEY`: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...` (длинная строка)

## Шаг 2: Создание таблиц

1. Откройте SQL Editor в Supabase Dashboard
2. Скопируйте содержимое файла `supabase_schema.sql`
3. Выполните SQL скрипт

Это создаст три таблицы:
- `receipts` - для хранения чеков
- `expenses` - для хранения трат/расходов
- `bank_transactions` - для хранения банковских транзакций

## Шаг 2.5: Настройка Storage buckets

1. Откройте SQL Editor в Supabase Dashboard
2. Скопируйте содержимое файла `setup_storage_buckets.sql`
3. Выполните SQL скрипт

Это создаст:
- `rejected-receipts` bucket - для хранения фото отклоненных/невалидных чеков
- `receipts` bucket - для хранения фото обычных чеков (опционально)
- Настроит политики доступа для service_role

**Важно:** После выполнения скрипта проверьте, что buckets созданы:
- Перейдите в Storage в Supabase Dashboard
- Убедитесь, что видны buckets `rejected-receipts` и `receipts`

## Шаг 3: Настройка переменных окружения

Добавьте в ваш `.env` файл:

```env
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key
```

## Шаг 4: Проверка работы

После настройки бот автоматически будет сохранять:
- Чеки в таблицу `receipts`
- Расходы в таблицу `expenses`
- Банковские транзакции в таблицу `bank_transactions`

## Структура данных

### receipts
- `user_id` - ID пользователя Telegram
- `store` - Название магазина
- `total` - Общая сумма
- `currency` - Валюта (RUB, KZT, USD и т.д.)
- `purchased_at` - Дата и время покупки
- `tax_amount` - Сумма налога
- `items` - JSON массив товаров
- `receipt_hash` - Уникальный хеш чека (для предотвращения дубликатов)
- `merchant_address` - Адрес магазина
- `external_id` - Внешний ID (например, из QR-кода)

### expenses
- `user_id` - ID пользователя Telegram
- `source` - Источник: 'receipt', 'bank', 'manual'
- `store` - Название магазина/места
- `amount` - Сумма расхода
- `currency` - Валюта
- `date` - Дата расхода
- `receipt_id` - Ссылка на чек (если есть)
- `expense_hash` - Уникальный хеш расхода
- `status` - Статус: 'pending_review', 'approved', 'rejected'
- `period` - Период в формате 'YYYY-MM' для группировки

### bank_transactions
- `user_id` - ID пользователя Telegram
- `amount` - Сумма транзакции
- `currency` - Валюта
- `transaction_date` - Дата транзакции
- `description` - Описание транзакции
- `transaction_hash` - Уникальный хеш транзакции
- `bank_name` - Название банка
- `account_number` - Номер счета

## Безопасность (опционально)

Если нужно включить Row Level Security (RLS), раскомментируйте соответствующие строки в `supabase_schema.sql`. Это позволит пользователям видеть только свои данные.

