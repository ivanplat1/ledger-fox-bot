-- Скрипт для удаления старых таблиц с префиксом ledgerfox_
-- Выполните этот SQL в Supabase SQL Editor

-- Удаляем триггеры (если они существуют)
DROP TRIGGER IF EXISTS update_receipts_updated_at ON ledgerfox_receipts;
DROP TRIGGER IF EXISTS update_expenses_updated_at ON ledgerfox_expenses;
DROP TRIGGER IF EXISTS update_bank_transactions_updated_at ON ledgerfox_bank_transactions;

-- Удаляем индексы
DROP INDEX IF EXISTS idx_receipts_user_id;
DROP INDEX IF EXISTS idx_receipts_purchased_at;
DROP INDEX IF EXISTS idx_receipts_receipt_hash;

DROP INDEX IF EXISTS idx_expenses_user_id;
DROP INDEX IF EXISTS idx_expenses_date;
DROP INDEX IF EXISTS idx_expenses_period;
DROP INDEX IF EXISTS idx_expenses_expense_hash;
DROP INDEX IF EXISTS idx_expenses_status;

DROP INDEX IF EXISTS idx_bank_transactions_user_id;
DROP INDEX IF EXISTS idx_bank_transactions_date;
DROP INDEX IF EXISTS idx_bank_transactions_hash;

-- Удаляем таблицы (в правильном порядке из-за внешних ключей)
DROP TABLE IF EXISTS ledgerfox_expenses CASCADE;
DROP TABLE IF EXISTS ledgerfox_bank_transactions CASCADE;
DROP TABLE IF EXISTS ledgerfox_receipts CASCADE;

-- Проверка: после выполнения этого скрипта старые таблицы должны быть удалены
-- Затем выполните supabase_schema.sql для создания новых таблиц

