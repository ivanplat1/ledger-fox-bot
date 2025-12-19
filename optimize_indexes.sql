-- Оптимизация индексов для ускорения запросов отчетов
-- Выполните этот SQL в Supabase SQL Editor

-- Составной индекс для user_id + period (для фильтрации по периоду)
-- Это ускорит запросы вида: WHERE user_id = X AND period LIKE 'YYYY-MM%'
CREATE INDEX IF NOT EXISTS idx_expenses_user_period 
ON expenses(user_id, period) 
WHERE period IS NOT NULL;

-- Составной индекс для user_id + date (для фильтрации по диапазону дат)
-- Это ускорит запросы вида: WHERE user_id = X AND date >= ... AND date <= ...
CREATE INDEX IF NOT EXISTS idx_expenses_user_date 
ON expenses(user_id, date);

-- Индекс для receipt_id в expenses (для JOIN с receipts)
CREATE INDEX IF NOT EXISTS idx_expenses_receipt_id 
ON expenses(receipt_id) 
WHERE receipt_id IS NOT NULL;

-- Индекс для user_id в receipts (для фильтрации чеков пользователя)
-- (уже должен быть, но на всякий случай)
CREATE INDEX IF NOT EXISTS idx_receipts_user_id 
ON receipts(user_id);

-- Анализ таблиц для обновления статистики планировщика
ANALYZE expenses;
ANALYZE receipts;

