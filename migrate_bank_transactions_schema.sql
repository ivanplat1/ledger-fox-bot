-- Миграция: обновление схемы таблицы bank_transactions
-- Добавляет недостающие колонки и переименовывает transaction_date в booked_at

-- Шаг 1: Добавляем колонку booked_at (если её нет)
-- Сначала проверяем, есть ли уже колонка booked_at
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'bank_transactions' AND column_name = 'booked_at'
    ) THEN
        -- Если есть transaction_date, копируем данные и переименовываем
        IF EXISTS (
            SELECT 1 FROM information_schema.columns 
            WHERE table_name = 'bank_transactions' AND column_name = 'transaction_date'
        ) THEN
            -- Добавляем новую колонку booked_at как TIMESTAMPTZ
            ALTER TABLE bank_transactions ADD COLUMN booked_at TIMESTAMPTZ;
            
            -- Копируем данные из transaction_date в booked_at
            UPDATE bank_transactions 
            SET booked_at = transaction_date::timestamp AT TIME ZONE 'UTC'
            WHERE booked_at IS NULL;
            
            -- Делаем колонку NOT NULL после заполнения данных
            ALTER TABLE bank_transactions ALTER COLUMN booked_at SET NOT NULL;
            
            -- Удаляем старую колонку transaction_date
            ALTER TABLE bank_transactions DROP COLUMN transaction_date;
        ELSE
            -- Если нет transaction_date, просто добавляем booked_at
            ALTER TABLE bank_transactions ADD COLUMN booked_at TIMESTAMPTZ NOT NULL DEFAULT NOW();
        END IF;
    END IF;
END $$;

-- Шаг 2: Добавляем колонку merchant (если её нет)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'bank_transactions' AND column_name = 'merchant'
    ) THEN
        ALTER TABLE bank_transactions ADD COLUMN merchant TEXT;
    END IF;
END $$;

-- Шаг 3: Добавляем колонку source_id (если её нет)
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'bank_transactions' AND column_name = 'source_id'
    ) THEN
        ALTER TABLE bank_transactions ADD COLUMN source_id TEXT;
    END IF;
END $$;

-- Шаг 4: Обновляем индекс для booked_at (если нужно)
DROP INDEX IF EXISTS idx_bank_transactions_date;
CREATE INDEX IF NOT EXISTS idx_bank_transactions_booked_at ON bank_transactions(booked_at);

-- Проверка: посмотреть структуру таблицы
-- SELECT column_name, data_type, is_nullable 
-- FROM information_schema.columns 
-- WHERE table_name = 'bank_transactions' 
-- ORDER BY ordinal_position;

