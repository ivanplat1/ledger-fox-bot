-- Миграция: изменение типа поля date в таблице expenses с DATE на TIMESTAMPTZ
-- Это позволит сохранять не только дату, но и время расхода

-- Шаг 1: Создаем временную колонку с типом TIMESTAMPTZ
ALTER TABLE expenses ADD COLUMN IF NOT EXISTS date_new TIMESTAMPTZ;

-- Шаг 2: Копируем данные из старой колонки в новую (добавляем время 00:00:00)
UPDATE expenses 
SET date_new = date::timestamp AT TIME ZONE 'UTC'
WHERE date_new IS NULL;

-- Шаг 3: Удаляем старую колонку
ALTER TABLE expenses DROP COLUMN IF EXISTS date;

-- Шаг 4: Переименовываем новую колонку
ALTER TABLE expenses RENAME COLUMN date_new TO date;

-- Шаг 5: Устанавливаем NOT NULL constraint
ALTER TABLE expenses ALTER COLUMN date SET NOT NULL;

-- Шаг 6: Обновляем индекс (он должен автоматически пересоздаться, но можно пересоздать явно)
DROP INDEX IF EXISTS idx_expenses_date;
CREATE INDEX IF NOT EXISTS idx_expenses_date ON expenses(date);

-- Проверка: посмотреть структуру таблицы
-- SELECT column_name, data_type, is_nullable 
-- FROM information_schema.columns 
-- WHERE table_name = 'expenses' AND column_name = 'date';

