-- Создание таблицы user_limits для монетизации
-- Выполните этот SQL в Supabase SQL Editor

-- Таблица лимитов пользователей (для монетизации)
-- Лимиты:
--   - Пробный период (trial): 10 чеков
--   - Premium подписка: 100 чеков в месяц
--   - Стоимость одного запроса в OpenAI: ~$0.0175 (GPT-4o)
--   - Стоимость 100 чеков: ~$1.75
CREATE TABLE IF NOT EXISTS user_limits (
    user_id BIGINT PRIMARY KEY,
    receipts_count INTEGER NOT NULL DEFAULT 0,
    limit_receipts INTEGER NOT NULL DEFAULT 10, -- Лимит чеков: 10 для trial, 100 для premium
    subscription_type TEXT DEFAULT 'trial', -- 'trial', 'premium', 'unlimited'
    expires_at TIMESTAMPTZ, -- Дата окончания подписки (NULL для пробного периода)
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Индексы для таблицы лимитов
CREATE INDEX IF NOT EXISTS idx_user_limits_user_id ON user_limits(user_id);
CREATE INDEX IF NOT EXISTS idx_user_limits_subscription_type ON user_limits(subscription_type);

-- Триггер для автоматического обновления updated_at
CREATE TRIGGER update_user_limits_updated_at BEFORE UPDATE ON user_limits
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

