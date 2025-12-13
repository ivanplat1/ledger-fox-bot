-- Создание таблицы feedback для обратной связи
-- Выполните этот SQL в Supabase SQL Editor

-- Таблица обратной связи (отзывы, ошибки, предложения, жалобы)
CREATE TABLE IF NOT EXISTS feedback (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    username TEXT,
    first_name TEXT,
    feedback_type TEXT NOT NULL, -- 'bug', 'suggestion', 'complaint'
    feedback_text TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Индексы для таблицы обратной связи
CREATE INDEX IF NOT EXISTS idx_feedback_user_id ON feedback(user_id);
CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback(feedback_type);
CREATE INDEX IF NOT EXISTS idx_feedback_created_at ON feedback(created_at);

