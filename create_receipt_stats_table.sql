-- Таблица статистики распознавания чеков
CREATE TABLE IF NOT EXISTS receipt_recognition_stats (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    recognition_method TEXT NOT NULL, -- 'qr', 'openai_photo', 'openai_qr_data'
    success BOOLEAN NOT NULL,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Индексы для быстрого поиска статистики
CREATE INDEX IF NOT EXISTS idx_receipt_stats_user_id ON receipt_recognition_stats(user_id);
CREATE INDEX IF NOT EXISTS idx_receipt_stats_method ON receipt_recognition_stats(recognition_method);
CREATE INDEX IF NOT EXISTS idx_receipt_stats_success ON receipt_recognition_stats(success);
CREATE INDEX IF NOT EXISTS idx_receipt_stats_created_at ON receipt_recognition_stats(created_at);
CREATE INDEX IF NOT EXISTS idx_receipt_stats_user_method_success ON receipt_recognition_stats(user_id, recognition_method, success);

-- Представление для удобного получения статистики
CREATE OR REPLACE VIEW receipt_stats_summary AS
SELECT 
    user_id,
    recognition_method,
    COUNT(*) FILTER (WHERE success = true) as successful_count,
    COUNT(*) FILTER (WHERE success = false) as failed_count,
    COUNT(*) as total_count,
    ROUND(100.0 * COUNT(*) FILTER (WHERE success = true) / NULLIF(COUNT(*), 0), 2) as success_rate_percent
FROM receipt_recognition_stats
GROUP BY user_id, recognition_method;

-- Представление для общей статистики по пользователю
CREATE OR REPLACE VIEW receipt_stats_by_user AS
SELECT 
    user_id,
    COUNT(*) FILTER (WHERE success = true) as total_successful,
    COUNT(*) FILTER (WHERE success = false) as total_failed,
    COUNT(*) FILTER (WHERE success = true AND recognition_method = 'qr') as qr_successful,
    COUNT(*) FILTER (WHERE success = false AND recognition_method = 'qr') as qr_failed,
    COUNT(*) FILTER (WHERE success = true AND recognition_method = 'openai_photo') as openai_photo_successful,
    COUNT(*) FILTER (WHERE success = false AND recognition_method = 'openai_photo') as openai_photo_failed,
    COUNT(*) FILTER (WHERE success = true AND recognition_method = 'openai_qr_data') as openai_qr_data_successful,
    COUNT(*) FILTER (WHERE success = false AND recognition_method = 'openai_qr_data') as openai_qr_data_failed,
    COUNT(*) as total_count,
    ROUND(100.0 * COUNT(*) FILTER (WHERE success = true) / NULLIF(COUNT(*), 0), 2) as overall_success_rate
FROM receipt_recognition_stats
GROUP BY user_id;


