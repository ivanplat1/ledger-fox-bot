-- Схема базы данных для LedgerFox Bot
-- Выполните этот SQL в Supabase SQL Editor для создания таблиц

-- Таблица чеков
CREATE TABLE IF NOT EXISTS receipts (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    store TEXT NOT NULL,
    total DECIMAL(10, 2) NOT NULL,
    currency TEXT NOT NULL DEFAULT 'RUB',
    purchased_at TIMESTAMPTZ NOT NULL,
    tax_amount DECIMAL(10, 2),
    items JSONB NOT NULL DEFAULT '[]'::jsonb,
    receipt_hash TEXT UNIQUE NOT NULL,
    external_id TEXT,
    merchant_address TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Индексы для таблицы чеков
CREATE INDEX IF NOT EXISTS idx_receipts_user_id ON receipts(user_id);
CREATE INDEX IF NOT EXISTS idx_receipts_purchased_at ON receipts(purchased_at);
CREATE INDEX IF NOT EXISTS idx_receipts_receipt_hash ON receipts(receipt_hash);

-- Таблица расходов (трат)
CREATE TABLE IF NOT EXISTS expenses (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    source TEXT NOT NULL, -- 'receipt', 'bank', 'manual'
    store TEXT,
    amount DECIMAL(10, 2) NOT NULL,
    currency TEXT NOT NULL DEFAULT 'RUB',
    date DATE NOT NULL,
    receipt_id BIGINT REFERENCES receipts(id) ON DELETE SET NULL,
    bank_transaction_id BIGINT, -- ссылка на bank_transactions если есть
    expense_hash TEXT UNIQUE NOT NULL,
    status TEXT DEFAULT 'pending_review', -- 'pending_review', 'approved', 'rejected'
    period TEXT, -- формат: 'YYYY-MM' для группировки
    note TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Индексы для таблицы расходов
CREATE INDEX IF NOT EXISTS idx_expenses_user_id ON expenses(user_id);
CREATE INDEX IF NOT EXISTS idx_expenses_date ON expenses(date);
CREATE INDEX IF NOT EXISTS idx_expenses_period ON expenses(period);
CREATE INDEX IF NOT EXISTS idx_expenses_expense_hash ON expenses(expense_hash);
CREATE INDEX IF NOT EXISTS idx_expenses_status ON expenses(status);

-- Таблица банковских транзакций
CREATE TABLE IF NOT EXISTS bank_transactions (
    id BIGSERIAL PRIMARY KEY,
    user_id BIGINT NOT NULL,
    amount DECIMAL(10, 2) NOT NULL,
    currency TEXT NOT NULL DEFAULT 'RUB',
    transaction_date DATE NOT NULL,
    description TEXT,
    transaction_hash TEXT UNIQUE NOT NULL,
    bank_name TEXT,
    account_number TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Индексы для таблицы банковских транзакций
CREATE INDEX IF NOT EXISTS idx_bank_transactions_user_id ON bank_transactions(user_id);
CREATE INDEX IF NOT EXISTS idx_bank_transactions_date ON bank_transactions(transaction_date);
CREATE INDEX IF NOT EXISTS idx_bank_transactions_hash ON bank_transactions(transaction_hash);

-- Функция для автоматического обновления updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Триггеры для автоматического обновления updated_at
CREATE TRIGGER update_receipts_updated_at BEFORE UPDATE ON receipts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_expenses_updated_at BEFORE UPDATE ON expenses
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_bank_transactions_updated_at BEFORE UPDATE ON bank_transactions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- RLS (Row Level Security) политики (опционально, для безопасности)
-- Раскомментируйте, если нужно включить RLS

-- ALTER TABLE receipts ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE expenses ENABLE ROW LEVEL SECURITY;
-- ALTER TABLE bank_transactions ENABLE ROW LEVEL SECURITY;

-- Политики доступа (пользователь может видеть только свои данные)
-- CREATE POLICY "Users can view own receipts" ON receipts
--     FOR SELECT USING (auth.uid() = user_id);
-- 
-- CREATE POLICY "Users can insert own receipts" ON receipts
--     FOR INSERT WITH CHECK (auth.uid() = user_id);
-- 
-- CREATE POLICY "Users can view own expenses" ON expenses
--     FOR SELECT USING (auth.uid() = user_id);
-- 
-- CREATE POLICY "Users can insert own expenses" ON expenses
--     FOR INSERT WITH CHECK (auth.uid() = user_id);
-- 
-- CREATE POLICY "Users can view own bank transactions" ON bank_transactions
--     FOR SELECT USING (auth.uid() = user_id);
-- 
-- CREATE POLICY "Users can insert own bank transactions" ON bank_transactions
--     FOR INSERT WITH CHECK (auth.uid() = user_id);

