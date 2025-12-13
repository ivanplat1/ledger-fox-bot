-- Тестовые данные для ExpenseCatBot (RUB - Российский рубль)
-- Выполните этот SQL в Supabase SQL Editor после создания таблиц

-- Очистка существующих данных (опционально)
-- TRUNCATE TABLE expenses CASCADE;
-- TRUNCATE TABLE receipts CASCADE;
-- TRUNCATE TABLE bank_transactions CASCADE;

-- Тестовый user_id (замените на реальный ID пользователя)
-- Для тестирования используем user_id = 188376853

-- Вставка чеков (receipts) - RUB
INSERT INTO receipts (user_id, store, total, currency, purchased_at, tax_amount, items, receipt_hash, external_id, merchant_address) VALUES
-- Декабрь 2025
(188376853, 'ООО "Продуктовый Дом"', 3500.00, 'RUB', '2025-12-20 14:30:00+00', 350.00,
 '[
   {"name": "Хлеб белый", "quantity": 2, "price": 120.00, "category": "Хлеб/Выпечка"},
   {"name": "Молоко 3.2% 1л", "quantity": 2, "price": 95.00, "category": "Молочные продукты"},
   {"name": "Яйца С1 10шт", "quantity": 1, "price": 120.00, "category": "Продукты"},
   {"name": "Курица целая 1.5кг", "quantity": 1, "price": 450.00, "category": "Мясо/Рыба"},
   {"name": "Картофель 2кг", "quantity": 1, "price": 150.00, "category": "Овощи/Фрукты"},
   {"name": "Морковь 1кг", "quantity": 1, "price": 80.00, "category": "Овощи/Фрукты"},
   {"name": "Лук репчатый 1кг", "quantity": 1, "price": 60.00, "category": "Овощи/Фрукты"},
   {"name": "Сметана 20% 400г", "quantity": 1, "price": 180.00, "category": "Молочные продукты"},
   {"name": "Сыр Российский 300г", "quantity": 1, "price": 350.00, "category": "Молочные продукты"},
   {"name": "Колбаса докторская 300г", "quantity": 1, "price": 280.00, "category": "Мясо/Рыба"},
   {"name": "Чай черный 100г", "quantity": 1, "price": 150.00, "category": "Напитки"},
   {"name": "Сахар 1кг", "quantity": 1, "price": 90.00, "category": "Продукты"},
   {"name": "Масло подсолнечное 1л", "quantity": 1, "price": 180.00, "category": "Продукты"},
   {"name": "Макароны 500г", "quantity": 2, "price": 120.00, "category": "Продукты"},
   {"name": "Томатная паста 200г", "quantity": 1, "price": 85.00, "category": "Продукты"},
   {"name": "Соль поваренная 1кг", "quantity": 1, "price": 30.00, "category": "Продукты"},
   {"name": "Перец черный молотый 50г", "quantity": 1, "price": 95.00, "category": "Продукты"},
   {"name": "Лавровый лист 10г", "quantity": 1, "price": 25.00, "category": "Продукты"},
   {"name": "Мыло туалетное", "quantity": 2, "price": 80.00, "category": "Косметика/Гигиена"},
   {"name": "Зубная паста 100мл", "quantity": 1, "price": 120.00, "category": "Косметика/Гигиена"},
   {"name": "Шампунь 250мл", "quantity": 1, "price": 200.00, "category": "Косметика/Гигиена"},
   {"name": "Туалетная бумага 8шт", "quantity": 1, "price": 150.00, "category": "Косметика/Гигиена"}
 ]'::jsonb,
 'hash_receipt_20251220_143000_3500_rub', 'ext_016', NULL),

(188376853, 'ООО "ТехноМаркет"', 25000.00, 'RUB', '2025-12-18 16:45:00+00', 2500.00,
 '[
   {"name": "Наушники беспроводные", "quantity": 1, "price": 25000.00, "category": "Электроника"}
 ]'::jsonb,
 'hash_receipt_20251218_164500_25000_rub', 'ext_017', NULL),

(188376853, 'Кафе "Московское"', 2800.00, 'RUB', '2025-12-16 19:20:00+00', 280.00,
 '[
   {"name": "Борщ с мясом", "quantity": 1, "price": 350.00, "category": "Ресторан/Кафе"},
   {"name": "Котлета по-киевски", "quantity": 1, "price": 450.00, "category": "Ресторан/Кафе"},
   {"name": "Салат Оливье", "quantity": 1, "price": 280.00, "category": "Ресторан/Кафе"},
   {"name": "Чай черный", "quantity": 2, "price": 120.00, "category": "Напитки"},
   {"name": "Хлеб", "quantity": 1, "price": 50.00, "category": "Хлеб/Выпечка"},
   {"name": "Пирожное Наполеон", "quantity": 1, "price": 180.00, "category": "Сладости"},
   {"name": "Кофе капучино", "quantity": 1, "price": 200.00, "category": "Напитки"},
   {"name": "Сок апельсиновый 0.3л", "quantity": 1, "price": 150.00, "category": "Напитки"},
   {"name": "Мороженое пломбир", "quantity": 1, "price": 120.00, "category": "Сладости"},
   {"name": "Вода минеральная 0.5л", "quantity": 1, "price": 80.00, "category": "Напитки"},
   {"name": "Сервисный сбор", "quantity": 1, "price": 280.00, "category": "Ресторан/Кафе"},
   {"name": "Чаевые", "quantity": 1, "price": 280.00, "category": "Ресторан/Кафе"}
 ]'::jsonb,
 'hash_receipt_20251216_192000_2800_rub', 'ext_018', NULL),

(188376853, 'АЗС "Роснефть"', 3500.00, 'RUB', '2025-12-12 10:15:00+00', 350.00,
 '[
   {"name": "Бензин АИ-95 30л", "quantity": 1, "price": 3500.00, "category": "Бензин/Топливо"}
 ]'::jsonb,
 'hash_receipt_20251212_101500_3500_rub', 'ext_019', NULL),

(188376853, 'Аптека "Здоровье+"', 1850.00, 'RUB', '2025-12-10 11:00:00+00', 185.00,
 '[
   {"name": "Парацетамол 500мг 20шт", "quantity": 1, "price": 120.00, "category": "Лекарства"},
   {"name": "Ибупрофен 200мг 20шт", "quantity": 1, "price": 150.00, "category": "Лекарства"},
   {"name": "Витамин D3 2000МЕ 60капс", "quantity": 1, "price": 450.00, "category": "Лекарства"},
   {"name": "Витамин C 1000мг 20шт", "quantity": 1, "price": 280.00, "category": "Лекарства"},
   {"name": "Бинт стерильный 5м", "quantity": 2, "price": 120.00, "category": "Аптека"},
   {"name": "Пластырь бактерицидный 10шт", "quantity": 1, "price": 80.00, "category": "Аптека"},
   {"name": "Йод 5% 10мл", "quantity": 1, "price": 50.00, "category": "Аптека"},
   {"name": "Перекись водорода 3% 100мл", "quantity": 1, "price": 60.00, "category": "Аптека"},
   {"name": "Вата стерильная 100г", "quantity": 1, "price": 90.00, "category": "Аптека"},
   {"name": "Термометр электронный", "quantity": 1, "price": 250.00, "category": "Аптека"}
 ]'::jsonb,
 'hash_receipt_20251210_110000_1850_rub', 'ext_020', NULL);

-- Вставка расходов (expenses) - RUB
-- Используем подзапросы для получения ID чеков по receipt_hash
INSERT INTO expenses (user_id, source, store, amount, currency, date, receipt_id, expense_hash, status, period, category) VALUES
(188376853, 'receipt', 'ООО "Продуктовый Дом"', 3500.00, 'RUB', '2025-12-20',
 (SELECT id FROM receipts WHERE receipt_hash = 'hash_receipt_20251220_143000_3500_rub' LIMIT 1),
 'exp_hash_20251220_3500_rub', 'approved', '2025-12', 'Продукты'),
(188376853, 'receipt', 'ООО "ТехноМаркет"', 25000.00, 'RUB', '2025-12-18',
 (SELECT id FROM receipts WHERE receipt_hash = 'hash_receipt_20251218_164500_25000_rub' LIMIT 1),
 'exp_hash_20251218_25000_rub', 'approved', '2025-12', 'Электроника'),
(188376853, 'receipt', 'Кафе "Московское"', 2800.00, 'RUB', '2025-12-16',
 (SELECT id FROM receipts WHERE receipt_hash = 'hash_receipt_20251216_192000_2800_rub' LIMIT 1),
 'exp_hash_20251216_2800_rub', 'approved', '2025-12', 'Ресторан/Кафе'),
(188376853, 'receipt', 'АЗС "Роснефть"', 3500.00, 'RUB', '2025-12-12',
 (SELECT id FROM receipts WHERE receipt_hash = 'hash_receipt_20251212_101500_3500_rub' LIMIT 1),
 'exp_hash_20251212_3500_rub', 'approved', '2025-12', 'Бензин/Топливо'),
(188376853, 'receipt', 'Аптека "Здоровье+"', 1850.00, 'RUB', '2025-12-10',
 (SELECT id FROM receipts WHERE receipt_hash = 'hash_receipt_20251210_110000_1850_rub' LIMIT 1),
 'exp_hash_20251210_1850_rub', 'approved', '2025-12', 'Лекарства');

-- Вставка банковских транзакций (bank_transactions) - RUB
INSERT INTO bank_transactions (user_id, amount, currency, transaction_date, description, transaction_hash, bank_name, account_number) VALUES
(188376853, -3500.00, 'RUB', '2025-12-20', 'Оплата в ООО "Продуктовый Дом"', 'txn_hash_20251220_016_rub', 'Test Bank', '****1234'),
(188376853, -25000.00, 'RUB', '2025-12-18', 'Оплата в ООО "ТехноМаркет"', 'txn_hash_20251218_017_rub', 'Test Bank', '****1234'),
(188376853, -2800.00, 'RUB', '2025-12-16', 'Оплата в Кафе "Московское"', 'txn_hash_20251216_018_rub', 'Test Bank', '****1234'),
(188376853, -3500.00, 'RUB', '2025-12-12', 'Оплата на АЗС "Роснефть"', 'txn_hash_20251212_019_rub', 'Test Bank', '****1234'),
(188376853, -1850.00, 'RUB', '2025-12-10', 'Оплата в Аптека "Здоровье+"', 'txn_hash_20251210_020_rub', 'Test Bank', '****1234');

-- Проверка данных
-- SELECT COUNT(*) FROM receipts WHERE currency = 'RUB';
-- SELECT COUNT(*) FROM expenses WHERE currency = 'RUB';
-- SELECT COUNT(*) FROM bank_transactions WHERE currency = 'RUB';
-- 
-- SELECT period, SUM(amount) as total FROM expenses WHERE currency = 'RUB' GROUP BY period ORDER BY period DESC;
-- SELECT category, SUM(amount) as total FROM expenses WHERE currency = 'RUB' GROUP BY category ORDER BY total DESC;

