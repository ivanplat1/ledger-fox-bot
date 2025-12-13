-- Тестовые данные для ExpenseCatBot
-- Выполните этот SQL в Supabase SQL Editor после создания таблиц

-- Очистка существующих данных (опционально)
-- TRUNCATE TABLE expenses CASCADE;
-- TRUNCATE TABLE receipts CASCADE;
-- TRUNCATE TABLE bank_transactions CASCADE;

-- Тестовый user_id (замените на реальный ID пользователя)
-- Для тестирования используем user_id = 188376853

-- Вставка чеков (receipts)
INSERT INTO receipts (user_id, store, total, currency, purchased_at, tax_amount, items, receipt_hash, external_id, merchant_address) VALUES
-- Декабрь 2025
(188376853, 'ТОО "СуперМарт"', 5420.00, 'KZT', '2025-12-15 18:45:00+00', 542.00, 
 '[
   {"name": "Молоко пастеризованное 1л", "quantity": 3, "price": 1200.00, "category": "Молочные продукты"},
   {"name": "Хлеб пшеничный", "quantity": 2, "price": 400.00, "category": "Хлеб/Выпечка"},
   {"name": "Яйца куриные С1 10шт", "quantity": 1, "price": 820.00, "category": "Продукты"},
   {"name": "Говядина вырезка 0.8кг", "quantity": 1, "price": 3000.00, "category": "Мясо/Рыба"}
 ]'::jsonb,
 'hash_receipt_20251215_184500_5420', 'ext_001', NULL),

(188376853, 'ЗАО "ЭлектронПлюс"', 125000.00, 'KZT', '2025-12-10 15:20:00+00', 12500.00,
 '[
   {"name": "Смартфон Galaxy Pro 128GB", "quantity": 1, "price": 125000.00, "category": "Компьютеры/Телефоны"}
 ]'::jsonb,
 'hash_receipt_20251210_152000_125000', 'ext_002', NULL),

(188376853, 'ООО "Фруктовый Рай"', 3450.00, 'KZT', '2025-12-08 11:30:00+00', 345.00,
 '[
   {"name": "Яблоки красные 1.5кг", "quantity": 1, "price": 1200.00, "category": "Овощи/Фрукты"},
   {"name": "Бананы 1кг", "quantity": 1, "price": 950.00, "category": "Овощи/Фрукты"},
   {"name": "Апельсины 1кг", "quantity": 1, "price": 1300.00, "category": "Овощи/Фрукты"}
 ]'::jsonb,
 'hash_receipt_20251208_113000_3450', 'ext_003', NULL),

(188376853, 'ИП Петров А.В.', 2800.00, 'KZT', '2025-12-05 19:15:00+00', 280.00,
 '[
   {"name": "Пицца Маргарита 30см", "quantity": 1, "price": 2500.00, "category": "Доставка еды"},
   {"name": "Доставка", "quantity": 1, "price": 300.00, "category": "Доставка еды"}
 ]'::jsonb,
 'hash_receipt_20251205_191500_2800', 'ext_004', NULL),

-- Ноябрь 2025
(188376853, 'ТОО "Аптека Здоровье"', 4200.00, 'KZT', '2025-11-25 14:10:00+00', 420.00,
 '[
   {"name": "Витамины Комплекс", "quantity": 1, "price": 2800.00, "category": "Лекарства"},
   {"name": "Бинт стерильный 5м", "quantity": 2, "price": 700.00, "category": "Аптека"}
 ]'::jsonb,
 'hash_receipt_20251125_141000_4200', 'ext_005', NULL),

(188376853, 'Кафе "Уютное"', 8500.00, 'KZT', '2025-11-20 20:30:00+00', 850.00,
 '[
   {"name": "Стейк из говядины", "quantity": 1, "price": 4500.00, "category": "Ресторан/Кафе"},
   {"name": "Салат Греческий", "quantity": 1, "price": 2000.00, "category": "Ресторан/Кафе"},
   {"name": "Вино красное 0.75л", "quantity": 1, "price": 2000.00, "category": "Алкоголь"}
 ]'::jsonb,
 'hash_receipt_20251120_203000_8500', 'ext_006', NULL),

(188376853, 'АЗС "Топливо+"', 12000.00, 'KZT', '2025-11-15 08:45:00+00', 1200.00,
 '[
   {"name": "Бензин АИ-95 30л", "quantity": 1, "price": 12000.00, "category": "Бензин/Топливо"}
 ]'::jsonb,
 'hash_receipt_20251115_084500_12000', 'ext_007', NULL),

-- Октябрь 2025
(188376853, 'ТОО "МебельМир"', 85000.00, 'KZT', '2025-10-25 16:00:00+00', 8500.00,
 '[
   {"name": "Стол обеденный", "quantity": 1, "price": 85000.00, "category": "Мебель"}
 ]'::jsonb,
 'hash_receipt_20251025_160000_85000', 'ext_008', NULL),

(188376853, 'Магазин "Чистота"', 6800.00, 'KZT', '2025-10-18 17:20:00+00', 680.00,
 '[
   {"name": "Стиральный порошок 4кг", "quantity": 1, "price": 3200.00, "category": "Бытовая химия"},
   {"name": "Средство для мытья посуды 1л", "quantity": 2, "price": 1800.00, "category": "Бытовая химия"},
   {"name": "Губки для посуды", "quantity": 1, "price": 1800.00, "category": "Бытовая химия"}
 ]'::jsonb,
 'hash_receipt_20251018_172000_6800', 'ext_009', NULL),

(188376853, 'Кинотеатр "Звезда"', 4500.00, 'KZT', '2025-10-12 21:00:00+00', 450.00,
 '[
   {"name": "Билет в кино", "quantity": 2, "price": 3000.00, "category": "Кино"},
   {"name": "Попкорн средний", "quantity": 1, "price": 1500.00, "category": "Развлечения"}
 ]'::jsonb,
 'hash_receipt_20251012_210000_4500', 'ext_010', NULL),

-- Сентябрь 2025
(188376853, 'Книжный магазин "Читай-Город"', 5500.00, 'KZT', '2025-09-28 13:45:00+00', 550.00,
 '[
   {"name": "Роман современный", "quantity": 1, "price": 3500.00, "category": "Книги"},
   {"name": "Тетрадь 48 листов", "quantity": 3, "price": 600.00, "category": "Канцтовары"},
   {"name": "Ручки шариковые набор", "quantity": 1, "price": 1400.00, "category": "Канцтовары"}
 ]'::jsonb,
 'hash_receipt_20250928_134500_5500', 'ext_011', NULL),

(188376853, 'Фитнес-клуб "Сила"', 30000.00, 'KZT', '2025-09-15 10:00:00+00', 3000.00,
 '[
   {"name": "Абонемент на 3 месяца", "quantity": 1, "price": 30000.00, "category": "Фитнес"}
 ]'::jsonb,
 'hash_receipt_20250915_100000_30000', 'ext_012', NULL),

(188376853, 'Магазин "Красота"', 5200.00, 'KZT', '2025-09-10 15:30:00+00', 520.00,
 '[
   {"name": "Шампунь для волос 500мл", "quantity": 1, "price": 1800.00, "category": "Косметика/Гигиена"},
   {"name": "Крем для лица 50мл", "quantity": 1, "price": 2400.00, "category": "Косметика/Гигиена"},
   {"name": "Зубная паста 100мл", "quantity": 1, "price": 1000.00, "category": "Косметика/Гигиена"}
 ]'::jsonb,
 'hash_receipt_20250910_153000_5200', 'ext_013', NULL),

-- Август 2025
(188376853, 'Такси "Быстрое"', 1500.00, 'KZT', '2025-08-22 23:15:00+00', 150.00,
 '[
   {"name": "Поездка по городу", "quantity": 1, "price": 1500.00, "category": "Такси"}
 ]'::jsonb,
 'hash_receipt_20250822_231500_1500', 'ext_014', NULL),

(188376853, 'Супермаркет "Продукты+"', 8900.00, 'KZT', '2025-08-15 12:00:00+00', 890.00,
 '[
   {"name": "Сыр твердый 400г", "quantity": 1, "price": 2200.00, "category": "Молочные продукты"},
   {"name": "Йогурт натуральный 500г", "quantity": 4, "price": 2000.00, "category": "Молочные продукты"},
   {"name": "Колбаса вареная 300г", "quantity": 1, "price": 1500.00, "category": "Мясо/Рыба"},
   {"name": "Сок апельсиновый 1л", "quantity": 2, "price": 1600.00, "category": "Напитки"},
   {"name": "Печенье шоколадное 200г", "quantity": 1, "price": 1600.00, "category": "Сладости"}
 ]'::jsonb,
 'hash_receipt_20250815_120000_8900', 'ext_015', NULL);

-- Вставка расходов (expenses) - связаны с чеками
-- Используем подзапросы для получения ID чеков по receipt_hash
INSERT INTO expenses (user_id, source, store, amount, currency, date, receipt_id, expense_hash, status, period, category) VALUES
(188376853, 'receipt', 'ТОО "СуперМарт"', 5420.00, 'KZT', '2025-12-15', 
 (SELECT id FROM receipts WHERE receipt_hash = 'hash_receipt_20251215_184500_5420' LIMIT 1),
 'exp_hash_20251215_5420', 'approved', '2025-12', 'Молочные продукты'),
(188376853, 'receipt', 'ЗАО "ЭлектронПлюс"', 125000.00, 'KZT', '2025-12-10',
 (SELECT id FROM receipts WHERE receipt_hash = 'hash_receipt_20251210_152000_125000' LIMIT 1),
 'exp_hash_20251210_125000', 'approved', '2025-12', 'Компьютеры/Телефоны'),
(188376853, 'receipt', 'ООО "Фруктовый Рай"', 3450.00, 'KZT', '2025-12-08',
 (SELECT id FROM receipts WHERE receipt_hash = 'hash_receipt_20251208_113000_3450' LIMIT 1),
 'exp_hash_20251208_3450', 'approved', '2025-12', 'Овощи/Фрукты'),
(188376853, 'receipt', 'ИП Петров А.В.', 2800.00, 'KZT', '2025-12-05',
 (SELECT id FROM receipts WHERE receipt_hash = 'hash_receipt_20251205_191500_2800' LIMIT 1),
 'exp_hash_20251205_2800', 'approved', '2025-12', 'Доставка еды'),
(188376853, 'receipt', 'ТОО "Аптека Здоровье"', 4200.00, 'KZT', '2025-11-25',
 (SELECT id FROM receipts WHERE receipt_hash = 'hash_receipt_20251125_141000_4200' LIMIT 1),
 'exp_hash_20251125_4200', 'approved', '2025-11', 'Лекарства'),
(188376853, 'receipt', 'Кафе "Уютное"', 8500.00, 'KZT', '2025-11-20',
 (SELECT id FROM receipts WHERE receipt_hash = 'hash_receipt_20251120_203000_8500' LIMIT 1),
 'exp_hash_20251120_8500', 'approved', '2025-11', 'Ресторан/Кафе'),
(188376853, 'receipt', 'АЗС "Топливо+"', 12000.00, 'KZT', '2025-11-15',
 (SELECT id FROM receipts WHERE receipt_hash = 'hash_receipt_20251115_084500_12000' LIMIT 1),
 'exp_hash_20251115_12000', 'approved', '2025-11', 'Бензин/Топливо'),
(188376853, 'receipt', 'ТОО "МебельМир"', 85000.00, 'KZT', '2025-10-25',
 (SELECT id FROM receipts WHERE receipt_hash = 'hash_receipt_20251025_160000_85000' LIMIT 1),
 'exp_hash_20251025_85000', 'approved', '2025-10', 'Мебель'),
(188376853, 'receipt', 'Магазин "Чистота"', 6800.00, 'KZT', '2025-10-18',
 (SELECT id FROM receipts WHERE receipt_hash = 'hash_receipt_20251018_172000_6800' LIMIT 1),
 'exp_hash_20251018_6800', 'approved', '2025-10', 'Бытовая химия'),
(188376853, 'receipt', 'Кинотеатр "Звезда"', 4500.00, 'KZT', '2025-10-12',
 (SELECT id FROM receipts WHERE receipt_hash = 'hash_receipt_20251012_210000_4500' LIMIT 1),
 'exp_hash_20251012_4500', 'approved', '2025-10', 'Кино'),
(188376853, 'receipt', 'Книжный магазин "Читай-Город"', 5500.00, 'KZT', '2025-09-28',
 (SELECT id FROM receipts WHERE receipt_hash = 'hash_receipt_20250928_134500_5500' LIMIT 1),
 'exp_hash_20250928_5500', 'approved', '2025-09', 'Книги'),
(188376853, 'receipt', 'Фитнес-клуб "Сила"', 30000.00, 'KZT', '2025-09-15',
 (SELECT id FROM receipts WHERE receipt_hash = 'hash_receipt_20250915_100000_30000' LIMIT 1),
 'exp_hash_20250915_30000', 'approved', '2025-09', 'Фитнес'),
(188376853, 'receipt', 'Магазин "Красота"', 5200.00, 'KZT', '2025-09-10',
 (SELECT id FROM receipts WHERE receipt_hash = 'hash_receipt_20250910_153000_5200' LIMIT 1),
 'exp_hash_20250910_5200', 'approved', '2025-09', 'Косметика/Гигиена'),
(188376853, 'receipt', 'Такси "Быстрое"', 1500.00, 'KZT', '2025-08-22',
 (SELECT id FROM receipts WHERE receipt_hash = 'hash_receipt_20250822_231500_1500' LIMIT 1),
 'exp_hash_20250822_1500', 'approved', '2025-08', 'Такси'),
(188376853, 'receipt', 'Супермаркет "Продукты+"', 8900.00, 'KZT', '2025-08-15',
 (SELECT id FROM receipts WHERE receipt_hash = 'hash_receipt_20250815_120000_8900' LIMIT 1),
 'exp_hash_20250815_8900', 'approved', '2025-08', 'Молочные продукты');

-- Вставка банковских транзакций (bank_transactions)
INSERT INTO bank_transactions (user_id, amount, currency, transaction_date, description, transaction_hash, bank_name, account_number) VALUES
(188376853, -5420.00, 'KZT', '2025-12-15', 'Оплата в ТОО "СуперМарт"', 'txn_hash_20251215_001', 'Test Bank', '****1234'),
(188376853, -125000.00, 'KZT', '2025-12-10', 'Оплата в ЗАО "ЭлектронПлюс"', 'txn_hash_20251210_002', 'Test Bank', '****1234'),
(188376853, -3450.00, 'KZT', '2025-12-08', 'Оплата в ООО "Фруктовый Рай"', 'txn_hash_20251208_003', 'Test Bank', '****1234'),
(188376853, -2800.00, 'KZT', '2025-12-05', 'Оплата в ИП Петров А.В.', 'txn_hash_20251205_004', 'Test Bank', '****1234'),
(188376853, -4200.00, 'KZT', '2025-11-25', 'Оплата в ТОО "Аптека Здоровье"', 'txn_hash_20251125_005', 'Test Bank', '****1234'),
(188376853, -8500.00, 'KZT', '2025-11-20', 'Оплата в Кафе "Уютное"', 'txn_hash_20251120_006', 'Test Bank', '****1234'),
(188376853, -12000.00, 'KZT', '2025-11-15', 'Оплата на АЗС "Топливо+"', 'txn_hash_20251115_007', 'Test Bank', '****1234'),
(188376853, -85000.00, 'KZT', '2025-10-25', 'Оплата в ТОО "МебельМир"', 'txn_hash_20251025_008', 'Test Bank', '****1234'),
(188376853, -6800.00, 'KZT', '2025-10-18', 'Оплата в Магазин "Чистота"', 'txn_hash_20251018_009', 'Test Bank', '****1234'),
(188376853, -4500.00, 'KZT', '2025-10-12', 'Оплата в Кинотеатр "Звезда"', 'txn_hash_20251012_010', 'Test Bank', '****1234'),
(188376853, -5500.00, 'KZT', '2025-09-28', 'Оплата в Книжный магазин "Читай-Город"', 'txn_hash_20250928_011', 'Test Bank', '****1234'),
(188376853, -30000.00, 'KZT', '2025-09-15', 'Оплата в Фитнес-клуб "Сила"', 'txn_hash_20250915_012', 'Test Bank', '****1234'),
(188376853, -5200.00, 'KZT', '2025-09-10', 'Оплата в Магазин "Красота"', 'txn_hash_20250910_013', 'Test Bank', '****1234'),
(188376853, -1500.00, 'KZT', '2025-08-22', 'Оплата в Такси "Быстрое"', 'txn_hash_20250822_014', 'Test Bank', '****1234'),
(188376853, -8900.00, 'KZT', '2025-08-15', 'Оплата в Супермаркет "Продукты+"', 'txn_hash_20250815_015', 'Test Bank', '****1234');

-- Проверка данных
-- SELECT COUNT(*) FROM receipts;
-- SELECT COUNT(*) FROM expenses;
-- SELECT COUNT(*) FROM bank_transactions;
-- 
-- SELECT period, SUM(amount) as total FROM expenses GROUP BY period ORDER BY period DESC;
-- SELECT category, SUM(amount) as total FROM expenses GROUP BY category ORDER BY total DESC;
