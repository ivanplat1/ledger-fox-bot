-- Настройка Storage buckets для ExpenseCatBot
-- Выполните этот скрипт в SQL Editor в Supabase Dashboard

-- 1. Создаем bucket для отклоненных чеков (если его еще нет)
-- Если старый bucket уже существует, удалите его вручную через Dashboard перед выполнением скрипта
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'rejected-receipts',
    'rejected-receipts',
    false,  -- Приватный bucket
    52428800,  -- 50 MB лимит на файл
    ARRAY['image/jpeg', 'image/jpg', 'image/png', 'image/webp', 'image/heic', 'image/heif']
)
ON CONFLICT (id) DO NOTHING;

-- 2. Настраиваем политики доступа для service_role на bucket rejected-receipts

-- Удаляем существующие политики, если они есть (для идемпотентности)
DROP POLICY IF EXISTS "Service role can read rejected receipts" ON storage.objects;
DROP POLICY IF EXISTS "Service role can upload rejected receipts" ON storage.objects;
DROP POLICY IF EXISTS "Service role can update rejected receipts" ON storage.objects;
DROP POLICY IF EXISTS "Service role can delete rejected receipts" ON storage.objects;

-- Политика для чтения (service_role может читать файлы)
CREATE POLICY "Service role can read rejected receipts"
ON storage.objects FOR SELECT
TO service_role
USING (bucket_id = 'rejected-receipts');

-- Политика для записи (service_role может загружать файлы)
CREATE POLICY "Service role can upload rejected receipts"
ON storage.objects FOR INSERT
TO service_role
WITH CHECK (bucket_id = 'rejected-receipts');

-- Политика для обновления (service_role может обновлять файлы)
CREATE POLICY "Service role can update rejected receipts"
ON storage.objects FOR UPDATE
TO service_role
USING (bucket_id = 'rejected-receipts');

-- Политика для удаления (service_role может удалять файлы)
CREATE POLICY "Service role can delete rejected receipts"
ON storage.objects FOR DELETE
TO service_role
USING (bucket_id = 'rejected-receipts');

-- 4. Примечание о доступе к buckets
-- Service role по умолчанию обходит RLS (Row Level Security) и имеет полный доступ
-- к таблице storage.buckets, поэтому явные политики не требуются.
-- Если list_buckets() все еще не работает, убедитесь, что используется service_role ключ.

-- Проверка: список всех buckets (должен показать созданные buckets)
SELECT id, name, public, created_at FROM storage.buckets ORDER BY created_at;

