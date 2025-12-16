# Деплой ExpenseCatBot на Railway

## Подготовка

### 1. Установка системных зависимостей

Railway использует Nixpacks для автоматической сборки. Для установки системных библиотек (Tesseract, zbar) создайте файл `nixpacks.toml`:

```toml
[phases.setup]
nixPkgs = ["tesseract", "zbar"]

[phases.install]
cmds = ["pip install -r requirements.txt"]
```

### 2. Переменные окружения

Убедитесь, что все необходимые переменные окружения настроены в Railway:

**Обязательные:**
- `EXPENSECAT_BOT_TOKEN` - токен Telegram бота
- `SUPABASE_URL` - URL вашего Supabase проекта
- `SUPABASE_SERVICE_KEY` - Service Role ключ Supabase
- `OPENAI_API_KEY` - API ключ OpenAI

**Опциональные:**
- `OCR_ENGINE` - движок OCR: "tesseract", "paddleocr" или "both" (по умолчанию: "both")
- `EXPENSECAT_LOG_LEVEL` - уровень логирования: DEBUG, INFO, WARNING, ERROR (по умолчанию: INFO)
- `EXPENSECAT_LOG_DIR` - директория для логов (по умолчанию: "logs")
- `FEEDBACK_CHAT_ID` - ID Telegram канала для обратной связи (опционально)
- `FAILED_RECEIPTS_CHAT_ID` - ID Telegram канала для отклоненных чеков (опционально)

## Деплой на Railway

### Вариант 1: Через GitHub (рекомендуется)

1. **Подключите репозиторий:**
   - Зайдите на [railway.app](https://railway.app)
   - Нажмите "New Project"
   - Выберите "Deploy from GitHub repo"
   - Выберите ваш репозиторий

2. **Настройте переменные окружения:**
   - В настройках проекта → Variables
   - Добавьте все необходимые переменные

3. **Настройте ресурсы:**
   - В настройках сервиса → Resources
   - Установите RAM: минимум 512 MB (рекомендуется 1 GB)
   - CPU: 1 vCPU достаточно

4. **Деплой:**
   - Railway автоматически задеплоит при push в main ветку
   - Или нажмите "Deploy" вручную

### Вариант 2: Через Railway CLI

1. **Установите Railway CLI:**
   ```bash
   npm i -g @railway/cli
   railway login
   ```

2. **Инициализируйте проект:**
   ```bash
   railway init
   ```

3. **Добавьте переменные окружения:**
   ```bash
   railway variables set EXPENSECAT_BOT_TOKEN=your_token
   railway variables set SUPABASE_URL=your_url
   railway variables set SUPABASE_SERVICE_KEY=your_key
   railway variables set OPENAI_API_KEY=your_key
   ```

4. **Деплой:**
   ```bash
   railway up
   ```

## Проверка работы

После деплоя проверьте:

1. **Логи:**
   - В Railway Dashboard → Deployments → View Logs
   - Должны увидеть: "Starting ExpenseCatBot"
   - Должны увидеть: "Available OCR engines: ..."

2. **Бот в Telegram:**
   - Отправьте `/start` боту
   - Должен ответить приветственным сообщением

3. **Обработка чеков:**
   - Отправьте фото чека
   - Проверьте, что бот обрабатывает его

## Оптимизация для Railway

### Ограничение памяти

Если используете бесплатный тариф (0.5 GB RAM):

1. **Отключите PaddleOCR:**
   ```bash
   railway variables set OCR_ENGINE=tesseract
   ```

2. **Ограничьте размер изображений:**
   - Код уже ограничивает до 1800px (см. `prepare_image_for_ocr`)

3. **Мониторинг памяти:**
   - В Railway Dashboard → Metrics можно видеть использование памяти

### Масштабирование

При росте нагрузки:

1. **Увеличьте RAM:**
   - Settings → Resources → RAM: 1 GB или 2 GB

2. **Добавьте больше CPU:**
   - Settings → Resources → CPU: 2 vCPU

## Troubleshooting

### Бот не запускается

1. Проверьте логи в Railway Dashboard
2. Убедитесь, что все переменные окружения установлены
3. Проверьте, что токен бота правильный

### Ошибки OCR

1. Проверьте, что Tesseract установлен (должно быть в логах)
2. Если PaddleOCR не работает - это нормально, используется Tesseract
3. Проверьте логи: `EXPENSECAT_LOG_LEVEL=DEBUG`

### Out of Memory (OOM)

1. Увеличьте RAM до 1 GB
2. Отключите PaddleOCR: `OCR_ENGINE=tesseract`
3. Ограничьте размер изображений

## Стоимость

**Бесплатный тариф:**
- $5 кредитов/месяц
- Достаточно для тестирования и небольшой нагрузки

**Платный тариф:**
- ~$5-20/месяц в зависимости от использования
- 1 GB RAM, 1 vCPU: ~$10/месяц

## Полезные ссылки

- [Railway Documentation](https://docs.railway.app)
- [Railway Pricing](https://railway.app/pricing)
- [Nixpacks Documentation](https://nixpacks.com)

