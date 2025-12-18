# Dockerfile для ExpenseCatBot
# Railway будет использовать Nixpacks, но Dockerfile нужен как fallback

FROM python:3.11-slim

# Установка системных зависимостей и локалей для поддержки казахского языка
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-kaz \
    libzbar0 \
    libgl1 \
    libglib2.0-0 \
    locales \
    fonts-dejavu-core \
    fonts-liberation \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Настройка локалей для поддержки UTF-8 и казахского языка
RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    sed -i '/ru_RU.UTF-8/s/^# //g' /etc/locale.gen && \
    sed -i '/kk_KZ.UTF-8/s/^# //g' /etc/locale.gen || true && \
    locale-gen en_US.UTF-8 ru_RU.UTF-8 kk_KZ.UTF-8 || true

# Установка переменных окружения для поддержки UTF-8
ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8
ENV PYTHONIOENCODING=utf-8
ENV PYTHONUNBUFFERED=1

# Установка рабочей директории
WORKDIR /app

# Копирование requirements.txt
COPY requirements.txt .

# Установка Python зависимостей
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копирование кода приложения
COPY . .

# Запуск бота
CMD ["python", "expense_cat_bot.py"]

