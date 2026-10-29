FROM python:3.11-slim

# Настройки Python для корректного вывода логов
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1


RUN apt-get update \
&& apt-get install -y --no-install-recommends \
    poppler-utils \
    curl \
&& rm -rf /var/lib/apt/lists/*


WORKDIR /app

# копирование файла с либами
COPY requirements.txt . 

# установка либ
RUN pip install --no-cache-dir -r requirements.txt

# копирование папок в докер
COPY ui ./ui
COPY mvp ./mvp 

# создание папок для выводов
RUN mkdir -p app/ui/output 

# указываем порт, на котором работает Streamlit
EXPOSE 8501 

# Проверяем, отвечает ли приложение
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Команда запуска приложения
CMD ["streamlit", "run", "ui/streamlit_app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]


