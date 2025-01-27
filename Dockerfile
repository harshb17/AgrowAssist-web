FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p static/uploads && chmod -R 777 static/uploads

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1", "app:app"]