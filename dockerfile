FROM python:3.12

WORKDIR /app

COPY app.py .
COPY models/ ./models/
COPY src/data_processing.py ./src/
COPY config.yml .

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]