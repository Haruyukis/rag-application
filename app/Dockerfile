FROM dustynv/llama-index:r35.4.1

WORKDIR /app

COPY requirements.txt .

RUN pip install -U -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.controllers:app", "--host", "0.0.0.0", "--port", "8000"]