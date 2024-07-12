FROM dustynv/llama-index:r35.4.1

WORKDIR app

COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "src.controllers:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]