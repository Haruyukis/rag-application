# For Jetson

FROM dustynv/llama-index:r35.4.1

WORKDIR /app

COPY requirements.txt .

RUN pip install -U -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "src.controllers:app", "--host", "0.0.0.0", "--port", "8000"]

# For NVIDIA Card

# FROM ubuntu:20.04

# RUN apt-get update && \
#     apt-get install -y python3 python3-pip && \
#     apt-get clean

# WORKDIR /app

# COPY requirements.txt .

# RUN pip install -U -r requirements.txt

# COPY . .

# EXPOSE 12312

# CMD ["uvicorn", "src.controllers:app", "--host", "0.0.0.0", "--port", "12312"]