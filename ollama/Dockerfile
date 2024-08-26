# For Jetson

FROM dustynv/llama-index:r35.4.1

WORKDIR /ollama

COPY model_pulling.sh .

RUN curl -fsSL https://ollama.com/install.sh | sh || (exit 0)

RUN chmod +x model_pulling.sh

RUN ./model_pulling.sh

EXPOSE 11434

CMD ollama serve

# For basic NVIDIA Card

# FROM ubuntu:20.04

# WORKDIR /ollama

# COPY model_pulling.sh .

# RUN apt-get update && \
#     apt-get install -y curl && \
#     apt-get clean

# RUN curl -fsSL https://ollama.com/install.sh | sh || (exit 0)

# RUN chmod +x model_pulling.sh

# RUN ./model_pulling.sh

# EXPOSE 11434

# CMD ollama serve