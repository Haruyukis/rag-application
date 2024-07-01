FROM dustynv/llama-index:r35.4.1

WORKDIR ollama

COPY model_pulling.sh .

RUN curl -fsSL https://ollama.com/install.sh | sh || (exit 0)

RUN chmod +x model_pulling.sh

RUN ./model_pulling.sh

CMD ollama serve