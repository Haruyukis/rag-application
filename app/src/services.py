from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from loguru import logger

from src.config import ollama_base_url


def generate_response(user_query: str):
    logger.info("Loading documents in process...")
    documents = SimpleDirectoryReader("./data/demo").load_data()
    logger.info("Loading documents done")

    logger.info("Setup the embedding model and llm")
    # bge-base embedding model and llm
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.llm = Ollama(
        model="llama3", request_timeout=360.0, base_url=ollama_base_url
    )

    logger.info("Indexing the documents")
    index = VectorStoreIndex.from_documents(documents, show_progress=True)

    logger.info("Generating a response...")
    query_engine = index.as_query_engine()
    response = query_engine.query(user_query)
    logger.info("The response is generated")
    return response
