from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from loguru import logger
import os


from src.config import ollama_base_url


def sentence_indexing(folder_path: str):
    # Initialization
    Settings.llm = Ollama(
        model="llama3", request_timeout=360.0, base_url=ollama_base_url
    )
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.callback_manager = CallbackManager()

    documents = SimpleDirectoryReader(folder_path).load_data()
    node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=20)

    nodes = node_parser.get_nodes_from_documents(documents)
    for idx, node in enumerate(nodes):
        node.id_ = f"node_{idx}"

    if not os.path.exists("database_index_storage"):
        # Indexing & Storing
        logger.info("Indexing and Storing")
        index = VectorStoreIndex(nodes, show_progress=True)
        index.storage_context.persist("database_index_storage")
    else:
        # Loading
        logger.info("Loading")
        storage_context = StorageContext.from_defaults(
            persist_dir="database_index_storage"
        )
        index = load_index_from_storage(storage_context)
    return index
