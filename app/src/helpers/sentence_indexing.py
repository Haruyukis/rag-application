from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.helpers.embedding_finetune import finetuning

from loguru import logger
import os


from src.config import ollama_base_url
from src.helpers.corpus_loader import sentence_load_corpus


def sentence_indexing(folder_path: str):
    # Initialization
    Settings.llm = Ollama(
        model="llama3", request_timeout=360.0, base_url=ollama_base_url
    )

    # Finetune Embedding
    Settings.callback_manager = CallbackManager()
    logger.info("Finetuning the embedding model")
    finetuning(folder_path, "auth.log")
    logger.info("Finetuning the embedding model done")
    logger.info("Loading the llama_model_v1")
    Settings.embed_model = HuggingFaceEmbedding(model_name="llama_model_v1")
    logger.info("Loading OK")

    nodes = sentence_load_corpus(directory=folder_path, chunk_size=75, chunk_overlap=50)
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
