from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from loguru import logger
import os

from src.config import ollama_base_url
from src.helpers.corpus_loader import sentence_load_corpus


def sentence_indexing(folder_path: str) -> VectorStoreIndex:
    # Initialization:
    Settings.llm = Ollama(
        model="llama3", request_timeout=360.0, base_url=ollama_base_url
    )
    # Finetune Embedding
    Settings.callback_manager = CallbackManager()

    logger.info("Starting to load model: llama_model_v1")
    Settings.embed_model = HuggingFaceEmbedding(model_name="llama_model_v1")
    logger.info("Successfully loaded model: llama_model_v1")

    # Sentence Loader
    nodes = sentence_load_corpus(directory=folder_path, chunk_size=75, chunk_overlap=50)

    for idx, node in enumerate(nodes):
        node.id_ = f"node_{idx}"

    if not os.path.exists("database_index_storage"):
        # Indexing & Storing
        index = VectorStoreIndex(nodes, show_progress=True)
        index.storage_context.persist("database_index_storage")
        logger.info("Successfully indexed and stored the index")
    else:
        # Loading
        storage_context = StorageContext.from_defaults(
            persist_dir="database_index_storage"
        )
        index = load_index_from_storage(storage_context)
        if not isinstance(index, VectorStoreIndex):
            raise TypeError("Loaded index is not of type VectorStoreIndex")
        logger.info("Successfully loaded the index")

    return index
