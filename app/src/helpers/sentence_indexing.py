import os

from llama_index.core import (
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.callbacks import CallbackManager
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from loguru import logger
from src.config import embedding_model, llm_model, ollama_base_url
from src.helpers.corpus_loader import sentence_load_corpus


def sentence_indexing(folder_path: str, file: str) -> VectorStoreIndex:
    # Initialization:
    Settings.llm = Ollama(
        model=llm_model, request_timeout=360.0, base_url=ollama_base_url
    )

    Settings.callback_manager = CallbackManager()

    Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)

    # Sentence Loader
    nodes = sentence_load_corpus(
        directory=folder_path, file=file, chunk_size=75, chunk_overlap=50
    )

    for idx, node in enumerate(nodes):
        node.id_ = f"node_{idx}"

    if not os.path.exists(f"database_index_storage/{file}"):
        # Indexing & Storing
        index = VectorStoreIndex(nodes, show_progress=True)
        index.storage_context.persist(f"database_index_storage/{file}")
        logger.info("Successfully indexed and stored the index")
    else:
        # Loading
        storage_context = StorageContext.from_defaults(
            persist_dir=f"database_index_storage/{file}"
        )
        index = load_index_from_storage(storage_context)
        if not isinstance(index, VectorStoreIndex):
            raise TypeError("Loaded index is not of type VectorStoreIndex")
        logger.info("Successfully loaded the index")

    return index
