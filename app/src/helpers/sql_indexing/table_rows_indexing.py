from llama_index.core import SQLDatabase, VectorStoreIndex, Settings
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


from typing import Dict, List
from sqlalchemy import text
from loguru import logger
from src.config import embedding_model


def index_all_tables(
    sql_database: SQLDatabase, table_names: List[str]
) -> Dict[str, VectorStoreIndex]:
    """Index all tables."""
    Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)

    # Creation d'un dictionnaire de VectorStoreIndex:
    vector_index_dict = {}
    engine = sql_database.engine

    for table_name in table_names:
        logger.info(f"Indexing rows in table: {table_name}")

        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT * FROM {table_name}"))
            rows = [row[1:] for row in result.fetchall()]

            rows_to_tuple = []
            for row in rows:
                rows_to_tuple.append(tuple(row))

        nodes = [TextNode(text=str(t)) for t in rows_to_tuple]

        index = VectorStoreIndex(nodes, show_progress=True)
        vector_index_dict[table_name] = index
        logger.info(f"Succesfully indexed rows in table {table_name}")

    return vector_index_dict
