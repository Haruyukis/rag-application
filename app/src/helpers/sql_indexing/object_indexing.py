from llama_index.core import VectorStoreIndex, SQLDatabase, Settings
from llama_index.core.objects import SQLTableNodeMapping, SQLTableSchema, ObjectIndex
from src.config import embedding_model
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def object_indexing(engine, table_infos):
    """Object Indexing for each table"""
    Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)

    sql_database = SQLDatabase(engine)

    table_node_mapping = SQLTableNodeMapping(sql_database)
    table_schema_objs = [
        SQLTableSchema(table_name=t.table_name, context_str=t.table_summary)
        for t in table_infos
    ]

    obj_index = ObjectIndex.from_objects(
        table_schema_objs, table_node_mapping, VectorStoreIndex
    )

    obj_retriever = obj_index.as_retriever(similarity_top_k=3)
    return obj_retriever
