from llama_index.core import (
    Settings,
    SQLDatabase,
    VectorStoreIndex,
    load_index_from_storage,
)

from llama_index.core.storage import StorageContext
from llama_index.core.schema import TextNode
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.objects import SQLTableNodeMapping, SQLTableSchema, ObjectIndex

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.callbacks import CallbackManager

from typing import Dict
from sqlalchemy import text

from src.ssh_analyse.ssh_model import TableInfo
from src.config import ollama_base_url

from pathlib import Path
import json, os
from loguru import logger


def structuring_table(table_names):
    """Associating each table with their metadata"""
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
    Settings.llm = Ollama(
        model="llama3", request_timeout=360.0, base_url=ollama_base_url
    )
    Settings.callback_manager = CallbackManager()

    prompt_str = """\
    You are given a ssh logs data.
    Give me a summary of the table only by their name.
    Name:
    {table_name}
    Summary:"""

    program = LLMTextCompletionProgram.from_defaults(
        output_cls=TableInfo,
        llm=Settings.llm,
        prompt_template_str=prompt_str,
    )
    table_infos = []

    for table_name in table_names:
        if os.path.exists(f"{table_name}.json"):
            # Loading
            table_info = TableInfo.parse_file(Path(f"{table_name}.json"))
        else:
            logger.info("Completing the value")
            table_info = program(table_name=table_name)
            logger.info("Program worked")

            # Storing:
            out_file = f"{table_name}.json"
            json.dump(table_info.dict(), open(out_file, "w"))

        table_infos.append(table_info)
    return table_infos


def object_indexing(engine, table_infos):
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


def index_all_tables(
    sql_database: SQLDatabase, table_index_dir: str = "table_index_dir"
) -> Dict[str, VectorStoreIndex]:
    """Index all tables."""

    if not Path(table_index_dir).exists():
        os.makedirs(table_index_dir)

    # Creation d'un dictionnaire de VectorStoreIndex:

    vector_index_dict = {}
    engine = sql_database.engine

    for table_name in [
        "successful_logins",
        "failed_logins",
    ]:  # Look why it doesn't work sql_database.get_usable_table_names()
        print(f"Indexing rows in table: {table_name}")

        if not os.path.exists(f"{table_index_dir}/{table_name}"):
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT * FROM {table_name}"))
                rows = result.fetchall()

                rows_to_tuple = []
                for row in rows:
                    rows_to_tuple.append(tuple(row))

            nodes = [TextNode(text=str(t)) for t in rows_to_tuple]

            index = VectorStoreIndex(nodes, show_progress=True)

            index.storage_context.persist(f"{table_index_dir}/{table_name}")
        else:
            storage_context = StorageContext.from_defaults(
                persist_dir=f"{table_index_dir}/{table_name}"
            )
            index = load_index_from_storage(storage_context)

        vector_index_dict[table_name] = index
    return vector_index_dict
