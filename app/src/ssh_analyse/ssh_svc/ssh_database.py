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

from typing import List, Dict
from sqlalchemy import Column, MetaData, Table, text

from ssh_analyse.ssh_model import TableInfo
from config import ollama_base_url

from pathlib import Path
import re, json, os


def create_table_from_data(
    path: str,
    table_names: List[str],
    metadata_obj: MetaData,
    columns: List[List[Column]],
    engine,
):
    pattern_failed = r"^(\w{3} \d{2} \d{2}:\d{2}:\d{2}) \S+ sshd\[\d+\]: Connection closed by (?:authenticating|invalid) user (\S+) (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}) port (\d+) \[preauth\]$"
    pattern_succeed = r"^(\w{3}\s+\d{1,2} \d{2}:\d{2}:\d{2}) \S+ sshd\[\d+\]: Accepted publickey for (\S+) from (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}) port (\d+)"
    # Creating all table with the defined columns
    tables = []
    for i, table_name in enumerate(table_names):
        tables.append(Table(table_name, metadata_obj, *(columns[i])))

    # Create the table in the db.
    metadata_obj.create_all(engine)

    with open(path, "r") as file:
        with engine.connect() as conn:
            for line in file:
                match_failed = re.search(pattern_failed, line)
                match_succeed = re.search(pattern_succeed, line)

                if match_failed:
                    username = match_failed.group(2)
                    attempt_time = match_failed.group(1)
                    insert_stmt = (
                        tables[1]
                        .insert()
                        .values(
                            user=username,
                            attempt_time=attempt_time,
                            log_message=line,
                        )
                    )
                    conn.execute(insert_stmt)
                elif match_succeed:
                    username = match_succeed.group(2)
                    login_time = match_succeed.group(1)
                    insert_stmt = (
                        tables[0]
                        .insert()
                        .values(
                            user=username,
                            login_time=login_time,
                            log_message=line,
                        )
                    )
                    conn.execute(insert_stmt)
            conn.commit()


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
            table_info = program(table_name=table_name)
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
