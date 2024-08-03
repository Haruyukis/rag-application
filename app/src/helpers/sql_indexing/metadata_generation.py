from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.callbacks import CallbackManager
from llama_index.core.program import LLMTextCompletionProgram

from sqlalchemy import create_engine, text, inspect
from loguru import logger
import time

from typing import List
from src.config import ollama_base_url
from src.ssh_analyse.ssh_model import TableInfo


def structuring_table(table_names: List[str]) -> List[str]:
    """Generate metadata for each table"""
    # Setting initialization
    Settings.llm = Ollama(
        model="llama3", request_timeout=360.0, base_url=ollama_base_url
    )
    Settings.callback_manager = CallbackManager()

    # Initialization
    prompt_str = """\
    You are given SSH logs data.
    Please provide a summary of the table based on its name and some example rows.
    
    Here is the table name:
    {table_name}

    Here are the columns:
    {columns}

    Here are some example rows in the same order as the columns:
    {rows}

    Please provide a detailed summary of the table, including the table name and a brief description of the data.
    Summary:"""

    program = LLMTextCompletionProgram.from_defaults(
        output_cls=TableInfo,
        llm=Settings.llm,
        prompt_template_str=prompt_str,
    )

    # SQL Initialization
    engine = create_engine("sqlite:///logs.db")
    inspector = inspect(engine)

    # Metadata Generation
    table_infos = []

    for table_name in table_names:
        with engine.connect() as conn:
            results = conn.execute(text(f"SELECT * from {table_name} LIMIT 2"))
            rows = results.fetchall()
            columns = inspector.get_columns(table_name)

        # Retry system if the generation fails, Generate the metadata
        attempts = 0
        while attempts < 3:
            try:
                table_info = program(
                    table_name=table_name, rows=str(rows), columns=str(columns)
                )
                if table_info is not None:
                    logger.info(
                        f"Successfully generated the metadata for: {table_name}"
                    )
                    break
            except:
                attempts += 1
                time.sleep(5)
                logger.info(
                    f"Failed program completion for the {attempts} times, retrying..."
                )

        table_infos.append(table_info)
    return table_infos
