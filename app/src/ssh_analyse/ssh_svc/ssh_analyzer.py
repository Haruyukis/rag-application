from typing import List

from llama_index.core import Settings, SQLDatabase
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import ChatResponse
from llama_index.core.objects import SQLTableSchema
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_pipeline import FnComponent, InputComponent
from llama_index.core.query_pipeline import QueryPipeline as QP
from llama_index.core.retrievers import SQLRetriever
from llama_index.llms.ollama import Ollama
from loguru import logger
from sqlalchemy import create_engine, inspect
from src.config import llm_model, ollama_base_url
from src.helpers.parser.parser_to_sql import parse_response_to_sql
from src.helpers.sql_indexing.metadata_generation import structuring_table
from src.helpers.sql_indexing.object_indexing import object_indexing
from src.helpers.sql_indexing.table_rows_indexing import index_all_tables


class SshAnalyzer:
    """Ssh Analyzer Tool"""

    def __init__(self):
        """Constructor"""
        # Init
        self.engine = create_engine("sqlite:///logs.db")
        self.sql_database = SQLDatabase(self.engine)

        Settings.llm = Ollama(
            model=llm_model, request_timeout=360.0, base_url=ollama_base_url
        )
        Settings.callback_manager = CallbackManager()

        # Create Table from Data
        self.table_names = inspect(self.engine).get_table_names()

        # Structured Retrieval Metadata
        self.table_infos = structuring_table(table_names=self.table_names)

        # Object Indexing
        self.obj_retriever = object_indexing(
            engine=self.engine, table_infos=self.table_infos
        )

        # Get Table Context
        self.sql_retriever = SQLRetriever(self.sql_database)

        self.table_parser_component = FnComponent(
            fn=self.get_table_context_and_rows_str
        )

        # Parse Response to SQL
        self.sql_parser_component = FnComponent(fn=self.parse_response_to_sql_from_chat)

        # text2sql_prompt_template
        self.text2sql_prompt = self.get_text2sql_prompt_template()

        # response_prompt_template
        self.response_synthesis_prompt = self.get_response_synthesis_prompt_template()

        # QP
        self.qp = self.get_query_pipeline()

    def run(self, query_str: str):
        """Generate an answer to the query_str."""
        return self.qp.run(query=query_str)[0].get_content()

    def get_table_context_and_rows_str(
        self, query_str: str, strtable_schema_objs: List[SQLTableSchema]
    ):
        """Get table context string."""
        context_strs = []
        vector_index_dict = index_all_tables(self.sql_database, self.table_names)

        for table_schema_obj in strtable_schema_objs:
            table_info = self.sql_database.get_single_table_info(
                table_schema_obj.table_name
            ).replace("id (INTEGER), ", "")

            if table_schema_obj.context_str:
                table_opt_context = " The table description is: "
                table_opt_context += table_schema_obj.context_str
                table_info += table_opt_context

            retriever = vector_index_dict[table_schema_obj.table_name].as_retriever(
                similarity_top_k=2
            )

            relevant_nodes = retriever.retrieve(query_str)
            if len(relevant_nodes) > 0:
                table_row_context = "\nHere are some relevant example rows (values in the same order as columns above)"
                for node in relevant_nodes:
                    table_row_context += str(node.get_content()) + "\n"
                    table_info += table_row_context
            context_strs.append(table_info)
        return "\n\n".join(context_strs)

    def parse_response_to_sql_from_chat(self, response: ChatResponse) -> str:
        """Parse response to SQL."""
        # response = response.message.content
        # sql_query_start = response.find("SQLQuery:")
        # if sql_query_start != -1:
        #     response = response[sql_query_start:]
        #     # TODO: move to removeprefix after Python 3.9+
        #     if response.startswith("SQLQuery:"):
        #         response = response[len("SQLQuery:") :]
        # sql_result_start = response.find("SQLResult:")
        # if sql_result_start != -1:
        #     response = response[:sql_result_start]
        # sql_query = response.strip().strip("```").strip()
        # if sql_query[-1] == ";":
        #     sql_query = sql_query[:-1]
        return parse_response_to_sql(str(response.message.content))

    def get_text2sql_prompt_template(self):
        """Text2SQL Prompting"""

        text2sql_prompt_str = """\
        Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
        Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Pay attention to which column is in which table. Also, qualify column names with the table name when needed. You are required to use the following format, each taking one line:

        Question: Question here
        SQLQuery: SQL Query to run
        SQLResult: Result of the SQLQuery
        Answer: Final answer here

        Only use tables listed below.
        {schema}

        
        Question: {query_str}
        SQLQuery:

        """

        text2sql_prompt = PromptTemplate(text2sql_prompt_str).partial_format(
            dialect=self.engine.dialect.name
        )
        return text2sql_prompt

    def get_response_synthesis_prompt_template(self):
        response_synthesis_prompt_str = (
            "Given an input question, generate a response to answer the query.\n"
            "Respect as much as possible the user query.\n"
            "Make sure to answer the query.\n"
            "Do not summarize the answer.\n"
            "Query: {query_str}\n"
            "SQL: {sql_query}\n"
            "SQL Response: {context_str}\n"
            "Response: "
        )

        response_synthesis_prompt = PromptTemplate(
            response_synthesis_prompt_str,
        )

        return response_synthesis_prompt

    def get_query_pipeline(self):
        qp = QP(
            modules={
                "input": InputComponent(),
                "table_retriever": self.obj_retriever,
                "table_output_parser": self.table_parser_component,
                "text2sql_prompt": self.text2sql_prompt,
                "text2sql_llm": Settings.llm,
                "sql_output_parser": self.sql_parser_component,
                "sql_retriever": self.sql_retriever,
            },
            verbose=True,
        )

        qp.add_link("input", "table_retriever")
        qp.add_link("input", "table_output_parser", dest_key="query_str")
        qp.add_link("input", "text2sql_prompt", dest_key="query_str")
        qp.add_link(
            "table_retriever", "table_output_parser", dest_key="strtable_schema_objs"
        )
        qp.add_link("table_output_parser", "text2sql_prompt", dest_key="schema")

        qp.add_chain(
            ["text2sql_prompt", "text2sql_llm", "sql_output_parser", "sql_retriever"]
        )

        return qp

    def get_sql_query(self):
        """Getter for sql_query"""
        return self.sql_query
