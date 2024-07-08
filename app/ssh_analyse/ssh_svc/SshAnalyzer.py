from llama_index.core import (
    Settings,
    SQLDatabase,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.retrievers import SQLRetriever
from llama_index.core.callbacks import CallbackManager
from llama_index.core.schema import TextNode
from llama_index.core.storage import StorageContext
from llama_index.core.query_pipeline import (
    FnComponent,
    QueryPipeline as QP,
    Link,
    InputComponent,
)
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import ChatResponse
from llama_index.core.program import LLMTextCompletionProgram
from llama_index.core.objects import SQLTableNodeMapping, ObjectIndex, SQLTableSchema
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama


from sqlalchemy import create_engine, MetaData, Column, String, Table, text
from typing import List, Dict
from pathlib import Path
import re, os, json

from ssh_analyse.ssh_model import TableInfo
from config import ollama_base_url


class SshAnalyzer:
    def __init__(self, path: str):
        """Constructor"""
        # Init
        self.engine = create_engine("sqlite:///ssh_log.db")
        self.sql_database = SQLDatabase(self.engine)

        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
        Settings.llm = Ollama(
            model="llama3", request_timeout=360.0, base_url=ollama_base_url
        )
        Settings.callback_manager = CallbackManager()

        # Create Table from Data
        self.metadata_obj = MetaData()
        self.table_names = ["successful_logins", "failed_logins"]
        self.columns = [
            [
                Column("user", String),
                Column("login_time", String),
                Column("log_message", String),
            ],
            [
                Column("user", String),
                Column("attempt_time", String),
                Column("log_message", String),
            ],
        ]
        self.create_table_from_data(path)

        # Structured Retrieval Metadata
        self.table_infos = self.structuring_table()

        # Object Indexing
        self.obj_retriever = self.object_indexing()

        # Get Table Context
        self.sql_retriever = SQLRetriever(self.sql_database)

        self.table_parser_component = FnComponent(
            fn=self.get_table_context_and_rows_str
        )

        # Parse Response to SQL
        self.sql_parser_component = FnComponent(fn=self.parse_response_to_sql)

        # text2sql_prompt_template
        self.text2sql_prompt = self.get_text2sql_prompt_template()

        # response_prompt_template
        self.response_synthesis_prompt = self.get_response_synthesis_prompt_template()

        # QP
        self.qp = self.get_query_pipeline()

    def run(self, user_query: str):
        return self.qp.run(query=user_query)

    def create_table_from_data(self, path: str):
        pattern_failed = r"^(\w{3} \d{2} \d{2}:\d{2}:\d{2}) \S+ sshd\[\d+\]: Connection closed by (?:authenticating|invalid) user (\S+) (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}) port (\d+) \[preauth\]$"
        pattern_succeed = r"^(\w{3}\s+\d{1,2} \d{2}:\d{2}:\d{2}) \S+ sshd\[\d+\]: Accepted publickey for (\S+) from (\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}) port (\d+)"
        # Creating all table with the defined columns
        tables = []
        for i, table_name in enumerate(self.table_names):
            tables.append(Table(table_name, self.metadata_obj, *(self.columns[i])))

        # Create the table in the db.
        self.metadata_obj.create_all(self.engine)

        with open(path, "r") as file:
            with self.engine.connect() as conn:
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

    def structuring_table(self):
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

        for table_name in self.table_names:
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

    def object_indexing(self):
        sql_database = SQLDatabase(self.engine)

        table_node_mapping = SQLTableNodeMapping(sql_database)
        table_schema_objs = [
            SQLTableSchema(table_name=t.table_name, context_str=t.table_summary)
            for t in self.table_infos
        ]

        obj_index = ObjectIndex.from_objects(
            table_schema_objs, table_node_mapping, VectorStoreIndex
        )

        obj_retriever = obj_index.as_retriever(similarity_top_k=3)
        return obj_retriever

    def get_table_context_and_rows_str(
        self, query_str: str, strtable_schema_objs: List[SQLTableSchema]
    ):
        """Get table context string."""

        def index_all_tables(
            sql_database: SQLDatabase, table_index_dir: str = "table_index_dir"
        ) -> Dict[str, VectorStoreIndex]:
            """Index all tables."""

            if not Path(table_index_dir).exists():
                os.makedirs(table_index_dir)

            # Creation d'un dictionnaire de VectorStoreIndex:

            vector_index_dict = {}
            engine = sql_database.engine
            print(len(sql_database.get_usable_table_names()))
            for table_name in sql_database.get_usable_table_names():
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

                    vector_index_dict[table_name] = index
                    index.storage_context.persist(f"{table_index_dir}/{table_name}")
                else:
                    storage_context = StorageContext.from_defaults(
                        persist_dir=f"{table_index_dir}/{table_name}"
                    )
                    index = load_index_from_storage(storage_context)

                    vector_index_dict[table_name] = index
            return vector_index_dict

        context_strs = []
        vector_index_dict = index_all_tables(self.sql_database)

        for table_schema_obj in strtable_schema_objs:
            table_info = self.sql_database.get_single_table_info(
                table_schema_obj.table_name
            )
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

    def parse_response_to_sql(self, response: ChatResponse) -> str:
        """Parse response to SQL."""
        response = response.message.content
        sql_query_start = response.find("SQLQuery:")
        if sql_query_start != -1:
            response = response[sql_query_start:]
            # TODO: move to removeprefix after Python 3.9+
            if response.startswith("SQLQuery:"):
                response = response[len("SQLQuery:") :]
        sql_result_start = response.find("SQLResult:")
        if sql_result_start != -1:
            response = response[:sql_result_start]
        return response.strip().strip("```").strip()

    def get_text2sql_prompt_template(self):
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
            "The response can be long but make sure to print the complete list of the SQL response without the SQLQuery.\n"
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
                "response_synthesis_prompt": self.response_synthesis_prompt,
                "response_synthesis_llm": Settings.llm,
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
        qp.add_link(
            "sql_output_parser", "response_synthesis_prompt", dest_key="sql_query"
        )
        qp.add_link(
            "sql_retriever", "response_synthesis_prompt", dest_key="context_str"
        )
        qp.add_link("input", "response_synthesis_prompt", dest_key="query_str")
        qp.add_link("response_synthesis_prompt", "response_synthesis_llm")

        return qp
