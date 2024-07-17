from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    InputComponent,
    FnComponent,
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import ChatResponse


from src.helpers.sentence_indexing import sentence_indexing
from src.config import ollama_base_url


class SshDatabase:
    """Ssh Database Generation"""

    def __init__(self, user_query: str, folder_path: str):
        """Constructor"""
        # Initialization
        Settings.llm = Ollama(
            model="llama3", request_timeout=360.0, base_url=ollama_base_url
        )
        Settings.callback_manager = CallbackManager()
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
        self.user_query = user_query

        # Indexing & Query Engine
        self.index = sentence_indexing(folder_path=folder_path)

        self.sentence_retriever = self.index.as_retriever(similarity_top_k=3)

        # Table Creation PromptTemplate
        self.table_creation_prompt = self.get_table_creation_prompt_template()

        # SQL Output Parser
        self.sql_parser_component = FnComponent(self.parse_response_to_sql)
        # Table Insertion PromptTemplate
        self.table_insert_prompt = self.get_table_insert_prompt_template()

        # Query Pipeline
        self.query_pipeline = self.get_query_pipeline()

    def run(self):
        """Answer generation"""
        return self.query_pipeline.run(query=self.user_query)

    def get_table_creation_prompt_template(self):
        """Create a Prompt Template for Table generation"""
        table_creation_prompt_str = """\
        Given an input question, generate SQL tables with their attributes needed to answer to the question. You are required to use the following format, each taking one line:
        Question: Question here
        SQLQuery: SQL Query to run to create the table
        Answer: Final answer here


        Here are some examples of logs.
        {retrieved_nodes}

        Question: {query_str}
        SQLQuery:
        
        """
        table_creation_prompt = PromptTemplate(table_creation_prompt_str)
        return table_creation_prompt

    def parse_response_to_sql(self, response: ChatResponse) -> str:
        """Parse response to SQL."""
        response = response.message.content
        sql_query_start = response.find("SQLQuery:")
        if sql_query_start != -1:
            response = response[sql_query_start:]
            # TODO: move to removeprefix after Python 3.9+
            if response.startswith("SQLQuery:"):
                response = response[len("SQLQuery:") :]
        sql_result_start = response.find("Answer:")
        if sql_result_start != -1:
            response = response[:sql_result_start]
        sql_query = response.strip().strip("```").strip()
        if sql_query[-1] == ";":
            sql_query = sql_query[:-1]
        self.sql_query = sql_query
        return sql_query

    def get_table_insert_prompt_template(self):
        """Create a Prompt Template for Table Insertion"""
        table_insert_prompt_str = """\
        Given an input question, first create a syntactically correct query to insert all important rows of the dataset inside the tables.
        Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Pay attention to which column is in which table. Also, qualify column names with the table name when needed. You are required to use the following format, each taking one line:

        Question: Question here
        SQLQuery: SQL Query to run
        Answer: Final answer here

        Only use tables listed below.
        {schema}

        
        Pay attention to all rows from the tables above.
        Question: {query_str}
        SQLQuery:

        """
        table_insert_prompt = PromptTemplate(table_insert_prompt_str)
        return table_insert_prompt

    def get_query_pipeline(self):
        """Create & Return the Query Pipeline of database generation"""

        qp = QP(
            modules={
                "input": InputComponent(),
                "sentence_retriever": self.sentence_retriever,
                "table_creation_prompt": self.table_creation_prompt,
                "llm1": Settings.llm,
                "sql_output_parser": self.sql_parser_component,
                # "table_insert_prompt": self.table_insert_prompt,
                # "llm2": Settings.llm,
            },
            verbose=True,
        )

        qp.add_link("input", "sentence_retriever")
        qp.add_link("input", "table_creation_prompt", dest_key="query_str")
        qp.add_link(
            "sentence_retriever", "table_creation_prompt", dest_key="retrieved_nodes"
        )

        qp.add_chain(["table_creation_prompt", "llm1", "sql_output_parser"])
        # qp.add_link("input", "table_insert_prompt", dest_key="query_str")
        # qp.add_link("sql_output_parser", "table_insert_prompt", dest_key="schema")
        # qp.add_chain(["table_insert_prompt", "llm2"])

        return qp
