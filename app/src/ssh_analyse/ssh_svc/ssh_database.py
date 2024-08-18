from llama_index.core import Settings, QueryBundle
from llama_index.core.callbacks import CallbackManager
from llama_index.core.query_pipeline import (
    QueryPipeline as QP,
    InputComponent,
    FnComponent,
)
from llama_index.llms.ollama import Ollama
from llama_index.core.prompts import PromptTemplate
from llama_index.core.llms import ChatResponse


from loguru import logger

from src.helpers.parser.parser_to_python import parse_response_to_python
from src.helpers.parser.llm_parser_to_python import parse_using_llm
from src.helpers.sentence_indexing import sentence_indexing
from src.config import ollama_base_url, llm_model
from src.helpers.custom_llmreranker import LlamaNodePostprocessor


class SshDatabase:
    """Ssh Database Generation"""

    def __init__(self, user_query: str, folder_path: str, file_name: str):
        """Constructor"""
        # Initialization
        Settings.callback_manager = CallbackManager()

        self.llm = Ollama(
            model=llm_model, request_timeout=600.0, base_url=ollama_base_url
        )
        self.user_query = user_query
        self.file_name = file_name

        # Indexing & Query Engine
        self.index = sentence_indexing(folder_path=folder_path, file=file_name)

        # Retriever processing
        self.process_retriever_component = FnComponent(
            self.process_retriever_component_fn
        )

        # Table Creation PromptTemplate
        self.table_creation_prompt = self.get_table_creation_prompt_template()

        # SQL Output Parser
        self.python_parser_component = FnComponent(
            self.parse_response_to_python_from_chat
        )

        self.python_parser_component1 = FnComponent(
            self.parse_response_to_python_from_chat1
        )

        # Table Insertion PromptTemplate
        self.table_insert_prompt = (
            self.get_table_insert_prompt_template().partial_format(
                path=f"./{folder_path}/{file_name}"
            )
        )

        # Query Pipeline
        self.query_pipeline = self.get_query_pipeline()

    def run(self):
        """Answer generation"""
        logger.info(
            f"Starting to run the query pipeline with the input: {self.user_query}"
        )
        return self.query_pipeline.run(query=self.user_query)

    def process_retriever_component_fn(self, user_query: str):
        """Transform the output of the sentence_retriver"""
        #with open("ranking_cache.txt", mode="r") as f:
        #    lines = f.readlines()
        #    try:
        #        if lines[0] == user_query + self.file_name + "\n":
        #            logger.info("Successfully retrieved nodes from cache")
        #            return "".join(lines[1:])
        #    except:
        #        logger.info("Failed to retrieve nodes from cache. No cache available")
        #        logger.info("Starting to retrieve nodes")

        #sentence_retriever = self.index.as_retriever(similarity_top_k=10)
        sentence_retriever = self.index.as_retriever(similarity_top_k=5)

        nodes = sentence_retriever.retrieve(user_query)

        # Create a QueryBundle from the user query
        # query_bundle = QueryBundle(query_str=user_query)

        # postprocessor = LlamaNodePostprocessor(
        #     top_n=5,
        #     llm=self.llm,
        # )
        # reranked_nodes = postprocessor.postprocess_nodes(
        #     nodes=nodes, query_bundle=query_bundle
        # )
        contexts = ""
        for node in nodes:
            contexts += str(node.text) + "\n"

        # for reranked_node in reranked_nodes:
        #     contexts += str(reranked_node.text) + "\n"
        # with open("ranking_cache.txt", mode="w") as f:
        #     logger.info("Starting to cache the relevant nodes")
        #     f.write(user_query + self.file_name + "\n")
        #     f.write(contexts)
        return contexts

    def get_table_creation_prompt_template(self):
        """Create a Prompt Template for Table generation"""

        table_creation_prompt_str = """\
        Given an input question and a Python code, complete the Python code by generating only the SQLAlchemy table definitions with their attributes to answer the input question.
        Each table needs to have an `id` attribute as the primary key. Do not use any ForeignKey and relationship. Do not remove or modify any existing Python code.

        You are required to use the following format:

        **Question**: Question here
        **Python Code**: Python Code here
        **Answer**: Final answer here



        Here are some examples of logs.
        {retrieved_nodes}

        **Question**: {query_str}
        **Python Code**: '''\
```python
import re
from sqlalchemy import Column, Integer, String, Boolean, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Engine
engine = create_engine("sqlite:///logs.db")
Base = declarative_base()

# The Table here




# Creating all table above in the database.
Base.metadata.create_all(engine)

# Creating the session to the database.
SessionMaker = sessionmaker(bind=engine)
session = SessionMaker()


```

        '''
        **Answer**:

        """
        logger.info("Starting to create the table")
        return PromptTemplate(table_creation_prompt_str)

    def parse_response_to_python_from_chat(self, response: ChatResponse) -> str:
        """Parse response to Python"""
        python_query = parse_response_to_python(
            str(response.message.content), file_name="creation.txt"
        )
        return python_query

    def parse_response_to_python_from_chat1(self, response: ChatResponse) -> str:
        """Parse response to Python"""
        python_query = parse_using_llm(str(response.message.content))
        return python_query

    def get_table_insert_prompt_template(self):
        """Create a Prompt Template for Table Insertion"""

        table_insert_prompt_str = """\
        Given an input question and a python code, complete the python code to insert each line in the database that answer the input question using regex.
        Pay attention to the regex pattern and use `re.search()`. When generating regex patterns that include literal parentheses, please ensure they are escaped (e.g., `\\(` and `\\)`). For capturing groups, use parentheses without escaping (e.g., `(\\w+)`). Be careful about whitespace in the regex pattern.
        When inserting inside the database, you must use `.add()`. Do not remove or modify any existing Python code.

        You are required to use the following format:

        **Question**: Question here
        **Python Code**: Python Code here
        **Answer**: Final answer here

        Here are some examples of logs.
        {retrieved_nodes}

        **Question**: {query_str}
        **Python Code**:
        ```python
        {python_code}


    with open({path}, "r") as logfile:
        for line in logfile:
            # Here is the regex
        
            # Here is the insertion in the database

    session.commit()

        
        ```
        
        **Answer**:

        """

        return PromptTemplate(table_insert_prompt_str)

    def get_query_pipeline(self):
        """Create & Return the Query Pipeline of database generation"""
        qp = QP(
            modules={
                "input": InputComponent(),
                "process_retriever": self.process_retriever_component,
                "table_creation_prompt": self.table_creation_prompt,
                "llm1": self.llm,
                "python_output_parser": self.python_parser_component,
                "table_insert_prompt": self.table_insert_prompt,
                "llm2": self.llm,
                "python_output_parser1": self.python_parser_component,
            },
            verbose=True,
        )

        qp.add_link("input", "process_retriever")
        qp.add_link("input", "table_creation_prompt", dest_key="query_str")
        qp.add_link(
            "process_retriever", "table_creation_prompt", dest_key="retrieved_nodes"
        )

        qp.add_chain(["table_creation_prompt", "llm1", "python_output_parser"])

        qp.add_link("input", "table_insert_prompt", dest_key="query_str")
        qp.add_link(
            "process_retriever", "table_insert_prompt", dest_key="retrieved_nodes"
        )
        qp.add_link(
            "python_output_parser", "table_insert_prompt", dest_key="python_code"
        )
        qp.add_chain(["table_insert_prompt", "llm2", "python_output_parser1"])

        return qp
