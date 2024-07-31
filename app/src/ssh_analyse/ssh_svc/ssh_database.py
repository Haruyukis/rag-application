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

from loguru import logger

from src.helpers.sentence_indexing import sentence_indexing
from src.config import ollama_base_url


class SshDatabase:
    """Ssh Database Generation"""

    def __init__(self, user_query: str, folder_path: str):
        """Constructor"""
        # Initialization
        self.llm = Ollama(
            model="llama3", request_timeout=600.0, base_url=ollama_base_url
        )

        Settings.callback_manager = CallbackManager()
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
        self.user_query = user_query

        # Indexing & Query Engine
        self.index = sentence_indexing(folder_path=folder_path)

        # Retriever processing
        self.process_retriever_component = FnComponent(
            self.process_retriever_component_fn
        )

        # Table Creation PromptTemplate
        self.table_creation_prompt = self.get_table_creation_prompt_template()

        # SQL Output Parser
        self.python_parser_component = FnComponent(self.parse_response_to_python)

        self.output_prompt = self.get_output_template()

        # Table Insertion PromptTemplate
        self.table_insert_prompt = (
            self.get_table_insert_prompt_template().partial_format(
                path=f"./{folder_path}/auth.log"
            )
        )

        # Query Pipeline
        self.query_pipeline = self.get_query_pipeline()

    def run(self):
        """Answer generation"""
        logger.info(f"Input: {self.user_query}")
        logger.info("Running the query pipeline...")
        return (
            self.query_pipeline.run(query=self.user_query)
            + """\
\n\n
from sqlalchemy import inspect
# Function to drop empty tables
def drop_empty_tables(engine, Base):
    inspector = inspect(engine)
    for table_name in inspector.get_table_names():

        table = Base.metadata.tables[table_name]
        if session.query(table).first() is None:
            print(f"Dropping empty table: {table_name}")
            table.drop(engine)

# Drop empty tables
drop_empty_tables(engine, Base)
"""
        )

    def process_retriever_component_fn(self, user_query: str):
        """Transform the output of the sentence_retriver"""

        logger.info("Sentence Retriever Output processing...")
        sentence_retriever = self.index.as_retriever(similarity_top_k=6)

        relevant_nodes = sentence_retriever.retrieve(user_query)
        logger.info("Relevant Node Retrieved...")
        contexts = ""
        for i in range(6):
            contexts += str(relevant_nodes[i]) + "\n"
        return contexts

    def get_table_creation_prompt_template(self):
        """Create a Prompt Template for Table generation"""

        table_creation_prompt_str = """\
        Given an input question and a python code, complete the python code by generating only SQL tables with their attributes needed to answer to the question.
        Pay attention to the table creation. Pay attention to the table name, they need to answer to the input question. You are required to use the following format, each taking one line:

        **Question**: Question here
        **Python Code**: Python Code here
        **Answer**: Final answer here



        Here are some examples of logs.
        {retrieved_nodes}

        **Question**: {query_str}
        **Python Code**: '''\
        ```python
        import re
        from sqlalchemy import Column, Integer, String, ForeignKey
        from sqlalchemy.ext.declarative import declarative_base
        from sqlalchemy.orm import sessionmaker
        from sqlalchemy import create_engine

        # Engine
        engine = create_engine("sqlite:///logs.db")
        Base = declarative_base()
        
        # The Table here
        

        # Creating all table above in the database.
        Base.metadata.create_all(engine)

        # Creating the session to the database.
        session = sessionmaker(bind=engine)()


        ```

        '''
        **Answer**:
        
        """
        table_creation_prompt = PromptTemplate(table_creation_prompt_str)
        logger.info("Starting Table Creation...")
        return table_creation_prompt

    def parse_response_to_python(self, response: ChatResponse) -> str:
        """Parse response to Python"""
        logger.info("Table Creation Done...")
        logger.info("Reading the python code generated")
        response = response.message.content
        python_query_start = response.find("**Python Code**:")
        if python_query_start != -1:
            response = response[python_query_start:]
            # TODO: move to removeprefix after Python 3.9+
            if response.startswith("**Python Code**:"):
                response = response[len("**Python Code**:") :]
        answer = response.find("**Answer**:")
        if answer != -1:
            response = response[:answer]

        python_query = response.strip().strip("```").strip()
        if python_query.startswith("python"):
            python_query = python_query[len("python") :]

        return python_query

    def get_table_insert_prompt_template(self):
        """Create a Prompt Template for Table Insertion"""
        logger.info("Starting Table Insertion...")

        table_insert_prompt_str = """\
        Given an input question, complete the python code to insert the rows of the file in the database using regex.
        Pay attention on the insertion. Be careful to not use re.match(). You are required to use the following format, each taking one line:

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
            # Here is the regex search

        
            # Here is the insertion in the database

    session.commit()

        
        ```


        **Answer**:


        """
        table_insert_prompt = PromptTemplate(table_insert_prompt_str)

        return table_insert_prompt

    def get_output_template(self):
        """Reformat the response in the right format"""
        output_prompt_str = """\
        Please keep only the python code and remove the other text
        {response}


        """
        output_prompt = PromptTemplate(output_prompt_str)
        return output_prompt

    def get_query_pipeline(self):
        """Create & Return the Query Pipeline of database generation"""

        qp = QP(
            modules={
                "input": InputComponent(),
                "process_retriever": self.process_retriever_component,
                # "table_creation_prompt": self.table_creation_prompt,
                # "llm1": self.llm,
                # "python_output_parser": self.python_parser_component,
                # "table_insert_prompt": self.table_insert_prompt,
                # "llm2": self.llm,
                # "python_output_parser1": self.python_parser_component,
                # "output_parser": self.output_prompt,
                # "llm3": Settings.llm,
                # "python_output_parser2": self.python_parser_component,
            },
            verbose=True,
        )

        qp.add_link("input", "process_retriever")
        # qp.add_link("input", "table_creation_prompt", dest_key="query_str")
        # qp.add_link(
        #     "process_retriever", "table_creation_prompt", dest_key="retrieved_nodes"
        # )

        # qp.add_chain(["table_creation_prompt", "llm1", "python_output_parser"])

        # qp.add_link("input", "table_insert_prompt", dest_key="query_str")
        # qp.add_link(
        #     "process_retriever", "table_insert_prompt", dest_key="retrieved_nodes"
        # )
        # qp.add_link(
        #     "python_output_parser", "table_insert_prompt", dest_key="python_code"
        # )
        # qp.add_chain(["table_insert_prompt", "llm2", "python_output_parser1"])
        # qp.add_link("python_output_parser1", "output_parser", dest_key="response")
        # qp.add_chain(["output_parser", "llm3"])

        return qp
