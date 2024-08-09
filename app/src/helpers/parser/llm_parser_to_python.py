from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.prompts import PromptTemplate
from src.config import ollama_base_url, llm_model
from src.helpers.parser.parser_to_python import parse_response_to_python
from loguru import logger


def parse_using_llm(python_code: str):
    """Parse the response into a Python Code"""
    Settings.llm = Ollama(
        model="llama3", request_timeout=360.0, base_url=ollama_base_url
    )

    llm = Settings.llm
    llm_str = """\
        Given a python code, you need to retrieve only the python code. Do not put any header in your answer.
        Be careful to not forget any python code. You are required to use the following format, each taking one line:

        **Python Code**: Python code here
        **Answer**: Answer here
        
        Here is the python code:
        {python_code}

        **Python Code**:
        **Answer**:

        
        """

    llm_prompt_template = PromptTemplate(llm_str).partial_format(
        python_code=python_code
    )
    logger.info("\033[95m > Running parser corrector. \033[0m")
    response_generated = llm.predict(llm_prompt_template)

    return parse_response_to_python(response_generated)
