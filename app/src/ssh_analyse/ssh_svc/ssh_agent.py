from llama_index.core import Settings, PromptTemplate, Response
from llama_index.core.callbacks import CallbackManager
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.chat_engine.types import AgentChatResponse

from llama_index.core.query_pipeline import StatefulFnComponent, QueryPipeline as QP

from src.ssh_analyse.ssh_svc.ssh_analyzer import SshAnalyzer

from llama_index.core.agent import (
    Task,
    QueryPipelineAgentWorker,
    AgentRunner,
)

from loguru import logger

from typing import Dict, Any, Tuple

from src.helpers.create_table import create_table_from_data
from src.config import ollama_base_url
from sqlalchemy import MetaData, String, Column, create_engine
from src.ssh_analyse.ssh_svc.helpers import structuring_table


class SshAgent:
    """Ssh Retry Agent"""

    def __init__(self, path: str):
        """Constructor"""
        # Init
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
        Settings.llm = Ollama(
            model="llama3", request_timeout=360.0, base_url=ollama_base_url
        )
        Settings.callback_manager = CallbackManager()

        # Agent Input Component
        self.agent_input = StatefulFnComponent(fn=self.agent_input_fn)

        # Retry Component
        self.retry_prompt = PromptTemplate(self.retry_prompt_str_fn())
        self.max_iter = 3
        # SQL Query Engine / Ssh Analyzer
        self.sql_query_engine = SshAnalyzer(path).qp

        # Validate Prompt Component
        self.validate_prompt = PromptTemplate(self.validate_prompt_str_fn())

        # Agent Output Component
        self.agent_output = StatefulFnComponent(fn=self.agent_output_fn)

        # Agent Query Pipeline
        self.qp = self.get_query_pipeline()

    # https://docs.llamaindex.ai/en/stable/examples/agent/agent_runner/query_pipeline_agent/?h=sql+agent#setup-simple-retry-agent-pipeline-for-text-to-sql

    def agent_input_fn(self, state: Dict[str, Any]) -> Dict:
        """Agent input function."""
        task = state["task"]
        if "convo_history" not in state:
            state["convo_history"] = []
        state["convo_history"].append(f"User: {task.input}")
        convo_history_str = "\n".join(state["convo_history"]) or "None"
        logger.info("Agent input FN")
        logger.info(task.input)
        logger.info("Agent input FN")
        return {"input": task.input, "convo_history": convo_history_str}

    # Retry Prompt.
    def retry_prompt_str_fn(self) -> str:
        retry_prompt_str = """\
        You are trying to generate a proper natural language query given a user input.

        This query will then be interpreted by a downstream text-to-SQL agent which
        will convert the query to a SQL statement. If the agent triggers an error,
        then that will be reflected in the current conversation history (see below).

        If the conversation history is None, use the user input. If its not None,
        generate a new SQL query that avoids the problems of the previous SQL query.

        Input: {input}
        Convo history (failed attempts): 
        {convo_history}

        New input: """

        return retry_prompt_str

    def validate_prompt_str_fn(self) -> str:
        validate_prompt_str = """\
        Given the user query, validate whether the inferred SQL query and response from executing the query is correct and answers the query.

        Answer with YES or NO.

        Query: {input}
        Inferred SQL query: {sql_query}
        SQL Response: {sql_response}

        Result: """
        return validate_prompt_str

    def agent_output_fn(
        self, state: Dict[str, Any], output: Response
    ) -> Tuple[AgentChatResponse, bool]:
        """Agent output component."""

        task = state["task"]
        print(f"> Inferred SQL Query: {output.metadata['sql_query']}")
        print(f"> SQL Response: {str(output)}")
        state["convo_history"].append(
            f"Assistant (inferred SQL query): {output.metadata['sql_query']}"
        )
        state["convo_history"].append(f"Assistant (response): {str(output)}")

        # run a mini chain to get response
        validate_prompt_partial = self.validate_prompt.as_query_component(
            partial={
                "sql_query": output.metadata["sql_query"],
                "sql_response": str(output),
            }
        )
        qp = QP(chain=[validate_prompt_partial, Settings.llm])
        validate_output = qp.run(input=task.input)

        state["count"] += 1
        is_done = False
        if state["count"] >= self.max_iter:
            is_done = True
        if "YES" in validate_output.message.content:
            is_done = True

        return str(output), is_done

    def get_query_pipeline(self):
        """Create Retry Agent Query Pipeline"""
        qp = QP(
            modules={
                "input": self.agent_input,
                "retry_prompt": self.retry_prompt,
                "llm": Settings.llm,
                "sql_query_engine": self.sql_query_engine,
                "output_component": self.agent_output,
            },
            verbose=True,
        )

        qp.add_link(
            "input", "retry_prompt", dest_key="input", input_fn=lambda x: x["input"]
        )
        qp.add_link(
            "input",
            "retry_prompt",
            dest_key="convo_history",
            input_fn=lambda x: x["convo_history"],
        )

        qp.add_chain(["retry_prompt", "llm"])

        qp.add_link("llm", "sql_query_engine", dest_key="query_str")

        qp.add_chain(["sql_query_engine", "output_component"])

        return qp

    def run(self, user_query: str) -> str:
        agent_worker = QueryPipelineAgentWorker(self.qp)
        agent = AgentRunner(
            agent_worker, callback_manager=CallbackManager([]), verbose=True
        )
        response = agent.chat(user_query)
        return response
