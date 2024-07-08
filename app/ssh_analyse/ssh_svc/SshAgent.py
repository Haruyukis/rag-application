from llama_cloud import ChatMessage
from llama_index.core.schema import QueryBundle
from ssh_analyse.ssh_svc.SshAnalyzer import SshAnalyzer

from llama_index.core.tools import QueryEngineTool, BaseTool
from llama_index.core.query_pipeline import QueryPipeline as QP

from llama_index.core.agent import Task, ReActChatFormatter
from llama_index.core.agent.react.types import (
    ObservationReasoningStep,
    ActionReasoningStep,
    ResponseReasoningStep,
)
from llama_index.core.llms import ChatMessage


from llama_index.core.query_engine import BaseQueryEngine

from llama_index.core.query_pipeline import AgentInputComponent, AgentFnComponent


from typing import Any, Dict, List


class CustomQueryEngine(BaseQueryEngine):
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def _query(self, query_bundle):
        return self.pipeline.run(query_bundle)

    def _get_prompt_modules(self):
        return super()._get_prompt_modules()

    async def _aquery(self, query_bundle: QueryBundle):
        return await super()._aquery(query_bundle)


class SshAgent:
    """Ssh Analyzer Agent"""

    def __init__(self, path: str):
        """Initialization"""
        # Get SQL_Tool
        self.sql_query_engine = CustomQueryEngine(SshAnalyzer(path).qp)

        self.sql_tool = QueryEngineTool.from_defaults(
            query_engine=self.sql_query_engine,
            name="sql_tool",
            description=("Useful for analyzing logs in SQL Database"),
        )
        # Agent Input Component
        self.agent_input_component = AgentInputComponent(
            fn=self.get_agent_input_component
        )

        # Agent Prompt
        self.react_prompt_component = AgentFnComponent(
            fn=self.get_react_prompt_fn, partial_dict={"tools": [self.sql_tool]}
        )

        # Setup ReAct Agent Pipeline

        # Get Query Pipeline
        self.qp = self.get_query_pipeline()

    def get_react(self):
        # TODO
        pass

    def get_agent_input_component(
        self, task: Task, state: Dict[str, Any]
    ) -> Dict[str, Any]:

        # Initialization for InputComponent()
        if "current_reasoning" not in state:
            state["current_reasoning"] = []
        reasoning_step = ObservationReasoningStep(observation=task.input)
        state["current_reasoning"].append(reasoning_step)

        return {"input": task.input}

    def get_react_prompt_fn(
        self, task: Task, state: Dict[str, Any], input: str, tools: List[BaseTool]
    ) -> List[ChatMessage]:
        chat_formatter = ReActChatFormatter()
        return chat_formatter.format(
            tools,
            chat_history=task.memory.get() + state["memory"].get_all(),
            current_reasoning=state["current_reasoning"],
        )

    def get_parse_react_output_fn():
        # TODO
        pass

    def get_conditional_link_1():
        # TODO - changer le nom de la méthode parce que c'est pas ça
        pass

    def get_conditional_link_2():
        # TODO
        pass

    def get_process_agent_response_fn():
        # TODO - permet de synthétiser une réponse.
        pass

    def get_query_pipeline():
        qp = QP(verbose=True)
        return qp

    def run():
        # TODO Agent runner
        pass
