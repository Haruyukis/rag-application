from llama_index.core import Settings
from llama_index.core.callbacks import CallbackManager
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage, ChatResponse
from llama_index.core.tools import BaseTool
from llama_index.core.chat_engine.types import AgentChatResponse
from llama_index.core.tools import FunctionTool

from llama_index.core.query_pipeline import (
    AgentInputComponent,
    AgentFnComponent,
    ToolRunnerComponent,
    QueryPipeline,
)
from llama_index.core.base.llms.types import MessageRole


from llama_index.core.agent.react.types import (
    ObservationReasoningStep,
    ActionReasoningStep,
    ResponseReasoningStep,
)
from llama_index.core.agent import (
    Task,
    ReActChatFormatter,
    QueryPipelineAgentWorker,
    AgentRunner,
)
from llama_index.core.agent.react.output_parser import ReActOutputParser

from typing import Dict, Any, List

from src.ssh_analyse.ssh_svc.ssh_analyzer import SshAnalyzer
from src.config import ollama_base_url


class SshAgent:
    """Ssh Agent"""

    def __init__(self, path: str):
        Settings.callback_manager = CallbackManager()
        Settings.llm = Ollama(
            model="llama3", request_timeout=360.0, base_url=ollama_base_url
        )

        self.sql_analyzer_tool = SshAnalyzer(path)

        self.tools = [FunctionTool.from_defaults(fn=self.sql_analyzer_tool.run)]
        # Setup ReAct Agent Pipeline
        self.agent_input_component = AgentInputComponent(fn=self.agent_input_fn)

        # Setup ReAct prompt
        self.react_prompt_component = AgentFnComponent(
            fn=self.react_prompt_fn,
            partial_dict={
                "tools": self.tools,
            },
        )

        # Setup ReAct output parser
        self.parse_react_output = AgentFnComponent(fn=self.parse_react_output_fn)

        # Setup Run Tools
        self.run_tool = AgentFnComponent(fn=self.run_tool_fn)

        # Setup process_response
        self.process_response = AgentFnComponent(fn=self.process_response_fn)

        # Setup agent process_response
        self.process_agent_response = AgentFnComponent(
            fn=self.process_agent_response_fn
        )

        # Query Pipeline
        self.qp = self.get_qp()

    def agent_input_fn(self, task: Task, state: Dict[str, Any]) -> Dict[str, Any]:
        """Agent input function.

        Returns:
            A Dictionary of output keys and values. If you are specifying
            src_key when defining links between this component and other
            components, make sure the src_key matches the specified output_key.

        """
        # initialize current_reasoning
        if "current_reasoning" not in state:
            state["current_reasoning"] = []
        reasoning_step = ObservationReasoningStep(observation=task.input)
        state["current_reasoning"].append(reasoning_step)
        return {"input": task.input}

    def react_prompt_fn(
        self, task: Task, state: Dict[str, Any], input: str, tools: List[BaseTool]
    ) -> List[ChatMessage]:
        # Add input to reasoning
        chat_formatter = ReActChatFormatter()
        return chat_formatter.format(
            tools,
            chat_history=task.memory.get() + state["memory"].get_all(),
            current_reasoning=state["current_reasoning"],
        )

    def parse_react_output_fn(
        self, task: Task, state: Dict[str, Any], chat_response: ChatResponse
    ):
        """Parse ReAct output into a reasoning step."""
        output_parser = ReActOutputParser()
        reasoning_step = output_parser.parse(chat_response.message.content)
        return {"done": reasoning_step.is_done, "reasoning_step": reasoning_step}

    def run_tool_fn(
        self, task: Task, state: Dict[str, Any], reasoning_step: ActionReasoningStep
    ):
        """Run tool and process tool output."""
        tool_runner_component = ToolRunnerComponent(
            self.tools, callback_manager=task.callback_manager
        )
        tool_output = tool_runner_component.run_component(
            tool_name=reasoning_step.action,
            tool_input=reasoning_step.action_input,
        )
        observation_step = ObservationReasoningStep(observation=str(tool_output))
        state["current_reasoning"].append(observation_step)
        # TODO: get output
        print(observation_step.get_content())
        return {"response_str": observation_step.get_content(), "is_done": False}

    def process_response_fn(
        self, task: Task, state: Dict[str, Any], response_step: ResponseReasoningStep
    ):
        """Process response."""
        state["current_reasoning"].append(response_step)
        response_str = response_step.response
        # Now that we're done with this step, put into memory
        state["memory"].put(ChatMessage(content=task.input, role=MessageRole.USER))
        state["memory"].put(
            ChatMessage(content=response_str, role=MessageRole.ASSISTANT)
        )

        return {"response_str": response_str, "is_done": True}

    def process_agent_response_fn(
        self, task: Task, state: Dict[str, Any], response_dict: dict
    ):
        """Process agent response."""
        return (
            AgentChatResponse(response_dict["response_str"]),
            response_dict["is_done"],
        )

    def get_qp(self):
        qp = QueryPipeline(verbose=True)

        qp.add_modules(
            {
                "agent_input": self.agent_input_component,
                "react_prompt": self.react_prompt_component,
                "llm": Settings.llm,
                "react_output_parser": self.parse_react_output,
                "run_tool": self.run_tool,
                "process_response": self.process_response,
                "process_agent_response": self.process_agent_response,
            }
        )
        # link input to react prompt to parsed out response (either tool action/input or observation)
        qp.add_chain(["agent_input", "react_prompt", "llm", "react_output_parser"])

        # add conditional link from react output to tool call (if not done)
        qp.add_link(
            "react_output_parser",
            "run_tool",
            condition_fn=lambda x: not x["done"],
            input_fn=lambda x: x["reasoning_step"],
        )
        # add conditional link from react output to final response processing (if done)
        qp.add_link(
            "react_output_parser",
            "process_response",
            condition_fn=lambda x: x["done"],
            input_fn=lambda x: x["reasoning_step"],
        )

        # whether response processing or tool output processing, add link to final agent response
        qp.add_link("process_response", "process_agent_response")
        qp.add_link("run_tool", "process_agent_response")

        return qp

    def run(self, user_query: str) -> str:
        agent_worker = QueryPipelineAgentWorker(self.qp)
        agent = AgentRunner(
            agent_worker, callback_manager=CallbackManager([]), verbose=True
        )
        response = agent.chat(user_query)
        return response
