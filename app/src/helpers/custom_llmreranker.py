from llama_index.core import QueryBundle
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.core.llms import LLM
from typing import List, Optional
from llama_index.core.prompts import PromptTemplate
from pydantic import Field
from loguru import logger

class LlamaNodePostprocessor(BaseNodePostprocessor):
    """Custom node postprocessor for llama3"""

    top_n: int = Field(description="Top N nodes to return.")
    llm: LLM = Field(description="The LLM to rerank with.")

    def __init__(
        self,
        llm: LLM,
        top_n: int = 5,
    ) -> None:

        llm = llm
        top_n = top_n

        super().__init__(
            llm=llm,
            top_n=top_n,
        )

    def _postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        logger.info("Starting to rerank nodes")
        if query_bundle is None:
            raise ValueError("Query bundle must be provided.")
        if len(nodes) == 0:
            return []

        llm_str = """\
            User Input: {query_str}
            What you need to do, is to re-rank (give new score) for each node depending if they are useful or not to the user input. You are required to use the following format, each taking one line:

            **New Score**: New score here
            **Answer**: Answer here

            Here is the node:
            {node}

            **New Score**:
            **Answer**:

            
            """
        llm_prompt_template = PromptTemplate(llm_str).partial_format(
            query_str=query_bundle.query_str
        )

        for node in nodes:
            new_score = self._parse_answer_fn(
                self.llm.predict(llm_prompt_template.partial_format(node=node))
            )
            node.score = new_score
        logger.info("Successfully reranked nodes...")

        return sorted(nodes, key=lambda x: x.score or 0.0, reverse=True)[: self.top_n]

    def _parse_answer_fn(self, response: str):
        """Parse the answer to retrieve the score"""

        ranking_query_start = response.find("**New Score**:")
        if ranking_query_start != -1:
            response = response[ranking_query_start:]
            if response.startswith("**New Score**:"):
                response = response[len("**New Score**:") :]
        answer = response.find("**Answer**:")
        if answer != -1:
            response = response[:answer]
        return float(response)
