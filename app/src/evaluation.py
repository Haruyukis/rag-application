import asyncio

from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.core.llama_dataset.generator import RagDatasetGenerator
from llama_index.core.llama_dataset import LabelledRagDataset
from llama_index.core.llama_pack import download_llama_pack
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.evaluation import (
    RetrieverEvaluator,
    EmbeddingQAFinetuneDataset,
    generate_question_context_pairs,
)
from llama_index.core.node_parser import SentenceSplitter

from loguru import logger

from src.config import ollama_base_url, embedding_model, llm_model
import os

import pandas as pd


# Code from Llama-index in RetrieverEvaluator
def display_results(name, eval_results):
    """Display results from evaluate."""

    metric_dicts = []
    for eval_result in eval_results:
        metric_dict = eval_result.metric_vals_dict
        metric_dicts.append(metric_dict)

    full_df = pd.DataFrame(metric_dicts)

    columns = {
        "retrievers": [name],
        **{k: [full_df[k].mean()] for k in ["hit_rate", "mrr"]},
    }

    metric_df = pd.DataFrame(columns)

    return metric_df


# Initialization
logger.info("Doing some initialisation...")
Settings.llm = Ollama(model=llm_model, request_timeout=360.0, base_url=ollama_base_url)
Settings.embed_model = HuggingFaceEmbedding(model_name=embedding_model)
RagEvaluatorPack = download_llama_pack("RagEvaluatorPack", "./pack")
logger.info("Loading and indexing data...")

# Loading step
documents = SimpleDirectoryReader("data").load_data()
node_parser = SentenceSplitter(chunk_size=512)
nodes = node_parser.get_nodes_from_documents(documents)
for idx, node in enumerate(nodes):
    node.id_ = f"node_{idx}"

index = VectorStoreIndex(nodes)

query_engine = index.as_query_engine()
retriever = index.as_retriever(similarity_top_k=2)

# Instantiate a DatasetGenerator and Create a QA dataset for the Generation.
dataset_generator = RagDatasetGenerator.from_documents(
    documents, llm=Settings.llm, num_questions_per_chunk=2, show_progress=True
)

if os.path.exists("rag_dataset.json"):
    logger.info("Generation QA dataset already exist")
    rag_dataset = LabelledRagDataset.from_json("rag_dataset.json")
else:
    logger.info("Generating QA dataset")
    rag_dataset = dataset_generator.generate_dataset_from_nodes()
    rag_dataset.save_json("rag_dataset.json")

# Instantiate a dataset for the Retrieval.
if os.path.exists("retriever_dataset.json"):
    logger.info("Retrieval QA dataset already exist")
    retriever_dataset = EmbeddingQAFinetuneDataset.from_json("retriever_dataset.json")
else:
    logger.info("Retrieval QA dataset already exist")
    retriever_dataset = generate_question_context_pairs(
        documents, llm=Settings.llm, num_questions_per_chunk=2
    )
    retriever_dataset.save_json("retriever_dataset.json")


# Evaluation
rag_evaluator = RagEvaluatorPack(
    query_engine=query_engine, rag_dataset=rag_dataset, judge_llm=Settings.llm
)
retriever_evaluator = RetrieverEvaluator.from_metric_names(
    ["hit_rate", "mrr"], retriever=retriever
)


async def get_retriever_evaluation():
    logger.info("Getting retriever evaluation result")
    eval_results = await retriever_evaluator.aevaluate_dataset(retriever_dataset)
    return eval_results


eval_results = asyncio.run(get_retriever_evaluation())

logger.info("Getting generation evaluation result")
benchmark_df = rag_evaluator.run()
# Display
print(display_results("top-2 eval", eval_results))
print(benchmark_df)
