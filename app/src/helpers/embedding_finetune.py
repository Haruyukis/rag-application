from src.helpers.corpus_loader import basic_load_corpus
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.core.evaluation import (
    EmbeddingQAFinetuneDataset,
    generate_qa_embedding_pairs,
)
from llama_index.finetuning import SentenceTransformersFinetuneEngine
from src.config import ollama_base_url

from loguru import logger
import os


def finetuning(folder_path, file_name):
    """Finetune the Embedding Model"""
    # Splitting into two dataset
    with open(folder_path + "/" + file_name, mode="r") as f:
        rows = f.readlines()
        with open(folder_path + "/train/" + file_name, mode="w") as training:
            for row in rows[: len(rows) // 2]:
                training.write(row)
        with open(folder_path + "/validate/" + file_name, mode="w") as validate:
            for row in rows[len(rows) // 2 :]:
                validate.write(row)
    logger.info(f"Successfully train-test split of the file {folder_path}/{file_name}")

    # Loading Nodes
    train_nodes = basic_load_corpus(folder_path + "/train/", verbose=True)
    val_nodes = basic_load_corpus(folder_path + "/validate/", verbose=True)

    # Generate, Storing or Loading the dataset
    Settings.llm = Ollama(
        model="llama3", request_timeout=360.0, base_url=ollama_base_url
    )
    if not os.path.exists("train_dataset.json"):
        logger.info("Starting to generate QA embedding pairs for training")
        train_dataset = generate_qa_embedding_pairs(
            train_nodes, Settings.llm, num_questions_per_chunk=1
        )
        train_dataset.save_json("train_dataset.json")
        logger.info("Successfully generated and stored QA embedding pairs for training")
    else:
        train_dataset = EmbeddingQAFinetuneDataset.from_json("train_dataset.json")
        logger.info("Successfully loaded QA embedding pairs for training")

    if not os.path.exists("val_dataset.json"):
        logger.info("Starting to generate QA embedding pairs for testing")

        val_dataset = generate_qa_embedding_pairs(
            val_nodes, Settings.llm, num_questions_per_chunk=1
        )
        val_dataset.save_json("val_dataset.json")
        logger.info("Successfully generated and stored QA embedding pairs for testing")
    else:
        val_dataset = EmbeddingQAFinetuneDataset.from_json("val_dataset.json")
        logger.info("Successfully loaded QA embedding pairs for testing")

    # Finetuning the HuggingFaceEmbeddingModel
    finetune_engine = SentenceTransformersFinetuneEngine(
        dataset=train_dataset,
        model_id="BAAI/bge-small-en-v1.5",
        model_output_path="llama_model_v1",
        val_dataset=val_dataset,
        epochs=3,
    )

    finetune_engine.finetune()
    logger.info("Successfully finetuned the embedding model")

    # Storing the model
    finetuned_embedding_model = finetune_engine.get_finetuned_model()
    finetuned_embedding_model.to_json()
    logger.info("Successfully stored the embedding model")
