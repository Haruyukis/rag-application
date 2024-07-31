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


def finetuning(folder_path, file_name):
    """Finetune the Embedding Model"""
    # Splitting into two dataset
    logger.info("Splitting into two dataset")
    with open(folder_path + "/" + file_name, mode="r") as f:
        rows = f.readlines()
        with open(folder_path + "/train/" + file_name, mode="w") as training:
            for row in rows[: len(rows) // 2]:
                training.write(row)
        with open(folder_path + "/validate/" + file_name, mode="w") as validate:
            for row in rows[len(rows) // 2 :]:
                validate.write(row)

    # Loading Nodes
    logger.info("Loading training and validate nodes")
    train_nodes = basic_load_corpus(folder_path + "/train/", verbose=True)
    val_nodes = basic_load_corpus(folder_path + "/validate/", verbose=True)

    # Generate the dataset
    Settings.llm = Ollama(
        model="llama3", request_timeout=360.0, base_url=ollama_base_url
    )
    logger.info("Training dataset generation")
    train_dataset = generate_qa_embedding_pairs(train_nodes, Settings.llm)
    train_dataset.save_json("train_dataset.json")
    logger.info("Storing train_dataset.json")
    logger.info("Validate dataset generation")
    val_dataset = generate_qa_embedding_pairs(val_nodes, Settings.llm)
    logger.info("Storing val_dataset.json")
    logger.info("Both dataset generation done")

    # Save the dataset generated
    logger.info("Storing both dataset")
    val_dataset.save_json("val_dataset.json")

    # Load the dataset generated
    logger.info("Loading both dataset")
    train_dataset = EmbeddingQAFinetuneDataset.from_json("train_dataset.json")
    val_dataset = EmbeddingQAFinetuneDataset.from_json("val_dataset.json")

    # Finetuning the HuggingFaceEmbeddingModel
    finetune_engine = SentenceTransformersFinetuneEngine(
        dataset=train_dataset,
        model_id="BAAI/bge-base-en-v1.5",
        model_output_path="llama_model_v1",
        val_dataset=val_dataset,
        epochs=2,
    )
    logger.info("Starting to finetune the embedding model")
    finetune_engine.finetune()
    logger.info("Finetuning done")

    # Storing the model
    logger.info("Storing the finetuned model")
    finetuned_embedding_model = finetune_engine.get_finetuned_model()
    finetuned_embedding_model.to_json()
