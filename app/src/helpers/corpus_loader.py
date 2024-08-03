from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser, SentenceSplitter
from loguru import logger


def basic_load_corpus(directory: str, verbose=False):
    """Transform a document into nodes with SimpleNodeParser"""
    documents = SimpleDirectoryReader(directory).load_data()
    node_parser = SimpleNodeParser.from_defaults()
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=verbose)
    logger.info(f"Successfully retrieved nodes from: {directory}")
    return nodes


def sentence_load_corpus(
    directory: str, chunk_size: int, chunk_overlap: int, verbose=False
):
    """Transform a document into nodes with SentenceSplitter"""
    documents = SimpleDirectoryReader(directory).load_data()
    node_parser = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = node_parser.get_nodes_from_documents(documents, show_progress=verbose)
    logger.info(f"Successfully retrieved nodes from: {directory}")
    return nodes
