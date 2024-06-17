from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from loguru import logger

from config import settings


def load_documents(file_path: str) -> List:
    """
    Load and split documents from a PDF file.

    Args:
        file_path (str): Path to the PDF file.

    Returns:
        List: List of document chunks.
    """

    logger.info(f"Loading documents from {file_path}")
    loader = PyPDFLoader(file_path)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP
    )

    return text_splitter.split_documents(document)
