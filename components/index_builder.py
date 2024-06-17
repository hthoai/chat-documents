import os
from typing import List

from langchain_community.vectorstores import Chroma
from langchain_qdrant import Qdrant
from langchain_voyageai import VoyageAIEmbeddings
from loguru import logger

from config import settings


def build_index(documents: List, persist_directory: str) -> Chroma:
    """Build a vectorstore index from documents.

    Args:
        documents (List): List of document chunks.
        persist_directory (str): Directory to persist the index.

    Returns:
        Chroma: Vectorstore object.
    """
    embeddings = VoyageAIEmbeddings(
        voyage_api_key=settings.VOYAGE_API_KEY, model=settings.EMBEDDING_MODEL
    )
    logger.info(f"Embeddings used: {settings.EMBEDDING_MODEL}")

    if os.path.exists(persist_directory):
        logger.info(f"Index already exists in {persist_directory}")
        vector_db = Qdrant.from_existing_collection(
            embedding=embeddings,
            path=persist_directory,
            collection_name="GPTs",
        )

        return vector_db

    logger.info("Building index ...")

    vector_db = Qdrant.from_documents(
        documents=documents,
        embedding=embeddings,
        path=persist_directory,
        collection_name="GPTs",
    )
    logger.info(f"Index built in {persist_directory}")

    return vector_db
