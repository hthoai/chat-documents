import chainlit as cl

from components.document_loader import load_documents
from components.index_builder import build_index


async def create_retriever(file):
    """
    Create the retriever.

    Args:
        file (File): The uploaded file.

    Returns:
        retriever (Retriever): The retriever.
    """
    async with cl.Step(name="Document Processor") as step:
        step.output = "Loading and processing the document."
        await step.update()
    documents = load_documents(file.path)

    async with cl.Step(name="Index Builder") as step:
        step.output = "Building the index."
        await step.update()
    vector_db = build_index(
        documents=documents,
        persist_directory="resources/qdrant_db/" + file.name,
    )

    retriever = vector_db.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.4}
    )

    return retriever
