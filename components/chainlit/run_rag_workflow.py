from typing import Dict, List, Literal, Tuple

from components.chainlit.answer_utils import update_answer_with_source
from components.chainlit.stream_steps import (
    stream_answer_grader_step,
    stream_decide_to_generate_step,
    stream_end_with_message_after_grade_answer,
    stream_final_answer,
    stream_hallucination_grader_step,
    stream_question_rewriter_step,
    stream_retrieval_grader_step,
    stream_retriever_step,
)


async def run_rag_workflow(
    app: Literal["CompiledGraph"], inputs: Dict, file_path: str
) -> Tuple[str, List]:
    """
    Run the RAG workflow.

    Args:
        app (CompiledGraph): The RAG workflow.
        inputs (Dict): The inputs for the workflow.
        file_path (str): The path to the PDF file.

    Returns:
        answer (str): The answer string.
        pdf_elements (List): List of PDF elements for the source documents.
    """
    async for event in app.astream_events(
        inputs,
        version="v1",
    ):

        await stream_retriever_step(event)
        await stream_retrieval_grader_step(event)

        await stream_decide_to_generate_step(event)
        await stream_question_rewriter_step(event)
        await stream_hallucination_grader_step(event)
        await stream_answer_grader_step(event)

        await stream_end_with_message_after_grade_answer(event)

    answer, source_documents = await stream_final_answer(event)

    answer, pdf_elements = update_answer_with_source(
        answer=answer, source_documents=source_documents, file_path=file_path
    )

    return answer, pdf_elements
