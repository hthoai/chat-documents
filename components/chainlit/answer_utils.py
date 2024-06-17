from typing import List, Tuple

import chainlit as cl


def update_answer_with_source(
    answer: str, source_documents: List, file_path: str
) -> Tuple[str, List]:
    """
    Update the answer with the source documents.

    Args:
        answer (str): The answer string.
        source_documents (List): List of source documents.
        file_path (str): Path to the PDF file.

    Returns:
        answer (str): The updated answer string.
        pdf_elements (List): List of PDF elements for the source documents.
    """
    pdf_elements = []

    if source_documents:
        for source_doc in source_documents:
            source_name = f"P{source_doc.metadata['page']}"
            pdf_elements.append(
                cl.Pdf(
                    name=source_name,
                    display="side",
                    path=file_path,
                    page=source_doc.metadata["page"],
                )
            )
        source_names = [pdf_el.name for pdf_el in pdf_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    return answer, pdf_elements
