from typing import List

from langchain_core.pydantic_v1 import BaseModel, Field
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    Represents the state of the graph.

    Attributes:
        messages : With user question, generation
        generation : Answer to user question
        documents : List of retrieved documents
        iterations : Number of tries
        logs : Logs
    """

    messages: List
    question: str
    generation: str
    documents: List
    iterations: int
    logs: str


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""

    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'",
        enum=["yes", "no"],
    )


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""

    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'",
        enum=["yes", "no"],
    )


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""

    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'",
        enum=["yes", "no"],
    )
