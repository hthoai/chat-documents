from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from components.schemas import GradeAnswer, GradeDocuments, GradeHallucinations


def create_rag_chain(llm: LLM):
    """
    Create the RAG chain.

    Returns:
        The RAG chain.
    """

    human = """You are an assistant for question-answering tasks. \n
        Use the following pieces of retrieved context to answer the question. \n
        If you don't know the answer, just say that you don't know. \n
        Question: {question} \n
        Context: {context} \n
        Answer:"""

    prompt = ChatPromptTemplate.from_messages([("human", human)])

    return prompt | llm | StrOutputParser()


def create_retrieval_grader(llm: LLM):
    """
    Create the retrieval grader.

    Returns:
        The retrieval grader.
    """

    system = """You are a grader assessing relevance of a retrieved document to a user question. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        {format_instructions}"""

    parser = PydanticOutputParser(pydantic_object=GradeDocuments)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n User question: {question}",
            ),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    return prompt | llm.with_config(run_name="retrieval_grader") | parser


def create_hallucination_grader(llm: LLM):
    """
    Create the hallucination grader.

    Returns:
        The hallucination grader.
    """

    system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
            Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts. \n
            {format_instructions}"""

    parser = PydanticOutputParser(pydantic_object=GradeHallucinations)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Set of facts: \n\n {documents} \n\n LLM generation: {generation}",
            ),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    return prompt | llm.with_config(run_name="hallucination_grader") | parser


def create_answer_grader(llm: LLM):
    """
    Create the answer grader.

    Returns:
        The answer grader.
    """

    system = """You are a grader assessing whether an answer addresses / resolves a question. \n
            Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question. \n
            {format_instructions}"""

    parser = PydanticOutputParser(pydantic_object=GradeAnswer)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", "User question: {question} \n\n LLM generation: {generation}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    return prompt | llm.with_config(run_name="answer_grader") | parser


def create_question_rewriter(llm: LLM):
    """
    Create the question rewriter.

    Returns:
        The question rewriter.
    """

    system = """You a question re-writer that converts an input question to a better version that is optimized \n
            for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning. \n
            Return the best version without any explanation."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    return prompt | llm.with_config(run_name="question_rewriter") | StrOutputParser()
