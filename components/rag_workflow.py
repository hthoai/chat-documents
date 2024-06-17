from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from loguru import logger

from components.chains import (
    create_answer_grader,
    create_hallucination_grader,
    create_question_rewriter,
    create_rag_chain,
    create_retrieval_grader,
)
from components.schemas import GraphState
from config import settings


class RAGWorkflow:
    """RAG Workflow using LangGraph."""

    def __init__(self, retriever, max_iterations=1):
        self.max_iterations = max_iterations
        self.retriever = retriever
        self.llm = ChatOpenAI(
            model="yi-large",
            temperature=0,
            api_key=settings.YI_API_KEY,
            base_url=settings.YI_BASE_URL,
        )
        # self.llm = ChatGoogleGenerativeAI(
        #     model="gemini-1.5-flash-latest",
        #     temperature=0.15,
        #     google_api_key=settings.GOOGLE_API_KEY,
        # )
        self.llm1 = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash-latest",
            temperature=0.15,
            google_api_key=settings.GOOGLE_API_KEY_B,
        )
        self.rag_chain = create_rag_chain(self.llm)
        self.retrieval_grader = create_retrieval_grader(self.llm1)
        self.hallucination_grader = create_hallucination_grader(self.llm)
        self.answer_grader = create_answer_grader(self.llm)
        self.question_rewriter = create_question_rewriter(self.llm)

    def retrieve(self, state: GraphState) -> GraphState:
        """
        Retrieve documents.

        Args:
            state (GraphState): The current graph state.

        Returns:
            state (GraphState): New key added to state, documents.
        """

        logger.info("RETRIEVE")
        question = state["question"]
        documents = self.retriever.invoke(question)
        logger.info(f"Retrieved {len(documents)} documents for question: {question}")

        return {"documents": documents}

    def generate(self, state: GraphState) -> GraphState:
        """
        Generate answer.

        Args:
            state (GraphState): The current graph state.

        Returns:
            state (GraphState): New key added to state, generation.
        """

        logger.info("GENERATE")
        question = state["question"]
        documents = state["documents"]
        generation = self.rag_chain.invoke({"context": documents, "question": question})

        return {"generation": generation, "documents": documents}

    def grade_documents(self, state: GraphState) -> GraphState:
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (GraphState): The current graph state.

        Returns:
            state (GraphState): Updates documents key with only filtered relevant documents.
        """

        logger.info("CHECK DOCUMENT RELEVANCE TO QUESTION")
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        for d in documents:
            score = self.retrieval_grader.invoke(
                {"question": question, "document": d.page_content}
            )
            grade = score.binary_score
            if grade == "yes":
                logger.info("GRADE: DOCUMENT RELEVANT")
                filtered_docs.append(d)
            else:
                logger.info("GRADE: DOCUMENT NOT RELEVANT")

        return {"documents": filtered_docs}

    def transform_query(self, state: GraphState) -> GraphState:
        """
        Transform the query to produce a better question.

        Args:
            state (GraphState): The current graph state.

        Returns:
            state (GraphState): Updates question key with a re-phrased question.
        """

        logger.info("TRANSFORM QUERY")
        iterations = state["iterations"]
        question = state["question"]

        iterations += 1

        better_question = self.question_rewriter.invoke({"question": question})
        logger.info(f"RE-WRITTEN QUESTION: {better_question}")

        return {
            "question": better_question,
            "iterations": iterations,
        }

    def decide_to_generate(self, state: GraphState) -> str:
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (GraphState): The current graph state.

        Returns:
            str: Binary decision for next node to call.
        """
        logger.info("ASSESS GRADED DOCUMENTS")
        filtered_documents = state["documents"]
        iterations = state["iterations"]

        if not filtered_documents:
            logger.info(
                "DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY"
            )
            if iterations == self.max_iterations:
                logger.info("DECISION: TRANSFORM QUERY LIMIT REACHED, ENDING")
                return "end_with_message"

            return "transform_query"

        logger.info("DECISION: GENERATE")
        return "generate"

    def grade_generation_v_documents_and_question(self, state: GraphState) -> str:
        """
        Determines whether the generation is grounded in the document and answers question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Decision for next node to call
        """
        logger.info("CHECK HALLUCINATIONS")
        documents = state["documents"]
        generation = state["generation"]
        question = state["question"]
        iterations = state["iterations"]

        score = self.hallucination_grader.invoke(
            {"documents": documents, "generation": generation}
        )
        grade = score.binary_score

        if grade == "yes":
            logger.info("DECISION: GENERATION IS GROUNDED IN DOCUMENTS")
            # Check question-answering
            logger.info("GRADE GENERATION vs QUESTION")
            score = self.answer_grader.invoke(
                {"question": question, "generation": generation}
            )

            grade = score.binary_score

            if grade == "yes":
                logger.info("DECISION: GENERATION ADDRESSES QUESTION")
                return "useful"
            else:
                logger.info("DECISION: GENERATION DOES NOT ADDRESS QUESTION, RE-TRY")
        else:
            logger.info("DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY")

        if iterations == self.max_iterations:
            logger.info("DECISION: TRANSFORM QUERY LIMIT REACHED, ENDING")
            return "end_with_message"

        return "not_useful"

    def end_with_message(self, state: GraphState) -> GraphState:
        """
        End the workflow with a message.

        Returns:
            GraphState: The final graph state with a message.
        """
        logger.info("END WITH MESSAGE")
        return {
            "documents": state["documents"],
            "generation": "Sorry, I couldn't find an answer for your question.",
        }

    def create_workflow(self) -> StateGraph:
        """
        Create the RAG workflow.

        Returns:
            StateGraph: The RAG workflow.
        """
        workflow = StateGraph(GraphState)

        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("transform_query", self.transform_query)
        workflow.add_node("end_with_message", self.end_with_message)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
                "end_with_message": "end_with_message",
            },
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation_v_documents_and_question,
            {
                "not_useful": "transform_query",
                "useful": END,
                "end_with_message": "end_with_message",
            },
        )
        workflow.add_edge("end_with_message", END)

        return workflow
