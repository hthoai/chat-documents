import chainlit as cl


async def stream_retriever_step(event):
    """Streams the retriever step based on the event received."""
    if event["event"] == "on_retriever_end":
        async with cl.Step(name="Retriever", language="python") as step:
            step_output = ""
            for doc in event["data"]["output"]["documents"]:
                step_output += str(doc) + "\n"
            step.output = step_output
            await step.update()


async def stream_retrieval_grader_step(event):
    """Streams the retrieval grader step based on the event received."""
    if event["event"] == "on_chat_model_start" and event["name"] == "retrieval_grader":
        async with cl.Step(name="Retrieval Grader") as step:
            await step.update()

    # if event["event"] == "on_chat_model_end" and event["name"] == "retrieval_grader":
    #     async with cl.Step(name="Retrieval Grader") as step:
    #         if "yes" in event["data"]["output"]["generations"][0][0]["text"]:
    #             step.output = "✅ Relevant"
    #         else:
    #             step.output = "❌ Not relevant"
    #         await step.update()


async def stream_hallucination_grader_step(event):
    """Streams the hallucination grader step based on the event received."""
    if (
        event["event"] == "on_chat_model_start"
        and event["name"] == "hallucination_grader"
    ):
        async with cl.Step(name="Hallucination Grader") as step:
            await step.update()

    if (
        event["event"] == "on_chat_model_end"
        and event["name"] == "hallucination_grader"
    ):
        async with cl.Step(name="Hallucination Grader") as step:
            if "yes" in event["data"]["output"]["generations"][0][0]["text"]:
                step.output = "✅ Decision: Generation is grounded in documents."
            else:
                step.output = "❌ Decision: Generation is not grounded in documents.\n🔄 Transform query."
            await step.update()


async def stream_answer_grader_step(event):
    """Streams the answer grader step based on the event received."""
    if event["event"] == "on_chat_model_start" and event["name"] == "answer_grader":
        async with cl.Step(name="Answer Grader") as step:
            await step.update()

    if event["event"] == "on_chat_model_end" and event["name"] == "answer_grader":
        async with cl.Step(name="Answer Grader") as step:
            if "yes" in event["data"]["output"]["generations"][0][0]["text"]:
                step.output = "✅ Decision: Answer addresses question."
            else:
                step.output = "❌ Decision: Answer does not address question.\n🔄 Transform query."
            await step.update()


async def stream_question_rewriter_step(event):
    """Streams the question rewriter step based on the event received."""
    if event["event"] == "on_chat_model_start" and event["name"] == "question_rewriter":
        async with cl.Step(name="Question Rewriter") as step:
            await step.update()

    if event["event"] == "on_chat_model_end" and event["name"] == "question_rewriter":
        async with cl.Step(name="Question Rewriter") as step:
            step.output = event["data"]["output"]["generations"][0][0]["text"]
            await step.update()


async def stream_decide_to_generate_step(event):
    """Streams the decide to generate step based on the event received."""
    if event["name"] == "decide_to_generate" and event["event"] == "on_chain_end":
        async with cl.Step(name="Retrieval Grader") as step:
            if event["data"]["output"] == "generate":
                step.output = (
                    "✅ Decision: Retrieved documents are relevant, generate answer."
                )
            elif event["data"]["output"] == "transform_query":
                step.output = "🔄 Decision: All documents are not relevant to question, transform query."
            elif event["data"]["output"] == "end_with_message":
                step.output = "❌ Decision: Transform query limit reached, ending."
            await step.update()


async def stream_end_with_message_after_grade_answer(event):
    """Streams the end with message after
    `grade_generation_v_documents_and_question`
    step based on the event received.
    """
    if (
        event["event"] == "on_chain_end"
        and event["name"] == "end_with_message"
        and event["metadata"]["langgraph_triggers"][0]
        == "branch:generate:grade_generation_v_documents_and_question:end_with_message"
    ):
        async with cl.Step(name="Hallucination/Answer Grader") as step:
            step.output = "❌ Decision: Transform query limit reached, ending."


async def stream_final_answer(event):
    """Streams the final answer based on the event received."""
    if (
        isinstance(event["data"]["output"], list)
        and "end_with_message" in event["data"]["output"][-1]
    ):
        answer = event["data"]["output"][-1]["end_with_message"]["generation"]
        source_documents = []
    elif "end_with_message" in event["data"]["output"]:
        answer = event["data"]["output"]["end_with_message"]["generation"]
        source_documents = []
    else:
        answer = event["data"]["output"]["generate"]["generation"]
        source_documents = event["data"]["output"]["generate"]["documents"]

    if source_documents:
        async with cl.Step(name="Answer Grader") as step:
            step.output = "Answer is generated based on the documents below:"
            await step.update()

        async with cl.Step(name="Answer Grader", language="python") as step:
            step_output = ""
            for doc in event["data"]["output"]["generate"]["documents"]:
                step_output += str(doc) + "\n"
            step.output = step_output
            await step.update()

    return answer, source_documents
