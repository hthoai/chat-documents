import io

import chainlit as cl
from PIL import Image

from components.chainlit.create_retriever import create_retriever
from components.chainlit.run_rag_workflow import run_rag_workflow
from components.rag_workflow import RAGWorkflow


@cl.on_chat_start
async def on_chat_start():
    files = None

    # Wait for the user to upload a file
    while files is None:
        files = await cl.AskFileMessage(
            content="Please upload a PDF file to begin!",
            accept=["application/pdf"],
            max_size_mb=20,
            timeout=180,
        ).send()

    file = files[0]

    msg = cl.Message(content=f"Processing `{file.name}`...")
    await msg.send()

    retriever = await create_retriever(file)

    rag_workflow = RAGWorkflow(retriever=retriever)
    workflow = rag_workflow.create_workflow()
    app = workflow.compile()

    # Generate and save the workflow graph
    img_data = app.get_graph().draw_mermaid_png()
    img = Image.open(io.BytesIO(img_data))
    img.save("resources/rag_workflow.png")
    image = cl.Image(
        path="resources/rag_workflow.png", name="rag_workflow", display="side"
    )
    # Attach the image to the message
    await cl.Message(
        content="Here is the workflow graph: rag_workflow",
        elements=[image],
    ).send()

    # Let the user know that the system is ready
    msg.content = f"Processing `{file.name}` done. You can now ask questions!"
    await msg.update()

    cl.user_session.set("app", app)
    cl.user_session.set("file_path", file.path)


@cl.on_message
async def main(message: cl.Message):
    app = cl.user_session.get("app")  # type: RAGWorkflow
    file_path = cl.user_session.get("file_path")
    inputs = {"question": message.content, "iterations": 0}

    answer, pdf_elements = await run_rag_workflow(app, inputs, file_path)

    await cl.Message(content=answer, elements=pdf_elements).send()
