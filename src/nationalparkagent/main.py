import json
import logging
import os
import time
import chromadb
from typing import Any
from typing import Literal
from uuid import uuid4
from io import BytesIO

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from pydantic import BaseModel, Field


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("nationalparkagent")
logging.getLogger("httpx").setLevel(logging.WARNING)

text_splitter = RecursiveCharacterTextSplitter(
      chunk_size=1000,
      chunk_overlap=150,
  )

app = FastAPI()
chroma_client = chromadb.PersistentClient(path="./chroma_db")
document_collection = chroma_client.get_or_create_collection("documents")


# Keep log values readable when prompts or model responses are large.
def preview(value: Any, limit: int = 500) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else f"{text[:limit]}..."


# A tool is a normal Python function that the model is allowed to call.
# `async def` lets it perform future network/database work without blocking.
# `@tool` exposes the function name, description, and arguments to the LLM.
@tool
async def check_weather(location: str) -> str:
    """Return the mock weather forecast for a location."""
    logger.info("TOOL check_weather START location=%s", location)
    result = f"It's always sunny in {location}."
    logger.info("TOOL check_weather END result=%s", result)
    return result


# This tool takes no arguments and returns the user's current location.
# Multiple async tools can run without blocking the FastAPI event loop.
@tool
async def get_location() -> str:
    """Return the user's mock current location."""
    logger.info("TOOL get_location START")
    result = "Yosemite National Park, California"
    logger.info("TOOL get_location END result=%s", result)
    return result


def search_document_chunks(query: str, limit: int = 5) -> list[dict[str, Any]]:
    """Search Chroma and return the closest document chunks."""
    document_count = document_collection.count()
    if not document_count:
        return []

    result = document_collection.query(
        query_texts=[query],
        n_results=min(limit, document_count),
    )
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    return [
        {
            "document": document,
            "metadata": metadata or {},
            "distance": distance,
        }
        for document, metadata, distance in zip(documents, metadatas, distances)
    ]


@tool
async def search_park_documents(query: str) -> str:
    """Search uploaded national park documents for relevant information."""
    results = search_document_chunks(query)
    if not results:
        return "No matching document content was found."

    return "\n\n".join(
        f"Source: {item['metadata'].get('filename', 'unknown')}"
        + (f" page {item['metadata']['page']}" if "page" in item["metadata"] else "")
        + "\n"
        f"{item['document']}"
        for item in results
    )


# Build the model from environment variables so the provider can be changed
# without changing the agent or API code. This supports Ollama locally today
# and OpenRouter later.
model_options = {"temperature": 0}
if base_url := os.getenv("LLM_BASE_URL"):
    model_options["base_url"] = base_url
if api_key := os.getenv("LLM_API_KEY"):
    model_options["api_key"] = api_key

model = init_chat_model(
    os.getenv("LLM_MODEL", "qwen3:1.7b"),
    model_provider=os.getenv("LLM_PROVIDER", "ollama"),
    **model_options,
)
agent = create_agent(
    model=model,
    tools=[check_weather, get_location, search_park_documents],
    system_prompt=(
        "You are a helpful national park agent. Use search_park_documents "
        "for questions about uploaded park documents."
    ),
)

@app.post("/v1/ingest")
async def ingest_document(file: UploadFile = File(...)):
    """Extract and split an uploaded PDF or text document."""
    allowed_types = {"application/pdf", "text/plain"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=415,
            detail="Only PDF and plain text documents are supported.",
        )

    raw_file = await file.read()
    chunks = []
    metadatas = []

    if file.content_type == "application/pdf":
        reader = PdfReader(BytesIO(raw_file))
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            for chunk in text_splitter.split_text(page_text):
                chunks.append(chunk)
                metadatas.append({"filename": file.filename, "page": page_number})
    else:
        document_text = raw_file.decode("utf-8", errors="replace")
        for chunk in text_splitter.split_text(document_text):
            chunks.append(chunk)
            metadatas.append({"filename": file.filename})

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="The document contains no readable text.",
        )

    document_collection.upsert(
        ids=[str(uuid4()) for _ in chunks],
        documents=chunks,
        metadatas=metadatas,
    )

    return {"status": "indexed", "filename": file.filename, "chunks": len(chunks)}


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1)


@app.post("/v1/retrieve")
async def retrieve_documents(request: RetrievalRequest):
    """Return the five most similar document chunks for a query."""
    return {
        "query": request.query,
        "results": search_document_chunks(request.query),
    }


# An async generator produces one response chunk at a time, so clients can
# display the answer while the model is still generating it.
async def stream_chat(messages):
    # Convert Pydantic messages into the dictionary format expected by the
    # LangGraph agent.
    inputs = {"messages": [message.model_dump() for message in messages]}

    # `astream` is the async version of `stream`. In `messages` mode it yields
    # LLM tokens as they arrive; v2 provides a consistent event shape.
    async for chunk in agent.astream(
        inputs,
        stream_mode="messages",
        version="v2",
    ):
        if chunk["type"] != "messages":
            continue

        # Each message event contains the token and metadata about its source.
        token, metadata = chunk["data"]
        text = getattr(token, "text", "")

        if text:
            # Send Server-Sent Events (SSE), used by OpenAI-style streaming
            # clients. [DONE] below tells the client the stream is complete.
            yield f"data: {json.dumps({'choices': [{'delta': {'content': text}}]})}\n\n"

    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    # Because this endpoint is async, it can handle other requests while the
    # LLM or a tool is waiting on I/O.
    logger.info("Request received")
    if request.stream:
        # StreamingResponse reads from the async generator and forwards each
        # yielded chunk immediately.
        return StreamingResponse(
            stream_chat(request.messages),
            media_type="text/event-stream",
        )

    # For non-streaming requests, wait asynchronously for the complete run.
    result = await agent.ainvoke({
        "messages": [message.model_dump() for message in request.messages]
    })

    return {"message": result["messages"][-1].content}
