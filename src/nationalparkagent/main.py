import json
import logging
import os
import time
import chromadb
import httpx
from typing import Any
from typing import Literal
from pathlib import Path
from uuid import uuid4
from io import BytesIO

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient
from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
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
chroma_path = os.getenv(
    "CHROMA_PATH",
    "/tmp/nationalparkagent-chroma" if os.getenv("VERCEL") else "./chroma_db",
)
chroma_client = chromadb.PersistentClient(path=chroma_path)
document_collection = chroma_client.get_or_create_collection("documents")


# Keep log values readable when prompts or model responses are large.
def preview(value: Any, limit: int = 500) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else f"{text[:limit]}..."


# A tool is a normal Python function that the model is allowed to call.
# `async def` lets it perform future network/database work without blocking.
# `@tool` exposes the function name, description, and arguments to the LLM.
@tool
async def check_air_quality(location: str) -> str:
    """Return current air quality for a location using Open-Meteo."""
    logger.info("TOOL check_air_quality START location=%s", location)
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            geocoding_response = await client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": location, "count": 1, "language": "en", "format": "json"},
            )
            geocoding_response.raise_for_status()
            locations = geocoding_response.json().get("results", [])
            if not locations:
                return f"I couldn't find a location named {location}."

            place = locations[0]
            air_quality_response = await client.get(
                "https://air-quality-api.open-meteo.com/v1/air-quality",
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current": "us_aqi,pm2_5,pm10,ozone,nitrogen_dioxide",
                    "timezone": "auto",
                },
            )
            air_quality_response.raise_for_status()
            current = air_quality_response.json()["current"]

        result = (
            f"Current air quality for {place['name']}, {place.get('admin1', '')}: "
            f"U.S. AQI {current['us_aqi']}, PM2.5 {current['pm2_5']} µg/m³, "
            f"PM10 {current['pm10']} µg/m³, ozone {current['ozone']} µg/m³, "
            f"nitrogen dioxide {current['nitrogen_dioxide']} µg/m³ "
            f"(as of {current['time']})."
        )
    except (httpx.HTTPError, KeyError) as error:
        logger.exception("TOOL check_air_quality FAILED location=%s", location)
        result = f"Air quality lookup failed for {location}: {error}"

    logger.info("TOOL check_air_quality END result=%s", result)
    return result


@tool
async def check_park_alerts(park_name: str) -> str:
    """Return current National Park Service alerts for a park."""
    logger.info("TOOL check_park_alerts START park=%s", park_name)
    api_key = os.getenv("NPS_API_KEY")
    if not api_key:
        return "NPS_API_KEY is not configured, so park alerts are unavailable."

    try:
        async with httpx.AsyncClient(
            timeout=10,
            headers={"X-Api-Key": api_key},
        ) as client:
            parks_response = await client.get(
                "https://developer.nps.gov/api/v1/parks",
                params={"q": park_name, "limit": 1},
            )
            parks_response.raise_for_status()
            parks = parks_response.json().get("data", [])
            if not parks:
                return f"I couldn't find an NPS park named {park_name}."

            park = parks[0]
            alerts_response = await client.get(
                "https://developer.nps.gov/api/v1/alerts",
                params={"parkCode": park["parkCode"], "limit": 10},
            )
            alerts_response.raise_for_status()
            alerts = alerts_response.json().get("data", [])

        if not alerts:
            return f"There are no current alerts for {park['fullName']}."

        return "\n\n".join(
            f"{alert.get('category', 'Alert')}: {alert.get('title', 'Untitled')}\n"
            f"{alert.get('description', 'No description available.')}"
            for alert in alerts
        )
    except (httpx.HTTPError, KeyError) as error:
        logger.exception("TOOL check_park_alerts FAILED park=%s", park_name)
        return f"Park alert lookup failed for {park_name}: {error}"


@tool
async def get_location() -> str:
    """Return the approximate location of the request's public IP address."""
    logger.info("TOOL get_location START")
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get("https://ipinfo.io/json")
            response.raise_for_status()
            location = response.json()

        city = location.get("city")
        region = location.get("region")
        country = location.get("country")
        result = ", ".join(part for part in (city, region, country) if part)
        if not result:
            result = "Location unavailable."
    except (httpx.HTTPError, KeyError, ValueError) as error:
        logger.exception("TOOL get_location FAILED")
        result = f"Location lookup failed: {error}"

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
eris_client = MultiServerMCPClient({
    "eris": {
        "transport": "streamable_http",
        "url": "https://weather-api.madadipouya.com/mcp",
    }
})
agent = None
project_root = Path(__file__).resolve().parents[2]


@app.on_event("startup")
async def configure_agent():
    global agent
    try:
        eris_tools = await eris_client.get_tools()
    except Exception:
        logger.exception("Eris MCP unavailable; starting without weather tools")
        eris_tools = []

    agent = create_deep_agent(
        model=model,
        backend=FilesystemBackend(root_dir=str(project_root), virtual_mode=True),
        skills=["skills/"],
        tools=[
            *eris_tools,
            check_air_quality,
            check_park_alerts,
            get_location,
            search_park_documents,
        ],
        system_prompt="You are a helpful National Park travel assistant.",
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
