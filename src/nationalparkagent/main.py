import logging
import os
import time
from uuid import uuid4
from typing import Any
from typing import Literal

from fastapi import FastAPI
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


# Keep log values readable when prompts or model responses are large.
def preview(value: Any, limit: int = 500) -> str:
    text = str(value).replace("\n", " ")
    return text if len(text) <= limit else f"{text[:limit]}..."


# Mock tools the agent can call while the real park APIs are not connected.
@tool
def check_weather(location: str) -> str:
    """Return the mock weather forecast for a location."""
    logger.info("TOOL check_weather START location=%s", location)
    result = f"It's always sunny in {location}."
    logger.info("TOOL check_weather END result=%s", result)
    return result


@tool
def get_location() -> str:
    """Return the user's mock current location."""
    logger.info("TOOL get_location START")
    result = "Yosemite National Park, California"
    logger.info("TOOL get_location END result=%s", result)
    return result


# Build the model from environment variables so the provider can be changed
# without changing the agent or API code.
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
    tools=[check_weather, get_location],
    system_prompt="You are a helpful national park agent.",
)

# FastAPI application and the request types for the OpenAI-style endpoint.
app = FastAPI()


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage] = Field(min_length=1)
    stream: bool = False


@app.post("/v1/chat/completions")
def chat_completions(request: ChatCompletionRequest):
    # Invoke the LangGraph agent and return its final assistant message.
    request_id = uuid4().hex[:8]
    started = time.perf_counter()
    logger.info(
        "[%s] REQUEST START model=%s messages=%d prompt=%s stream=%s",
        request_id,
        request.model or os.getenv("LLM_MODEL", "qwen3:1.7b"),
        len(request.messages),
        preview(request.messages[-1].content),
        request.stream,
    )
    try:
        result = agent.invoke(
            {"messages": [message.model_dump() for message in request.messages]},
        )
        response = result["messages"][-1].content
        logger.info(
            "[%s] REQUEST END duration_ms=%.0f response=%s",
            request_id,
            (time.perf_counter() - started) * 1000,
            preview(response),
        )
        return {"message": response}
    except Exception:
        logger.exception("[%s] REQUEST FAILED duration_ms=%.0f", request_id, (time.perf_counter() - started) * 1000)
        raise
