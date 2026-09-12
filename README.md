# National Park Agent

National Park Agent helps users plan visits to national parks using an AI assistant. It is designed to combine:

- Weather conditions
- Park alerts and closures
- Air quality (AQI)
- Hiking trails and trip difficulty
- Electric vehicle charging locations

The project currently provides a basic LangChain agent with mock weather and location tools. Real park, weather, AQI, trail, and EV data sources will be connected as the project grows.

## Local setup

```bash
uv sync
source .venv/bin/activate
```

Create a `.env` file for the model provider:

```env
LLM_PROVIDER=ollama
LLM_MODEL=qwen3:4b
LLM_BASE_URL=http://localhost:11434
```

For OpenRouter, install its provider package and change the configuration:

```bash
uv add langchain-openrouter
```

```env
LLM_PROVIDER=openrouter
LLM_MODEL=provider/model-name
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=your-openrouter-api-key
```

## Run the API

```bash
uv run uvicorn nationalparkagent.main:app --reload
```

The chat endpoint is:

```text
POST /v1/chat/completions
```

Example request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:4b",
    "messages": [{"role": "user", "content": "What is the weather in Yosemite?"}],
    "stream": false
  }'
```

## Debugging

Open the project in VS Code, select **Debug National Park Agent**, and press `F5`. The launch configuration uses the project `.venv` and loads `.env`.
