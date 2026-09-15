# National Park Agent

National Park Agent helps users plan visits to national parks using an AI assistant. It is designed to combine:

- Weather conditions
- Park alerts and closures
- Air quality (AQI)
- Hiking trails and trip difficulty
- Electric vehicle charging locations

The project provides a LangGraph-based agent with live weather, air-quality,
park-alert, location, and document-retrieval tools. Weather is connected to the
hosted Eris MCP server, and documents are stored in a local vector database.

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

For live National Park Service alerts, request a free API key from the
[NPS Developer Resources](https://www.nps.gov/subjects/developer/get-started.htm)
and add it to `.env`:

```env
NPS_API_KEY=your-nps-api-key
```

The agent's `check_park_alerts` tool uses this key to find the park and retrieve
its current alerts. Keep the key private and do not commit `.env`.

## LangSmith observability

LangSmith automatically traces LangChain and LangGraph runs when tracing is
enabled. Add these variables to `.env`:

```env
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=national-park-agent
```

Restart the API after changing `.env`, then open the project in
[LangSmith](https://smith.langchain.com/). You can inspect complete agent
traces, tool calls, model calls, latency, errors, and token usage when the model
provider reports token data. The default endpoint is the LangSmith cloud API;
set `LANGSMITH_ENDPOINT` only for another LangSmith deployment.

## RAG and vector database

RAG (retrieval-augmented generation) lets the agent answer questions using
your documents. The flow is:

1. Read a PDF or text document.
2. Split it into smaller pieces.
3. Convert each piece into an embedding (a list of numbers representing its meaning).
4. Store the embeddings in a vector database.
5. Search for relevant pieces when a user asks a question and provide them to the agent.

This project uses Chroma as a simple local vector database. It is already
included in the project dependencies:

```bash
uv sync
```

Chroma stores its data in the local `chroma_db/` directory. It does not need a
separate database server for learning. A persistent client keeps the data
between program runs:

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("documents")

collection.add(
    ids=["1"],
    documents=["Yosemite National Park has beautiful hiking trails."],
)

results = collection.query(
    query_texts=["Where are the hiking trails?"],
    n_results=1,
)
print(results)
```

Chroma automatically downloads its default local embedding model the first
time this example runs. The `POST /v1/ingest` endpoint extracts, chunks,
embeds, and stores uploaded documents in Chroma. The `POST /v1/retrieve`
endpoint searches those stored chunks and returns the five closest matches.

Retrieve relevant chunks with:

```bash
curl --location 'http://127.0.0.1:8000/v1/retrieve' \
  --header 'Content-Type: application/json' \
  --data '{"query":"Where is Yosemite Valley parking?"}'
```

The response includes each matching chunk, its filename/page metadata, and its
Chroma distance score.

## Run the API

```bash
uv run uvicorn nationalparkagent.main:app --reload --env-file .env
```

The chat endpoint is:

```text
POST /v1/chat/completions
```

The document ingestion endpoint is:

```text
POST /v1/ingest
```

The document retrieval endpoint is:

```text
POST /v1/retrieve
```

Upload a PDF or plain text file using the multipart field named `file`:

```bash
curl --location 'http://127.0.0.1:8000/v1/ingest' \
  --header 'accept: application/json' \
  --form 'file=@/Users/ashokmarannan/Downloads/Yosemite-Guide-1.pdf;type=application/pdf'
```

In Postman:

1. Create a `POST` request to `http://127.0.0.1:8000/v1/ingest`.
2. Open **Body** → **form-data**.
3. Add a field named `file`, change its type from **Text** to **File**, and choose the PDF or `.txt` file.
4. Remove any manually added `Content-Type: application/json` header. Postman will set the multipart content type.
5. Select **Send**. A valid upload returns `200` with `{"status":"indexed"}`.

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
