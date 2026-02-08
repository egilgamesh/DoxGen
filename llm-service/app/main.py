import os
import httpx
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="LLM Service")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

class ChatRequest(BaseModel):
    system: str
    user: str

@app.post("/llm/chat")
async def chat(req: ChatRequest):
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": req.system},
            {"role": "user", "content": req.user}
        ],
        "stream": False
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{OLLAMA_URL}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        return {"content": data["message"]["content"], "model": MODEL}
