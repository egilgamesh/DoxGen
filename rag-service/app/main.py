import os
import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

VECTOR_URL = os.environ["VECTOR_SERVICE_URL"]
EMBED_URL = os.environ["EMBED_SERVICE_URL"]
LLM_URL = os.environ["LLM_SERVICE_URL"]

app = FastAPI(title="RAG Service")

class AskRequest(BaseModel):
    tenantId: str
    workspaceId: str
    question: str
    topK: int = 6

@app.post("/rag/ask")
async def ask(req: AskRequest):
    # 1) embed query
    async with httpx.AsyncClient(timeout=120) as client:
        emb_res = await client.post(f"{EMBED_URL}/embed", json={"chunks": [{"chunkId": "q", "text": req.question}]})
        emb_res.raise_for_status()
        qvec = emb_res.json()["vectors"][0]["vector"]

        # 2) vector search (filtered)
        search_res = await client.post(f"{VECTOR_URL}/vectors/search", json={
            "tenantId": req.tenantId,
            "workspaceId": req.workspaceId,
            "queryVector": qvec,
            "topK": req.topK
        })
        search_res.raise_for_status()
        matches = search_res.json()["matches"]

        # 3) Build grounded prompt with sources
        sources = []
        citations = []
        for i, m in enumerate(matches, start=1):
            sources.append(f"[{i}] (doc={m['document_id']} p={m['page']}) {m['content']}")
            citations.append({
                "rank": i,
                "documentId": m["document_id"],
                "page": m["page"],
                "chunkId": m["chunk_id"],
                "snippet": (m["content"][:180] + "…") if len(m["content"]) > 180 else m["content"]
            })

        system = (
            "You are DoxForge assistant. Answer ONLY using SOURCES.\n"
            "If the answer is not in the sources, say: 'I don’t know based on the provided documents.'\n"
            "Cite sources using [1], [2] next to each claim."
        )
        user = f"QUESTION:\n{req.question}\n\nSOURCES:\n" + "\n\n".join(sources)

        # 4) Ask LLM
        llm_res = await client.post(f"{LLM_URL}/llm/chat", json={"system": system, "user": user})
        llm_res.raise_for_status()

        return {"answer": llm_res.json()["content"], "citations": citations}
