from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from sentence_transformers import SentenceTransformer

app = FastAPI(title="Embedding Service")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")  # dim=384

class EmbedChunk(BaseModel):
    chunkId: str
    text: str

class EmbedRequest(BaseModel):
    chunks: List[EmbedChunk]

@app.post("/embed")
def embed(req: EmbedRequest):
    texts = [c.text for c in req.chunks]
    embs = model.encode(texts, normalize_embeddings=True).tolist()
    vectors = [{"chunkId": req.chunks[i].chunkId, "vector": embs[i]} for i in range(len(req.chunks))]
    return {"dim": len(embs[0]) if embs else 0, "vectors": vectors}
