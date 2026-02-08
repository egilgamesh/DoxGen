from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from app.db import exec_sql, query

app = FastAPI(title="Vector Service")

# Adjust after first embed call; default is MiniLM dim=384
EMBED_DIM = 384

@app.on_event("startup")
def startup():
    exec_sql("CREATE EXTENSION IF NOT EXISTS vector;")
    exec_sql("""
    CREATE TABLE IF NOT EXISTS doc_vectors (
      chunk_id TEXT PRIMARY KEY,
      tenant_id TEXT NOT NULL,
      workspace_id TEXT NOT NULL,
      document_id TEXT NOT NULL,
      page INT NOT NULL,
      content TEXT NOT NULL,
      embedding vector(384) NOT NULL
    );
    """)
    exec_sql("CREATE INDEX IF NOT EXISTS idx_scope ON doc_vectors(tenant_id, workspace_id, document_id);")
    exec_sql("""
    CREATE INDEX IF NOT EXISTS idx_vec_ivfflat
    ON doc_vectors USING ivfflat (embedding vector_cosine_ops) WITH (lists=100);
    """)

class UpsertItem(BaseModel):
    chunkId: str
    tenantId: str
    workspaceId: str
    documentId: str
    page: int
    text: str
    vector: List[float]

class UpsertRequest(BaseModel):
    items: List[UpsertItem]

@app.post("/vectors/upsert")
def upsert(req: UpsertRequest):
    for it in req.items:
        exec_sql("""
        INSERT INTO doc_vectors(chunk_id, tenant_id, workspace_id, document_id, page, content, embedding)
        VALUES (:chunk_id, :tenant_id, :workspace_id, :document_id, :page, :content, :emb::vector)
        ON CONFLICT (chunk_id) DO UPDATE
        SET content = EXCLUDED.content, embedding = EXCLUDED.embedding;
        """, {
            "chunk_id": it.chunkId,
            "tenant_id": it.tenantId,
            "workspace_id": it.workspaceId,
            "document_id": it.documentId,
            "page": it.page,
            "content": it.text,
            "emb": it.vector
        })
    return {"upserted": len(req.items)}

class SearchRequest(BaseModel):
    tenantId: str
    workspaceId: str
    queryVector: List[float]
    topK: int = 6

@app.post("/vectors/search")
def search(req: SearchRequest):
    rows = query("""
    SELECT chunk_id, document_id, page, content
    FROM doc_vectors
    WHERE tenant_id = :tenant_id AND workspace_id = :workspace_id
    ORDER BY embedding <=> (:qvec::vector)
    LIMIT :top_k
    """, {"tenant_id": req.tenantId, "workspace_id": req.workspaceId, "qvec": req.queryVector, "top_k": req.topK})
    return {"matches": list(rows)}
