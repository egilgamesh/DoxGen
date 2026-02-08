from pydantic import BaseModel
from typing import List, Optional

class Page(BaseModel):
    page: int
    text: str

class Chunk(BaseModel):
    chunkId: str
    tenantId: str
    workspaceId: str
    documentId: str
    page: int
    text: str

class VectorItem(BaseModel):
    chunkId: str
    tenantId: str
    workspaceId: str
    documentId: str
    page: int
    text: str
    vector: List[float]

class Citation(BaseModel):
    documentId: str
    page: int
    chunkId: str
    snippet: str
