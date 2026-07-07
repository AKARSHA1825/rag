"""Configuration management for RAG application."""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


load_dotenv()


@dataclass
class VectorDBConfig:
    """Vector database configuration."""

    db_path: str = "./chroma_db"
    embedding_model: str = "BAAI/bge-large-en-v1.5"


@dataclass
class BM25Config:
    """BM25 retriever configuration."""

    bm25_path: str = "./bm25_chunks.pkl"


@dataclass
class RetrieverConfig:
    """Hybrid retriever configuration."""

    vector_k: int = 20
    bm25_k: int = 20
    rrf_k: int = 60
    neighbor_window: int = 2


@dataclass
class IngestConfig:
    """Document ingestion configuration."""

    pdf_path: str = "data/Manomay Rates.pdf"
    chunk_size: int = 12000
    chunk_overlap: int = 3000


@dataclass
class ClaudeConfig:
    """Claude API configuration."""

    endpoint: str = "https://manomaysonnet-resource.services.ai.azure.com/anthropic"
    model_name: str = "claude-sonnet-4-6-1"
    api_key: Optional[str] = None
    max_tokens: int = 4096

    def __post_init__(self) -> None:
        """Initialize API key from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.getenv("API_KEY")
        if not self.api_key:
            raise ValueError("API_KEY environment variable not set")


@dataclass
class RAGConfig:
    """Main RAG configuration."""

    vector_db: VectorDBConfig = None
    bm25: BM25Config = None
    retriever: RetrieverConfig = None
    ingest: IngestConfig = None
    claude: ClaudeConfig = None

    def __post_init__(self) -> None:
        """Initialize default configs."""
        if self.vector_db is None:
            self.vector_db = VectorDBConfig()
        if self.bm25 is None:
            self.bm25 = BM25Config()
        if self.retriever is None:
            self.retriever = RetrieverConfig()
        if self.ingest is None:
            self.ingest = IngestConfig()
        if self.claude is None:
            self.claude = ClaudeConfig()


# Global config instance
config = RAGConfig()
