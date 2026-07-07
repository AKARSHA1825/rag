"""Hybrid retriever combining BM25 and vector search."""

import pickle
from collections import defaultdict
from typing import List, Tuple

from langchain_community.vectorstores import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document


class HybridRetriever:
    """
    Hybrid retriever combining BM25 keyword search with vector similarity search
    using Reciprocal Rank Fusion (RRF) for ranking.
    """

    def __init__(
        self,
        db_path: str = "./chroma_db",
        bm25_path: str = "./bm25_chunks.pkl",
        embedding_model: str = "BAAI/bge-large-en-v1.5",
        vector_k: int = 20,
        bm25_k: int = 20,
        rrf_k: int = 60,
        neighbor_window: int = 2,
    ) -> None:
        """
        Initialize the hybrid retriever.

        Args:
            db_path: Path to Chroma vector database
            bm25_path: Path to BM25 documents pickle file
            embedding_model: HuggingFace embedding model name
            vector_k: Number of vector search results
            bm25_k: Number of BM25 search results
            rrf_k: RRF constant for ranking
            neighbor_window: Window size for neighbor expansion
        """
        self.vector_k = vector_k
        self.bm25_k = bm25_k
        self.rrf_k = rrf_k
        self.neighbor_window = neighbor_window

        print("Loading embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )

        print("Loading Chroma vector database...")
        self.db = Chroma(
            persist_directory=db_path,
            embedding_function=self.embeddings,
        )

        print("Loading BM25 documents...")
        with open(bm25_path, "rb") as f:
            self.documents = pickle.load(f)

        self.bm25 = BM25Retriever.from_documents(self.documents)
        self.bm25.k = self.bm25_k

        print("Hybrid retriever ready.\n")

    def bm25_search(self, query: str) -> List[Document]:
        """Search using BM25 keyword matching."""
        return self.bm25.invoke(query)

    def vector_search(self, query: str) -> List[Document]:
        """Search using vector similarity."""
        return self.db.similarity_search(query, k=self.vector_k)

    def reciprocal_rank_fusion(
        self, bm25_docs: List[Document], vector_docs: List[Document]
    ) -> List[Tuple[Document, float]]:
        """
        Combine BM25 and vector search results using Reciprocal Rank Fusion.

        Args:
            bm25_docs: Documents from BM25 search
            vector_docs: Documents from vector search

        Returns:
            List of (document, score) tuples ranked by RRF score
        """
        scores = defaultdict(float)
        unique_docs = {}

        # Score BM25 results
        for rank, doc in enumerate(bm25_docs, start=1):
            chunk_id = doc.metadata["chunk_id"]
            unique_docs[chunk_id] = doc
            scores[chunk_id] += 1 / (self.rrf_k + rank)

        # Score vector results
        for rank, doc in enumerate(vector_docs, start=1):
            chunk_id = doc.metadata["chunk_id"]
            unique_docs[chunk_id] = doc
            scores[chunk_id] += 1 / (self.rrf_k + rank)

        # Sort by combined score
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        return [
            (unique_docs[chunk_id], score) for chunk_id, score in ranked
        ]

    def hybrid_search(
        self, query: str, debug: bool = False
    ) -> List[Tuple[Document, float]]:
        """
        Perform hybrid search combining BM25 and vector search.

        Args:
            query: Search query
            debug: Print debug information

        Returns:
            Ranked documents with RRF scores
        """
        bm25_docs = self.bm25_search(query)
        vector_docs = self.vector_search(query)
        ranked_docs = self.reciprocal_rank_fusion(bm25_docs, vector_docs)

        if debug:
            self._print_rrf_results(ranked_docs)

        return ranked_docs

    def expand_neighbors(
        self, ranked_docs: List[Tuple[Document, float]], debug: bool = False
    ) -> List[Document]:
        """
        Expand retrieval results by including neighboring chunks.

        Uses dynamic window sizes - larger windows for top-ranked results.

        Args:
            ranked_docs: Ranked documents from hybrid search
            debug: Print debug information

        Returns:
            List of documents including neighbors, without scores
        """
        expanded_docs = []
        visited = set()

        for rank, (doc, _score) in enumerate(ranked_docs):
            chunk_id = doc.metadata["chunk_id"]

            # Dynamic neighbor window based on rank
            if rank < 3:
                window = 3
            elif rank < 8:
                window = 2
            else:
                window = self.neighbor_window

            start = max(0, chunk_id - window)
            end = min(len(self.documents), chunk_id + window + 1)

            for idx in range(start, end):
                if idx not in visited:
                    visited.add(idx)
                    expanded_docs.append(self.documents[idx])

        if debug:
            self._print_expansion_results(expanded_docs)

        return expanded_docs

    def remove_duplicates(self, docs: List[Document]) -> List[Document]:
        """Remove duplicate chunks by chunk_id."""
        unique_docs = []
        seen = set()

        for doc in docs:
            chunk_id = doc.metadata["chunk_id"]
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique_docs.append(doc)

        return unique_docs

    def sort_chunks(self, docs: List[Document]) -> List[Document]:
        """Sort documents by chunk_id."""
        return sorted(docs, key=lambda d: d.metadata["chunk_id"])

    def retrieve(
        self, query: str, debug: bool = False
    ) -> List[Document]:
        """
        Retrieve documents for a query using full pipeline.

        Pipeline:
            1. Hybrid search (BM25 + vector + RRF)
            2. Neighbor expansion
            3. Duplicate removal
            4. Sorting by chunk_id

        Args:
            query: Search query
            debug: Print debug information

        Returns:
            Processed list of documents
        """
        ranked_docs = self.hybrid_search(query, debug)
        docs = self.expand_neighbors(ranked_docs, debug)
        docs = self.remove_duplicates(docs)
        docs = self.sort_chunks(docs)

        if debug:
            self._print_final_results(docs)

        return docs

    def build_context(self, docs: List[Document]) -> str:
        """Combine document contents into a single context string."""
        return "\n\n".join(doc.page_content for doc in docs)

    def _print_rrf_results(self, ranked_docs: List[Tuple[Document, float]]) -> None:
        """Print RRF ranking results."""
        print("\n" + "=" * 80)
        print("RRF RESULTS")
        print("=" * 80)
        for rank, (doc, score) in enumerate(ranked_docs, start=1):
            print(
                f"{rank:02d} | Chunk {doc.metadata['chunk_id']:03d} | "
                f"Score {score:.6f}"
            )
        print("=" * 80)

    def _print_expansion_results(self, docs: List[Document]) -> None:
        """Print neighbor expansion results."""
        print("\n" + "=" * 80)
        print("AFTER NEIGHBOR EXPANSION")
        print("=" * 80)
        for doc in docs:
            print(
                f"Chunk {doc.metadata['chunk_id']:03d} | "
                f"Page {doc.metadata['page']}"
            )
        print("=" * 80)

    def _print_final_results(self, docs: List[Document]) -> None:
        """Print final retrieval results."""
        print("\n" + "=" * 80)
        print("FINAL CHUNKS")
        print("=" * 80)
        for doc in docs:
            print(
                f"Chunk {doc.metadata['chunk_id']:03d} | "
                f"Page {doc.metadata['page']}"
            )
        print("=" * 80)
        pages = sorted(set(doc.metadata["page"] for doc in docs))
        print(f"Total chunks: {len(docs)} | Pages: {pages}\n")