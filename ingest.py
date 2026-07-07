"""Document ingestion pipeline for building vector and BM25 indexes."""

import pickle
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

from config import config


def load_pdf(pdf_path: str) -> List[Document]:
    """
    Load PDF documents.

    Args:
        pdf_path: Path to PDF file

    Returns:
        List of documents from PDF
    """
    print(f"Loading PDF from {pdf_path}...")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages")
    return docs


def chunk_documents(
    docs: List[Document], chunk_size: int, chunk_overlap: int
) -> List[Document]:
    """
    Split documents into chunks.

    Args:
        docs: Documents to chunk
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of document chunks
    """
    print(f"Chunking documents (size={chunk_size}, overlap={chunk_overlap})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")
    return chunks


def add_metadata(chunks: List[Document], pdf_path: str) -> None:
    """
    Add metadata to chunks.

    Args:
        chunks: Document chunks to annotate
        pdf_path: Source PDF path
    """
    print("Adding metadata...")
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx
        chunk.metadata["source"] = pdf_path
        if "page" not in chunk.metadata:
            chunk.metadata["page"] = -1


def save_bm25_chunks(chunks: List[Document], output_path: str) -> None:
    """
    Save chunks for BM25 retrieval.

    Args:
        chunks: Document chunks
        output_path: Path to save pickled chunks
    """
    print(f"Saving BM25 chunks to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(chunks, f)
    print("BM25 chunks saved")


def create_vector_db(
    chunks: List[Document], embedding_model: str, db_path: str
) -> None:
    """
    Create Chroma vector database.

    Args:
        chunks: Document chunks
        embedding_model: HuggingFace model name
        db_path: Path to save database
    """
    print(f"Creating vector embeddings with {embedding_model}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )

    print(f"Creating Chroma database at {db_path}...")
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path,
    )
    print("Vector database created")


def print_sample_metadata(chunks: List[Document], num_samples: int = 5) -> None:
    """
    Print sample metadata from chunks.

    Args:
        chunks: Document chunks
        num_samples: Number of samples to display
    """
    print("\nSample Metadata:")
    print("-" * 50)
    for chunk in chunks[:num_samples]:
        print(f"Chunk ID: {chunk.metadata['chunk_id']}")
        print(f"Page: {chunk.metadata['page']}")
        print(f"Source: {chunk.metadata['source']}")
        print("-" * 50)


def ingest_documents() -> None:
    """Run the complete document ingestion pipeline."""
    print("Starting document ingestion...")
    print("=" * 50)

    # Load PDF
    docs = load_pdf(config.ingest.pdf_path)

    # Chunk documents
    chunks = chunk_documents(
        docs,
        chunk_size=config.ingest.chunk_size,
        chunk_overlap=config.ingest.chunk_overlap,
    )

    # Add metadata
    add_metadata(chunks, config.ingest.pdf_path)

    # Save BM25 chunks
    save_bm25_chunks(chunks, config.bm25.bm25_path)

    # Create vector database
    create_vector_db(
        chunks,
        embedding_model=config.vector_db.embedding_model,
        db_path=config.vector_db.db_path,
    )

    # Print sample metadata
    print_sample_metadata(chunks)

    print("\n" + "=" * 50)
    print("Document ingestion complete!")


if __name__ == "__main__":
    ingest_documents()
