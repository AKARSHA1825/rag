"""RAG pipeline using Claude for answering questions."""

from anthropic import AnthropicFoundry

from config import config
from retriever import HybridRetriever


def _initialize_client() -> AnthropicFoundry:
    """Initialize Anthropic Foundry client."""
    return AnthropicFoundry(
        api_key=config.claude.api_key,
        base_url=config.claude.endpoint,
    )


def _initialize_retriever() -> HybridRetriever:
    """Initialize hybrid retriever."""
    return HybridRetriever(
        db_path=config.vector_db.db_path,
        bm25_path=config.bm25.bm25_path,
        embedding_model=config.vector_db.embedding_model,
        vector_k=config.retriever.vector_k,
        bm25_k=config.retriever.bm25_k,
        rrf_k=config.retriever.rrf_k,
        neighbor_window=config.retriever.neighbor_window,
    )


# Initialize clients
client = _initialize_client()
retriever = _initialize_retriever()


def invoke_claude(prompt: str) -> str:
    """
    Invoke Claude model with a prompt.

    Args:
        prompt: Input prompt

    Returns:
        Model response text
    """
    response = client.messages.create(
        model=config.claude.model_name,
        max_tokens=config.claude.max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def ask(query: str, debug: bool = False) -> str:
    """
    Ask a question using RAG pipeline.

    Pipeline:
        1. Retrieve relevant documents using hybrid search
        2. Build context from retrieved documents
        3. Invoke Claude with context and query

    Args:
        query: User question
        debug: Print debug information

    Returns:
        Claude's answer
    """
    # Retrieve documents
    docs = retriever.retrieve(query, debug=debug)

    # Build context
    context = retriever.build_context(docs)

    # Create prompt with context
    prompt = f"""You are an expert insurance underwriter.

Answer ONLY using the supplied context.

Do not hallucinate.

Do not infer.

If the answer is not present in the context, respond exactly with:

I don't know

Preserve wording from the underwriting guide whenever possible.

------------------------------------------------------------

Context

{context}

------------------------------------------------------------

Question

{query}

------------------------------------------------------------

Answer
"""

    # Get answer from Claude
    answer = invoke_claude(prompt)

    return answer


if __name__ == "__main__":
    print("RAG Pipeline Ready")
    print("-" * 50)

    while True:
        query = input("\nQuestion: ")

        if query.lower() in ["exit", "quit"]:
            break

        try:
            answer = ask(query, debug=True)
            print("\n" + "=" * 80)
            print("ANSWER")
            print("=" * 80)
            print(answer)
        except Exception as e:
            print(f"Error: {e}")
