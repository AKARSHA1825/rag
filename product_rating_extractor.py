"""
product_rating_extractor.py
------------------------------------------------------------------
Given a product name (e.g. "Private Motor - Comprehensive"), this
module:
  1. Pulls EVERY chunk of the underwriting guide that could relate
     to that product (not just the top-k from hybrid search -- see
     gather_product_context() below for why).
  2. Sends that full context to Claude with a strict, JSON-only
     extraction prompt so nothing is missed and nothing is invented.
  3. Returns a plain python dict ready for rating_excel_builder.py.

WHY A SEPARATE FILE:
Same reason as rating_excel_builder.py -- this repo has 2 contributors
pushing to GitHub, so new features live in new files. This module only
IMPORTS from retriver.py / rag.py / config.py, it never edits them.
------------------------------------------------------------------
"""

import re
from typing import Any, Dict, List

from config import config
from retriever import HybridRetriever


# ------------------------------------------------------------------
# STEP 1: Gather the full context for a product
# ------------------------------------------------------------------
def gather_product_context(retriever: HybridRetriever, product_name: str, debug: bool = False) -> str:
    """
    The existing HybridRetriever.retrieve() is tuned for short Q&A
    (top-k + a small neighbor window), which is great for chat but
    risks silently dropping some rating rules for a full product
    extraction. For this use case we need EVERYTHING about the
    product, so we combine two passes:

        Pass A - Hybrid search with a much larger k (so ranking-based
                  misses are minimized).
        Pass B - A raw keyword scan across every stored chunk
                  (self.documents) for the product name / its
                  synonyms, so even chunks that ranked poorly in
                  Pass A are still included.

    Both passes are unioned by chunk_id, neighbor-expanded (so partial
    tables/steps at chunk boundaries are captured whole), de-duplicated,
    and sorted back into original document order before being joined
    into one context string.
    """
    original_vector_k = retriever.vector_k
    original_bm25_k = retriever.bm25_k

    try:
        # --- Pass A: widen the search temporarily for this call only ---
        retriever.vector_k = max(original_vector_k, 40)
        retriever.bm25_k = max(original_bm25_k, 40)
        retriever.bm25.k = retriever.bm25_k
        ranked_docs = retriever.hybrid_search(product_name, debug=debug)
        pass_a_chunk_ids = {doc.metadata["chunk_id"] for doc, _score in ranked_docs}
    finally:
        # restore original config so normal chat Q&A is unaffected
        retriever.vector_k = original_vector_k
        retriever.bm25_k = original_bm25_k
        retriever.bm25.k = original_bm25_k

    # --- Pass B: brute-force keyword scan across ALL stored chunks ---
    # Splits "Private Motor - Comprehensive" into significant words so we
    # catch chunks that mention the product with slightly different phrasing.
    keywords = [w.lower() for w in re.split(r"[\s\-/,]+", product_name) if len(w) > 2]
    pass_b_chunk_ids = set()
    for doc in retriever.documents:
        text_lower = doc.page_content.lower()
        if any(kw in text_lower for kw in keywords):
            pass_b_chunk_ids.add(doc.metadata["chunk_id"])

    all_chunk_ids = pass_a_chunk_ids | pass_b_chunk_ids

    # Build fake (doc, score) tuples so we can reuse expand_neighbors(),
    # which pulls in the chunks immediately before/after each match --
    # this is what prevents a rating table from being cut in half.
    id_to_doc = {doc.metadata["chunk_id"]: doc for doc in retriever.documents}
    ranked_for_expansion = [(id_to_doc[cid], 1.0) for cid in all_chunk_ids if cid in id_to_doc]

    expanded = retriever.expand_neighbors(ranked_for_expansion, debug=debug)
    deduped = retriever.remove_duplicates(expanded)
    ordered = retriever.sort_chunks(deduped)

    if debug:
        print(f"[gather_product_context] Pass A chunks: {len(pass_a_chunk_ids)} | "
              f"Pass B chunks: {len(pass_b_chunk_ids)} | Final (with neighbors): {len(ordered)}")

    return retriever.build_context(ordered)


# ------------------------------------------------------------------
# STEP 2: Ask Claude to extract structured data using TOOL CALLING
# ------------------------------------------------------------------
# WHY THIS CHANGED FROM "ask the model to type JSON as text":
# Asking a model to hand-write raw JSON and then json.loads() it is
# fundamentally fragile -- it can add markdown fences despite being
# told not to, leave a trailing comma, forget to escape a quote inside
# a source phrase (e.g. a 6" wheel), etc. We hit two different flavors
# of that same root problem already (truncated strings, then a
# trailing-comma parse error). Patching the text parser for each new
# failure mode is whack-a-mole.
#
# The correct fix is to stop asking the model to WRITE JSON as text at
# all. Anthropic's tool-calling ("function calling") lets us give the
# model a JSON Schema and force it to call that tool -- the API parses
# and validates the arguments FOR us, so what we get back
# (tool_use_block.input) is already a plain Python dict. No markdown
# fences are possible, and malformed JSON simply cannot reach us,
# because the model isn't generating free-form text for this part at
# all; it's filling in structured tool arguments.
# ------------------------------------------------------------------

STEPS_TOOL = {
    "name": "record_rating_steps",
    "description": (
        "Records every rating step, premium component, condition and formula "
        "extracted for one insurance product from the underwriting guide."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "description": "Every rating step for this product, in source order. Do not omit any.",
                "items": {
                    "type": "object",
                    "properties": {
                        "step_title": {
                            "type": "string",
                            "description": "e.g. 'Step 1 - Base Premium Computation'",
                        },
                        "rows": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "component": {
                                        "type": "string",
                                        "description": (
                                            "Premium component name. Leave as an empty string if this "
                                            "row is a continuation of the same component as the row above "
                                            "(i.e. the source spec would merge these cells)."
                                        ),
                                    },
                                    "condition": {"type": "string"},
                                    "rate_reference": {
                                        "type": "string",
                                        "description": "Empty string if a continuation of the row above.",
                                    },
                                    "calc_logic": {"type": "string"},
                                    "output": {
                                        "type": "string",
                                        "description": "Empty string if a continuation of the row above.",
                                    },
                                },
                                "required": ["component", "condition", "rate_reference", "calc_logic", "output"],
                            },
                        },
                    },
                    "required": ["step_title", "rows"],
                },
            }
        },
        "required": ["steps"],
    },
}

TABLES_TOOL = {
    "name": "record_rating_tables",
    "description": (
        "Records every lookup / rate table relevant to one insurance product "
        "from the underwriting guide (base amount tables, loading tables, "
        "discount tables, limit tables, etc)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "tables": {
                "type": "array",
                "description": "Every relevant lookup table for this product. Include every row, do not truncate.",
                "items": {
                    "type": "object",
                    "properties": {
                        "table_name": {"type": "string"},
                        "columns": {"type": "array", "items": {"type": "string"}},
                        "rows": {
                            "type": "array",
                            "items": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                    "required": ["table_name", "columns", "rows"],
                },
            }
        },
        "required": ["tables"],
    },
}

STEPS_SYSTEM_PROMPT = """You are an expert insurance underwriter and data extraction specialist.
You ONLY work with insurance underwriting rating documents.

You will be given raw text from an underwriting rating guide and the name of
ONE product. Call the record_rating_steps tool with EVERY rating step,
premium component, condition and formula that belongs to that product ONLY.
Do not skip or summarize -- if the source lists 40 steps, output all 40, and
if a component has 8 conditions, output all 8 rows.

Do not hallucinate or infer values not present in the text. Preserve the
original wording of conditions and formulas as closely as possible. Group
rows under their correct step. Do NOT include lookup tables here -- only
reference them by name in rate_reference (tables are extracted separately)."""

TABLES_SYSTEM_PROMPT = """You are an expert insurance underwriter and data extraction specialist.
You ONLY work with insurance underwriting rating documents.

You will be given raw text from an underwriting rating guide and the name of
ONE product. Call the record_rating_tables tool with EVERY lookup / rate
table relevant to that product ONLY. Include every row of every table, do
not truncate or summarize. Do not hallucinate values -- preserve numbers and
labels exactly as shown in the source."""


def _call_claude_tool(
    client,
    system_prompt: str,
    user_content: str,
    tool_schema: Dict[str, Any],
    max_tokens: int = 8192,
    max_size_retries: int = 2,
) -> Dict[str, Any]:
    """
    Calls Claude with a single forced tool call and returns the already-
    parsed dict from tool_use_block.input -- no json.loads(), no markdown
    fence stripping, no trailing-comma issues, because the API itself
    guarantees the arguments match `tool_schema` before we ever see them.

    If the response is cut off (stop_reason == "max_tokens") we do NOT try
    to stitch a partial tool call back together -- that's not a safe or
    well-defined operation for structured tool arguments the way it is for
    plain text. Instead we simply retry the SAME request with a larger
    token budget (doubling each time, up to max_size_retries attempts).
    This is simple and safe because each attempt is a clean, complete
    generation rather than a fragile stitched-together one.
    """
    tokens = max_tokens
    tool_name = tool_schema["name"]

    for attempt in range(max_size_retries + 1):
        response = client.messages.create(
            model=config.claude.model_name,
            max_tokens=tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_content}],
            tools=[tool_schema],
            tool_choice={"type": "tool", "name": tool_name},
        )

        if response.stop_reason == "max_tokens":
            tokens = min(tokens * 2, 32000)
            continue  # retry from scratch with a bigger budget

        tool_block = next(
            (block for block in response.content if getattr(block, "type", None) == "tool_use"),
            None,
        )
        if tool_block is None:
            raise RuntimeError(
                f"Expected the model to call '{tool_name}' but it didn't "
                f"(stop_reason={response.stop_reason}). This usually means the "
                f"endpoint doesn't support forced tool_choice -- check the "
                f"Azure/Foundry deployment's tool-calling support."
            )
        return tool_block.input

    raise RuntimeError(
        f"'{tool_name}' response was still truncated (max_tokens) after "
        f"{max_size_retries} retries, up to {tokens} tokens. This product may "
        f"genuinely be too large for one extraction pass -- consider narrowing "
        f"the product name, or splitting extraction by step-range."
    )


def extract_structured_rating(client, product_name: str, context: str, max_tokens: int = 8192) -> Dict[str, Any]:
    """
    Runs the two extraction calls (steps, then tables) and merges them into
    the {"steps": [...], "tables": [...]} dict that rating_excel_builder.py
    expects. `client` is the already-initialized Anthropic client from
    rag.py (reused, not recreated).
    """
    context_block = f"""Product to extract: {product_name}

------------------------------------------------------------
Underwriting guide context
------------------------------------------------------------
{context}
------------------------------------------------------------"""

    steps_prompt = context_block + "\n\nExtract this product's rating steps by calling record_rating_steps."
    tables_prompt = context_block + "\n\nExtract this product's lookup tables by calling record_rating_tables."

    steps_result = _call_claude_tool(client, STEPS_SYSTEM_PROMPT, steps_prompt, STEPS_TOOL, max_tokens=max_tokens)
    tables_result = _call_claude_tool(client, TABLES_SYSTEM_PROMPT, tables_prompt, TABLES_TOOL, max_tokens=max_tokens)

    return {
        "steps": steps_result.get("steps", []),
        "tables": tables_result.get("tables", []),
    }


def get_product_rating_data(client, retriever: HybridRetriever, product_name: str, debug: bool = False) -> Dict[str, Any]:
    """Convenience wrapper: gather context + extract structured data (steps + tables)."""
    context = gather_product_context(retriever, product_name, debug=debug)
    return extract_structured_rating(client, product_name, context)