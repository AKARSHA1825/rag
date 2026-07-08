"""
intent_classifier.py
"""

import json
import re
from typing import Optional, TypedDict

from config import config


class IntentResult(TypedDict):
    is_export_request: bool
    product_name: Optional[str]


_CLASSIFIER_SYSTEM_PROMPT = """You classify one user message from an insurance
underwriting chat assistant.

The assistant can do two things:
1. Answer a normal question about the underwriting guide (Q&A).
2. Export a full rating specification for ONE named product to an Excel file
   -- this is only what the user wants if they are clearly asking to
   generate/build/export/download/create/pull/get a rating sheet, rating
   section, rating specification, or Excel/spreadsheet for a specific
   product. Phrasing varies a lot (typos, casual language, any word order) --
   judge the INTENT, not exact wording.

If it is an export request, also extract the product name exactly as the
user referred to it (e.g. "private motor", "Private Motor - Comprehensive").
If no specific product is named, product_name must be null -- do not guess.

If it is a normal question (asking what a rate is, how something works,
explaining a rule, etc.) -- even if it mentions "rating" -- this is NOT
an export request.

Respond with STRICT JSON ONLY, no markdown fences, no commentary:
{"is_export_request": true|false, "product_name": "string or null"}
"""


def _parse_json_response(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(json)?", "", cleaned).strip()
    cleaned = re.sub(r"```$", "", cleaned).strip()
    return json.loads(cleaned)


def classify_intent(client, query: str) -> IntentResult:
    """
    Classifies a single user query. `client` is the already-initialized
    Anthropic client from rag.py (reused, not recreated).

    Fails safe: if the model call or JSON parsing fails for any reason,
    we treat it as a normal question (is_export_request=False) rather
    than crashing the chat loop or silently swallowing an export the
    user actually asked for -- worst case they just get a normal answer
    and can re-ask more explicitly.
    """
    try:
        response = client.messages.create(
            model=config.claude.model_name,
            max_tokens=200,
            system=_CLASSIFIER_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": query}],
        )
        raw_text = response.content[0].text
        parsed = _parse_json_response(raw_text)
        return IntentResult(
            is_export_request=bool(parsed.get("is_export_request", False)),
            product_name=parsed.get("product_name") or None,
        )
    except Exception:
        # Fail safe -> treat as a normal question rather than blowing up
        # the chat loop or guessing.
        return IntentResult(is_export_request=False, product_name=None)