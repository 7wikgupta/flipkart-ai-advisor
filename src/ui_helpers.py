"""Display helpers for the Streamlit app, kept free of Streamlit so they can be unit tested.

The pipeline cites products as ``[product_id]``. The app numbers the sources in rank
order and shows ``[1]``, ``[2]`` ... in the answer, matching the numbered source cards.
"""

from __future__ import annotations

import re

from src.query_understanding import is_follow_up
from src.retrieval import RetrievedChunk

_CITATION = re.compile(r"\[([A-Za-z0-9_\-]+)\]")
_NON_WORD = re.compile(r"[^\w\s]")

# Short, user-facing explanations for each RagResult.reason.
REASON_LABELS = {
    "answered": "Grounded answer from catalogue records",
    "no_support": "The catalogue evidence does not support an answer",
    "no_evidence": "No matching products in the catalogue",
    "weak_evidence": "Nothing in the catalogue is close enough to this question",
    "no_budget_match": "No products with a known price match this budget",
}

EXAMPLE_PROMPTS = [
    "Suggest a power bank under 1500",
    "Compare the TP-LINK TL-WR841N and the iBall 150M router",
    "Good wireless headphones under 2000?",
    "Which portable bluetooth speaker should I buy?",
    "What can you help me with?",
]


def source_numbers(sources: list[RetrievedChunk]) -> dict[str, int]:
    """Map each product ID to its 1-based position in the source list."""
    numbers: dict[str, int] = {}
    for s in sources:
        numbers.setdefault(s.product_id, len(numbers) + 1)
    return numbers


def number_citations(answer: str, sources: list[RetrievedChunk]) -> str:
    """Replace ``[product_id]`` with ``[n]`` for retrieved products; drop citations to unknown IDs."""
    numbers = source_numbers(sources)

    def repl(m: re.Match) -> str:
        n = numbers.get(m.group(1))
        if n is not None:
            return f"[{n}]"
        # Not a retrieved product: an invented ID is removed, other bracketed text is kept.
        return "" if m.group(1).startswith("P-") else m.group(0)

    text = _CITATION.sub(repl, answer)
    return re.sub(r"(\[\d+\])(?:\s*\1)+", r"\1", text)  # collapse "[1] [1]" left by repeated citations


def _normalise(text: str) -> str:
    return " ".join(_NON_WORD.sub(" ", text.lower()).split())


def cache_key(question: str, history: list[dict]) -> tuple[str, ...]:
    """Key for reusing an answer. A new-topic question is keyed on its text alone. A follow-up
    depends on the conversation, so the last three earlier user questions are part of its key."""
    if not history or not is_follow_up(question):
        return (_normalise(question),)
    earlier = [_normalise(m["content"]) for m in history if m.get("role") == "user"][-3:]
    return (_normalise(question), *earlier)


def format_price(price: float | None) -> str:
    """Historical catalogue price. ``None`` means unknown and is never shown as zero."""
    if price is None:
        return "Price unknown"
    return f"INR {price:,.0f} (2015-16 price)"


def format_rating(rating: float | None) -> str:
    """Average catalogue rating out of 5. Most records have none."""
    if rating is None:
        return "No rating"
    return f"★ {rating:g} / 5"
