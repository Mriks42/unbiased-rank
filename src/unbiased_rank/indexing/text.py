"""Tokenization for lexical retrieval.

Deliberately simple: lowercase, split on non-alphanumerics, drop very short
tokens. No stemming and no stopword list.

Both omissions are choices, not oversights. Stemming would help recall
slightly but introduces a dependency whose behaviour varies by version, and
the experiment measures *position bias*, not lexical matching quality — a
stronger BM25 would shift every arm equally and change none of the
conclusions. Keeping the tokenizer trivial makes it fully reproducible.
"""

from __future__ import annotations

import re
from typing import Final

_TOKEN_PATTERN: Final[re.Pattern[str]] = re.compile(r"[a-z0-9]+")
MIN_TOKEN_LENGTH: Final[int] = 2


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric tokens."""
    if not text:
        return []
    return [t for t in _TOKEN_PATTERN.findall(text.lower()) if len(t) >= MIN_TOKEN_LENGTH]


def product_text(title: str, brand: str | None = None) -> str:
    """Compose the indexed text for a product.

    Title plus brand only. Descriptions and bullet points are excluded: they
    are long, sparsely populated, and inflate the index by an order of
    magnitude for a marginal retrieval gain that is irrelevant to what this
    experiment measures.
    """
    parts = [title or ""]
    if brand:
        parts.append(brand)
    return " ".join(parts)


__all__ = ["MIN_TOKEN_LENGTH", "product_text", "tokenize"]
