"""Grounding- and scope-aware prompt construction."""

from __future__ import annotations

import json

from app.rag.retriever import RetrievedChunk

SYSTEM_PROMPT_TEMPLATE = '''You are the Esports Intelligence assistant for BGMI competition rules.

Answer the user's question using only the supplied rulebook context. Do not use prior knowledge, assumptions, or outside information. Do not infer rules from unrelated passages. If the context does not contain enough information to answer, reply exactly:
"I don't know based on the provided rulebook."

The user-selected scope is: {scope_description}
Respect this scope. For "General Rules", answer only from chunks labeled general.
For "Competition-Specific Rules", answer only from chunks labeled event_specific.
For "Auto scope", the supplied context may contain both general and event-specific rules. Clearly distinguish them when relevant and do not treat one scope as overriding the other unless the context explicitly says so.

Every factual claim must be supported by one or more supplied context entries. Cite claims using only the supplied citation IDs in square brackets, such as [C1]. Do not invent citation IDs, page numbers, headings, or rule text. If you cannot support a claim with a supplied entry, omit that claim.

Return a JSON object with exactly these keys:
{{"answer": "...", "citation_ids": ["C1"]}}
The answer must contain citations for its factual claims. If the answer is the required "I don't know" response, return an empty citation_ids array.'''


def scope_description(scope: str) -> str:
    """Return the exact human-readable scope name for the prompt."""
    return {
        "auto": "Auto scope",
        "general": "General Rules",
        "event_specific": "Competition-Specific Rules",
    }[scope]


def build_messages(
    question: str,
    scope: str,
    chunks: list[RetrievedChunk],
) -> list[dict[str, str]]:
    """Build system/user messages with stable IDs and trusted source metadata."""
    system = SYSTEM_PROMPT_TEMPLATE.format(scope_description=scope_description(scope))
    entries: list[dict[str, object]] = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk.metadata
        entries.append(
            {
                "citation_id": f"C{index}",
                "page_number": metadata.get("page_number"),
                "heading": metadata.get("heading") or "Section heading unavailable",
                "doc_scope": metadata.get("doc_scope"),
                "content": chunk.content,
            }
        )
    user = (
        f"Selected scope: {scope_description(scope)} ({scope}).\n"
        f"Question: {question}\n"
        "Use only these retrieved rulebook passages:\n"
        f"{json.dumps(entries, ensure_ascii=False, sort_keys=True)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]