"""Versioned prompts.

Why versioned: a prompt edit changes model behaviour exactly as much as a model
swap does. PROMPT_VERSION is returned in every response and written to the
feedback table, so a drop in thumbs-up rate can be attributed to the prompt that
produced it. Editing a prompt without bumping the version destroys that link.

Grounding is enforced in three layers, because prompt instructions alone are not
a control:
  1. Prompt: answer only from the numbered context, cite ids, refuse otherwise.
  2. Schema: the reply must parse as GroundedAnswer JSON.
  3. Code: cited ids must be a subset of the ids actually supplied; unknown ids
     are dropped and an answer left with zero valid citations is discarded
     (`app/router.py`). Layer 3 is the one that actually holds.
"""

from __future__ import annotations

PROMPT_VERSION = "grounded-v1.2"

SYSTEM_PROMPT = """You are NovaBank's customer support assistant. NovaBank is a UK app-only \
digital bank.

Rules you must follow:
1. Answer ONLY using the numbered context passages provided. They are NovaBank's \
official policy documents.
2. If the context does not contain the answer, set "insufficient_context" to true \
and say plainly that you do not have that information. Never guess a fee, limit, \
timeframe or process that is not written in the context.
3. Cite the id of every passage you used in the "cited" array. Cite only ids that \
appear in the context.
4. Never ask for or repeat a PIN, passcode, one-time code, full card number or CVV. \
If the user includes one, do not echo it back.
5. You cannot see the customer's account, balance or transactions, and you cannot \
take any action on the account. Tell the user which screen in the app to use instead.
6. Be concise and practical: lead with what the customer should do, then the \
relevant numbers or timeframes. Plain English, no marketing language.

Return ONLY a JSON object with this exact shape, and nothing else:
{"answer": "<your reply to the customer>", "cited": ["<id>", ...], \
"insufficient_context": false}"""

USER_TEMPLATE = """Context passages:
{context}

Detected intent: {intent} (confidence {confidence:.2f})

Customer question: {question}

Return the JSON object now."""

REPAIR_SUFFIX = (
    "\n\nYour previous reply was not valid JSON matching the required shape. "
    'Return ONLY the JSON object: {"answer": "...", "cited": ["..."], '
    '"insufficient_context": false}'
)

REFUSAL_NO_CONTEXT = (
    "I don't have anything in NovaBank's published policies that answers that, so I'd "
    "rather not guess. You can reach a person in the app under Help > Contact us, or "
    "call 0800 000 0000 if it's about a lost card or fraud."
)

REFUSAL_LLM_DOWN = (
    "I can't generate an answer right now because of a temporary technical problem. "
    "Please try again shortly, or contact support in the app under Help > Contact us. "
    "For anything urgent about a lost card or fraud, call 0800 000 0000 (24/7)."
)


def format_context(passages: list[tuple[str, str, str]]) -> str:
    """passages: (chunk_id, heading, text) -> numbered block for the prompt."""
    blocks = []
    for i, (chunk_id, heading, text) in enumerate(passages, start=1):
        blocks.append(f"[{i}] id: {chunk_id}\nsection: {heading}\n{text}")
    return "\n\n".join(blocks)
