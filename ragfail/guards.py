"""Guards on the generation side: strip instructions hidden in retrieved text, cap input size."""

from __future__ import annotations

import re

MAX_QUESTION_CHARS = 2000
# Sentences that address an AI system or try to change its instructions.
INSTRUCTION = re.compile(
    r"\b(ignore|disregard|forget)\b[^.]{0,60}\b(instructions?|question|documents?|rules)\b"
    r"|\b(ai|automated)\s+(assistants?|systems?)\b|\bsystem\s+override\b|\breply\s+only\s+with\b"
    r"|\banswer\s+every\s+question\b", re.I)


def sentences(text: str) -> list[str]:
    """Split at whitespace that follows . ! or ? (optionally with a closing quote). Text inside double
    quotes is never split, and "0.4" or "site.example" are not sentence ends because no space follows."""
    out, start, inq = [], 0, False
    for i, ch in enumerate(text):
        if ch == '"':
            inq = not inq
        elif ch.isspace() and not inq and i > 0:
            prev = text[i - 1]
            if prev in ".!?" or (prev == '"' and i > 1 and text[i - 2] in ".!?"):
                if text[start:i].strip():
                    out.append(text[start:i].strip())
                start = i + 1
    if text[start:].strip():
        out.append(text[start:].strip())
    return out


def strip_instructions(text: str) -> tuple[str, int]:
    """Remove sentences that look like instructions to an AI. Returns (clean text, sentences removed)."""
    removed = 0
    paras = []
    for para in text.split("\n\n"):
        if not INSTRUCTION.search(para):
            paras.append(para)  # untouched paragraphs keep their exact text
            continue
        keep = []
        for s in sentences(para):
            if INSTRUCTION.search(s):
                removed += 1
            else:
                keep.append(s)
        if keep:
            paras.append(" ".join(keep))
    return "\n\n".join(paras), removed


def check_question(q: str) -> str:
    q = q.strip()
    if not q:
        raise ValueError("empty question")
    if len(q) > MAX_QUESTION_CHARS:
        raise ValueError(f"question longer than {MAX_QUESTION_CHARS} characters")
    return q
