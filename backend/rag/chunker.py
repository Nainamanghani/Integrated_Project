import re
from typing import List


def chunk_text(text: str, max_chars: int = 1200, overlap_chars: int = 200) -> List[str]:
    if len(text) <= max_chars:
        return [text.strip()]

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    chunks: List[str] = []
    current = ""

    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = sentence
        elif len(sentence) > max_chars:
            for start in range(0, len(sentence), max_chars - overlap_chars):
                end = min(len(sentence), start + max_chars)
                chunks.append(sentence[start:end].strip())
            current = ""
        else:
            current = sentence

    if current:
        chunks.append(current)

    if overlap_chars > 0 and len(chunks) > 1:
        overlapped = []
        for i in range(len(chunks)):
            if i == 0:
                overlapped.append(chunks[i])
                continue
            prefix = chunks[i - 1][-overlap_chars:].strip()
            overlapped.append(f"{prefix} {chunks[i]}".strip())
        return overlapped

    return chunks
