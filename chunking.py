"""Estrategias de chunking compartidas entre la precomputación y la app.

Cada función devuelve una lista de dicts: {"text", "start", "end"} con los
offsets sobre el texto original, para poder pintar las fronteras en la UI.
"""

import re

# Configuraciones disponibles en la app. Las claves son estables: los
# embeddings precomputados se indexan por (doc, config_id).
CONFIGS = {
    "fixed_200_0": {"label": "Tamaño fijo · 200 chars · sin solape", "fn": "fixed", "size": 200, "overlap": 0},
    "fixed_400_0": {"label": "Tamaño fijo · 400 chars · sin solape", "fn": "fixed", "size": 400, "overlap": 0},
    "fixed_400_100": {"label": "Tamaño fijo · 400 chars · solape 100", "fn": "fixed", "size": 400, "overlap": 100},
    "fixed_800_200": {"label": "Tamaño fijo · 800 chars · solape 200", "fn": "fixed", "size": 800, "overlap": 200},
    "sent_3_0": {"label": "Por frases · 3 frases · sin solape", "fn": "sentences", "n": 3, "overlap": 0},
    "sent_3_1": {"label": "Por frases · 3 frases · solape 1", "fn": "sentences", "n": 3, "overlap": 1},
    "sent_6_1": {"label": "Por frases · 6 frases · solape 1", "fn": "sentences", "n": 6, "overlap": 1},
    "paragraph": {"label": "Por párrafos", "fn": "paragraphs"},
}

_SENT_RE = re.compile(r"(?<=[.!?…])\s+")


def chunk_fixed(text, size, overlap):
    chunks = []
    step = max(1, size - overlap)
    pos = 0
    while pos < len(text):
        end = min(pos + size, len(text))
        piece = text[pos:end]
        if piece.strip():
            chunks.append({"text": piece, "start": pos, "end": end})
        if end == len(text):
            break
        pos += step
    return chunks


def _sentences_with_offsets(text):
    """Divide en frases conservando offsets. Los encabezados markdown
    (líneas que empiezan por #) cuentan como frase propia."""
    sents = []
    for para_match in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", text):
        block = para_match.group(0)
        base = para_match.start()
        if block.lstrip().startswith("#"):
            sents.append({"text": block, "start": base, "end": base + len(block)})
            continue
        pos = 0
        for m in _SENT_RE.finditer(block):
            piece = block[pos:m.start()]
            if piece.strip():
                sents.append({"text": piece, "start": base + pos, "end": base + m.start()})
            pos = m.end()
        piece = block[pos:]
        if piece.strip():
            sents.append({"text": piece, "start": base + pos, "end": base + len(block)})
    return sents


def chunk_sentences(text, n, overlap):
    sents = _sentences_with_offsets(text)
    chunks = []
    step = max(1, n - overlap)
    for i in range(0, len(sents), step):
        group = sents[i:i + n]
        if not group:
            break
        start, end = group[0]["start"], group[-1]["end"]
        chunks.append({"text": text[start:end], "start": start, "end": end})
        if i + n >= len(sents):
            break
    return chunks


def chunk_paragraphs(text):
    chunks = []
    for m in re.finditer(r"[^\n]+(?:\n(?!\n)[^\n]+)*", text):
        if m.group(0).strip():
            chunks.append({"text": m.group(0), "start": m.start(), "end": m.end()})
    return chunks


def apply_config(text, config_id):
    cfg = CONFIGS[config_id]
    if cfg["fn"] == "fixed":
        return chunk_fixed(text, cfg["size"], cfg["overlap"])
    if cfg["fn"] == "sentences":
        return chunk_sentences(text, cfg["n"], cfg["overlap"])
    return chunk_paragraphs(text)
