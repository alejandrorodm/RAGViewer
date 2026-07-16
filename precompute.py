"""Precomputa todo lo que la app estática no puede calcular en el navegador:
embeddings (model2vec), scores de cross-encoder (reranking) y proyecciones
PCA 2D. El resultado se escribe en data/data.json.

Uso: .venv/bin/python precompute.py
"""

import json
from pathlib import Path

import numpy as np
from model2vec import StaticModel
from sentence_transformers import CrossEncoder
from sklearn.decomposition import PCA

from chunking import CONFIGS, apply_config

ROOT = Path(__file__).parent
EMB_MODEL = "minishlab/potion-multilingual-128M"
CE_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

DOCS = {
    "rag": {"title": "Sistemas RAG", "file": "corpus/rag.md"},
    "espacio": {"title": "Exploración espacial", "file": "corpus/espacio.md"},
    "cafe": {"title": "El café", "file": "corpus/cafe.md"},
}

QUERIES = [
    {"text": "¿Qué es el solapamiento en el chunking y para qué sirve?", "target": "rag"},
    {"text": "¿Cómo funciona un cross-encoder en el reranking?", "target": "rag"},
    {"text": "¿Qué diferencia hay entre similitud coseno y distancia euclídea?", "target": "rag"},
    {"text": "¿Quién fue el primer ser humano en viajar al espacio?", "target": "espacio"},
    {"text": "¿Qué telescopio sustituyó al Hubble y qué observa?", "target": "espacio"},
    {"text": "¿Cómo consiguió SpaceX abaratar los lanzamientos?", "target": "espacio"},
    {"text": "¿Qué diferencias hay entre café arábica y robusta?", "target": "cafe"},
    {"text": "¿Cuánto tiempo se macera el cold brew?", "target": "cafe"},
    {"text": "¿Qué reacción química desarrolla los aromas durante el tueste?", "target": "cafe"},
    {"text": "¿Qué ayuda a mantenerse despierto por la noche?", "target": "cafe"},
]


def r4(arr):
    return [round(float(x), 4) for x in arr]


def main():
    print(f"Cargando modelos: {EMB_MODEL} + {CE_MODEL}")
    emb_model = StaticModel.from_pretrained(EMB_MODEL)
    ce_model = CrossEncoder(CE_MODEL)

    docs_out = {}
    for doc_id, meta in DOCS.items():
        docs_out[doc_id] = {
            "title": meta["title"],
            "text": (ROOT / meta["file"]).read_text(encoding="utf-8"),
        }

    query_texts = [q["text"] for q in QUERIES]
    query_embs = emb_model.encode(query_texts)

    chunks_out = {}
    queries_out = [
        {"text": q["text"], "target": q["target"], "emb": r4(e), "xy": {}, "ce": {}}
        for q, e in zip(QUERIES, query_embs)
    ]

    for config_id in CONFIGS:
        chunk_list = []
        for doc_id, doc in docs_out.items():
            for ch in apply_config(doc["text"], config_id):
                chunk_list.append({"doc": doc_id, "start": ch["start"], "end": ch["end"], "text": ch["text"]})

        texts = [c["text"] for c in chunk_list]
        embs = emb_model.encode(texts)
        print(f"  {config_id}: {len(chunk_list)} chunks")

        # Proyección 2D compartida entre chunks y queries de esta config
        pca = PCA(n_components=2, random_state=0)
        chunk_xy = pca.fit_transform(embs)
        query_xy = pca.transform(query_embs)

        # Cross-encoder: cada query contra todos los chunks de la config
        pairs = [(q, t) for q in query_texts for t in texts]
        ce_scores = ce_model.predict(pairs, batch_size=64, show_progress_bar=False)
        ce_scores = np.array(ce_scores).reshape(len(query_texts), len(texts))

        chunks_out[config_id] = [
            {"doc": c["doc"], "start": c["start"], "end": c["end"], "emb": r4(e), "xy": r4(xy)}
            for c, e, xy in zip(chunk_list, embs, chunk_xy)
        ]
        for qi, q_out in enumerate(queries_out):
            q_out["xy"][config_id] = r4(query_xy[qi])
            q_out["ce"][config_id] = r4(ce_scores[qi])

    data = {
        "emb_model": EMB_MODEL,
        "ce_model": CE_MODEL,
        "dims": int(query_embs.shape[1]),
        "docs": docs_out,
        "configs": {cid: cfg["label"] for cid, cfg in CONFIGS.items()},
        "chunks": chunks_out,
        "queries": queries_out,
    }

    out = ROOT / "data" / "data.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"Escrito {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
