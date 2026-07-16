# 🔬 RAG Viewer

Visor interactivo de las **entrañas de un RAG**: chunking, embeddings, métricas de
similitud (coseno, euclídea, producto escalar) y reranking con cross-encoder — todo
visual y **100 % estático**, pensado para GitHub Pages.

La app es Streamlit, pero en producción corre **dentro del navegador** con
[stlite](https://github.com/whitphx/stlite) (Streamlit sobre WebAssembly/Pyodide).
No hay servidor: lo costoso (modelos) se ejecuta una sola vez en la fase de
precomputación y el navegador solo hace álgebra con numpy.

## Qué se puede explorar

| Pestaña | Qué enseña |
|---|---|
| 🧩 Chunking | 8 estrategias (tamaño fijo, por frases, por párrafos, con/sin solape) pintadas sobre el texto real, con los solapes resaltados. Se calcula en vivo. |
| 🗺️ Mapa de embeddings | Los chunks proyectados a 2D con PCA: se ve cómo se agrupan por tema y dónde cae la consulta. |
| 📐 Métricas de similitud | Diagrama geométrico exacto (ángulo θ real) comparando coseno, euclídea y dot, y si producen o no el mismo ranking. |
| 🔎 Retrieval + Reranking | Ranking vectorial vs. ranking tras el cross-encoder, con las subidas y bajadas de cada chunk. |

## Estructura

```
corpus/          3 documentos de ejemplo (RAG, espacio, café)
chunking.py      estrategias de chunking (compartido: precompute + app)
precompute.py    genera data/data.json (embeddings, scores CE, PCA)
data/data.json   datos precomputados (~0.7 MB) — se commitea
app.py           la app Streamlit
index.html       monta la app con stlite para servirla estática
```

- **Embeddings**: [`minishlab/potion-multilingual-128M`](https://huggingface.co/minishlab/potion-multilingual-128M) (model2vec, 256 dims)
- **Reranker**: [`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1) (multilingüe)

## Desarrollo local

```bash
uv venv .venv
uv pip install --python .venv/bin/python streamlit numpy plotly
.venv/bin/streamlit run app.py
```

Para probar la versión estática (la que verá GitHub Pages):

```bash
python3 -m http.server 8000   # y abrir http://localhost:8000
```

## Regenerar los datos precomputados

Solo hace falta si cambias el corpus, las queries o las configs de chunking:

```bash
uv pip install --python .venv/bin/python model2vec scikit-learn sentence-transformers
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python precompute.py
```

## Deploy en GitHub Pages

1. Sube el repo a GitHub.
2. En **Settings → Pages**, elige *Source: GitHub Actions*.
3. Haz push a `main`: el workflow `.github/workflows/deploy.yml` publica el sitio tal cual (no hay build).

## Limitaciones (por diseño)

Es una demo estática: las consultas son un menú cerrado, porque aceptar texto libre
exigiría ejecutar el modelo de embeddings en un servidor. A cambio, no cuesta nada
hostearla y no carga ningún backend.
