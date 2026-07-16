"""RAG Viewer — las entrañas de un RAG, visualizadas.

Funciona igual en local (streamlit run app.py) y en el navegador vía stlite.
Todo lo costoso (embeddings, cross-encoder, PCA) viene precomputado en
data/data.json; aquí solo hay numpy y plotly.
"""

import html
import json
import math

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from chunking import CONFIGS, apply_config

st.set_page_config(page_title="RAG Viewer", page_icon="🔬", layout="wide")

st.markdown("""<style>
.block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1200px;}
h1, h2, h3 {letter-spacing: -0.015em;}
.rv-sub {color: #6b7280; margin: 0.15rem 0 0.7rem; font-size: 1.02rem;}
.chip {display: inline-block; padding: 2px 11px; border-radius: 999px;
       background: rgba(99,110,250,0.10); color: #4a54c4;
       font-size: 0.76rem; font-family: monospace; margin: 0 6px 12px 0;}
[data-testid="stMetric"] {background: rgba(128,128,128,0.06);
       border-radius: 10px; padding: 10px 14px;}
</style>""", unsafe_allow_html=True)

DOC_COLORS = {"rag": "#636efa", "espacio": "#ef553b", "cafe": "#00cc96"}
CHUNK_BG = ["rgba(99,110,250,0.18)", "rgba(0,204,150,0.18)"]
OVERLAP_BG = "rgba(239,85,59,0.35)"


@st.cache_data
def load_data():
    with open("data/data.json", encoding="utf-8") as f:
        return json.load(f)


data = load_data()
DOCS = data["docs"]


def cosine(q, M):
    return (M @ q) / (np.linalg.norm(M, axis=1) * np.linalg.norm(q))


def euclidean(q, M):
    return np.linalg.norm(M - q, axis=1)


def dot(q, M):
    return M @ q


def spearman(a, b):
    """Correlación de Spearman entre dos criterios de ordenación (1 = rankings idénticos)."""
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def chunk_arrays(config_id):
    chunks = data["chunks"][config_id]
    embs = np.array([c["emb"] for c in chunks])
    return chunks, embs


def chunk_text(c):
    return DOCS[c["doc"]]["text"][c["start"]:c["end"]]


def score_bar(value, vmin, vmax, color):
    pct = 0 if vmax == vmin else (value - vmin) / (vmax - vmin) * 100
    return (
        f'<div style="background:rgba(128,128,128,0.15);border-radius:4px;height:10px;width:100%">'
        f'<div style="background:{color};border-radius:4px;height:10px;width:{max(pct, 2):.0f}%"></div></div>'
    )


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🔬 RAG Viewer")
st.sidebar.caption("Las entrañas de un RAG, visualizadas paso a paso.")

config_id = st.sidebar.selectbox(
    "Estrategia de chunking",
    list(CONFIGS),
    index=5,
    format_func=lambda cid: data["configs"][cid],
)
query_idx = st.sidebar.selectbox(
    "Consulta (query)",
    range(len(data["queries"])),
    format_func=lambda i: data["queries"][i]["text"],
)
metric_name = st.sidebar.radio(
    "Métrica de similitud", ["Coseno", "Euclídea", "Producto escalar"]
)
top_k = st.sidebar.slider("Top-k a recuperar", 3, 10, 5)
normalize = st.sidebar.toggle(
    "Normalizar embeddings (L2)",
    value=False,
    help="Reescala cada vector a longitud 1 antes de calcular las métricas. "
    "Con vectores unitarios, coseno y euclídea producen el mismo ranking: "
    "la explicación, en la pestaña 📐.",
)

st.sidebar.divider()
with st.sidebar.expander("⚙️ Detalles técnicos"):
    st.markdown(
        f"**Embeddings**: `{data['emb_model']}` ({data['dims']} dims)\n\n"
        f"**Reranker**: `{data['ce_model']}`\n\n"
        "Los modelos se ejecutaron una sola vez en la fase de precomputación; "
        "esta página es 100 % estática y el álgebra corre en tu navegador."
    )

query = data["queries"][query_idx]
q_raw = np.array(query["emb"])
chunks, embs_raw = chunk_arrays(config_id)

if normalize:
    embs = embs_raw / np.linalg.norm(embs_raw, axis=1, keepdims=True)
    q_emb = q_raw / np.linalg.norm(q_raw)
else:
    embs, q_emb = embs_raw, q_raw

cos_scores = cosine(q_emb, embs)
euc_dists = euclidean(q_emb, embs)
dot_scores = dot(q_emb, embs)
ce_scores = np.array(query["ce"][config_id])

if metric_name == "Coseno":
    order = np.argsort(-cos_scores)
    metric_vals = cos_scores
elif metric_name == "Euclídea":
    order = np.argsort(euc_dists)
    metric_vals = euc_dists
else:
    order = np.argsort(-dot_scores)
    metric_vals = dot_scores

# ---------------------------------------------------------------- header
st.markdown(
    f'<h1 style="margin-bottom:0">🔬 RAG Viewer</h1>'
    f'<p class="rv-sub">Chunking · embeddings · similitud · reranking — '
    f'el pipeline de un RAG, abierto en canal.</p>'
    f'<span class="chip">embeddings · {data["emb_model"].split("/")[-1]} · {data["dims"]}d</span>'
    f'<span class="chip">reranker · {data["ce_model"].split("/")[-1]}</span>'
    f'<span class="chip">chunks · {len(chunks)}</span>'
    f'<span class="chip">100 % estático</span>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------- tabs
tab_chunk, tab_map, tab_sim, tab_ret, tab_info = st.tabs(
    ["🧩 Chunking", "🗺️ Mapa de embeddings", "📐 Métricas de similitud",
     "🔎 Retrieval + Reranking", "ℹ️ El pipeline"]
)

# ================================================================ chunking
with tab_chunk:
    st.subheader("Cómo se trocea un documento")
    st.markdown(
        "El **chunking** parte cada documento en fragmentos que luego se convierten "
        "en vectores. Cambia la estrategia en la barra lateral y observa dónde caen "
        "las fronteras. Las zonas rojas son **solape**: texto repetido en dos chunks "
        "consecutivos para no perder la información que cae justo en una frontera."
    )
    doc_id = st.selectbox("Documento", list(DOCS), format_func=lambda d: DOCS[d]["title"])
    text = DOCS[doc_id]["text"]
    doc_chunks = apply_config(text, config_id)

    lens = [len(c["text"]) for c in doc_chunks]
    c1, c2, c3 = st.columns(3)
    c1.metric("Chunks", len(doc_chunks))
    c2.metric("Longitud media", f"{int(np.mean(lens))} chars")
    c3.metric("Solape", "sí" if any(
        doc_chunks[i + 1]["start"] < doc_chunks[i]["end"] for i in range(len(doc_chunks) - 1)
    ) else "no")

    # Pintamos el texto con un fondo alterno por chunk y rojo en los solapes
    events = []
    for i, c in enumerate(doc_chunks):
        events.append((c["start"], c["end"], i))
    parts = []
    pos = 0
    boundaries = sorted({b for s, e, _ in events for b in (s, e)} | {0, len(text)})
    for a, b in zip(boundaries, boundaries[1:]):
        owners = [i for s, e, i in events if s <= a and b <= e]
        piece = html.escape(text[a:b]).replace("\n", "<br>")
        if not owners:
            parts.append(piece)
        elif len(owners) > 1:
            parts.append(f'<span style="background:{OVERLAP_BG}">{piece}</span>')
        else:
            bg = CHUNK_BG[owners[0] % 2]
            parts.append(f'<span style="background:{bg}">{piece}</span>')
        pos = b
    st.markdown(
        f'<div style="line-height:1.7;font-size:0.95rem">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )

# ================================================================ mapa
with tab_map:
    st.subheader("El espacio vectorial, aplastado a 2D")
    st.markdown(
        f"Cada punto es un chunk convertido en un vector de **{data['dims']} dimensiones** "
        "y proyectado a 2D con PCA. Los chunks del mismo tema forman **grupos**: el modelo "
        "de embeddings no sabe de qué documento vienen, solo de qué *hablan*. "
        "La ⭐ es la consulta seleccionada: el retrieval consiste en encontrar sus vecinos."
    )
    fig = go.Figure()
    for doc_id, meta in DOCS.items():
        pts = [(c["xy"][0], c["xy"][1], chunk_text(c)) for c in chunks if c["doc"] == doc_id]
        fig.add_trace(go.Scatter(
            x=[p[0] for p in pts], y=[p[1] for p in pts],
            mode="markers", name=meta["title"],
            marker=dict(size=10, color=DOC_COLORS[doc_id], opacity=0.75),
            hovertext=[p[2][:160] + "…" for p in pts], hoverinfo="text",
        ))
    top_idx = set(order[:top_k].tolist())
    fig.add_trace(go.Scatter(
        x=[chunks[i]["xy"][0] for i in top_idx], y=[chunks[i]["xy"][1] for i in top_idx],
        mode="markers", name=f"Top-{top_k} recuperados",
        marker=dict(size=16, symbol="circle-open", color="#ffa15a", line=dict(width=3)),
        hoverinfo="skip",
    ))
    qx, qy = query["xy"][config_id]
    fig.add_trace(go.Scatter(
        x=[qx], y=[qy], mode="markers+text", name="Consulta",
        marker=dict(size=18, symbol="star", color="#ffd700", line=dict(width=1, color="#333")),
        text=["  consulta"], textposition="middle right",
    ))
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="PCA 1", yaxis_title="PCA 2")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Ojo: PCA descarta información al pasar de "
        f"{data['dims']} a 2 dimensiones — dos puntos que aquí se ven juntos pueden no ser "
        "los más cercanos en el espacio original. Las distancias reales se calculan siempre "
        "en el espacio completo."
    )

# ================================================================ métricas
with tab_sim:
    st.subheader("Tres formas de medir «cerca»")
    best = int(order[0])
    pick = st.selectbox(
        "Compara la consulta con un chunk",
        range(len(chunks)),
        index=best,
        format_func=lambda i: f"[{DOCS[chunks[i]['doc']]['title']}] {chunk_text(chunks[i])[:90]}…",
    )
    a, b = q_emb, embs[pick]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    cos_v = float(a @ b / (na * nb))
    euc_v = float(np.linalg.norm(a - b))
    dot_v = float(a @ b)
    theta = math.acos(max(-1, min(1, cos_v)))

    col_fig, col_vals = st.columns([3, 2])
    with col_fig:
        # Diagrama exacto: dos vectores con sus magnitudes reales y el ángulo real
        ax, ay = na, 0.0
        bx, by = nb * math.cos(theta), nb * math.sin(theta)
        fig = go.Figure()
        for (x, y, name, color) in [(ax, ay, "consulta", "#ffd700"), (bx, by, "chunk", "#636efa")]:
            fig.add_trace(go.Scatter(x=[0, x], y=[0, y], mode="lines+markers", name=name,
                                     line=dict(width=4, color=color),
                                     marker=dict(size=[0, 12])))
        fig.add_trace(go.Scatter(x=[ax, bx], y=[ay, by], mode="lines", name="dist. euclídea",
                                 line=dict(width=2, color="#ef553b", dash="dot")))
        arc_t = np.linspace(0, theta, 30)
        r = min(na, nb) * 0.25
        fig.add_trace(go.Scatter(x=r * np.cos(arc_t), y=r * np.sin(arc_t), mode="lines",
                                 showlegend=False, line=dict(color="#999")))
        fig.add_annotation(x=r * 1.25 * math.cos(theta / 2), y=r * 1.25 * math.sin(theta / 2),
                           text=f"θ = {math.degrees(theta):.1f}°", showarrow=False)
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis=dict(scaleanchor="x", scaleratio=1))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Geometría exacta: el plano que contiene a los dos vectores. "
            "El ángulo θ, las longitudes y la distancia entre puntas son los valores reales "
            f"en las {data['dims']} dimensiones."
        )
    with col_vals:
        st.metric("Similitud coseno", f"{cos_v:.4f}", help="cos(θ). 1 = misma dirección.")
        st.latex(r"\cos(\theta)=\frac{\vec{a}\cdot\vec{b}}{\|\vec{a}\|\,\|\vec{b}\|}")
        st.metric("Distancia euclídea", f"{euc_v:.4f}", help="Longitud del segmento rojo. 0 = idénticos.")
        st.latex(r"d(\vec{a},\vec{b})=\|\vec{a}-\vec{b}\|")
        st.metric("Producto escalar", f"{dot_v:.4f}", help="Sensible al ángulo Y a la magnitud.")
        st.latex(r"\vec{a}\cdot\vec{b}=\|\vec{a}\|\,\|\vec{b}\|\cos(\theta)")

    st.divider()
    st.markdown("**¿Dan el mismo ranking?** Top-5 de cada métrica para la consulta actual:")
    cols = st.columns(3)
    for col, (name, vals, asc) in zip(cols, [
        ("Coseno ↓", cos_scores, False), ("Euclídea ↑", euc_dists, True), ("Dot ↓", dot_scores, False),
    ]):
        idx = np.argsort(vals if asc else -vals)[:5]
        with col:
            st.markdown(f"**{name}**")
            for rank, i in enumerate(idx, 1):
                c = chunks[i]
                col.markdown(
                    f"<small>{rank}. <b style='color:{DOC_COLORS[c['doc']]}'>"
                    f"[{DOCS[c['doc']]['title']}]</b> {html.escape(chunk_text(c)[:70])}… "
                    f"<code>{vals[i]:.3f}</code></small>",
                    unsafe_allow_html=True,
                )
    if normalize:
        st.caption(
            "Embeddings **normalizados**: coseno y euclídea dan exactamente el mismo ranking, "
            "y el producto escalar coincide con el coseno."
        )
    else:
        st.caption(
            "Sin normalizar: la magnitud de cada vector «contamina» a la euclídea y al producto "
            "escalar, y sus rankings divergen del coseno. Activa **Normalizar embeddings (L2)** "
            "en la barra lateral y vuelve a mirar."
        )

    # ---------------- experimento: normalización -----------------
    st.divider()
    st.markdown("### 🧪 Por qué normalizar pone a coseno y euclídea de acuerdo")

    cos_all = cosine(q_raw, embs_raw)
    euc_raw_all = euclidean(q_raw, embs_raw)
    embs_n = embs_raw / np.linalg.norm(embs_raw, axis=1, keepdims=True)
    q_n = q_raw / np.linalg.norm(q_raw)
    euc_norm_all = euclidean(q_n, embs_n)

    rho_raw = spearman(-cos_all, euc_raw_all)
    rho_norm = spearman(-cos_all, euc_norm_all)
    top_cos = set(np.argsort(-cos_all)[:top_k].tolist())
    ov_raw = len(top_cos & set(np.argsort(euc_raw_all)[:top_k].tolist()))
    ov_norm = len(top_cos & set(np.argsort(euc_norm_all)[:top_k].tolist()))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ρ Spearman · crudos", f"{rho_raw:.3f}",
              help="Acuerdo entre el ranking por coseno y por euclídea sobre los vectores tal cual. 1.000 = rankings idénticos.")
    m2.metric("ρ Spearman · normalizados", f"{rho_norm:.3f}",
              help="El mismo acuerdo tras reescalar todos los vectores a longitud 1.")
    m3.metric(f"Top-{top_k} compartido · crudos", f"{ov_raw}/{top_k}",
              help="Cuántos chunks aparecen a la vez en el top-k del coseno y en el de la euclídea.")
    m4.metric(f"Top-{top_k} compartido · normalizados", f"{ov_norm}/{top_k}")

    col_exp, col_math = st.columns([3, 2])
    with col_exp:
        xs = np.linspace(float(cos_all.min()), float(cos_all.max()), 100)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=cos_all, y=euc_raw_all, mode="markers", name="vectores crudos",
            marker=dict(color="#636efa", size=8, opacity=0.6),
        ))
        fig2.add_trace(go.Scatter(
            x=cos_all, y=euc_norm_all, mode="markers", name="normalizados (L2)",
            marker=dict(color="#00cc96", size=8, opacity=0.8),
        ))
        fig2.add_trace(go.Scatter(
            x=xs, y=np.sqrt(np.maximum(0, 2 - 2 * xs)), mode="lines",
            name="teoría: d = √(2 − 2·cos θ)",
            line=dict(color="#ef553b", dash="dash", width=2),
        ))
        fig2.update_layout(
            height=440, margin=dict(l=10, r=10, t=30, b=10),
            xaxis_title="similitud coseno (no cambia al normalizar)",
            yaxis_title="distancia euclídea",
            legend=dict(orientation="h", yanchor="bottom", y=1.0),
        )
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(
            "Cada punto es un chunk comparado con la consulta actual. Crudos, forman una nube: "
            "a igual ángulo hay distancias distintas. Normalizados, **todos caen sobre la curva "
            "teórica**: la distancia queda completamente determinada por el ángulo."
        )
    with col_math:
        st.markdown("**La derivación, en dos pasos.** Desarrollando el cuadrado de la distancia:")
        st.latex(r"\|\vec a-\vec b\|^2=\underbrace{\|\vec a\|^2+\|\vec b\|^2}_{\text{magnitudes}}-2\,\vec a\cdot\vec b")
        st.markdown(
            "Con vectores unitarios ($\\|\\vec a\\|=\\|\\vec b\\|=1$) el término de magnitudes "
            "se congela en $2$ y el producto escalar pasa a ser exactamente $\\cos\\theta$:"
        )
        st.latex(r"d=\sqrt{2-2\cos\theta}")
        st.markdown(
            "Esa función es **estrictamente decreciente** en $\\cos\\theta$: más coseno implica "
            "siempre menos distancia. Ordenar por distancia mínima y por coseno máximo se vuelve "
            "la *misma* ordenación — de ahí el ρ = 1.000 exacto.\n\n"
            "Sin normalizar, $\\|\\vec b\\|$ varía de chunk a chunk (suele crecer con la longitud "
            "del texto y la frecuencia de sus tokens), así que la euclídea mezcla *significado* "
            "(ángulo) con *magnitud* (ruido) y los rankings se separan.\n\n"
            "Por eso muchos modelos de embeddings devuelven vectores ya normalizados y la mayoría "
            "de bases de datos vectoriales normalizan por defecto: coseno, euclídea y dot se "
            "vuelven intercambiables y permiten usar el índice más rápido disponible."
        )

# ================================================================ retrieval
with tab_ret:
    st.subheader("Recuperar barato, reordenar caro")
    st.markdown(
        f"**Etapa 1 — búsqueda vectorial** ({metric_name.lower()}): compara la consulta con los "
        f"{len(chunks)} chunks a la vez. Rápida, pero comprime cada chunk en un solo vector. "
        "**Etapa 2 — reranking**: un cross-encoder lee consulta y chunk *juntos* y puntúa "
        "su relevancia. Observa qué chunks suben ⬆ y cuáles caen ⬇ al aplicarlo."
    )
    st.info(f"**Consulta:** {query['text']}")
    if normalize:
        st.caption(
            "Normalización L2 activada: con vectores unitarios, coseno y euclídea "
            "devuelven exactamente el mismo top-k."
        )

    retrieved = order[:top_k]
    ce_order = retrieved[np.argsort(-ce_scores[retrieved])]

    col_v, col_r = st.columns(2)
    with col_v:
        st.markdown(f"#### 1️⃣ Ranking vectorial ({metric_name.lower()})")
        vmin, vmax = float(metric_vals[retrieved].min()), float(metric_vals[retrieved].max())
        if metric_name == "Euclídea":
            vmin, vmax = vmax, vmin  # menor distancia = barra más larga
        for rank, i in enumerate(retrieved, 1):
            c = chunks[i]
            st.markdown(
                f"**{rank}.** <b style='color:{DOC_COLORS[c['doc']]}'>[{DOCS[c['doc']]['title']}]</b> "
                f"<code>{metric_vals[i]:.3f}</code><br>"
                f"{score_bar(metric_vals[i], vmin, vmax, '#636efa')}"
                f"<small>{html.escape(chunk_text(c)[:180])}…</small>",
                unsafe_allow_html=True,
            )
    with col_r:
        st.markdown("#### 2️⃣ Tras el reranking (cross-encoder)")
        sub = ce_scores[retrieved]
        vmin, vmax = float(sub.min()), float(sub.max())
        old_pos = {int(i): r for r, i in enumerate(retrieved, 1)}
        for rank, i in enumerate(ce_order, 1):
            c = chunks[int(i)]
            delta = old_pos[int(i)] - rank
            arrow = f"⬆ +{delta}" if delta > 0 else (f"⬇ {delta}" if delta < 0 else "＝")
            st.markdown(
                f"**{rank}.** <code>{arrow}</code> "
                f"<b style='color:{DOC_COLORS[c['doc']]}'>[{DOCS[c['doc']]['title']}]</b> "
                f"<code>{ce_scores[int(i)]:.3f}</code><br>"
                f"{score_bar(float(ce_scores[int(i)]), vmin, vmax, '#00cc96')}"
                f"<small>{html.escape(chunk_text(c)[:180])}…</small>",
                unsafe_allow_html=True,
            )
    st.caption(
        "En producción, la etapa 1 filtraría millones de chunks a unas decenas y el "
        "cross-encoder solo vería esos candidatos: por eso el embudo barato→caro escala."
    )

# ================================================================ pipeline
with tab_info:
    st.subheader("El pipeline completo")
    st.markdown(f"""
```
 INGESTA (offline)                        CONSULTA (online)
 ─────────────────                        ─────────────────
 📄 Documentos                            ❓ Pregunta del usuario
      │  chunking                              │  mismo modelo de embeddings
      ▼                                        ▼
 🧩 Chunks ({data['dims']}d c/u)               ⭐ Vector de la consulta
      │  modelo de embeddings                  │  búsqueda por similitud
      ▼                                        ▼
 🗄️ Base de datos vectorial  ────────────▶ 🔎 Top-k candidatos
                                               │  cross-encoder (reranking)
                                               ▼
                                          🏆 Mejores 3-5 chunks
                                               │  se insertan en el prompt
                                               ▼
                                          🤖 LLM genera la respuesta
```

**Lo que estás viendo en esta demo:**

| Pestaña | Parte del pipeline | Se calcula… |
|---|---|---|
| 🧩 Chunking | ingesta | en vivo, en tu navegador |
| 🗺️ Mapa | espacio vectorial | embeddings precomputados, PCA precomputado |
| 📐 Métricas | búsqueda por similitud | en vivo con numpy sobre vectores precomputados |
| 🔎 Retrieval | top-k + reranking | similitud en vivo; scores del cross-encoder precomputados |

**Por qué es 100 % estática:** los modelos (`{data['emb_model']}` para embeddings y
`{data['ce_model']}` como reranker) se ejecutaron una sola vez en la fase de
precomputación. El navegador solo hace álgebra con numpy sobre los resultados,
así que la demo puede vivir en GitHub Pages sin ningún servidor.

**Lo que NO puede hacer una demo estática:** aceptar consultas libres — haría falta
ejecutar el modelo de embeddings en un servidor (o en el navegador, que es posible
pero pesado). Por eso las consultas son un menú cerrado.
""")
