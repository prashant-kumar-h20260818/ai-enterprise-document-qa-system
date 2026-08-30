from __future__ import annotations

import os
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.chunking import split_documents
from src.config import RAGConfig
from src.embeddings import EmbeddingManager
from src.evaluation import evaluate_dataset
from src.loaders import SUPPORTED_EXTENSIONS, load_uploaded_files
from src.rag import RAGPipeline, create_llm
from src.retriever import RAGRetriever
from src.vector_store import VectorStore

load_dotenv()
CONFIG = RAGConfig()

st.set_page_config(page_title="Enterprise Document Q&A", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_manager(model_name: str) -> EmbeddingManager:
    return EmbeddingManager(model_name)


def secret_value(name: str, default: str = "") -> str:
    value = os.getenv(name, "")
    if value:
        return value
    try:
        value = str(st.secrets.get(name, ""))
        return value or default
    except Exception:
        return default


def init_state() -> None:
    defaults = {
        "session_id": uuid.uuid4().hex[:16],
        "messages": [],
        "retriever": None,
        "index_info": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def collection_name() -> str:
    return f"rag_{st.session_state.session_id}"


def get_pipeline(api_key: str, llm_model: str) -> RAGPipeline | None:
    if st.session_state.retriever is None:
        return None
    if not api_key:
        return RAGPipeline(st.session_state.retriever, None)
    llm = create_llm(api_key, llm_model, CONFIG.temperature, CONFIG.max_tokens)
    return RAGPipeline(st.session_state.retriever, llm)


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    with st.expander("Sources"):
        for src in sources:
            semantic = float(src.get("score", 0.0))
            lexical = float(src.get("lexical_score", 0.0))
            mode = str(src.get("retrieval_mode", "semantic"))
            st.markdown(
                f"**[{src['label']}] {src['source']}** — {src['locator']} — "
                f"{src.get('content_type', 'text')}"
            )
            st.caption(
                f"Retrieval: {mode} · semantic similarity {semantic:.3f} · "
                f"lexical similarity {lexical:.3f}"
            )
            st.caption(src["preview"])


init_state()
embedding_manager = get_embedding_manager(CONFIG.embedding_model)

st.title("📚 AI-Powered Enterprise Document Question Answering System")
st.caption(
    "Ask grounded questions across PDFs, Word files, presentations, spreadsheets, tables, "
    "images, charts, diagrams, scanned pages and other common enterprise documents."
)

with st.sidebar:
    st.header("RAG configuration")
    st.write(f"**Embedding:** `{CONFIG.embedding_model}`")
    st.write(f"**Dimensions:** `{embedding_manager.dimension}`")
    st.caption("Retrieval uses dense semantic search + lexical TF-IDF with reciprocal-rank fusion.")

    llm_model = st.text_input("Groq answer model", CONFIG.llm_model)
    vision_model = st.text_input("Groq vision model", secret_value("GROQ_VISION_MODEL", CONFIG.vision_model))
    chunk_size = st.number_input("Chunk size (characters)", 300, 3000, CONFIG.chunk_size, 100)
    chunk_overlap = st.number_input("Chunk overlap", 0, int(chunk_size) - 1, min(CONFIG.chunk_overlap, int(chunk_size) - 1), 50)
    top_k = st.slider("Top-K retrieval", 1, 12, CONFIG.top_k)
    threshold = st.slider("Minimum cosine similarity", -1.0, 1.0, CONFIG.score_threshold, 0.05)

    api_key = secret_value("GROQ_API_KEY")
    if api_key:
        st.success("Groq API key loaded from environment/secrets.")
    else:
        api_key = st.text_input("Groq API key (session only)", type="password")
        st.warning("For deployment, store GROQ_API_KEY in Streamlit secrets, not in code.")

    enable_vision = st.checkbox(
        "Multimodal extraction",
        value=bool(api_key),
        help="Uses the Groq vision model for scanned pages, images, charts and diagrams.",
    )
    max_vision_items = st.number_input(
        "Vision analyses per file (0 = unlimited)", 0, 500, CONFIG.max_vision_items_per_file, 5
    )

    st.divider()
    if st.button("Clear session index", use_container_width=True):
        try:
            VectorStore(collection_name(), CONFIG.persist_directory).reset()
        except Exception:
            pass
        st.session_state.messages = []
        st.session_state.retriever = None
        st.session_state.index_info = {}
        st.success("Index cleared.")

chat_tab, eval_tab, architecture_tab = st.tabs(["💬 Chat", "📏 Evaluation", "🧠 Architecture"])

with chat_tab:
    uploader_types = sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS)
    files = st.file_uploader(
        "Upload enterprise documents",
        type=uploader_types,
        accept_multiple_files=True,
        help="Supports PDF, DOCX, PPTX, XLSX/XLSM, CSV/TSV, TXT/Markdown, HTML, JSON/XML/YAML, EML and common image formats.",
    )

    if st.button("Process & index documents", type="primary", disabled=not files):
        try:
            with st.status("Building multimodal RAG index...", expanded=True) as status:
                st.write("1/4 Extracting text, tables and visual evidence")
                docs = load_uploaded_files(
                    files,
                    api_key=api_key,
                    vision_model=vision_model,
                    enable_vision=enable_vision,
                    max_vision_items=int(max_vision_items),
                )
                if not docs:
                    raise ValueError("No extractable content was found in the uploaded files.")

                st.write("2/4 Chunking documents while preserving metadata")
                chunks = split_documents(docs, int(chunk_size), int(chunk_overlap))

                st.write("3/4 Creating normalized MiniLM embeddings")
                vectors = embedding_manager.encode([c.page_content for c in chunks])

                st.write("4/4 Upserting into ChromaDB and building hybrid retriever")
                store = VectorStore(collection_name(), CONFIG.persist_directory, reset_collection=True)
                store.add_documents(chunks, vectors)
                st.session_state.retriever = RAGRetriever(store, embedding_manager)

                content_types = {}
                for doc in docs:
                    kind = str(doc.metadata.get("content_type", "text"))
                    content_types[kind] = content_types.get(kind, 0) + 1

                st.session_state.index_info = {
                    "files": len(files),
                    "units": len(docs),
                    "chunks": len(chunks),
                    "sources": store.list_sources(),
                    "content_types": content_types,
                }
                st.session_state.messages = []
                status.update(label="Index ready", state="complete")
        except Exception as exc:
            st.error(f"Indexing failed: {exc}")

    if st.session_state.index_info:
        info = st.session_state.index_info
        c1, c2, c3 = st.columns(3)
        c1.metric("Files", info["files"])
        c2.metric("Extracted units", info["units"])
        c3.metric("Chunks", info["chunks"])
        with st.expander("Indexed sources"):
            st.write("**Content types:**", info["content_types"])
            for source in info["sources"]:
                st.write("•", source)

    # Always render completed history first. After each new answer we st.rerun(),
    # so the chat input below remains the final element in the conversation.
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if message.get("max_retrieval_similarity") is not None and message.get("sources"):
                    caption = (
                        f"Max retrieval similarity: {float(message['max_retrieval_similarity']):.3f} "
                        "(retrieval signal, not answer confidence)"
                    )
                    if message.get("model_used"):
                        caption += f" · model: `{message['model_used']}`"
                    st.caption(caption)
                render_sources(message.get("sources", []))

    question = st.chat_input(
        "Ask a question about the indexed documents",
        disabled=not st.session_state.index_info,
    )

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        if not api_key:
            output = {
                "answer": "Add GROQ_API_KEY to generate an answer. Retrieval evaluation can still be run without generation.",
                "sources": [],
                "max_retrieval_similarity": 0.0,
                "model_used": None,
            }
        else:
            try:
                pipeline = get_pipeline(api_key, llm_model)
                with st.spinner("Retrieving evidence and generating a grounded answer..."):
                    output = pipeline.answer(question, top_k=top_k, score_threshold=threshold)
            except Exception as exc:
                output = {
                    "answer": f"Request failed: {exc}",
                    "sources": [],
                    "max_retrieval_similarity": 0.0,
                    "model_used": None,
                }

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": output["answer"],
                "sources": output.get("sources", []),
                "max_retrieval_similarity": output.get("max_retrieval_similarity"),
                "model_used": output.get("model_used"),
            }
        )
        # Critical UX fix: rerender the full conversation before drawing the next
        # chat input, keeping the input below every completed Q&A turn.
        st.rerun()

with eval_tab:
    st.subheader("Measure RAG accuracy correctly")
    st.markdown(
        """
Do not use one vague accuracy number. Evaluate the pipeline in layers:

1. **Extraction quality** — was text/table/chart/diagram/scanned evidence captured correctly?
2. **Hit@K / Recall@K** — did the gold source/location occur among retrieved chunks?
3. **MRR** — how highly was the first correct source ranked?
4. **Answer correctness** — did the generated answer match the gold answer?
5. **Faithfulness** — are generated claims supported by retrieved evidence?
6. **No-answer accuracy** — does the system refuse when the documents do not contain the answer?
7. **Operational metrics** — latency, cost/query, failures and user task-success rate.

Use separate benchmark slices for text, tables, charts, diagrams, scans and unanswerable questions.
"""
    )

    template = Path("data/evaluation_template.csv")
    if template.exists():
        st.download_button("Download evaluation template", template.read_bytes(), "evaluation_template.csv", "text/csv")

    eval_file = st.file_uploader("Upload labeled evaluation CSV", type=["csv"], key="evaluation_file")
    run_generation = st.checkbox("Run end-to-end answer generation", value=True)
    use_judge = st.checkbox("Run optional LLM judge for correctness/faithfulness", value=False, disabled=not run_generation)

    if st.button("Run evaluation", disabled=eval_file is None or not st.session_state.index_info):
        if run_generation and not api_key:
            st.error("GROQ_API_KEY is required for answer-generation evaluation.")
        else:
            try:
                frame = pd.read_csv(eval_file)
                pipeline = get_pipeline(api_key if run_generation else "", llm_model)
                with st.spinner("Running benchmark..."):
                    results, summary = evaluate_dataset(
                        frame,
                        pipeline,
                        embedding_manager,
                        top_k=top_k,
                        score_threshold=threshold,
                        run_generation=run_generation,
                        use_llm_judge=use_judge,
                    )
                cols = st.columns(len(summary))
                for col, (name, value) in zip(cols, summary.items()):
                    col.metric(name.replace("_", " ").title(), f"{value:.3f}")
                st.dataframe(results, use_container_width=True)
                st.download_button(
                    "Download results",
                    results.to_csv(index=False).encode(),
                    "rag_evaluation_results.csv",
                    "text/csv",
                )
            except Exception as exc:
                st.error(f"Evaluation failed: {exc}")

with architecture_tab:
    st.subheader("End-to-end architecture")
    st.code(
        """Enterprise documents (PDF/DOCX/PPTX/XLSX/CSV/images/...)
        ↓
Format-aware parsing + optional Groq Vision
        ↓
Text + tables + chart/diagram/image/scanned-page evidence
        ↓
LangChain Documents + source/location metadata
        ↓
RecursiveCharacterTextSplitter (default 1000 / 200)
        ↓
all-MiniLM-L6-v2 → 384-d normalized embeddings
        ↓
Persistent ChromaDB → HNSW cosine dense index
        +
TF-IDF lexical index for exact words/acronyms/identifiers
        ↓
Question → dense + lexical retrieval → reciprocal-rank fusion → Top-K
        ↓
Source-labeled retrieved context [S1], [S2], ...
        ↓
Groq answer model with automatic model-access fallback (temperature 0.1)
        ↓
Grounded Markdown/LaTeX answer + source citations
        ↓
Streamlit UI + layered RAG evaluation workflow""",
        language="text",
    )
    st.info(
        "For production: add SSO/tenant authorization, encrypted object storage, malware scanning, "
        "background ingestion, audit logs, deletion policies, a production BM25/sparse index, "
        "cross-encoder reranking, and a curated regression benchmark."
    )
