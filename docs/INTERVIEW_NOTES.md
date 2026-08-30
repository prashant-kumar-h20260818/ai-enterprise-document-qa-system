# Interview Notes — Actual Implementation Choices

This project was refactored from `document3.ipynb` and extended into a
multimodal enterprise-document RAG application.

## Architecture you should be able to explain

- **Input formats:** PDF, DOCX, PPTX, XLSX/XLSM, CSV/TSV, text/Markdown,
  HTML, JSON/XML/YAML, EML and common image formats.
- **Multimodal ingestion:** deterministic parsing first; Groq Vision is used for
  scanned pages, images, charts, diagrams, screenshots and formulas when enabled.
- **Structured evidence:** tables are converted to Markdown; visual content is
  converted into detailed textual evidence before embedding.
- **Location metadata:** page, slide, sheet, table/image or row-range locators
  are preserved for citations and evaluation.
- **Chunking:** `RecursiveCharacterTextSplitter`, default 1000 characters with
  200-character overlap.
- **Embeddings:** `all-MiniLM-L6-v2`, 384 dimensions, normalized.
- **Vector store:** persistent ChromaDB with HNSW cosine distance.
- **Retrieval:** query is embedded with the same model; Chroma returns cosine
  distances; similarity is computed as `1 - distance`; Top-K + minimum threshold
  are configurable.
- **Generation:** Groq `llama-3.3-70b-versatile`, temperature 0.1.
- **Grounding:** answer only from retrieved context; cite `[S1]`, `[S2]`; refuse
  when evidence is insufficient.
- **UI:** Streamlit multi-document upload, chat history, source previews,
  multimodal parsing settings and an evaluation tab.

## Likely "why" questions

### Why do multimodal extraction before embeddings?
The MiniLM embedding model is text-only. Visual content therefore has to be
converted into textual/structured evidence first. A chart can become a
representation containing its title, axes, series and values; a diagram can
become nodes/arrows/relationships; a table becomes Markdown. The same retrieval
pipeline can then search all evidence with one query embedding.

### Why not just OCR every document?
OCR only recognizes text pixels. It does not reliably capture chart semantics,
diagram relationships, table structure or the meaning of an image. The pipeline
therefore uses native document parsers where possible, and vision only where it
adds information that ordinary text extraction may miss.

### Why parse formats locally first instead of sending every page to a vision model?
Local parsing is faster, cheaper and deterministic for text and structured
tables. Vision calls add latency/cost and are probabilistic. The design uses
vision selectively for visual pages/images and scanned content.

### Why 1000 / 200 chunking?
It is a practical baseline: enough local context while keeping retrieval units
specific. The overlap reduces context loss at boundaries. It is not claimed to
be optimal; tune it against Hit@K/MRR and answer quality.

### Why MiniLM?
It is small, fast and strong enough for a portfolio/local semantic-search
baseline. The 384-d vectors reduce memory/latency versus larger embedding
models. Benchmark domain-specific or larger models if retrieval quality is weak.

### Why normalize embeddings and use cosine?
With L2-normalized vectors, cosine similarity equals the dot product. Normalizing
makes vector magnitudes irrelevant and keeps comparisons focused on direction.
Chroma is explicitly configured with cosine distance for consistency.

### Why Chroma?
It is simple, persistent and local, with metadata support and HNSW indexing.
For large multi-tenant production deployments, evaluate managed/distributed
stores or pgvector depending on requirements.

### Why temperature 0.1?
Document QA is factual rather than creative. Low temperature reduces variation
and supports more repeatable answers; retrieval/prompt quality still matters.

### Why isn't retrieval similarity "confidence"?
A high query-to-chunk cosine similarity only says the retrieved text is close to
the question in embedding space. It does not prove the generated answer is
correct. The UI therefore calls it `max_retrieval_similarity`.

## How to discuss tables/images/diagrams

A strong answer:

> "I don't treat a document as only a string of text. During ingestion I first
> use the native structure available in the file. For example, Word and Excel
> tables are kept as Markdown tables and PowerPoint chart series are extracted
> where possible. For scanned pages, embedded images, charts and diagrams I use
> a configurable vision model that converts the visual evidence into structured
> text. I keep source and location metadata with every unit, then chunk and
> embed that evidence in the same retrieval index. This lets one query retrieve
> evidence from a paragraph, a table, a chart or a diagram."

## How to evaluate multimodal accuracy

Do not only test paragraph questions. Build a gold dataset with separate slices:

- text-only questions
- table lookup/calculation questions
- chart questions
- diagram/flow questions
- scanned-page questions
- cross-modal questions that require text + table/chart evidence
- intentionally unanswerable questions

Report Hit@K/MRR for retrieval and correctness/faithfulness for answers for each
slice. This tells you whether visual extraction is genuinely helping or hiding
failures behind one average score.

## Important limitations to acknowledge

- Vision extraction is probabilistic and can misread tiny labels/numbers.
- Sending visual pages to an external model may not satisfy some enterprise
  privacy requirements.
- Native Office parsing cannot fully reproduce every visual relationship or macro.
- Legacy binary Office formats need conversion.
- Public deployment needs real authentication and tenant-level authorization.
- LLM prompt injection cannot be solved by prompting alone; production systems
  need stronger content isolation and security controls.
- LLM-as-judge evaluation is useful but biased/noisy; human audit is required.
- The default parameters are baselines and should be tuned empirically.
