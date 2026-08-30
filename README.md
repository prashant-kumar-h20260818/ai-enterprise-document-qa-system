# Enterprise Multimodal RAG Document Assistant

An end-to-end, source-grounded document Q&A application refactored from the
original `document3.ipynb` and extended to handle **multimodal enterprise documents**.

The application does not assume a document is plain text. It can ingest common
business formats containing combinations of:

- paragraphs and headings
- tables and spreadsheets
- charts and plots
- embedded images and screenshots
- diagrams and flowcharts
- scanned pages
- forms and labels
- formulas and other visual/layout information

## Supported file formats

| Category | Formats |
|---|---|
| PDF | `.pdf` |
| Word | `.docx` |
| PowerPoint | `.pptx` |
| Excel | `.xlsx`, `.xlsm` |
| Tabular | `.csv`, `.tsv` |
| Text / markup | `.txt`, `.md`, `.markdown`, `.html`, `.htm`, `.xml`, `.yaml`, `.yml` |
| Structured | `.json` |
| Email | `.eml` |
| Images | `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`, `.tiff` |

Legacy binary Office formats such as `.doc`, `.ppt`, and `.xls` are not parsed
directly because reliable support normally requires an Office/LibreOffice
conversion service. Convert those files to modern formats before upload.

## Multimodal extraction strategy

The ingestion layer combines **deterministic format-aware parsing** with an
optional **Groq vision model**.

### PDF

- extracts ordinary PDF text
- extracts recognizable tables with PyMuPDF
- retains page metadata
- uses page-level vision when a page contains images or has little extractable text
- vision can recover scanned text and describe charts, diagrams, formulas and layout relationships

### DOCX

- extracts paragraphs
- preserves tables as Markdown
- analyzes embedded images with the vision model

### PPTX

- extracts slide text
- preserves tables
- extracts chart categories/series where accessible
- analyzes embedded pictures
- retains text-box positions as an extra signal for diagrams/layouts

### XLSX / CSV

- converts sheets/tables to structured Markdown
- preserves sheet and row-range locators for citations

### Images / scanned documents

The visual extractor is instructed to preserve evidence instead of producing a
vague image caption. It attempts to:

- transcribe important text
- convert tables to Markdown
- record chart titles, axes, series, values and trends
- describe diagram nodes, arrows, hierarchy and relationships
- preserve form fields and values
- capture formulas and captions

## How I measure accuracy

A RAG system should **not** be described with one vague accuracy number. Measure
two stages independently.

### 1. Retrieval quality

- **Hit@K / Recall@K:** did the correct source and location appear in Top-K?
- **MRR:** how high was the first correct source ranked?

For multimodal files, the gold location can be a page, slide, sheet, table or
image locator. The evaluation CSV therefore supports both `expected_page` and
`expected_locator`.

### 2. Generation quality

- answer correctness
- faithfulness to retrieved evidence
- semantic answer similarity as an automated proxy
- no-answer accuracy for questions absent from the documents

For visual questions, include questions whose answers are found specifically in
charts, tables, diagrams, screenshots or scanned pages. This measures whether
the multimodal ingestion stage actually adds value.

For a proper benchmark, create 50–100 labeled questions with expected answers
and source/location references. The Streamlit **Evaluation** tab can run the
dataset and export results. See `docs/ACCURACY_AND_EVALUATION.md`.

## Architecture

```text
PDF / DOCX / PPTX / XLSX / CSV / HTML / JSON / Images / ...
                         |
                         v
        Format-aware parsing + Groq Vision (optional)
                         |
                         v
 Text + tables + chart/diagram/image/scanned-page evidence
                         |
                         v
   LangChain Documents + source/location/content-type metadata
                         |
                         v
         RecursiveCharacterTextSplitter
          default 1000 chars / 200 overlap
                         |
                         v
              all-MiniLM-L6-v2
        384-d normalized text embeddings
                         |
                         v
             Persistent ChromaDB
             HNSW + cosine distance
                         |
                         v
             Query embedding
                         |
                         v
     Top-K retrieval + similarity threshold
                         |
                         v
 Source-labeled context [S1], [S2], ... with location
                         |
                         v
          Groq / Llama 3.3 70B
              temperature 0.1
                         |
                         v
           Grounded answer + citations
                         |
                         v
                 Streamlit UI
```

### Why use text embeddings after vision?

The vision model converts visual evidence into structured textual evidence at
ingestion time. That text is then embedded using the same MiniLM model as other
document content. This keeps the retrieval layer simple and lets a user ask one
natural-language question across text, tables and visual information.

For a larger production system, a future upgrade would be true multimodal
embeddings that jointly index image and text representations.

## What was improved from the notebook

- Added an end-to-end Streamlit application.
- Added multimodal/format-aware ingestion.
- Added tables, charts, diagrams, images and scanned-page extraction.
- Added PDF/DOCX/PPTX/XLSX/CSV/image and common structured-text support.
- Removed hard-coded API-key handling.
- Added deterministic chunk IDs so upserts do not duplicate chunks.
- Added per-session vector collections to reduce accidental cross-user retrieval.
- Added grounded prompts and source labels.
- Replaced the misleading `confidence` name with retrieval similarity.
- Added a real evaluation workflow (Hit@K, MRR, semantic answer similarity,
  optional LLM-as-judge correctness/faithfulness).
- Added tests, `.gitignore`, `.env.example`, and interview documentation.

## Local setup

Python 3.11 is recommended.

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Create `.env` from `.env.example`:

```env
GROQ_API_KEY=your_real_key_here
GROQ_VISION_MODEL=qwen/qwen3.6-27b
```

The vision model name is configurable because provider model availability can
change. If your Groq account exposes a different vision-capable model, set
`GROQ_VISION_MODEL` accordingly.

Never commit the real API key.

Run:

```bash
streamlit run app.py
```

## Multimodal ingestion controls

The Streamlit sidebar includes:

- **Multimodal document extraction** — turns visual analysis on/off.
- **Groq vision model** — configurable vision model name.
- **Vision analyses per file** — prevents accidental high API usage on very
  visual files; `0` means unlimited.

Text and locally extractable tables can still be processed if visual extraction
is disabled. Image-only documents require visual extraction.

## Evaluation CSV

Use `data/evaluation_template.csv`:

```csv
question,expected_answer,expected_source,expected_page,expected_locator
"What is the resignation date?","Gold answer","resume.pdf",1,"page 1"
"What value is shown in the revenue chart?","Gold answer","annual_report.pdf",,"page 7"
"What is the total in the Sales sheet?","Gold answer","financials.xlsx",,"sheet Sales"
```

Upload the same underlying documents to the app, build the index, then upload
the labeled evaluation CSV in the Evaluation tab.

## Security

The original notebook contained an API key directly in a code cell. This repo
does **not** include that secret. Rotate any key that has been exposed in a
notebook or Git history.

The app reads `GROQ_API_KEY` from `.env`, environment variables, Streamlit
secrets, or a session-only password field.

Remember that multimodal ingestion can send document images/pages to the
configured external vision provider. For confidential enterprise documents,
review provider data-handling requirements before enabling this feature.

## Important limitations

"Any document" should be interpreted as broad support for common enterprise
formats, not literally every proprietary file format. Examples requiring a
specialized pipeline include:

- password-protected/encrypted files
- corrupted documents
- legacy `.doc/.ppt/.xls` without conversion
- CAD/GIS/medical-image formats
- handwritten documents with very poor image quality
- highly complex spreadsheet charts/macros
- specialized scientific diagrams where domain-specific extraction is required

Vision extraction is also probabilistic. Critical numerical values in charts or
scans should be validated in the evaluation set.

## Production upgrades

For a real enterprise deployment, add:

- SSO/authentication and tenant-level document authorization
- encrypted object storage
- malware/content scanning on upload
- extraction caching by file hash
- background ingestion workers
- provider-level privacy controls or self-hosted multimodal models
- rate limiting
- structured audit logs and monitoring
- deletion/retention policies
- hybrid lexical + vector retrieval and reranking
- managed/distributed vector infrastructure when required
- a larger curated multimodal evaluation set and regression testing

## Publish to GitHub

Suggested repository name:

`ai-enterprise-document-qa-system`

With GitHub CLI installed and authenticated:

```bash
git init
git add .
git commit -m "Add multimodal end-to-end enterprise RAG assistant"
git branch -M main
gh repo create prashant-kumar-h20260818/ai-enterprise-document-qa-system \
  --public --source=. --remote=origin --push
```

Before publishing, verify that `.env`, `.streamlit/secrets.toml`, `vector_store/`
and private source documents are not tracked.
