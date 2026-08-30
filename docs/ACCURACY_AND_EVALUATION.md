# Measuring Accuracy in a Multimodal RAG Application

A RAG system does **not** have one honest "accuracy" metric. It has at least
three systems that can fail independently in this project:

1. **Extraction** — was the relevant text/table/chart/diagram/image information
   extracted correctly from the source document?
2. **Retriever** — did it find the right extracted evidence?
3. **Generator** — did the LLM use that evidence correctly?

For multimodal documents, separating these stages is especially important. If a
chart value was extracted incorrectly, the retriever and generator can both work
perfectly and still return the wrong answer.

## 1. Build a gold evaluation set

Create at least 50–100 representative questions before tuning the system. For
every question, label:

- `question`
- `expected_answer`
- `expected_source`
- `expected_page` for PDF page-level cases, where practical
- `expected_locator` for generalized locations such as `page 7`, `slide 3`,
  `sheet Sales`, `table 2`, or `embedded image 1`

Use a balanced set containing:

- normal paragraph questions
- table questions
- chart/plot questions
- diagram/flowchart questions
- scanned-page questions
- spreadsheet questions
- paraphrased questions
- multi-document questions
- cross-modal questions
- deliberately unanswerable questions

## 2. Extraction evaluation

For visual/structured sources, manually inspect a sample of the extracted
`Document.page_content` before evaluating retrieval.

Useful extraction-level measures include:

- exact numeric value accuracy for tables/charts
- table cell preservation rate
- OCR/transcription word accuracy for scanned pages
- diagram relation accuracy (correct nodes/edges/direction)
- chart metadata accuracy (title/axis/legend/series/value)

A simple portfolio benchmark can label 20–30 visual items and record whether the
critical evidence was extracted correctly. In a production system, maintain a
larger extraction regression set.

## 3. Retrieval metrics

### Hit@K / Recall@K

For each question, ask whether the known correct source/location appears
anywhere in the top K retrieved results.

`Hit@5 = successful questions / total questions`

This is usually the first retrieval metric to optimize because a generator
cannot reliably answer from evidence it never receives.

### MRR — Mean Reciprocal Rank

If the first correct source is ranked first, reciprocal rank = 1. If it is
second, 1/2. If fifth, 1/5. Average across questions.

MRR rewards ranking the correct evidence near the top.

### Context precision (optional)

If you label all relevant chunks, measure how many retrieved chunks are truly
relevant. This helps detect a high-recall but noisy retriever.

## 4. Generation metrics

### Answer correctness

Compare the generated answer with the gold answer. Human review is strongest.
This repo also provides embedding-based semantic similarity as a quick proxy.

For numerical/table/chart questions, use exact or tolerance-based numeric checks
where possible instead of only semantic similarity.

### Faithfulness

Check whether every factual claim in the generated answer is supported by the
retrieved context. This is the key hallucination metric.

The optional LLM-as-judge evaluator returns correctness and faithfulness scores,
but judge scores are not ground truth. Manually audit a sample.

### No-answer accuracy

Include questions whose answer is absent from the documents. Measure whether the
assistant correctly refuses instead of inventing an answer.

## 5. Report results by modality

Do not hide multimodal failure modes inside one overall average. Report a table
such as:

| Question type | Count | Hit@5 | Answer correctness | Faithfulness |
|---|---:|---:|---:|---:|
| Text | 30 | ... | ... | ... |
| Tables | 20 | ... | ... | ... |
| Charts | 15 | ... | ... | ... |
| Diagrams | 10 | ... | ... | ... |
| Scanned pages | 10 | ... | ... | ... |
| Unanswerable | 15 | n/a | no-answer accuracy | ... |

This lets you say exactly where the application is strong or weak.

## 6. Product metrics

Offline evaluation is not enough for a production system. Track:

- task-success rate
- user feedback
- p50 / p95 latency
- ingestion latency per page/file
- vision API usage/cost per document
- text-generation API cost per query
- no-answer rate
- repeated/rephrased queries
- source-click rate
- retrieval misses
- extraction failures

## Recommended interview framing

Say:

> "Because this is multimodal RAG, I evaluate three stages. First I verify that
> the relevant evidence was extracted correctly from text, tables, charts or
> diagrams. Then I evaluate retrieval with Hit@K and MRR. Finally I evaluate the
> generated answer for correctness, faithfulness and no-answer behavior. I also
> report results by modality rather than hiding chart/table failures in one
> overall number. That lets me diagnose whether a bad answer came from
> extraction, retrieval or generation."

Do not say a single semantic-similarity score is "the accuracy" of the RAG
system.
