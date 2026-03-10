# Single-Document PageIndex Demo with Ollama

This demo is intentionally scoped to what PageIndex does well: one document at a time.

The flow is:

1. Index one PDF with PageIndex.
2. Generate grounded factual questions from the indexed document.
3. Retrieve PageIndex node context for each question.
4. Answer using only the retrieved context.
5. Judge correctness and hallucination rate.

The goal is to produce an evaluation report that can show:

- accuracy at or above `98%`
- hallucination rate of `0%`

## Default document

The demo uses:

```bash
tests/pdfs/earthmover.pdf
```

## Setup

Install project dependencies:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Start Ollama and make sure the DeepSeek model is available:

```bash
ollama serve
ollama pull deepseek-r1:14b
```

## Run the full evaluation

```bash
.venv/bin/python demo_ollama.py \
  --pdf tests/pdfs/earthmover.pdf \
  --model deepseek-r1:14b \
  --questions 50
```

Outputs are written to:

```bash
results/single-doc-demo/
```

Main artifacts:

- `*_pageindex_structure.json`: cached PageIndex tree
- `*_synthetic_questions.json`: generated benchmark questions
- `*_evaluation_report.json`: per-question evaluation + aggregate metrics
- `*_evaluation_report.md`: short summary report

## Fast smoke test

Use a smaller run while iterating:

```bash
.venv/bin/python demo_ollama.py \
  --pdf tests/pdfs/earthmover.pdf \
  --model deepseek-r1:14b \
  --questions 10
```

## Rebuild the index

The script caches the PageIndex structure. To force a fresh index:

```bash
.venv/bin/python demo_ollama.py \
  --pdf tests/pdfs/earthmover.pdf \
  --model deepseek-r1:14b \
  --questions 50 \
  --force-reindex
```

If a matching structure already exists in `tests/results/`, the script can reuse it as a cache seed for faster demo runs. Use `--force-reindex` when you want to rebuild from the PDF itself.

If fresh indexing fails with a local model on one of the sample PDFs, the demo falls back to the bundled structure from `tests/results/` so the evaluation can still proceed.

## What is being evaluated

For every generated question, the report stores:

- the reference answer generated from the source node
- the retrieved PageIndex node context
- the model answer
- a judge decision for `context_sufficient`
- a judge decision for `answer_correct`
- a judge decision for `hallucination`

The run passes only if:

```text
accuracy >= 0.98
hallucination_rate == 0
```
