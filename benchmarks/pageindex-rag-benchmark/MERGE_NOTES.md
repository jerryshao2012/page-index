# Merge Notes: pageindex-rag-benchmark

This directory is merged from:
- [adorosario/pageindex-rag-benchmark](https://github.com/adorosario/pageindex-rag-benchmark)

## Local Adaptations in This Repo

1. Script paths were changed from Docker-specific `/app/...` to repo-relative paths.
2. `scripts/fair_benchmark.py` now supports CLI flags for:
- `--questions-file`
- `--index-dir`
- `--output-dir`
- `--answer-model`
- `--judge-model`
- `--penalty-ratio`
3. `scripts/two_stage_search.py` now supports:
- `--index-dir`
- `--embedding-model`
4. `scripts/generate_chart.py` now saves output to `article/benchmark_results.png` using a repo-relative path.

## Run From This Repo

Install benchmark-only dependencies:

```bash
.venv/bin/pip install -r benchmarks/pageindex-rag-benchmark/requirements.txt
```

Run benchmark:

```bash
.venv/bin/python benchmarks/pageindex-rag-benchmark/scripts/fair_benchmark.py \
  --limit 100 \
  --index-dir benchmarks/pageindex-rag-benchmark/faiss_index
```

Run retrieval smoke check:

```bash
.venv/bin/python benchmarks/pageindex-rag-benchmark/scripts/two_stage_search.py \
  --index-dir benchmarks/pageindex-rag-benchmark/faiss_index
```
