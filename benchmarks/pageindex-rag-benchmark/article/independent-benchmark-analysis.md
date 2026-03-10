# No, PageIndex Will Not "Kill" RAG, But It Is Indeed Excellent In Some Cases

An independent benchmark revealing when tree-based RAG outperforms vector RAG, and when it cannot be used.

A viral tweet claimed that PageIndex, a reasoning-based RAG system, achieved 98.7% accuracy on a financial benchmark without vector databases, chunking, or similarity search.

The benchmark results here present a more nuanced conclusion and highlight a key limitation in multi-document settings.

## What Is PageIndex?

PageIndex by VectifyAI uses a different approach than standard chunk-embed-retrieve pipelines:

1. Build a hierarchical tree index (semantic table of contents)
2. Use LLM reasoning to navigate the tree and identify relevant sections
3. Extract content from selected sections for answer generation

This can outperform similarity search for within-document relevance. For example, a question about certification dates may be answered better by navigating timeline structure than by nearest-neighbor chunk similarity.

VectifyAI's Mafin 2.5 reports 98.7% on FinanceBench, a single-document benchmark where each question targets one financial report.

## The Scalability Problem

In multi-document scenarios (for example, ~1000 documents), tree-index construction is a bottleneck. In this benchmark setup, tree-based retrieval could not be applied practically at that scale, so the pipeline used FAISS vector retrieval fallback.

PageIndex team statements indicate the open-source implementation is optimized for single long-document QA and sequential indexing (proof-of-concept orientation).

## The Benchmark: 100 Questions, 1000 Documents

This benchmark compares PageIndex fallback pipeline against commercial RAG providers using:

- 100 questions from SimpleQA-Verified
- 2,795 source documents
- Provider-specific retrieval/generation pipelines

Scoring:

`Quality = (correct - 4 x incorrect) / total`

The 4x incorrect penalty favors precision-oriented systems that abstain when uncertain.

## Core Trade-off

### Single-Document: Strong Fit

PageIndex is strong when document identity is known and structural reasoning can be applied deeply within one document.

### Multi-Document: Fallback to Standard RAG

At large corpus scale, tree construction is the constraint, and retrieval falls back to vector search. In that mode, PageIndex behaves like other vector-RAG pipelines.

## Where PageIndex Excels

- Single-document deep analysis (legal, financial, technical)
- Hierarchical/structured documents
- Retrieval traceability and auditability
- Conservative abstention behavior in high-stakes contexts

## Takeaway

PageIndex does not replace all RAG patterns. It is highly effective in targeted, high-stakes single-document workflows, while multi-document retrieval still depends on scalable discovery methods (often vector search). Hybrid designs are likely the practical path: vector retrieval for document discovery, tree-based reasoning for within-document extraction.

## Methodology Notes

- Questions: 100 from SimpleQA-Verified
- Documents: 2,795 (with FAISS index over 81,868 chunks)
- Answer model (PageIndex fallback run): GPT-5.1, temperature 0
- Judge: GPT-4.1-mini using simple-evals style grading
- Penalty ratio: 4.0

At this sample size, rankings are directional rather than definitive.

## Disclosure

Benchmark design and execution were performed independently by Alden Do Rosario (CustomGPT.ai), one evaluated provider. Audit artifacts are included for transparency and reproducibility.
