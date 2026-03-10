import argparse
import ast
import json
import os
import re
from copy import deepcopy
from pathlib import Path

import openai

from pageindex import config, page_index_main
from pageindex.utils import (
    _resolve_client_config,
    add_node_text,
    get_page_tokens,
    structure_to_list,
    write_node_id,
)


DEFAULT_MODEL = "deepseek-r1:14b"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_PDF = "tests/pdfs/earthmover.pdf"
DEFAULT_OUTPUT_DIR = "results/single-doc-demo"

QUESTION_SCHEMA = {
    "questions": [
        {
            "question": "string",
            "answer": "string",
            "evidence": "string",
        }
    ]
}

JUDGE_SCHEMA = {
    "context_sufficient": False,
    "answer_correct": False,
    "hallucination": False,
    "explanation": "string",
}

TITLE_BLACKLIST = {
    "abstract",
    "references",
    "acknowledgment",
    "acknowledgement",
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-document PageIndex grounded QA demo using local Ollama models."
    )
    parser.add_argument(
        "--pdf",
        default=DEFAULT_PDF,
        help="Single PDF to index and evaluate.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OLLAMA_MODEL", DEFAULT_MODEL),
        help="Ollama model for indexing, QA generation, answering, and judging.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Optional separate judge model. Defaults to --model.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("PAGEINDEX_API_BASE_URL", DEFAULT_BASE_URL),
        help="OpenAI-compatible API base URL for Ollama.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for cached structures and evaluation reports.",
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=50,
        help="Number of synthetic grounded questions to evaluate.",
    )
    parser.add_argument(
        "--qa-per-node",
        type=int,
        default=4,
        help="How many QA pairs to request from each candidate node.",
    )
    parser.add_argument(
        "--min-node-chars",
        type=int,
        default=800,
        help="Minimum node text length for QA generation.",
    )
    parser.add_argument(
        "--top-k-nodes",
        type=int,
        default=2,
        help="How many retrieved nodes to include in the generated context.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=6000,
        help="Maximum context characters passed to the answer model.",
    )
    parser.add_argument(
        "--target-accuracy",
        type=float,
        default=0.98,
        help="Accuracy threshold required for the demo to pass.",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Ignore any cached PageIndex structure and rebuild it from the PDF.",
    )
    return parser.parse_args()


def normalize_text(text):
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = cleaned.replace("```json", "").replace("```", "")
    return cleaned.strip()


def sanitize_json_candidate(text):
    text = text.replace("\r", "\n")
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", text)
    text = re.sub(r",(\s*[}\]])", r"\1", text)
    return text.strip()


def extract_json_candidates(text):
    candidates = []
    normalized = normalize_text(text)
    for opener, closer in (("{", "}"), ("[", "]")):
        start = normalized.find(opener)
        end = normalized.rfind(closer)
        if start != -1 and end != -1 and end > start:
            candidates.append(normalized[start : end + 1])
    candidates.append(normalized)
    return candidates


def try_parse_json(text):
    for candidate in extract_json_candidates(text):
        cleaned = sanitize_json_candidate(candidate)
        if not cleaned:
            continue
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        try:
            return ast.literal_eval(cleaned)
        except (SyntaxError, ValueError):
            pass
    return None


def repair_json_with_model(client, model, raw_text, schema):
    prompt = f"""
Convert the following content into valid JSON.

Rules:
- Return JSON only.
- Preserve the original meaning.
- Do not add new facts.
- Use this schema exactly:
{json.dumps(schema)}

Content to repair:
{raw_text}
"""
    repaired = chat(client, model, prompt)
    return try_parse_json(repaired)


def parse_json_response(text, fallback=None, client=None, model=None, schema=None):
    parsed = try_parse_json(text)
    if parsed is not None:
        return parsed
    if client is not None and model is not None and schema is not None:
        repaired = repair_json_with_model(client, model, text, schema)
        if repaired is not None:
            return repaired
    return fallback


def chat(client, model, prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return normalize_text(response.choices[0].message.content or "")


def tokenize(text):
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in STOPWORDS]


def score_node(question_tokens, node):
    title_tokens = set(tokenize(node["title"]))
    text_tokens = set(tokenize(node["text"][:4000]))
    overlap_title = len(question_tokens & title_tokens)
    overlap_text = len(question_tokens & text_tokens)
    return overlap_title * 4 + overlap_text


def get_cache_paths(output_dir, pdf_path):
    stem = pdf_path.stem
    return {
        "structure": output_dir / f"{stem}_pageindex_structure.json",
        "qa": output_dir / f"{stem}_synthetic_questions.json",
        "report": output_dir / f"{stem}_evaluation_report.json",
        "report_md": output_dir / f"{stem}_evaluation_report.md",
    }


def load_bundled_structure(pdf_path):
    bundled = pdf_path.parent.parent / "results" / f"{pdf_path.stem}_structure.json"
    if not bundled.exists():
        return None
    with open(bundled, "r", encoding="utf-8") as f:
        bundled_structure = json.load(f)
    return {
        "doc_name": pdf_path.name,
        "structure": bundled_structure.get("structure", bundled_structure),
    }


def ensure_structure(pdf_path, cache_file, model, force_reindex):
    if cache_file.exists() and not force_reindex:
        with open(cache_file, "r", encoding="utf-8") as f:
            cached = json.load(f)
        structure_doc = deepcopy(cached)
    elif not force_reindex:
        bundled_structure = load_bundled_structure(pdf_path)
        if bundled_structure is not None:
            structure_doc = bundled_structure
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(structure_doc, f, indent=2, ensure_ascii=False)
        else:
            options = config(
                model=model,
                toc_check_page_num=20,
                max_page_num_each_node=10,
                max_token_num_each_node=20000,
                if_add_node_id="yes",
                if_add_node_summary="no",
                if_add_doc_description="no",
                if_add_node_text="no",
            )
            try:
                structure_doc = page_index_main(str(pdf_path), options)
            except Exception as exc:
                raise RuntimeError(
                    f"Step 1 indexing failed for {pdf_path.name}: {exc}"
                ) from exc
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(structure_doc, f, indent=2, ensure_ascii=False)
    else:
        options = config(
            model=model,
            toc_check_page_num=20,
            max_page_num_each_node=10,
            max_token_num_each_node=20000,
            if_add_node_id="yes",
            if_add_node_summary="no",
            if_add_doc_description="no",
            if_add_node_text="no",
        )
        try:
            structure_doc = page_index_main(str(pdf_path), options)
        except Exception as exc:
            bundled_structure = load_bundled_structure(pdf_path)
            if bundled_structure is None:
                raise RuntimeError(
                    f"Step 1 indexing failed for {pdf_path.name} and no bundled fallback exists: {exc}"
                ) from exc
            print(
                "Fresh indexing failed. Falling back to bundled structure from tests/results "
                f"for {pdf_path.name}. Original error: {exc}"
            )
            structure_doc = bundled_structure
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(structure_doc, f, indent=2, ensure_ascii=False)

    structure = structure_doc["structure"]
    write_node_id(structure)
    page_list = get_page_tokens(str(pdf_path), model=model)
    add_node_text(structure, page_list)
    return structure_doc


def candidate_nodes(structure, min_node_chars):
    nodes = []
    for node in structure_to_list(structure):
        title = (node.get("title") or "").strip()
        text = (node.get("text") or "").strip()
        if node.get("nodes"):
            continue
        if not title or not text:
            continue
        if title.lower() in TITLE_BLACKLIST:
            continue
        if len(text) < min_node_chars:
            continue
        nodes.append(node)
    return nodes


def generate_questions(client, model, nodes, total_questions, qa_per_node):
    generated = []
    seen_questions = set()

    for node in nodes:
        remaining = total_questions - len(generated)
        if remaining <= 0:
            break

        requested = min(qa_per_node, remaining)
        excerpt = node["text"][:5000]
        prompt = f"""
You are creating a strict grounded QA benchmark from one document section.

Section title: {node["title"]}
Pages: {node.get("start_index")} to {node.get("end_index")}
Section text:
{excerpt}

Generate up to {requested} factual questions answerable directly from this section.
Rules:
- Questions must be answerable from the section text alone.
- Answers must be short and factual.
- Do not create yes/no questions.
- Do not ask about citations, acknowledgments, or references.
- Keep questions specific enough that one answer is clearly best.

Return JSON only in this shape:
{json.dumps(QUESTION_SCHEMA)}
"""
        raw = chat(client, model, prompt)
        payload = parse_json_response(
            raw,
            fallback={"questions": []},
            client=client,
            model=model,
            schema=QUESTION_SCHEMA,
        )
        for item in payload.get("questions", []):
            question = (item.get("question") or "").strip()
            answer = (item.get("answer") or "").strip()
            evidence = (item.get("evidence") or "").strip()
            if not question or not answer:
                continue
            dedupe_key = question.lower()
            if dedupe_key in seen_questions:
                continue
            seen_questions.add(dedupe_key)
            generated.append(
                {
                    "question": question,
                    "reference_answer": answer,
                    "reference_evidence": evidence,
                    "source_node_id": node["node_id"],
                    "source_title": node["title"],
                    "source_pages": [node.get("start_index"), node.get("end_index")],
                }
            )
            if len(generated) >= total_questions:
                return generated

    return generated


def retrieve_context(question, nodes, top_k_nodes, max_context_chars):
    question_tokens = set(tokenize(question))
    scored = []
    for node in nodes:
        score = score_node(question_tokens, node)
        if score > 0:
            scored.append((score, node))

    if not scored:
        return {
            "context": "",
            "selected_nodes": [],
        }

    scored.sort(key=lambda item: (-item[0], item[1]["node_id"]))
    selected = [node for _, node in scored[:top_k_nodes]]

    parts = []
    current_chars = 0
    for node in selected:
        header = f"[Node {node['node_id']}] {node['title']} (pages {node.get('start_index')}-{node.get('end_index')})\n"
        body_budget = max_context_chars - current_chars - len(header)
        if body_budget <= 0:
            break
        body = node["text"][:body_budget]
        parts.append(header + body)
        current_chars += len(header) + len(body)
        if current_chars >= max_context_chars:
            break

    return {
        "context": "\n\n".join(parts),
        "selected_nodes": [
            {
                "node_id": node["node_id"],
                "title": node["title"],
                "pages": [node.get("start_index"), node.get("end_index")],
            }
            for node in selected
        ],
    }


def answer_question(client, model, question, context):
    prompt = f"""
Answer the question using only the provided context.

Rules:
- If the answer is not fully supported by the context, reply exactly: I don't know.
- Keep the answer concise.
- Do not use outside knowledge.

Context:
{context}

Question: {question}
"""
    return chat(client, model, prompt)


def judge_answer(client, model, question, reference_answer, context, answer):
    prompt = f"""
You are evaluating a single-document RAG answer.

Question: {question}
Reference answer: {reference_answer}
Retrieved context:
{context}

Model answer: {answer}

Return JSON only with this exact shape:
{json.dumps(JUDGE_SCHEMA)}

Definitions:
- context_sufficient: true if the retrieved context contains enough information to answer the question correctly.
- answer_correct: true if the model answer matches the reference answer in meaning.
- hallucination: true if the model answer includes unsupported claims or uses outside knowledge beyond the retrieved context.
"""
    raw = chat(client, model, prompt)
    return parse_json_response(
        raw,
        fallback=deepcopy(JUDGE_SCHEMA),
        client=client,
        model=model,
        schema=JUDGE_SCHEMA,
    )


def build_markdown_report(pdf_name, model, judge_model, metrics, report_file):
    content = [
        f"# Single-Document PageIndex Evaluation: {pdf_name}",
        "",
        f"- Answer model: `{model}`",
        f"- Judge model: `{judge_model}`",
        f"- Questions evaluated: `{metrics['total_questions']}`",
        f"- Accuracy: `{metrics['accuracy']:.2%}`",
        f"- Context sufficient rate: `{metrics['context_sufficient_rate']:.2%}`",
        f"- Hallucination rate: `{metrics['hallucination_rate']:.2%}`",
        f"- Passes 98% / zero-hallucination gate: `{metrics['passes_gate']}`",
        "",
        f"JSON report: `{report_file.name}`",
    ]
    return "\n".join(content) + "\n"


def main():
    args = parse_args()

    os.environ["PAGEINDEX_API_BASE_URL"] = args.base_url
    os.environ.setdefault("CHATGPT_API_KEY", "ollama")

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_paths = get_cache_paths(output_dir, pdf_path)

    client = openai.OpenAI(**_resolve_client_config())
    judge_model = args.judge_model or args.model

    print(f"PDF: {pdf_path}")
    print(f"Model: {args.model}")
    print(f"Judge model: {judge_model}")
    print(f"Output directory: {output_dir}")

    print("\nStep 1/4: building or loading the PageIndex tree...")
    indexed_doc = ensure_structure(pdf_path, cache_paths["structure"], args.model, args.force_reindex)
    nodes = candidate_nodes(indexed_doc["structure"], args.min_node_chars)
    print(f"Candidate nodes for QA generation: {len(nodes)}")

    print("Step 2/4: generating grounded questions from indexed nodes...")
    qa_pairs = generate_questions(client, args.model, nodes, args.questions, args.qa_per_node)
    if not qa_pairs:
        raise RuntimeError("No QA pairs were generated from the indexed document.")

    with open(cache_paths["qa"], "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    print(f"Generated {len(qa_pairs)} questions")

    print("Step 3/4: retrieving context and answering each question...")
    evaluations = []
    for idx, qa in enumerate(qa_pairs, start=1):
        retrieval = retrieve_context(
            qa["question"],
            nodes,
            args.top_k_nodes,
            args.max_context_chars,
        )
        answer = answer_question(client, args.model, qa["question"], retrieval["context"])
        judgment = judge_answer(
            client,
            judge_model,
            qa["question"],
            qa["reference_answer"],
            retrieval["context"],
            answer,
        )
        evaluations.append(
            {
                **qa,
                "retrieved_context": retrieval["context"],
                "selected_nodes": retrieval["selected_nodes"],
                "model_answer": answer,
                "judgment": judgment,
            }
        )
        print(
            f"  [{idx:02d}/{len(qa_pairs)}] "
            f"correct={judgment.get('answer_correct')} "
            f"context={judgment.get('context_sufficient')} "
            f"hallucination={judgment.get('hallucination')}"
        )

    print("Step 4/4: aggregating evaluation metrics...")
    total = len(evaluations)
    correct = sum(1 for item in evaluations if item["judgment"].get("answer_correct") is True)
    sufficient = sum(1 for item in evaluations if item["judgment"].get("context_sufficient") is True)
    hallucinations = sum(1 for item in evaluations if item["judgment"].get("hallucination") is True)

    metrics = {
        "total_questions": total,
        "correct_answers": correct,
        "context_sufficient": sufficient,
        "hallucinations": hallucinations,
        "accuracy": correct / total if total else 0.0,
        "context_sufficient_rate": sufficient / total if total else 0.0,
        "hallucination_rate": hallucinations / total if total else 0.0,
    }
    metrics["passes_gate"] = (
        metrics["accuracy"] >= args.target_accuracy and metrics["hallucination_rate"] == 0.0
    )

    report = {
        "pdf": str(pdf_path),
        "doc_name": indexed_doc.get("doc_name", pdf_path.name),
        "model": args.model,
        "judge_model": judge_model,
        "metrics": metrics,
        "evaluations": evaluations,
    }

    with open(cache_paths["report"], "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    markdown_report = build_markdown_report(pdf_path.name, args.model, judge_model, metrics, cache_paths["report"])
    with open(cache_paths["report_md"], "w", encoding="utf-8") as f:
        f.write(markdown_report)

    print("\nEvaluation complete.")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Hallucination rate: {metrics['hallucination_rate']:.2%}")
    print(f"Passes 98% / zero-hallucination gate: {metrics['passes_gate']}")
    print(f"Question set: {cache_paths['qa']}")
    print(f"Report JSON: {cache_paths['report']}")
    print(f"Report MD: {cache_paths['report_md']}")


if __name__ == "__main__":
    main()
