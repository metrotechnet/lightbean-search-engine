#!/usr/bin/env python3
"""Query every question in the Meilisearch dataset and save the Meilisearch hits to JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib import error, request


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def http_json(method: str, url: str, payload: dict | None = None, api_key: str | None = None):
    data = None
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")

    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach Meilisearch at {url}: {exc}") from exc


def load_questions(path: str):
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    if not isinstance(data, list):
        raise ValueError(f"{file_path} must contain a JSON array of question documents")
    return data


def ensure_index(host: str, api_key: str | None, index_name: str, documents: list[dict]):
    endpoint = f"{host.rstrip('/')}/indexes/{index_name}/documents"
    try:
        http_json("POST", endpoint, documents, api_key)
    except RuntimeError as exc:
        print(f"Warning: document upload skipped or failed: {exc}", file=sys.stderr)


def run_searches(host: str, api_key: str | None, index_name: str, questions: list[dict], limit: int):
    results = []
    for question in questions:
        query = question.get("search_text") or question.get("question_en") or question.get("question_fr") or ""
        payload = {"q": query, "limit": limit}
        endpoint = f"{host.rstrip('/')}/indexes/{index_name}/search"
        response = http_json("POST", endpoint, payload, api_key)

        results.append(
            {
                "id": question.get("id"),
                "category_id": question.get("category_id"),
                "category_name_en": question.get("category_name_en"),
                "category_name_fr": question.get("category_name_fr"),
                "question_fr": question.get("question_fr"),
                "question_en": question.get("question_en"),
                "query": query,
                "estimated_total_hits": response.get("estimatedTotalHits"),
                "total_hits": response.get("hits", []).__len__(),
                "hits": response.get("hits", []),
            }
        )
    return results


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    env_path = project_root / ".env"
    env_values = load_env(env_path)

    default_input = project_root / "data" / "meilisearch-search-questions.json"
    default_output = project_root / "results" / "meilisearch-search-results.json"

    parser = argparse.ArgumentParser(description="Send all questions in a JSON file to Meilisearch and save the results.")
    parser.add_argument("--host", default=os.getenv("MEILI_HOST", env_values.get("MEILI_HOST", "http://localhost:7700")), help="Meilisearch host URL")
    parser.add_argument("--key", default=os.getenv("MEILI_KEY", env_values.get("MEILI_KEY")), help="Optional Meilisearch API key")
    parser.add_argument("--index", default=os.getenv("MEILI_INDEX", env_values.get("MEILI_INDEX", "search-questions")), help="Name of the Meilisearch index to query")
    parser.add_argument("--input", default=os.getenv("INPUT_PATH", str(default_input)), help="Input JSON array file")
    parser.add_argument("--output", default=os.getenv("OUTPUT_PATH", str(default_output)), help="Output JSON file for search results")
    parser.add_argument("--limit", type=int, default=int(os.getenv("QUERY_LIMIT", env_values.get("QUERY_LIMIT", "10"))), help="Number of matches to include per search")
    parser.add_argument("--no-upload", action="store_true", help="Skip uploading the documents before search")
    args = parser.parse_args()

    if os.getenv("UPLOAD", env_values.get("UPLOAD", "true")).lower() in {"1", "true", "yes", "y"} and not args.no_upload:
        upload_enabled = True
    else:
        upload_enabled = False

    questions = load_questions(args.input)

    if upload_enabled:
        ensure_index(args.host, args.key, args.index, questions)

    results = run_searches(args.host, args.key, args.index, questions, args.limit)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"Saved {len(results)} search result sets to {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
