#!/usr/bin/env python3
"""Send every question to the Meilisearch chat API and save the generated answers."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib import request, error


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


def load_questions(path: str) -> list[dict[str, Any]]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"{file_path} must contain a JSON array of question documents")
    return data


def extract_streamed_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    pieces: list[str] = []
    for choice in choices:
        delta = choice.get("delta") or {}
        if isinstance(delta, dict) and "content" in delta and delta["content"]:
            pieces.append(str(delta["content"]))
        message = choice.get("message") or {}
        if isinstance(message, dict) and "content" in message and message["content"]:
            pieces.append(str(message["content"]))
    return "".join(pieces)


def call_chat_completion(host: str, api_key: str, workspace: str, model: str, question: str) -> str:
    url = f"{host.rstrip('/')}/chats/{workspace}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": question}
        ],
        "stream": True,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers=headers, method="POST")

    try:
        with request.urlopen(req, timeout=120) as resp:
            chunks: list[str] = []
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if not line.startswith("data:"):
                    continue
                content = line[5:].strip()
                if content == "[DONE]":
                    break
                try:
                    item = json.loads(content)
                except json.JSONDecodeError:
                    continue
                text = extract_streamed_content(item)
                if text:
                    chunks.append(text)
            return "".join(chunks).strip()
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Chat request failed with HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach Meilisearch chat endpoint at {url}: {exc}") from exc


def run_chat_for_all(host: str, api_key: str, workspace: str, model: str, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in questions:
        question_text = item.get("question_en") or item.get("question_fr") or item.get("search_text") or ""
        answer = call_chat_completion(host, api_key, workspace, model, question_text)
        results.append(
            {
                "id": item.get("id"),
                "category_id": item.get("category_id"),
                "category_name_en": item.get("category_name_en"),
                "category_name_fr": item.get("category_name_fr"),
                "question_fr": item.get("question_fr"),
                "question_en": item.get("question_en"),
                "query": question_text,
                "answer": answer,
            }
        )
    return results


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    env_path = project_root / ".env"
    env_values = load_env(env_path)

    default_input = project_root / "data" / "meilisearch-search-questions.json"
    default_output = project_root / "results" / "meilisearch-chat-answers.json"

    parser = argparse.ArgumentParser(description="Send all questions to the Meilisearch chat API and save the answers.")
    parser.add_argument("--host", default=os.getenv("MEILI_HOST", env_values.get("MEILI_HOST", "http://localhost:7700")), help="Meilisearch host URL")
    parser.add_argument("--api-key", default=os.getenv("MEILI_CHAT_API_KEY", env_values.get("MEILI_CHAT_API_KEY")), help="Meilisearch chat API key")
    parser.add_argument("--workspace", default=os.getenv("MEILI_CHAT_WORKSPACE", env_values.get("MEILI_CHAT_WORKSPACE")), help="Meilisearch chat workspace uid")
    parser.add_argument("--model", default=os.getenv("MEILI_CHAT_MODEL", env_values.get("MEILI_CHAT_MODEL", "gpt-4o-mini")), help="LLM model to use in the chat workspace")
    parser.add_argument("--input", default=os.getenv("INPUT_PATH", str(default_input)), help="Input JSON array file")
    parser.add_argument("--output", default=os.getenv("MEILI_CHAT_OUTPUT_PATH", env_values.get("MEILI_CHAT_OUTPUT_PATH", str(default_output))), help="Output JSON file for chat answers")
    args = parser.parse_args()

    if not args.api_key:
        raise ValueError("Missing Meilisearch chat API key. Set MEILI_CHAT_API_KEY in .env or pass --api-key.")
    if not args.workspace:
        raise ValueError("Missing Meilisearch chat workspace UID. Set MEILI_CHAT_WORKSPACE in .env or pass --workspace.")

    questions = load_questions(args.input)
    answers = run_chat_for_all(args.host, args.api_key, args.workspace, args.model, questions)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(answers, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(f"Saved {len(answers)} answers to {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
