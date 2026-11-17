#!/usr/bin/env python3
"""Create a Meilisearch index from Lightbeans product exports and support embedding-based searches.

Examples:
  python scripts/classification/meilisearch_products_manager.py create-index --host http://localhost:7700 --key <MEILI_KEY>
  python scripts/classification/meilisearch_products_manager.py search --host http://localhost:7700 --key <MEILI_KEY> --query "terracotta wall tile" --limit 10
  python scripts/classification/meilisearch_products_manager.py search --host http://localhost:7700 --key <MEILI_KEY> --query "green tile" --use-embedding --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib import error, request

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency when embeddings search is used.
    SentenceTransformer = None


FIELD_ALIASES = {
    "id": "id",
    "productName": "product_name",
    "brandName": "brand_name",
    "productSlug": "product_slug",
    "productCategory": "product_category",
    "materialType": "material_type",
    "location": "location",
    "application": "application",
    "color": "color",
    "organizationSlug": "organization_slug",
    "featured": "featured",
    "publicationState": "publication_state",
    "productPageV2": "product_page_v2",
    "previewPath": "preview_path",
    "createdAt": "created_at",
    "locale": "locale",
    "tid": "tid",
    "brandSlug": "brand_slug",
}


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
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=60) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach Meilisearch at {url}: {exc}") from exc


def load_products(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {file_path}")
    return [item for item in data if isinstance(item, dict)]


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(value).strip()] if str(value).strip() else []


def flatten_product(record: dict[str, Any]) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "id": record.get("id"),
        "product_id": record.get("id"),
        "locale": record.get("locale"),
        "location": as_list(record.get("location")),
        "application": as_list(record.get("application")),
        "product_category": as_list(record.get("productCategory")),
        "material_type": as_list(record.get("materialType")),
        "color": as_list(record.get("color")),
    }

    return doc


def ensure_index(host: str, api_key: str | None, index_name: str, primary_key: str = "product_id") -> None:
    url = f"{host.rstrip('/')}/indexes"
    payload = {"uid": index_name, "primaryKey": primary_key}
    try:
        http_json("POST", url, payload, api_key)
    except RuntimeError:
        # Meilisearch may already have the index, so we continue without failing.
        pass

    settings_url = f"{host.rstrip('/')}/indexes/{index_name}/settings"
    settings = {
        "searchableAttributes": [
            "location",
            "application",
            "product_category",
            "material_type",
            "color",
        ],
        "filterableAttributes": [
            "product_id",
            "location",
            "application",
            "product_category",
            "material_type",
            "color",
            "locale",
        ],
        "sortableAttributes": ["product_id"],
        "rankingRules": [
            "words",
            "typo",
            "proximity",
            "attribute",
            "sort",
            "exactness",
        ],
    }
    http_json("PATCH", settings_url, settings, api_key)


def upload_documents(host: str, api_key: str | None, index_name: str, documents: list[dict[str, Any]]) -> None:
    endpoint = f"{host.rstrip('/')}/indexes/{index_name}/documents"
    http_json("POST", endpoint, documents, api_key)


def create_index(host: str, api_key: str | None, index_name: str, input_path: str | Path) -> int:
    products = load_products(input_path)
    documents = [flatten_product(record) for record in products]
    ensure_index(host, api_key, index_name)
    upload_documents(host, api_key, index_name, documents)
    print(f"Uploaded {len(documents)} documents to index '{index_name}'")
    return 0


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers is required for embedding search. Install it with: pip install sentence-transformers")
    return SentenceTransformer(model_name)


def search_with_embedding(host: str, api_key: str | None, index_name: str, query: str, limit: int = 10, use_embedding: bool = True, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "q": query,
        "limit": limit,
        "attributesToRetrieve": ["*"],
    }

    if use_embedding:
        model = get_embedding_model(model_name)
        vector = model.encode(query, convert_to_numpy=True, normalize_embeddings=True).tolist()
        payload["vector"] = vector
        payload["hybrid"] = {"semanticRatio": 0.7, "embedder": "default"}

    endpoint = f"{host.rstrip('/')}/indexes/{index_name}/search"
    return http_json("POST", endpoint, payload, api_key)


def get_env_defaults() -> dict[str, str]:
    project_root = Path(__file__).resolve().parents[2]
    return load_env(project_root / ".env")


def build_parser() -> argparse.ArgumentParser:
    env_values = get_env_defaults()
    parser = argparse.ArgumentParser(description="Create and query a Meilisearch index from the Lightbeans product export.")
    parser.add_argument(
        "--host",
        default=env_values.get("MEILI_HOST") or os.getenv("MEILI_HOST") or "http://localhost:7700",
        help="Meilisearch host URL. Defaults to MEILI_HOST from .env.",
    )
    parser.add_argument(
        "--key",
        default=env_values.get("MEILI_KEY") or os.getenv("MEILI_KEY"),
        help="Meilisearch API key. Defaults to MEILI_KEY from .env.",
    )
    parser.add_argument("--index", default=env_values.get("MEILI_INDEX") or os.getenv("MEILI_INDEX") or "lightbeans-products", help="Meilisearch index name")
    subparsers = parser.add_subparsers(dest="command", required=True)

    env_values = get_env_defaults()
    create_parser = subparsers.add_parser("create-index", help="Create the Meilisearch index and upload product documents")
    create_parser.add_argument("--input", default=env_values.get("INPUT_PATH") or "data/platform-products-production-v2-en.json", help="Path to the Lightbeans product JSON export")

    search_parser = subparsers.add_parser("search", help="Search the index using text or embeddings")
    search_parser.add_argument("--query", required=True, help="Text query to search")
    search_parser.add_argument("--limit", type=int, default=10, help="Number of matches to return")
    search_parser.add_argument("--use-embedding", action="store_true", help="Generate a vector embedding for the query before searching")
    search_parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="SentenceTransformers model to use when --use-embedding is enabled")
    search_parser.add_argument("--output", help="Optional output JSON path for search results")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    env_values = load_env(project_root / ".env")
    if not args.key:
        args.key = env_values.get("MEILI_KEY") or os.getenv("MEILI_KEY")
    if not args.host:
        args.host = env_values.get("MEILI_HOST") or os.getenv("MEILI_HOST") or "http://localhost:7700"
    if not args.index:
        args.index = env_values.get("MEILI_INDEX") or os.getenv("MEILI_INDEX") or "lightbeans-products"

    if args.command == "create-index":
        return create_index(args.host, args.key, args.index, args.input)

    if args.command == "search":
        result = search_with_embedding(
            host=args.host,
            api_key=args.key,
            index_name=args.index,
            query=args.query,
            limit=args.limit,
            use_embedding=args.use_embedding,
            model_name=args.model,
        )
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(f"Saved search results to {out_path}")
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    parser.error(f"Unsupported command: {args.command}")
    return 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - top-level CLI error reporting.
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
