#!/usr/bin/env python3
"""Build DINOv2 texture index, then retrieve embeddings by product id or product name."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib import error, request

DEFAULT_INDEX = "lightbeans-texture2-dinov2-views"


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


def http_json(method: str, url: str, payload: dict | list | None = None, api_key: str | None = None) -> Any:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach Meilisearch at {url}: {exc}") from exc


def escape_filter_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    env_values = load_env(project_root / ".env")

    parser = argparse.ArgumentParser(description="Build DINOv2 index, then retrieve texture embeddings by product id or name.")
    parser.add_argument("--host", default=env_values.get("MEILI_HOST") or os.getenv("MEILI_HOST") or "http://localhost:7700", help="Meilisearch host.")
    parser.add_argument("--key", default=env_values.get("MEILI_KEY") or os.getenv("MEILI_KEY"), help="Meilisearch API key.")
    parser.add_argument("--index", default=env_values.get("MEILI_INDEX") or os.getenv("MEILI_INDEX") or DEFAULT_INDEX, help="Meilisearch index to query.")
    parser.add_argument("--product-export", type=Path, default=Path("texture/lightbeans_main_page_textures.json"), help="Product export JSON used for index generation.")
    parser.add_argument("--textures-dir", type=Path, default=Path("texture/views"), help="Texture views directory used for index generation.")
    parser.add_argument("--mapping-json", type=Path, help="Optional mapping JSON for index generation.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device used by the index generator.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size used by the index generator.")
    parser.add_argument(
        "--build-index",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build/update the index before retrieval.",
    )
    parser.add_argument("--product-id", help="Product id to match (matches product_id field).")
    parser.add_argument("--product-name", help="Product name to match.")
    parser.add_argument("--exact-name", action=argparse.BooleanOptionalAction, default=False, help="Use exact match on name field instead of full-text query.")
    parser.add_argument("--limit", type=int, default=10, help="Maximum number of matching documents to return.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()

    if not args.product_id and not args.product_name:
        raise ValueError("Provide at least one of --product-id or --product-name.")
    if args.limit <= 0:
        raise ValueError(f"--limit must be > 0, got {args.limit}.")
    if args.batch_size <= 0:
        raise ValueError(f"--batch-size must be > 0, got {args.batch_size}.")
    return args


def build_index(
    project_root: Path,
    host: str,
    api_key: str | None,
    index: str,
    product_export: Path,
    textures_dir: Path,
    mapping_json: Path | None,
    device: str,
    batch_size: int,
) -> None:
    script_path = project_root / "scripts" / "classification" / "texture_indexer_dinov2_meilisearch.py"
    cmd: list[str] = [
        sys.executable,
        str(script_path),
        "--product-export",
        str(product_export),
        "--textures-dir",
        str(textures_dir),
        "--host",
        host,
        "--index",
        index,
        "--device",
        device,
        "--batch-size",
        str(batch_size),
    ]
    if api_key:
        cmd.extend(["--key", api_key])
    if mapping_json:
        cmd.extend(["--mapping-json", str(mapping_json)])
    subprocess.run(cmd, check=True)


def query_embeddings(
    host: str,
    api_key: str | None,
    index: str,
    product_id: str | None,
    product_name: str | None,
    exact_name: bool,
    limit: int,
) -> dict[str, Any]:
    filters: list[str] = []
    query_text = ""

    if product_id:
        filters.append(f'product_id = "{escape_filter_value(product_id.strip())}"')
    if product_name:
        name = product_name.strip()
        if exact_name:
            filters.append(f'name = "{escape_filter_value(name)}"')
        else:
            query_text = name

    payload: dict[str, Any] = {
        "q": query_text,
        "limit": limit,
        "attributesToRetrieve": [
            "id",
            "product_id",
            "name",
            "image_name",
            "image_path",
            "embedding",
            "_vectors",
        ],
    }
    if filters:
        payload["filter"] = " AND ".join(filters)

    endpoint = f"{host.rstrip('/')}/indexes/{index}/search"
    result = http_json("POST", endpoint, payload, api_key)
    if not isinstance(result, dict):
        return {"hits": []}
    return result


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.build_index:
        build_index(
            project_root=project_root,
            host=args.host,
            api_key=args.key,
            index=args.index,
            product_export=args.product_export,
            textures_dir=args.textures_dir,
            mapping_json=args.mapping_json,
            device=args.device,
            batch_size=args.batch_size,
        )

    result = query_embeddings(
        host=args.host,
        api_key=args.key,
        index=args.index,
        product_id=args.product_id,
        product_name=args.product_name,
        exact_name=args.exact_name,
        limit=args.limit,
    )

    hits = result.get("hits", []) if isinstance(result, dict) else []
    output = {
        "query": {
            "index": args.index,
            "product_id": args.product_id,
            "product_name": args.product_name,
            "exact_name": args.exact_name,
            "limit": args.limit,
        },
        "count": len(hits) if isinstance(hits, list) else 0,
        "hits": hits if isinstance(hits, list) else [],
    }

    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
