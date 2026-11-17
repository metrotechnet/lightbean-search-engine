#!/usr/bin/env python3
"""Evaluate Meilisearch retrieval quality using the image-prediction outputs and product ground truth."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
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
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request_obj = request.Request(url, data=data, headers=headers, method=method)
    try:
        with request.urlopen(request_obj, timeout=60) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Could not reach Meilisearch at {url}: {exc}") from exc


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(value).strip()] if str(value).strip() else []


def get_product_records(path: Path) -> dict[str, dict[str, Any]]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    records = {}
    for record in data:
        if isinstance(record, dict):
            product_id = record.get("id")
            if product_id is not None:
                records[str(product_id)] = record
    return records


def build_query_from_predictions(predictions: dict[str, Any]) -> str:
    selected = []
    for field in ["location", "application", "productCategory", "materialType", "color"]:
        value = predictions.get(field, {}).get("best") if isinstance(predictions.get(field), dict) else None
        if value:
            selected.append(str(value))
    return " ".join(selected)


def search_meilisearch(host: str, api_key: str | None, index: str, query: str, limit: int = 10) -> dict[str, Any]:
    payload = {
        "q": query,
        "limit": limit,
        "attributesToRetrieve": ["product_id", "location", "application", "product_category", "material_type", "color"],
    }
    endpoint = f"{host.rstrip('/')}/indexes/{index}/search"
    return http_json("POST", endpoint, payload, api_key)


def compute_summary(records: list[dict[str, Any]]) -> dict[str, float | int]:
    if not records:
        return {
            "total": 0,
            "top1_accuracy": 0.0,
            "top3_accuracy": 0.0,
            "top10_accuracy": 0.0,
            "mrr": 0.0,
            "attribute_accuracy": 0.0,
        }

    total = len(records)
    top1 = sum(1 for r in records if r["top1"]) / total
    top3 = sum(1 for r in records if r["top3"]) / total
    top10 = sum(1 for r in records if r["top10"]) / total
    mrr = sum(r["mrr"] for r in records) / total
    attribute_accuracy = sum(r["attribute_accuracy"] for r in records) / total

    return {
        "total": total,
        "top1_accuracy": round(top1, 6),
        "top3_accuracy": round(top3, 6),
        "top10_accuracy": round(top10, 6),
        "mrr": round(mrr, 6),
        "attribute_accuracy": round(attribute_accuracy, 6),
    }


def evaluate_image(
    filename: str,
    prediction: dict[str, Any],
    ground_truth: dict[str, str],
    product_records: dict[str, dict[str, Any]],
    host: str,
    api_key: str | None,
    index: str,
    limit: int,
) -> dict[str, Any]:
    actual_product_id = ground_truth.get(filename)
    if not actual_product_id:
        return {
            "filename": filename,
            "actual_product_id": None,
            "skipped": True,
            "reason": "missing_ground_truth",
        }

    query = build_query_from_predictions(prediction.get("predicted", {}))
    if not query:
        return {
            "filename": filename,
            "actual_product_id": actual_product_id,
            "skipped": True,
            "reason": "empty_prediction_query",
        }

    response = search_meilisearch(host, api_key, index, query, limit=limit)
    hits = response.get("hits", [])
    result_ids = []
    for hit in hits:
        result_id = hit.get("product_id") or hit.get("id")
        if result_id is not None:
            result_ids.append(str(result_id))

    actual_record = product_records.get(actual_product_id)
    if actual_record is None:
        return {
            "filename": filename,
            "actual_product_id": actual_product_id,
            "skipped": True,
            "reason": "missing_ground_truth_record",
        }

    top1 = bool(result_ids and result_ids[0] == actual_product_id)
    top3 = actual_product_id in result_ids[:3]
    top10 = actual_product_id in result_ids[:10]

    mrr = 0.0
    for idx, result_id in enumerate(result_ids, start=1):
        if result_id == actual_product_id:
            mrr = 1.0 / idx
            break

    attribute_accuracy = 0.0
    field_names = ["location", "application", "productCategory", "materialType", "color"]
    if prediction.get("predicted"):
        score_total = 0
        score_count = 0
        for field_name in field_names:
            predicted_best = prediction["predicted"].get(field_name, {}).get("best")
            if predicted_best is None:
                continue
            score_count += 1
            expected_values = set(as_list(actual_record.get(field_name)))
            if predicted_best in expected_values:
                score_total += 1
        if score_count:
            attribute_accuracy = score_total / score_count

    return {
        "filename": filename,
        "actual_product_id": actual_product_id,
        "query": query,
        "search_result_ids": result_ids[:10],
        "top1": top1,
        "top3": top3,
        "top10": top10,
        "mrr": mrr,
        "attribute_accuracy": attribute_accuracy,
        "skipped": False,
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    env_values = load_env(project_root / ".env")

    parser = argparse.ArgumentParser(description="Evaluate Meilisearch retrieval quality against the image prediction file and product ground truth.")
    parser.add_argument("--predictions", default="SigLIP2_image_predictions.json", help="Path to the image predictions JSON file")
    parser.add_argument("--ground-truth", default="ground_truth.json", help="Path to the filename-to-product-id JSON mapping")
    parser.add_argument("--product-export", default="data/platform-products-production-v2-en.json", help="Path to the original product export used to derive ground truth")
    parser.add_argument("--host", default=env_values.get("MEILI_HOST") or os.getenv("MEILI_HOST"), help="Meilisearch host URL (defaults to MEILI_HOST from .env or environment)")
    parser.add_argument("--key", default=env_values.get("MEILI_KEY") or os.getenv("MEILI_KEY"), help="Meilisearch API key (defaults to MEILI_KEY from .env or environment)")
    parser.add_argument("--index", default=env_values.get("MEILI_INDEX") or os.getenv("MEILI_INDEX", "lightbean-index"), help="Meilisearch index name")
    parser.add_argument("--limit", type=int, default=10, help="How many Meilisearch hits to inspect per query")
    parser.add_argument("--max-images", type=int, default=0, help="Process only the first N images (0 = all)")
    parser.add_argument("--output", default="evaluation_results.json", help="Path to save the evaluation results JSON")
    args = parser.parse_args()

    if not args.host:
        raise ValueError("Missing Meilisearch host. Set MEILI_HOST or pass --host.")
    if not args.key:
        raise ValueError("Missing Meilisearch key. Set MEILI_KEY or pass --key.")

    predictions_path = Path(args.predictions)
    ground_truth_path = Path(args.ground_truth)
    product_export_path = Path(args.product_export)

    with predictions_path.open("r", encoding="utf-8") as fh:
        predictions = json.load(fh)
    if not isinstance(predictions, dict):
        raise ValueError(f"Expected a dictionary from {predictions_path}")

    with ground_truth_path.open("r", encoding="utf-8") as fh:
        ground_truth = json.load(fh)
    if not isinstance(ground_truth, dict):
        raise ValueError(f"Expected a dictionary from {ground_truth_path}")

    product_records = get_product_records(product_export_path)

    items = list(predictions.items())
    if args.max_images > 0:
        items = items[: args.max_images]

    results = []
    for filename, prediction in items:
        result = evaluate_image(
            filename=filename,
            prediction=prediction,
            ground_truth=ground_truth,
            product_records=product_records,
            host=args.host,
            api_key=args.key,
            index=args.index,
            limit=args.limit,
        )
        results.append(result)

    summary = compute_summary([item for item in results if not item.get("skipped", False)])
    payload = {
        "host": args.host,
        "index": args.index,
        "limit": args.limit,
        "summary": summary,
        "results": results,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved evaluation results to {output_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
