#!/usr/bin/env python3
"""Index SigLIP2 image embeddings in Meilisearch and evaluate similarity retrieval.

Workflow:
1) Load train/test image split (train: first occurrence per product, test: other occurrences).
2) Generate normalized SigLIP2 embeddings for train images.
3) Index train embeddings + product metadata into Meilisearch.
4) Query test image embeddings against Meilisearch vector search.
5) Report Recall@1/5/10 and top-10 visually similar products with cosine scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from meilisearch_products import http_json, load_env
from siglip2_predict_attributes import MODEL_NAME


@dataclass(frozen=True)
class ImageRecord:
    image_name: str
    image_path: Path
    product_id: str
    metadata: dict[str, Any]


def make_document_id(image_name: str) -> str:
    digest = hashlib.sha1(image_name.encode("utf-8")).hexdigest()
    return f"img_{digest}"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [str(value).strip()] if str(value).strip() else []


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return device_arg


def build_product_lookup(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        product_id = record.get("id")
        if product_id is None:
            continue
        lookup[str(product_id)] = record
    return lookup


def build_metadata(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "product_name": record.get("productName"),
        "brand_name": record.get("brandName"),
        "location": as_list(record.get("location")),
        "application": as_list(record.get("application")),
        "product_category": as_list(record.get("productCategory")),
        "material_type": as_list(record.get("materialType")),
        "color": as_list(record.get("color")),
        "locale": record.get("locale"),
    }


def collect_split_records(
    image_names: list[str],
    images_dir: Path,
    ground_truth: dict[str, str],
    product_lookup: dict[str, dict[str, Any]],
) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    for image_name in image_names:
        product_id = ground_truth.get(image_name)
        if not product_id:
            raise ValueError(f"Image '{image_name}' is missing from ground truth mapping.")
        product_id = str(product_id)
        product_record = product_lookup.get(product_id)
        if product_record is None:
            raise ValueError(f"Product '{product_id}' not found in product export for image '{image_name}'.")

        image_path = images_dir / image_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        records.append(
            ImageRecord(
                image_name=image_name,
                image_path=image_path,
                product_id=product_id,
                metadata=build_metadata(product_record),
            )
        )
    return records


def extract_image_embeddings(outputs: Any) -> torch.Tensor:
    image_embeds = getattr(outputs, "image_embeds", None)
    if image_embeds is not None:
        return image_embeds

    pooler_output = getattr(outputs, "pooler_output", None)
    if pooler_output is not None:
        return pooler_output

    last_hidden_state = getattr(outputs, "last_hidden_state", None)
    if last_hidden_state is not None and last_hidden_state.ndim == 3:
        return last_hidden_state[:, 0, :]

    raise RuntimeError("SigLIP2 output does not expose a usable image embedding.")


def compute_embeddings(
    records: list[ImageRecord],
    processor: AutoProcessor,
    model: AutoModel,
    device: str,
    batch_size: int,
) -> dict[str, list[float]]:
    embeddings: dict[str, list[float]] = {}
    model.eval()

    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        images: list[Image.Image] = []
        for record in batch:
            image = Image.open(record.image_path).convert("RGB")
            images.append(image)

        try:
            image_inputs = processor(images=images, return_tensors="pt")
            image_inputs = {key: value.to(device) for key, value in image_inputs.items()}

            with torch.no_grad():
                if hasattr(model, "get_image_features") and "pixel_values" in image_inputs:
                    features = model.get_image_features(pixel_values=image_inputs["pixel_values"])
                    if isinstance(features, torch.Tensor):
                        image_embeds = features
                    else:
                        image_embeds = extract_image_embeddings(features)
                else:
                    fallback_inputs = processor(
                        images=images,
                        text=["image"] * len(images),
                        return_tensors="pt",
                        padding=True,
                    )
                    fallback_inputs = {key: value.to(device) for key, value in fallback_inputs.items()}
                    outputs = model(**fallback_inputs)
                    image_embeds = extract_image_embeddings(outputs)

                image_embeds = torch.nn.functional.normalize(image_embeds, p=2, dim=-1)

            cpu_embeddings = image_embeds.detach().cpu().tolist()
            for record, embedding in zip(batch, cpu_embeddings):
                embeddings[record.image_name] = [float(value) for value in embedding]
        finally:
            for image in images:
                image.close()

    return embeddings


def wait_for_task(host: str, api_key: str | None, task_uid: int, timeout_seconds: int = 600, poll_seconds: float = 0.5) -> None:
    start = time.time()
    endpoint = f"{host.rstrip('/')}/tasks/{task_uid}"
    while True:
        task = http_json("GET", endpoint, api_key=api_key)
        status = task.get("status")
        if status == "succeeded":
            return
        if status == "failed":
            error_block = task.get("error")
            raise RuntimeError(f"Meilisearch task {task_uid} failed: {error_block}")

        if time.time() - start > timeout_seconds:
            raise RuntimeError(f"Timed out waiting for Meilisearch task {task_uid} after {timeout_seconds}s.")

        time.sleep(poll_seconds)


def extract_task_uid(response: dict[str, Any], operation_name: str) -> int:
    task_uid = response.get("taskUid")
    if not isinstance(task_uid, int):
        raise RuntimeError(f"{operation_name} did not return a valid taskUid: {response}")
    return task_uid


def ensure_similarity_index(host: str, api_key: str | None, index_name: str, vector_dimensions: int) -> None:
    create_endpoint = f"{host.rstrip('/')}/indexes"
    create_payload = {"uid": index_name, "primaryKey": "id"}

    try:
        create_response = http_json("POST", create_endpoint, create_payload, api_key)
        wait_for_task(host, api_key, extract_task_uid(create_response, "create-index"))
    except RuntimeError as exc:
        if "index_already_exists" not in str(exc):
            raise

    settings_payload = {
        "filterableAttributes": ["product_id", "location", "application", "product_category", "material_type", "color", "locale"],
        "searchableAttributes": ["product_name", "brand_name", "location", "application", "product_category", "material_type", "color"],
        "embedders": {
            "default": {
                "source": "userProvided",
                "dimensions": vector_dimensions,
            }
        },
    }
    settings_endpoint = f"{host.rstrip('/')}/indexes/{index_name}/settings"
    settings_response = http_json("PATCH", settings_endpoint, settings_payload, api_key)
    wait_for_task(host, api_key, extract_task_uid(settings_response, "update-settings"))


def clear_index_documents(host: str, api_key: str | None, index_name: str) -> None:
    endpoint = f"{host.rstrip('/')}/indexes/{index_name}/documents"
    response = http_json("DELETE", endpoint, api_key=api_key)
    wait_for_task(host, api_key, extract_task_uid(response, "clear-documents"))


def index_training_embeddings(
    host: str,
    api_key: str | None,
    index_name: str,
    train_records: list[ImageRecord],
    train_embeddings: dict[str, list[float]],
    batch_size: int,
) -> dict[str, list[float]]:
    if not train_records:
        raise ValueError("No training records to index.")

    vector_dimensions = len(train_embeddings[train_records[0].image_name])
    ensure_similarity_index(host, api_key, index_name, vector_dimensions)
    clear_index_documents(host, api_key, index_name)

    embedding_lookup_by_doc_id: dict[str, list[float]] = {}
    endpoint = f"{host.rstrip('/')}/indexes/{index_name}/documents"

    for start in range(0, len(train_records), batch_size):
        batch = train_records[start : start + batch_size]
        documents: list[dict[str, Any]] = []

        for record in batch:
            embedding = train_embeddings.get(record.image_name)
            if embedding is None:
                raise RuntimeError(f"Missing embedding for train image '{record.image_name}'.")

            doc_id = make_document_id(record.image_name)
            embedding_lookup_by_doc_id[doc_id] = embedding

            document = {
                "id": doc_id,
                "image_name": record.image_name,
                "product_id": record.product_id,
                **record.metadata,
                "_vectors": {"default": embedding},
            }
            documents.append(document)

        response = http_json("POST", endpoint, documents, api_key)
        wait_for_task(host, api_key, extract_task_uid(response, "index-documents"))

    return embedding_lookup_by_doc_id


def vector_search(
    host: str,
    api_key: str | None,
    index_name: str,
    query_embedding: list[float],
    limit: int = 10,
) -> list[dict[str, Any]]:
    endpoint = f"{host.rstrip('/')}/indexes/{index_name}/search"
    payload = {
        "q": "",
        "vector": query_embedding,
        "hybrid": {"embedder": "default", "semanticRatio": 1.0},
        "limit": limit,
        "showRankingScore": True,
        "attributesToRetrieve": [
            "id",
            "image_name",
            "product_id",
            "product_name",
            "brand_name",
            "location",
            "application",
            "product_category",
            "material_type",
            "color",
            "locale",
        ],
    }
    response = http_json("POST", endpoint, payload, api_key)
    hits = response.get("hits")
    if not isinstance(hits, list):
        raise RuntimeError(f"Invalid search response from Meilisearch: {response}")
    return hits


def cosine_from_normalized_vectors(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embedding dimensions do not match for cosine computation.")
    return float(sum(a * b for a, b in zip(left, right)))


def evaluate_similarity(
    host: str,
    api_key: str | None,
    index_name: str,
    test_records: list[ImageRecord],
    test_embeddings: dict[str, list[float]],
    indexed_embeddings_by_doc_id: dict[str, list[float]],
) -> dict[str, Any]:
    evaluated = 0
    recall1 = 0
    recall5 = 0
    recall10 = 0
    per_query_results: list[dict[str, Any]] = []

    for record in test_records:
        query_embedding = test_embeddings.get(record.image_name)
        if query_embedding is None:
            raise RuntimeError(f"Missing embedding for test image '{record.image_name}'.")

        hits = vector_search(host, api_key, index_name, query_embedding, limit=10)
        top10: list[dict[str, Any]] = []
        predicted_product_ids: list[str] = []

        for rank, hit in enumerate(hits, start=1):
            doc_id = str(hit.get("id", ""))
            candidate_embedding = indexed_embeddings_by_doc_id.get(doc_id)
            if candidate_embedding is None:
                raise RuntimeError(f"Missing indexed embedding for retrieved document id '{doc_id}'.")

            cosine_similarity = cosine_from_normalized_vectors(query_embedding, candidate_embedding)
            product_id = str(hit.get("product_id", ""))
            predicted_product_ids.append(product_id)

            top10.append(
                {
                    "rank": rank,
                    "image_name": hit.get("image_name"),
                    "product_id": product_id,
                    "product_name": hit.get("product_name"),
                    "brand_name": hit.get("brand_name"),
                    "similarity": round(cosine_similarity, 6),
                    "meili_ranking_score": hit.get("_rankingScore"),
                }
            )

        evaluated += 1
        if record.product_id in predicted_product_ids[:1]:
            recall1 += 1
        if record.product_id in predicted_product_ids[:5]:
            recall5 += 1
        if record.product_id in predicted_product_ids[:10]:
            recall10 += 1

        per_query_results.append(
            {
                "query_image": record.image_name,
                "query_product_id": record.product_id,
                "top10": top10,
            }
        )

    if evaluated == 0:
        raise RuntimeError("No test records were evaluated.")

    metrics = {
        "evaluated_queries": evaluated,
        "recall_at_1": round(recall1 / evaluated, 6),
        "recall_at_5": round(recall5 / evaluated, 6),
        "recall_at_10": round(recall10 / evaluated, 6),
    }
    return {"metrics": metrics, "queries": per_query_results}


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent.parent
    env_values = load_env(project_root / ".env")

    parser = argparse.ArgumentParser(
        description="Build SigLIP2 image-similarity index in Meilisearch and evaluate retrieval on test split."
    )
    parser.add_argument("--split", type=Path, default=Path("results/image_split_first_occurrence.json"), help="Path to train/test split JSON.")
    parser.add_argument("--images-dir", type=Path, default=Path("images"), help="Directory containing local image files.")
    parser.add_argument("--ground-truth", type=Path, default=Path("ground_truth.json"), help="Path to filename->product_id JSON.")
    parser.add_argument("--product-export", type=Path, default=Path("data/platform-products-production-v2-en.json"), help="Path to product export JSON.")
    parser.add_argument("--host", default=env_values.get("MEILI_HOST") or os.getenv("MEILI_HOST") or "http://localhost:7700", help="Meilisearch host URL.")
    parser.add_argument("--key", default=env_values.get("MEILI_KEY") or os.getenv("MEILI_KEY"), help="Meilisearch API key.")
    parser.add_argument("--index", default="lightbeans-siglip2-image-similarity", help="Meilisearch index name for image vectors.")
    parser.add_argument("--batch-size", type=int, default=16, help="Image embedding batch size.")
    parser.add_argument("--index-batch-size", type=int, default=100, help="Document indexing batch size.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device for SigLIP2.")
    parser.add_argument("--max-test-images", type=int, default=0, help="Optionally limit number of test images for quicker runs (0 = all).")
    parser.add_argument("--output", type=Path, default=Path("results/siglip2_image_similarity_eval.json"), help="Output JSON report path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent

    split = load_json(project_root / args.split)
    if not isinstance(split, dict):
        raise ValueError(f"Expected split JSON object in {args.split}")
    train_names = split.get("train")
    test_names = split.get("test")
    if not isinstance(train_names, list) or not isinstance(test_names, list):
        raise ValueError("Split JSON must contain 'train' and 'test' lists.")

    ground_truth_data = load_json(project_root / args.ground_truth)
    if not isinstance(ground_truth_data, dict):
        raise ValueError(f"Expected ground-truth JSON object in {args.ground_truth}")
    ground_truth = {str(name): str(product_id) for name, product_id in ground_truth_data.items()}

    product_export = load_json(project_root / args.product_export)
    if not isinstance(product_export, list):
        raise ValueError(f"Expected product export JSON list in {args.product_export}")
    product_lookup = build_product_lookup(product_export)

    images_dir = project_root / args.images_dir
    train_records = collect_split_records([str(name) for name in train_names], images_dir, ground_truth, product_lookup)
    test_input_names = [str(name) for name in test_names]
    if args.max_test_images > 0:
        test_input_names = test_input_names[: args.max_test_images]
    test_records = collect_split_records(test_input_names, images_dir, ground_truth, product_lookup)

    device = choose_device(args.device)
    print(f"Using device: {device}")
    print(f"Loading SigLIP2 model: {MODEL_NAME}")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    print(f"Computing embeddings for {len(train_records)} train images...")
    train_embeddings = compute_embeddings(train_records, processor, model, device, args.batch_size)
    print(f"Computing embeddings for {len(test_records)} test images...")
    test_embeddings = compute_embeddings(test_records, processor, model, device, args.batch_size)

    print(f"Indexing train embeddings into Meilisearch index '{args.index}'...")
    indexed_embeddings = index_training_embeddings(
        host=args.host,
        api_key=args.key,
        index_name=args.index,
        train_records=train_records,
        train_embeddings=train_embeddings,
        batch_size=args.index_batch_size,
    )

    print("Evaluating similarity retrieval on test split...")
    evaluation = evaluate_similarity(
        host=args.host,
        api_key=args.key,
        index_name=args.index,
        test_records=test_records,
        test_embeddings=test_embeddings,
        indexed_embeddings_by_doc_id=indexed_embeddings,
    )

    report = {
        "model": MODEL_NAME,
        "device": device,
        "index": args.index,
        "train_images_indexed": len(train_records),
        "test_images_evaluated": len(test_records),
        **evaluation,
    }

    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    metrics = report["metrics"]
    print(
        "Evaluation complete: "
        f"Recall@1={metrics['recall_at_1']}, "
        f"Recall@5={metrics['recall_at_5']}, "
        f"Recall@10={metrics['recall_at_10']}"
    )
    print(f"Saved report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
