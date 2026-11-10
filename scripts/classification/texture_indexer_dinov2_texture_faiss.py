#!/usr/bin/env python3
"""Build and query a local FAISS texture index using DINOv2 ViT-S/14 embeddings.

This module provides:
- DINOv2TextureEmbedder: reusable embedding API
- DINOv2TextureSearchIndex: incremental local indexing + similarity search
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

EmbeddingMode = Literal["cls", "mean", "texture_stats"]
MODEL_REPO = "facebookresearch/dinov2"
MODEL_NAME = "dinov2_vits14"
DEFAULT_INDEX_ROOT = "texture/indexes/dinov2_faiss"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def choose_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this machine.")
    return device_arg


def validate_input_size(input_size: int) -> int:
    if input_size <= 0:
        raise ValueError(f"input_size must be > 0, got {input_size}.")
    if input_size % 14 != 0:
        raise ValueError(f"input_size must be a multiple of 14 for ViT-S/14, got {input_size}.")
    return input_size


def l2_normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


class DINOv2TextureEmbedder:
    """DINOv2 texture embedder.

    Resizing/cropping choices change texture scale and local geometry statistics. Keep
    input_size and preprocessing consistent between index build and query time.
    """

    def __init__(
        self,
        embedding_mode: EmbeddingMode = "texture_stats",
        input_size: int = 224,
        device: str = "auto",
        batch_size: int = 16,
    ):
        self.embedding_mode: EmbeddingMode = embedding_mode
        self.input_size = validate_input_size(int(input_size))
        self.device = choose_device(device)
        self.batch_size = max(1, int(batch_size))
        self._transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.model = torch.hub.load(MODEL_REPO, MODEL_NAME, pretrained=True, trust_repo=True)
        self.model.to(self.device)
        self.model.eval()
        self.embedding_dim = 768 if self.embedding_mode == "texture_stats" else 384

    def _coerce_rgb_image(self, value: Image.Image | Path | str | np.ndarray) -> Image.Image:
        if isinstance(value, Image.Image):
            return value.convert("RGB")
        if isinstance(value, (Path, str)):
            with Image.open(Path(value)) as loaded:
                return loaded.convert("RGB")
        if isinstance(value, np.ndarray):
            array = np.asarray(value)
            if array.ndim == 2:
                return Image.fromarray(array.astype(np.uint8), mode="L").convert("RGB")
            if array.ndim == 3 and array.shape[2] in {3, 4}:
                mode = "RGBA" if array.shape[2] == 4 else "RGB"
                return Image.fromarray(array.astype(np.uint8), mode=mode).convert("RGB")
            raise ValueError(f"Unsupported ndarray shape for image: {array.shape}")
        raise TypeError(f"Unsupported image input type: {type(value)!r}")

    def _embed_tensor_batch(self, batch_tensor: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            features = self.model.forward_features(batch_tensor)
            patch_tokens = features["x_norm_patchtokens"]
            if self.embedding_mode == "cls":
                cls_token = features["x_norm_clstoken"]
                if cls_token.ndim == 3:
                    cls_token = cls_token[:, 0, :]
                embedding = cls_token
            elif self.embedding_mode == "mean":
                embedding = patch_tokens.mean(dim=1)
            else:
                mean = patch_tokens.mean(dim=1)
                std = patch_tokens.std(dim=1, unbiased=False)
                embedding = torch.cat([mean, std], dim=-1)
            embedding = F.normalize(embedding, p=2, dim=-1)
            return embedding

    def embed_batch(self, images: Sequence[Image.Image | Path | str | np.ndarray]) -> np.ndarray:
        if not images:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        rows: list[np.ndarray] = []
        for start in range(0, len(images), self.batch_size):
            chunk = images[start : start + self.batch_size]
            tensors: list[torch.Tensor] = []
            for value in chunk:
                rgb = self._coerce_rgb_image(value)
                try:
                    tensors.append(self._transform(rgb))
                finally:
                    rgb.close()
            batch = torch.stack(tensors, dim=0).to(self.device)
            embedded = self._embed_tensor_batch(batch)
            rows.append(embedded.detach().cpu().numpy().astype(np.float32))
        output = np.vstack(rows)
        return l2_normalize_rows(output.astype(np.float32))

    def embed_image(self, image: Image.Image | Path | str | np.ndarray) -> np.ndarray:
        batch = self.embed_batch([image])
        if batch.shape != (1, self.embedding_dim):
            raise RuntimeError(f"Unexpected embedding shape {batch.shape}, expected (1, {self.embedding_dim}).")
        return batch[0]

    @staticmethod
    def compute_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
        a = np.asarray(embedding_a, dtype=np.float32).reshape(1, -1)
        b = np.asarray(embedding_b, dtype=np.float32).reshape(1, -1)
        a = l2_normalize_rows(a)[0]
        b = l2_normalize_rows(b)[0]
        return float(np.dot(a, b))


class DINOv2TextureSearchIndex:
    def __init__(self, index_root: Path, embedder: DINOv2TextureEmbedder):
        self.index_root = Path(index_root)
        self.embedder = embedder
        self._faiss = None
        self.index = None
        self.records: list[dict[str, Any]] = []

    @property
    def mode(self) -> EmbeddingMode:
        return self.embedder.embedding_mode

    @property
    def embedding_dim(self) -> int:
        return self.embedder.embedding_dim

    @property
    def base_name(self) -> str:
        return f"{MODEL_NAME}_{self.mode}_{self.embedder.input_size}"

    @property
    def faiss_path(self) -> Path:
        return self.index_root / f"{self.base_name}.faiss"

    @property
    def metadata_path(self) -> Path:
        return self.index_root / f"{self.base_name}.json"

    def _require_faiss(self):
        if self._faiss is not None:
            return self._faiss
        try:
            import faiss  # type: ignore
        except ModuleNotFoundError as exc:
            raise RuntimeError("FAISS is required. Install with: pip install faiss-cpu") from exc
        self._faiss = faiss
        return faiss

    def _load_cached_metadata(self) -> dict[str, Any]:
        if not self.metadata_path.exists():
            return {"items": []}
        with self.metadata_path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if not isinstance(payload, dict):
            return {"items": []}
        items = payload.get("items")
        if not isinstance(items, list):
            payload["items"] = []
        return payload

    def _save_metadata(self, items: list[dict[str, Any]], textures_dir: Path) -> None:
        payload = {
            "model": MODEL_NAME,
            "embedding_mode": self.mode,
            "input_size": self.embedder.input_size,
            "embedding_dim": self.embedding_dim,
            "device": self.embedder.device,
            "textures_dir": str(textures_dir),
            "items": items,
        }
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _collect_texture_files(self, textures_dir: Path, recursive: bool) -> list[Path]:
        if recursive:
            candidates = [path for path in textures_dir.rglob("*") if path.is_file()]
        else:
            candidates = [path for path in textures_dir.glob("*") if path.is_file()]
        filtered = [path for path in candidates if path.suffix.lower() in IMAGE_EXTENSIONS]
        return sorted(filtered)

    def _metadata_lookup(self, metadata_payload: Any) -> dict[str, dict[str, Any]]:
        lookup: dict[str, dict[str, Any]] = {}
        if isinstance(metadata_payload, dict):
            for key, value in metadata_payload.items():
                if isinstance(value, dict):
                    lookup[str(key)] = value
        elif isinstance(metadata_payload, list):
            for row in metadata_payload:
                if not isinstance(row, dict):
                    continue
                keys = [row.get("image"), row.get("image_name"), row.get("filename"), row.get("path")]
                for key in keys:
                    if isinstance(key, str) and key.strip():
                        lookup[key.strip()] = row
        return lookup

    def _load_extra_metadata(self, metadata_json: Path | None) -> dict[str, dict[str, Any]]:
        if metadata_json is None or not metadata_json.exists():
            return {}
        with metadata_json.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
        return self._metadata_lookup(payload)

    def build_or_update_index(self, textures_dir: Path, metadata_json: Path | None = None, recursive: bool = True) -> dict[str, int]:
        faiss = self._require_faiss()
        textures_dir = textures_dir.resolve()
        files = self._collect_texture_files(textures_dir, recursive=recursive)
        cached_payload = self._load_cached_metadata()
        cached_items = cached_payload.get("items", [])
        cached_by_image = {str(item.get("image")): item for item in cached_items if isinstance(item, dict) and item.get("image")}
        extra_metadata = self._load_extra_metadata(metadata_json)

        items: list[dict[str, Any]] = []
        to_compute_paths: list[Path] = []
        to_compute_indices: list[int] = []
        reused = 0

        for path in files:
            rel = path.relative_to(textures_dir).as_posix()
            stat = path.stat()
            cached = cached_by_image.get(rel)
            cached_embedding = cached.get("embedding") if isinstance(cached, dict) else None
            can_reuse = (
                isinstance(cached_embedding, list)
                and len(cached_embedding) == self.embedding_dim
                and int(cached.get("size_bytes", -1)) == int(stat.st_size)
                and int(cached.get("mtime_ns", -1)) == int(stat.st_mtime_ns)
            )
            item = {
                "image": rel,
                "filename": path.name,
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "metadata": {},
                "embedding": None,
            }
            metadata_match = extra_metadata.get(rel) or extra_metadata.get(path.name) or extra_metadata.get(path.stem)
            if isinstance(metadata_match, dict):
                item["metadata"] = metadata_match
            if can_reuse:
                item["embedding"] = [float(v) for v in cached_embedding]
                reused += 1
            else:
                to_compute_paths.append(path)
                to_compute_indices.append(len(items))
            items.append(item)

        if to_compute_paths:
            embeddings = self.embedder.embed_batch(to_compute_paths)
            for idx, vec in zip(to_compute_indices, embeddings, strict=True):
                items[idx]["embedding"] = [float(v) for v in vec.tolist()]

        index = faiss.IndexFlatIP(self.embedding_dim)
        if items:
            vectors = np.asarray([item["embedding"] for item in items if isinstance(item.get("embedding"), list)], dtype=np.float32)
            if vectors.shape[0] != len(items):
                raise RuntimeError("Index build failed: one or more embeddings are missing.")
            if vectors.shape[1] != self.embedding_dim:
                raise RuntimeError(f"Index build failed: embedding dimension mismatch ({vectors.shape[1]} != {self.embedding_dim}).")
            index.add(vectors.astype(np.float32))
        self.index_root.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.faiss_path))
        self.index = index
        self.records = items
        self._save_metadata(items, textures_dir=textures_dir)

        return {
            "total": len(items),
            "reused": reused,
            "computed": len(to_compute_paths),
            "removed": max(0, len(cached_by_image) - len(items)),
        }

    def load(self) -> None:
        faiss = self._require_faiss()
        if not self.faiss_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {self.faiss_path}")
        payload = self._load_cached_metadata()
        items = payload.get("items", [])
        if not isinstance(items, list):
            raise RuntimeError(f"Invalid metadata format: {self.metadata_path}")
        dim = int(payload.get("embedding_dim") or 0)
        mode = str(payload.get("embedding_mode") or "")
        if dim and dim != self.embedding_dim:
            raise RuntimeError(f"Embedding dimension mismatch in metadata ({dim} != {self.embedding_dim}).")
        if mode and mode != self.mode:
            raise RuntimeError(f"Embedding mode mismatch in metadata ({mode} != {self.mode}).")

        self.index = faiss.read_index(str(self.faiss_path))
        self.records = items
        if self.index.ntotal != len(self.records):
            raise RuntimeError(
                f"Index/metadata mismatch: index contains {self.index.ntotal} vectors but metadata has {len(self.records)} records."
            )

    def search_by_embedding(self, embedding: np.ndarray, top_k: int = 10) -> list[dict[str, Any]]:
        if top_k <= 0:
            raise ValueError(f"top_k must be > 0, got {top_k}.")
        if self.index is None:
            self.load()
        if self.index is None:
            raise RuntimeError("FAISS index is not loaded.")

        query = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        if query.shape[1] != self.embedding_dim:
            raise ValueError(f"Query embedding dimension {query.shape[1]} does not match index dimension {self.embedding_dim}.")
        query = l2_normalize_rows(query)
        distances, neighbors = self.index.search(query, min(top_k, len(self.records)))

        results: list[dict[str, Any]] = []
        for rank, (idx, score) in enumerate(zip(neighbors[0], distances[0]), start=1):
            if idx < 0:
                continue
            record = self.records[int(idx)]
            merged = dict(record.get("metadata") or {})
            merged["image"] = record.get("image")
            merged["filename"] = record.get("filename")
            results.append(
                {
                    "rank": rank,
                    "image": merged.get("image"),
                    "similarity": float(score),
                    "metadata": merged,
                }
            )
        return results

    def search_similar(self, query_image: Image.Image | Path | str | np.ndarray, top_k: int = 10) -> list[dict[str, Any]]:
        query_embedding = self.embedder.embed_image(query_image)
        return self.search_by_embedding(query_embedding, top_k=top_k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/search a DINOv2 texture FAISS index.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--index-root", type=Path, default=Path(DEFAULT_INDEX_ROOT), help="Directory that stores mode-specific FAISS index files.")
    common.add_argument("--embedding-mode", choices=["cls", "mean", "texture_stats"], default="texture_stats", help="Embedding strategy.")
    common.add_argument("--input-size", type=int, default=224, help="Input image size (must be multiple of 14).")
    common.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Inference device.")
    common.add_argument("--batch-size", type=int, default=16, help="Embedding batch size.")

    index_parser = subparsers.add_parser("index", parents=[common], help="Build or update the FAISS texture index.")
    index_parser.add_argument("--textures-dir", type=Path, required=True, help="Directory containing texture images.")
    index_parser.add_argument("--metadata-json", type=Path, help="Optional JSON file with per-image metadata.")
    index_parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True, help="Recursively scan --textures-dir.")

    search_parser = subparsers.add_parser("search", parents=[common], help="Search the FAISS texture index.")
    search_parser.add_argument("--query-image", type=Path, required=True, help="Query image file.")
    search_parser.add_argument("--top-k", type=int, default=10, help="Top K neighbors to return.")
    search_parser.add_argument("--output", type=Path, help="Optional output JSON file.")

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    embedder = DINOv2TextureEmbedder(
        embedding_mode=args.embedding_mode,
        input_size=args.input_size,
        device=args.device,
        batch_size=args.batch_size,
    )
    search_index = DINOv2TextureSearchIndex(index_root=args.index_root, embedder=embedder)

    if args.command == "index":
        if not args.textures_dir.exists():
            raise FileNotFoundError(f"Textures directory not found: {args.textures_dir}")
        stats = search_index.build_or_update_index(
            textures_dir=args.textures_dir,
            metadata_json=args.metadata_json,
            recursive=bool(args.recursive),
        )
        print(json.dumps({"index_root": str(args.index_root), "embedding_mode": args.embedding_mode, **stats}, ensure_ascii=False, indent=2))
        return 0

    if not args.query_image.exists():
        raise FileNotFoundError(f"Query image not found: {args.query_image}")
    search_index.load()
    results = search_index.search_similar(args.query_image, top_k=args.top_k)
    payload = {
        "query_image": str(args.query_image),
        "embedding_mode": args.embedding_mode,
        "embedding_dim": embedder.embedding_dim,
        "results": results,
    }
    rendered = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
