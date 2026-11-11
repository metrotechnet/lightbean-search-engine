# Lightbean Search Engine

This project evaluates product discovery and retrieval using Meilisearch and a chat-based workflow for Lightbeans materials.

## Contents
- `data/` — datasets and prompt configuration
- `scripts/crawler/` — crawling and download scripts
- `scripts/classification/` — indexing, feature extraction, prediction, and evaluation scripts
- `scripts/segmentation/` — segmentation and masked-region matching scripts
- `scripts/viewer/` — visualization and interactive viewer scripts
- `results/` — generated outputs from search and chat runs
- `start-search.ps1` and `start-chat.ps1` — launcher scripts for local execution

## Setup
1. Copy `.env.example` if present or create a local `.env` file with your Meilisearch credentials.
2. Install Python dependencies:
   `pip install python-dotenv python-docx`
3. Run the scripts from the project root.

## Notes
- `.env` is intentionally ignored and should never be committed to source control.
- The project is designed for local experimentation and evaluation of search relevance and chat responses.

## Useful scripts
- Generate scale + rotation texture views (max-fit square after transform) and a contact sheet:
  - Default batch mode (all images in `texture/main_page` → `texture/views`):
    `python scripts/classification/generate_views.py`
  - Default uses scales `0.5,1.0,1.5,2.0` and rotations `0,45,90,135,180`.
  - Single image mode:
    `python scripts/classification/generate_views.py --input texture.jpg --output ./views`
- Build ground truth mapping:
  `python scripts/classification/ground_truth_generator.py`
- Split downloaded images into train/test (train = first occurrence per product, test = remaining occurrences):
  `python scripts/classification/image_split_generator.py --ground-truth ground_truth.json --images-dir images --output image_split_first_occurrence.json`
- Build SigLIP2 image-similarity index in Meilisearch and evaluate retrieval on test images:
  `python scripts/classification/image_similarity_indexer_siglip2_meilisearch.py --split results/image_split_first_occurrence.json --images-dir images --ground-truth ground_truth.json --product-export data/platform-products-production-v2-en.json --index lightbeans-siglip2-image-similarity --output results/siglip2_image_similarity_eval.json`
- Segment images into regions with Meta SAM (SAM 3 checkpoint by default):
  `python scripts/segment_images_sam3.py --images-dir images --output-dir results/sam3_segments --batch-size 4 --device auto`
- Segment all images in `images/` with Meta SAM 2:
  `python scripts/segmentation/sam2_image_segmenter.py --images-dir images --output-dir results/sam2_segments --batch-size 4 --device auto`
- Visualize SAM2 segmented regions as semi-transparent overlays with Previous/Next navigation:
  `python scripts/viewer/sam2_segments_viewer.py --metadata results/sam2_segments/segments_metadata.json --alpha 110`
  - In the UI Method selector, `hsi_dinov2` runs a two-step search: HSI prefilter (top 100), then DINOv2 similarity scoring per candidate; final ranking uses `hsi_score * dinov2_score`. Query-time DINOv2 scoring uses a single resized crop (`--hsi-dinov2-size`, default `224`).
  - In the UI Method selector, `dinov2_texture_faiss` uses a local FAISS index built from DINOv2 ViT-S/14 patch-token texture statistics.
- Download texture images from `previewPath` and create file-to-product mapping:
  `python scripts/crawler/texture_images_downloader.py data/platform-products-production-v2-en.json --output-dir texture --mapping-output texture/texture_product_mapping.json`
- Build a DINOv2 texture embedding index in Meilisearch with product features and vector embeddings:
  `python scripts/classification/texture_indexer_dinov2_meilisearch.py --product-export texture/lightbeans_main_page_textures.json --textures-dir texture/views --index lightbeans-texture2-dinov2-views`
  - When indexing generated views, `contact_sheet` images are automatically skipped.
  - Use `--upload-batch-size` (default `256`) to control upload chunk size and avoid oversized requests.
  - This script automatically reads `MEILI_HOST`, `MEILI_KEY`, and `MEILI_INDEX` from the project `.env` file when present.
- Retrieve stored embeddings directly by product id or product name:
  - This helper builds/updates the DINOv2 views index first by default (use `--no-build-index` to only query).
  - By product id:
    `python scripts/classification/texture_embedding_getter_meilisearch.py --index lightbeans-texture2-dinov2-views --product-id 001-Montana-Timber-Products-AquaFir-Cedar-Shale-Smooth-Shiplap-6-Texture`
  - By product name:
    `python scripts/classification/texture_embedding_getter_meilisearch.py --index lightbeans-texture2-dinov2-views --product-name "AquaFir Cedar Shale"`
- Build and query a local DINOv2 ViT-S/14 FAISS texture index (CLS, mean, or texture stats):
  - Build/update:
    `python scripts/classification/texture_indexer_dinov2_texture_faiss.py index --textures-dir texture/main_page --index-root texture/indexes/dinov2_faiss --embedding-mode texture_stats --input-size 224`
  - Search:
    `python scripts/classification/texture_indexer_dinov2_texture_faiss.py search --query-image path/to/query.png --index-root texture/indexes/dinov2_faiss --embedding-mode texture_stats --input-size 224 --top-k 10`
  - Install requirement for local FAISS search:
    `pip install faiss-cpu`
  - Keep the same `--embedding-mode` and `--input-size` for index and query. Resizing/cropping changes texture scale and local geometric patterns, so retrieval quality depends on consistent preprocessing.
- Build a DeepTEN (MINC-2500 pretrained) texture embedding index in Meilisearch:
  `python scripts/classification/texture_indexer_deepten_minc2500_meilisearch.py --product-export data/platform-products-production-v2-en.json --textures-dir texture --mapping-json texture/texture_product_mapping.json`
- Match a masked image region to the texture embedding index and return the top matches with a texture link:
  `python scripts/viewer/masked_region_texture_matcher.py --image path/to/input.png --mask path/to/mask.png --limit 5`
  - This script reads the same `.env` values for Meilisearch and Hugging Face access, extracts a mask bounding box, computes a DINOv2 embedding, and returns the best texture matches.
  - To use local DINOv2 texture-stat FAISS search instead of Meilisearch:
    `python scripts/viewer/masked_region_texture_matcher.py --image path/to/input.png --mask path/to/mask.png --method dinov2_texture_faiss --faiss-index-root texture/indexes/dinov2_faiss --faiss-embedding-mode texture_stats --faiss-input-size 224 --limit 10`
