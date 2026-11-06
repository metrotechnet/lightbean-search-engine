# Lightbean Search Engine

This project evaluates product discovery and retrieval using Meilisearch and a chat-based workflow for Lightbeans materials.

## Contents
- `data/` — datasets and prompt configuration
- `scripts/` — conversion, search, chat, and report generation scripts
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
- Build ground truth mapping:
  `python scripts/generate_ground_truth.py`
- Split downloaded images into train/test (train = first occurrence per product, test = remaining occurrences):
  `python scripts/split_images_train_test.py --ground-truth ground_truth.json --images-dir images --output image_split_first_occurrence.json`
- Build SigLIP2 image-similarity index in Meilisearch and evaluate retrieval on test images:
  `python scripts/siglip2_image_similarity_meilisearch.py --split results/image_split_first_occurrence.json --images-dir images --ground-truth ground_truth.json --product-export data/platform-products-production-v2-en.json --index lightbeans-siglip2-image-similarity --output results/siglip2_image_similarity_eval.json`
- Segment images into regions with Meta SAM (SAM 3 checkpoint by default):
  `python scripts/segment_images_sam3.py --images-dir images --output-dir results/sam3_segments --batch-size 4 --device auto`
- Segment all images in `images/` with Meta SAM 2:
  `python scripts/segment_images_sam2.py --images-dir images --output-dir results/sam2_segments --batch-size 4 --device auto`
- Visualize SAM2 segmented regions as semi-transparent overlays with Previous/Next navigation:
  `python scripts/view_sam2_segments.py --metadata results/sam2_segments/segments_metadata.json --alpha 110`
- Download texture images from `previewPath` and create file-to-product mapping:
  `python scripts/download_texture_images.py data/platform-products-production-v2-en.json --output-dir texture --mapping-output texture/texture_product_mapping.json`
- Build a DINOv2 texture embedding index in Meilisearch with product features and vector embeddings:
  `python scripts/dinov2_texture_meilisearch.py --product-export data/platform-products-production-v2-en.json --textures-dir texture --mapping-json texture/texture_product_mapping.json`
  - This script automatically reads `MEILI_HOST`, `MEILI_KEY`, and `MEILI_INDEX` from the project `.env` file when present.
- Match a masked image region to the texture embedding index and return the top matches with a texture link:
  `python scripts/match_masked_region_to_texture.py --image path/to/input.png --mask path/to/mask.png --limit 5`
  - This script reads the same `.env` values for Meilisearch and Hugging Face access, extracts a mask bounding box, computes a DINOv2 embedding, and returns the best texture matches.
