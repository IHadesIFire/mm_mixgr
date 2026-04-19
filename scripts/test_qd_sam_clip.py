import os
import json
import hashlib
import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np
import torch
from PIL import Image, ImageDraw
from datasets import load_from_disk
#from transformers import CLIPModel, CLIPProcessor
from transformers import CLIPModel, CLIPProcessor, AutoModel, AutoProcessor
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# =========================
# 工具函数
# =========================
def unwrap_dataset(ds, name="dataset"):
    from datasets import DatasetDict
    if isinstance(ds, DatasetDict):
        if "test" in ds:
            return ds["test"]
        first_key = list(ds.keys())[0]
        print(f"[WARN] {name} is DatasetDict, using split: {first_key}")
        return ds[first_key]
    return ds

def extract_images_from_row(row):
    images = []
    for key in ["image", "image 1", "image 2", "image 3", "image 4", "vision"]:
        val = row.get(key)
        if val is None:
            continue
        if isinstance(val, Image.Image):
            images.append(val.convert("RGB"))
        elif isinstance(val, dict) and "bytes" in val:
            from io import BytesIO
            images.append(Image.open(BytesIO(val["bytes"])).convert("RGB"))
    return images


def matches_domain(row, domains):
    qid = str(row.get("id", ""))
    domain_field = str(row.get("domain", ""))

    domain_prefixes = {
        "Science": ["Agriculture", "Geography", "Chemistry", "Biology"],
        "Medicine": ["Diagnostics", "Clinical_Medicine", "Basic_Medical", "Pharmacy", "Diagnostics_and_Laboratory"],
    }

    prefixes = []
    for d in domains:
        prefixes.extend(domain_prefixes.get(d, [d]))

    if domain_field:
        for d in domains:
            if d.lower() in domain_field.lower():
                return True

    for p in prefixes:
        if p.lower() in qid.lower():
            return True

    return False


def md5_cache_key(task: str, text: str) -> str:
    return hashlib.md5(f"{task}:{text}".encode("utf-8")).hexdigest()


def load_text_units_from_cache(cache_dir: str, task: str, text: str) -> List[str]:
    key = md5_cache_key(task, text)
    cache_file = Path(cache_dir) / f"{key}.json"
    if not cache_file.exists():
        return []

    with open(cache_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return [str(x).strip() for x in data if str(x).strip()]
    return []


def build_region_views(image: Image.Image, ann: dict, context_ratio: float = 0.25):
    """
    给一个 SAM 区域构造三种视图：
    - masked
    - box
    - context
    """
    image = image.convert("RGB")
    img_np = np.array(image)

    seg = ann["segmentation"].astype(bool)
    x, y, w, h = ann["bbox"]
    x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

    H, W = img_np.shape[:2]
    x1 = max(0, min(x1, W - 1))
    y1 = max(0, min(y1, H - 1))
    x2 = max(x1 + 1, min(x2, W))
    y2 = max(y1 + 1, min(y2, H))

    # masked
    masked = np.zeros_like(img_np)
    masked[seg] = img_np[seg]
    masked_img = Image.fromarray(masked)

    # box
    box_img = image.crop((x1, y1, x2, y2))

    # context
    bw, bh = x2 - x1, y2 - y1
    ex, ey = int(bw * context_ratio), int(bh * context_ratio)
    cx1 = max(0, x1 - ex)
    cy1 = max(0, y1 - ey)
    cx2 = min(W, x2 + ex)
    cy2 = min(H, y2 + ey)
    context_img = image.crop((cx1, cy1, cx2, cy2))

    return {
        "masked": masked_img,
        "box": box_img,
        "context": context_img,
        "bbox_xyxy": [x1, y1, x2, y2],
    }


def draw_overlay(image: Image.Image, anns: List[dict], top_idx: int, save_path: str):
    img = image.convert("RGB").copy()
    draw = ImageDraw.Draw(img)

    for i, ann in enumerate(anns):
        x, y, w, h = ann["bbox"]
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)

        color = "red" if i == top_idx else "yellow"
        width = 4 if i == top_idx else 2
        draw.rectangle((x1, y1, x2, y2), outline=color, width=width)
        draw.text((x1 + 2, y1 + 2), str(i), fill=color)

    img.save(save_path)


# =========================
# CLIP scorer
# =========================

class CLIPScorer:
    def __init__(self, model_name="openai/clip-vit-large-patch14", model_family="clip", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.model_family = model_family

        if self.model_family == "clip":
            self.model = CLIPModel.from_pretrained(model_name).to(self.device)
            self.processor = CLIPProcessor.from_pretrained(model_name)
        elif self.model_family == "siglip":
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_name)
        else:
            raise ValueError(f"Unsupported model_family: {self.model_family}")

        self.model.eval()

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> np.ndarray:
        if self.model_family != "clip":
            raise NotImplementedError("encode_texts is only used for model_family='clip'")

        if len(texts) == 0:
            return np.zeros((0, 1), dtype=np.float32)

        batch_size = 8
        all_feats = []

        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start:start + batch_size]

            inputs = self.processor(
                text=batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            try:
                feats = self.model.get_text_features(**inputs)
            except Exception:
                outputs = self.model.text_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask", None),
                )
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    feats = outputs.pooler_output
                else:
                    feats = outputs.last_hidden_state[:, 0, :]

                if hasattr(self.model, "text_projection") and self.model.text_projection is not None:
                    in_dim = self.model.text_projection.weight.shape[1]
                    if feats.shape[-1] == in_dim:
                        feats = self.model.text_projection(feats)

            if not torch.is_tensor(feats):
                if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                    feats = feats.pooler_output
                elif hasattr(feats, "last_hidden_state"):
                    feats = feats.last_hidden_state[:, 0, :]
                else:
                    raise TypeError(f"Unsupported text feature type: {type(feats)}")

                if hasattr(self.model, "text_projection") and self.model.text_projection is not None:
                    in_dim = self.model.text_projection.weight.shape[1]
                    if feats.shape[-1] == in_dim:
                        feats = self.model.text_projection(feats)

            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.detach().cpu())

            del inputs, feats
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return torch.cat(all_feats, dim=0).numpy()

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        if self.model_family != "clip":
            raise NotImplementedError("encode_images is only used for model_family='clip'")

        if len(images) == 0:
            return np.zeros((0, 1), dtype=np.float32)

        batch_size = 4
        all_feats = []

        for start in range(0, len(images), batch_size):
            batch_images = images[start:start + batch_size]

            inputs = self.processor(
                images=batch_images,
                return_tensors="pt",
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            try:
                feats = self.model.get_image_features(**inputs)
            except Exception:
                outputs = self.model.vision_model(
                    pixel_values=inputs["pixel_values"]
                )
                if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                    feats = outputs.pooler_output
                else:
                    feats = outputs.last_hidden_state[:, 0, :]

                if hasattr(self.model, "visual_projection") and self.model.visual_projection is not None:
                    in_dim = self.model.visual_projection.weight.shape[1]
                    if feats.shape[-1] == in_dim:
                        feats = self.model.visual_projection(feats)

            if not torch.is_tensor(feats):
                if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                    feats = feats.pooler_output
                elif hasattr(feats, "last_hidden_state"):
                    feats = feats.last_hidden_state[:, 0, :]
                else:
                    raise TypeError(f"Unsupported image feature type: {type(feats)}")

                if hasattr(self.model, "visual_projection") and self.model.visual_projection is not None:
                    in_dim = self.model.visual_projection.weight.shape[1]
                    if feats.shape[-1] == in_dim:
                        feats = self.model.visual_projection(feats)

            feats = feats / feats.norm(dim=-1, keepdim=True)
            all_feats.append(feats.detach().cpu())

            del inputs, feats
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return torch.cat(all_feats, dim=0).numpy()

    def similarity(self, image_feats: np.ndarray, text_feats: np.ndarray) -> np.ndarray:
        if self.model_family != "clip":
            raise NotImplementedError("similarity is only used for model_family='clip'")
        return image_feats @ text_feats.T

    @torch.no_grad()
    def similarity_from_images_and_texts(self, images: List[Image.Image], texts: List[str]):
        """
        仅用于 siglip：
        返回两个矩阵：
        - raw_logits: [num_images, num_texts]
        - sigmoid_scores: [num_images, num_texts]，范围 0~1
        """
        if self.model_family != "siglip":
            raise NotImplementedError("similarity_from_images_and_texts is only used for model_family='siglip'")

        if len(images) == 0 or len(texts) == 0:
            empty = np.zeros((len(images), len(texts)), dtype=np.float32)
            return empty, empty

        img_batch_size = 4
        txt_batch_size = 8

        all_logit_rows = []
        all_score_rows = []

        for i in range(0, len(images), img_batch_size):
            image_batch = images[i:i + img_batch_size]

            row_logit_blocks = []
            row_score_blocks = []

            for j in range(0, len(texts), txt_batch_size):
                text_batch = texts[j:j + txt_batch_size]

                inputs = self.processor(
                    text=text_batch,
                    images=image_batch,
                    padding="max_length",
                    return_tensors="pt",
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(**inputs)

                if hasattr(outputs, "logits_per_image"):
                    logits = outputs.logits_per_image
                else:
                    raise ValueError("SigLIP outputs do not contain logits_per_image")

                scores = torch.sigmoid(logits)

                row_logit_blocks.append(logits.detach().cpu())
                row_score_blocks.append(scores.detach().cpu())

                del inputs, outputs, logits, scores
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            all_logit_rows.append(torch.cat(row_logit_blocks, dim=1))
            all_score_rows.append(torch.cat(row_score_blocks, dim=1))

        raw_logits = torch.cat(all_logit_rows, dim=0).numpy()
        sigmoid_scores = torch.cat(all_score_rows, dim=0).numpy()

        return raw_logits, sigmoid_scores


# =========================
# 主逻辑
# =========================
def process_one_sample(
    row,
    images,
    text_units,
    out_dir: Path,
    clip_scorer: CLIPScorer,
    mask_generator,
    max_regions: int,
    item_prefix: str = "query",
):
    item_id = str(row["id"])
    text = row.get("text", "")

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / f"{item_prefix}.txt", "w", encoding="utf-8") as f:
        f.write(text)

    with open(out_dir / "text_units.json", "w", encoding="utf-8") as f:
        json.dump(text_units, f, ensure_ascii=False, indent=2)

    print(f"\n[{item_prefix}:{item_id}] num_text_units = {len(text_units)}")

    for img_idx, image in enumerate(images):
        image = image.convert("RGB")
        image.save(out_dir / f"image_{img_idx}_orig.jpg")

        anns = mask_generator.generate(np.array(image))

        filtered_anns = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            if w < 8 or h < 8:
                continue
            if w * h < 100:
                continue
            filtered_anns.append(ann)

        anns = filtered_anns[:max_regions]
        print(f"[{item_prefix}:{item_id}] image_{img_idx}: {len(anns)} regions")

        if len(anns) == 0:
            continue

        region_views = []
        region_meta = []

        for rid, ann in enumerate(anns):
            views = build_region_views(image, ann)
            region_views.append(views)
            region_meta.append({
                "region_id": rid,
                "bbox_xyxy": views["bbox_xyxy"],
                "area": int(ann["area"]),
            })

        masked_imgs = [rv["masked"] for rv in region_views]
        box_imgs = [rv["box"] for rv in region_views]
        context_imgs = [rv["context"] for rv in region_views]

        if clip_scorer.model_family == "clip":
            text_feats = clip_scorer.encode_texts(text_units)

            masked_feats = clip_scorer.encode_images(masked_imgs)
            box_feats = clip_scorer.encode_images(box_imgs)
            context_feats = clip_scorer.encode_images(context_imgs)

            sim_masked = clip_scorer.similarity(masked_feats, text_feats)
            sim_box = clip_scorer.similarity(box_feats, text_feats)
            sim_context = clip_scorer.similarity(context_feats, text_feats)

            raw_masked = sim_masked
            raw_box = sim_box
            raw_context = sim_context

        elif clip_scorer.model_family == "siglip":
            raw_masked, sim_masked = clip_scorer.similarity_from_images_and_texts(masked_imgs, text_units)
            raw_box, sim_box = clip_scorer.similarity_from_images_and_texts(box_imgs, text_units)
            raw_context, sim_context = clip_scorer.similarity_from_images_and_texts(context_imgs, text_units)

        else:
            raise ValueError(f"Unsupported model_family: {clip_scorer.model_family}")

        all_results = []
        global_best = {
            "region_id": None,
            "view": None,
            "text_idx": None,
            "text_unit": None,
            "score": -1e9,
            "raw_score": -1e9,
        }

        for rid in range(len(region_views)):
            per_region_best = {
                "region_id": rid,
                "bbox_xyxy": region_meta[rid]["bbox_xyxy"],
                "area": region_meta[rid]["area"],
                "best_view": None,
                "best_text_idx": None,
                "best_text_unit": None,
                "best_score": -1e9,
                "view_scores": {},
            }

            for view_name, raw_sim, sim in [
                ("masked", raw_masked, sim_masked),
                ("box", raw_box, sim_box),
                ("context", raw_context, sim_context),
            ]:
                row_scores = sim[rid]          # 用 sigmoid 后的分数做排序
                row_raw_scores = raw_sim[rid]  # 保留原始 logit

                txt_idx = int(np.argmax(row_scores))
                score = float(row_scores[txt_idx])
                raw_score = float(row_raw_scores[txt_idx])

                per_region_best["view_scores"][view_name] = {
                    "text_idx": txt_idx,
                    "text_unit": text_units[txt_idx],
                    "score": score,
                    "raw_score": raw_score,
                }

                if score > per_region_best["best_score"]:
                    per_region_best["best_score"] = score
                    per_region_best["best_view"] = view_name
                    per_region_best["best_text_idx"] = txt_idx
                    per_region_best["best_text_unit"] = text_units[txt_idx]
                    per_region_best["best_raw_score"] = raw_score

                if score > global_best["score"]:
                    global_best["region_id"] = rid
                    global_best["view"] = view_name
                    global_best["text_idx"] = txt_idx
                    global_best["text_unit"] = text_units[txt_idx]
                    global_best["score"] = score
                    global_best["raw_score"] = raw_score

            all_results.append(per_region_best)

        all_results = sorted(all_results, key=lambda x: x["best_score"], reverse=True)

        with open(out_dir / f"image_{img_idx}_scores.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        region_root = out_dir / f"image_{img_idx}_regions"
        region_root.mkdir(parents=True, exist_ok=True)

        for rid, rv in enumerate(region_views):
            rid_dir = region_root / f"region_{rid:03d}"
            rid_dir.mkdir(parents=True, exist_ok=True)
            rv["masked"].save(rid_dir / "masked.jpg")
            rv["box"].save(rid_dir / "box.jpg")
            rv["context"].save(rid_dir / "context.jpg")

        best_rid = global_best["region_id"]
        best_view = global_best["view"]
        best_text_unit = global_best["text_unit"]

        best_region_dir = out_dir / f"image_{img_idx}_best_match"
        best_region_dir.mkdir(parents=True, exist_ok=True)

        region_views[best_rid]["masked"].save(best_region_dir / "best_masked.jpg")
        region_views[best_rid]["box"].save(best_region_dir / "best_box.jpg")
        region_views[best_rid]["context"].save(best_region_dir / "best_context.jpg")

        draw_overlay(
            image=image,
            anns=anns,
            top_idx=best_rid,
            save_path=best_region_dir / "best_overlay.jpg",
        )

        with open(best_region_dir / "best_match.txt", "w", encoding="utf-8") as f:
            f.write(f"{item_prefix}_id: {item_id}\n")
            f.write(f"image_index: {img_idx}\n")
            f.write(f"best_region_id: {best_rid}\n")
            f.write(f"best_view: {best_view}\n")
            f.write(f"best_text_idx: {global_best['text_idx']}\n")
            f.write(f"best_score: {global_best['score']:.6f}\n")
            f.write(f"best_raw_score: {global_best['raw_score']:.6f}\n")
            f.write(f"best_text_unit: {best_text_unit}\n")

        print(
            f"[{item_prefix}:{item_id}] image_{img_idx} BEST => "
            f"region={best_rid}, view={best_view}, "
            f"score={global_best['score']:.4f}, text_unit={best_text_unit}"
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_path", type=str, required=True)
    parser.add_argument("--corpus_path", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True)
    parser.add_argument("--sam_checkpoint", type=str, required=True)
    parser.add_argument("--sam_model_type", type=str, default="vit_b")
    parser.add_argument("--clip_model", type=str, default="openai/clip-vit-large-patch14")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--domains", nargs="+", default=["Science", "Medicine"])
    parser.add_argument("--num_query_samples", type=int, default=3)
    parser.add_argument("--num_doc_samples", type=int, default=3)
    parser.add_argument("--output_dir", type=str, default="./results/test_query_sam_clip")
    parser.add_argument("--max_regions", type=int, default=30)
    parser.add_argument(
        "--query_indices",
        type=int,
        nargs="+",
        default=[0, 7, 15],
        help="指定要选的 query idx"
    )

    parser.add_argument(
        "--doc_indices",
        type=int,
        nargs="+",
        default=[0, 7, 15],
        help="指定要选的 doc idx"
    )
    parser.add_argument(
        "--model_family",
        type=str,
        default="clip",
        choices=["clip", "siglip"],
        help="选择图文模型家族"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading query dataset from: {args.query_path}")
    ds_query = load_from_disk(args.query_path)
    ds_query = unwrap_dataset(ds_query, "query")
    print(f"Loaded query split, len = {len(ds_query)}")

    print(f"Loading corpus dataset from: {args.corpus_path}")
    ds_corpus = load_from_disk(args.corpus_path)
    ds_corpus = unwrap_dataset(ds_corpus, "corpus")
    print(f"Loaded corpus split, len = {len(ds_corpus)}")

    print("Loading SAM...")
    sam = sam_model_registry[args.sam_model_type](checkpoint=args.sam_checkpoint)
    sam.to(device=args.device)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=256,
    )

    print("Loading CLIP...")
    #clip_scorer = CLIPScorer(model_name=args.clip_model, device=args.device)
    clip_scorer = CLIPScorer(
        model_name=args.clip_model,
        model_family=args.model_family,
        device=args.device,
    )

    # =========================
    # 先处理 query
    # =========================
    query_candidates = []
    for row in ds_query:
        if not matches_domain(row, args.domains):
            continue

        # 这里只检查有没有图，不解码
        has_image = any(
            row.get(k) is not None
            for k in ["image", "image 1", "image 2", "image 3", "image 4", "vision"]
        )
        if not has_image:
            continue

        query_candidates.append(row)

    print(f"Total valid query candidates = {len(query_candidates)}")

    selected_queries = []
    for idx in args.query_indices:
        if 0 <= idx < len(query_candidates):
            row = query_candidates[idx]
            images = extract_images_from_row(row)  # 只对选中的样本解码
            if len(images) > 0:
                selected_queries.append((idx, row, images))
            else:
                print(f"[WARN] query idx {idx} selected but decoded 0 images")
        else:
            print(f"[WARN] query idx {idx} out of range")

    print(f"Selected {len(selected_queries)} query samples (by indices)")

    for query_idx, row, images in selected_queries:
        qid = str(row.get("id", ""))
        query_text = row.get("text", "")

        subqueries = load_text_units_from_cache(args.cache_dir, "subquery", query_text)
        if len(subqueries) == 0:
            subqueries = [query_text]
        # subqueries = subqueries[:16]

        sample_dir = output_dir / f"query_{query_idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # 保存文本单元：query 部分必须保存 subqueries
        with open(sample_dir / "subdocs.json", "w", encoding="utf-8") as f:
            json.dump(subqueries, f, ensure_ascii=False, indent=2)

        # 可选：保存原始信息，方便排查
        with open(sample_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "selected_idx": query_idx,
                    "orig_id": qid,
                    "num_images": len(images),
                    "text_preview": query_text[:300]
                },
                f,
                ensure_ascii=False,
                indent=2
            )

        process_one_sample(
            row=row,
            images=images,
            text_units=subqueries,
            out_dir=sample_dir,
            clip_scorer=clip_scorer,
            mask_generator=mask_generator,
            max_regions=args.max_regions,
            item_prefix="query",
        )


    # =========================
    # 再处理 doc
    # =========================
    doc_candidates = []
    for row in ds_corpus:
        # 这里只检查有没有图，不解码
        has_image = any(
            row.get(k) is not None
            for k in ["image", "image 1", "image 2", "image 3", "image 4", "vision"]
        )
        if not has_image:
            continue

        doc_candidates.append(row)

    print(f"Total valid doc candidates = {len(doc_candidates)}")

    selected_docs = []
    for idx in args.doc_indices:
        if 0 <= idx < len(doc_candidates):
            row = doc_candidates[idx]
            images = extract_images_from_row(row)  # 只对选中的样本解码
            if len(images) > 0:
                selected_docs.append((idx, row, images))
            else:
                print(f"[WARN] doc idx {idx} selected but decoded 0 images")
        else:
            print(f"[WARN] doc idx {idx} out of range")

    print(f"Selected {len(selected_docs)} doc samples (by indices)")

    for doc_idx, row, images in selected_docs:
        did = str(row.get("id", ""))
        doc_text = row.get("text", "")

        propositions = load_text_units_from_cache(args.cache_dir, "proposition", doc_text)
        if len(propositions) == 0:
            propositions = [doc_text]
        # propositions = propositions[:16]

        sample_dir = output_dir / f"doc_{doc_idx}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        # 保存文本单元：doc 部分保存 propositions
        with open(sample_dir / "subdocs.json", "w", encoding="utf-8") as f:
            json.dump(propositions, f, ensure_ascii=False, indent=2)

        # 可选：保存原始信息，方便排查
        with open(sample_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "selected_idx": doc_idx,
                    "orig_id": did,
                    "num_images": len(images),
                    "text_preview": doc_text[:300]
                },
                f,
                ensure_ascii=False,
                indent=2
            )

        process_one_sample(
            row=row,
            images=images,
            text_units=propositions,
            out_dir=sample_dir,
            clip_scorer=clip_scorer,
            mask_generator=mask_generator,
            max_regions=args.max_regions,
            item_prefix="doc",
        )


if __name__ == "__main__":
    main()