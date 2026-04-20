# postprocess_regions.py
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional

import numpy as np
from PIL import Image


@dataclass
class PostProcessConfig:
    # 1) 太小碎片
    min_width: int = 12
    min_height: int = 12
    min_box_area: int = 180
    min_mask_area: int = 120

    # 2) 太细长的框
    # aspect_ratio = max(w/h, h/w)
    max_aspect_ratio: float = 10.0

    # 3) 重复区域
    dedup_by_box_iou: bool = True
    box_iou_thresh: float = 0.65

    dedup_by_mask_iou: bool = False
    mask_iou_thresh: float = 0.80

    # 保留上限
    max_regions: int = 30


def _clip_bbox_xywh(bbox, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    x, y, w, h = bbox
    x1 = int(max(0, min(round(x), img_w - 1)))
    y1 = int(max(0, min(round(y), img_h - 1)))
    x2 = int(max(x1 + 1, min(round(x + w), img_w)))
    y2 = int(max(y1 + 1, min(round(y + h), img_h)))
    return x1, y1, x2, y2


def _xyxy_area(box: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def _box_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2

    ix1 = max(x11, x21)
    iy1 = max(y11, y21)
    ix2 = min(x12, x22)
    iy2 = min(y12, y22)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    a1 = _xyxy_area(box1)
    a2 = _xyxy_area(box2)
    union = a1 + a2 - inter
    if union <= 0:
        return 0.0

    return inter / union


def _mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    inter = np.logical_and(mask1, mask2).sum()
    if inter == 0:
        return 0.0
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def _quality_score(ann: Dict[str, Any]) -> float:
    """
    用 SAM 自带质量分数排序：
    predicted_iou 越高越好，stability_score 越高越好。
    """
    pred_iou = float(ann.get("predicted_iou", 0.0))
    stability = float(ann.get("stability_score", 0.0))
    area = float(ann.get("area", 0.0))

    # 面积只给一个很小的加成，避免大框天然占优
    area_bonus = np.log1p(max(area, 0.0)) * 0.01

    return pred_iou + stability + area_bonus


def postprocess_sam_regions(
    image: Image.Image,
    anns: List[Dict[str, Any]],
    cfg: PostProcessConfig,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    只做三类规则后处理：
    1) 太小碎片
    2) 太细长的框
    3) 重复区域
    """
    image = image.convert("RGB")
    img_w, img_h = image.size

    debug_info = {
        "num_input_regions": len(anns),
        "removed_small": 0,
        "removed_slender": 0,
        "removed_duplicate": 0,
        "num_output_regions": 0,
    }

    candidates = []

    # -------------------------------------------------
    # step1: 小碎片过滤 + 细长框过滤
    # -------------------------------------------------
    for ann in anns:
        if "bbox" not in ann or "segmentation" not in ann:
            continue

        seg = ann["segmentation"]
        if not isinstance(seg, np.ndarray):
            seg = np.array(seg, dtype=bool)
        else:
            seg = seg.astype(bool)

        x1, y1, x2, y2 = _clip_bbox_xywh(ann["bbox"], img_w, img_h)
        bw, bh = x2 - x1, y2 - y1
        box_area = bw * bh
        mask_area = int(seg.sum())

        # 1) 太小碎片
        if bw < cfg.min_width or bh < cfg.min_height:
            debug_info["removed_small"] += 1
            continue

        if box_area < cfg.min_box_area:
            debug_info["removed_small"] += 1
            continue

        if mask_area < cfg.min_mask_area:
            debug_info["removed_small"] += 1
            continue

        # 2) 太细长的框
        aspect_ratio = max(bw / max(bh, 1), bh / max(bw, 1))
        if aspect_ratio > cfg.max_aspect_ratio:
            debug_info["removed_slender"] += 1
            continue

        ann_copy = dict(ann)
        ann_copy["segmentation"] = seg
        ann_copy["_pp_bbox_xyxy"] = [x1, y1, x2, y2]
        ann_copy["_pp_box_area"] = int(box_area)
        ann_copy["_pp_mask_area"] = int(mask_area)
        ann_copy["_pp_aspect_ratio"] = float(aspect_ratio)
        ann_copy["_pp_quality_score"] = float(_quality_score(ann_copy))

        candidates.append(ann_copy)

    # -------------------------------------------------
    # step2: 先按质量排序，再去重
    # -------------------------------------------------
    candidates = sorted(candidates, key=lambda x: x["_pp_quality_score"], reverse=True)

    kept = []
    for ann in candidates:
        keep_flag = True
        box1 = tuple(ann["_pp_bbox_xyxy"])
        seg1 = ann["segmentation"]

        for kept_ann in kept:
            if cfg.dedup_by_box_iou:
                box2 = tuple(kept_ann["_pp_bbox_xyxy"])
                iou_box = _box_iou(box1, box2)
                if iou_box >= cfg.box_iou_thresh:
                    keep_flag = False
                    break

            if cfg.dedup_by_mask_iou:
                seg2 = kept_ann["segmentation"]
                iou_mask = _mask_iou(seg1, seg2)
                if iou_mask >= cfg.mask_iou_thresh:
                    keep_flag = False
                    break

        if not keep_flag:
            debug_info["removed_duplicate"] += 1
            continue

        kept.append(ann)
        if len(kept) >= cfg.max_regions:
            break

    debug_info["num_output_regions"] = len(kept)
    return kept, debug_info