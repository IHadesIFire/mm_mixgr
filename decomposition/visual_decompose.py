# decomposition/visual_decompose.py

"""
Visual decomposition with SAM.

用途：
1. 对文档图像做自动分割
2. 过滤过小/过大/重复 mask
3. 构造三种 region 视图：
   - masked: 只保留区域，其余置黑
   - box: bbox crop
   - context: 带上下文的 bbox crop
"""

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class RegionCandidate:
    """
    单个区域候选。
    """
    region_id: int
    mask: np.ndarray               # H x W, bool
    bbox_xyxy: Tuple[int, int, int, int]
    area: int
    score: float = 0.0            # 预留给后续排序用


class SAMVisualDecomposer:
    """
    使用 Segment Anything 做自动分割。

    依赖：
        pip install segment-anything opencv-python
    并准备好 SAM checkpoint，例如：
        sam_vit_h_4b8939.pth
    """

    def __init__(
        self,
        sam_checkpoint: str,
        model_type: str = "vit_h",
        device: str = "cuda",
        min_region_area: int = 256,
        max_region_area_ratio: float = 0.7,
        pred_iou_thresh: float = 0.86,
        stability_score_thresh: float = 0.92,
        crop_n_layers: int = 1,
        crop_n_points_downscale_factor: int = 2,
    ):
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        self.device = device
        self.min_region_area = min_region_area
        self.max_region_area_ratio = max_region_area_ratio
        self.pred_iou_thresh = pred_iou_thresh
        self.stability_score_thresh = stability_score_thresh
        self.crop_n_layers = crop_n_layers
        self.crop_n_points_downscale_factor = crop_n_points_downscale_factor

        self._mask_generator = None

    def _lazy_load(self):
        if self._mask_generator is not None:
            return

        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        logger.info(
            f"Loading SAM model: type={self.model_type}, checkpoint={self.sam_checkpoint}"
        )
        sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        sam.to(device=self.device)

        self._mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            pred_iou_thresh=self.pred_iou_thresh,
            stability_score_thresh=self.stability_score_thresh,
            crop_n_layers=self.crop_n_layers,
            crop_n_points_downscale_factor=self.crop_n_points_downscale_factor,
            min_mask_region_area=self.min_region_area,
        )

    def segment(self, image: Image.Image) -> List[RegionCandidate]:
        """
        对单张图像做自动分割，返回过滤后的候选区域。
        """
        self._lazy_load()

        rgb = image.convert("RGB")
        img_np = np.array(rgb)
        h, w = img_np.shape[:2]
        img_area = h * w

        raw_masks = self._mask_generator.generate(img_np)
        logger.info(f"SAM produced {len(raw_masks)} raw masks")

        results: List[RegionCandidate] = []
        seen_boxes = set()

        for idx, ann in enumerate(raw_masks):
            seg = ann["segmentation"].astype(bool)
            area = int(ann["area"])

            if area < self.min_region_area:
                continue
            if area > img_area * self.max_region_area_ratio:
                continue

            x, y, bw, bh = ann["bbox"]
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + bw), int(y + bh)

            # 边界修正
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))

            # 简单去重：完全相同 bbox 的只保留一个
            box_key = (x1, y1, x2, y2)
            if box_key in seen_boxes:
                continue
            seen_boxes.add(box_key)

            results.append(
                RegionCandidate(
                    region_id=len(results),
                    mask=seg,
                    bbox_xyxy=(x1, y1, x2, y2),
                    area=area,
                    score=float(ann.get("predicted_iou", 0.0)),
                )
            )

        logger.info(f"Filtered to {len(results)} region candidates")
        return results


def make_masked_view(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """
    只保留 mask 区域，其余位置置黑。
    """
    img_np = np.array(image.convert("RGB"))
    out = np.zeros_like(img_np)
    out[mask] = img_np[mask]
    return Image.fromarray(out)


def make_box_view(image: Image.Image, bbox_xyxy: Tuple[int, int, int, int]) -> Image.Image:
    """
    直接裁剪 bbox。
    """
    x1, y1, x2, y2 = bbox_xyxy
    return image.crop((x1, y1, x2, y2))


def make_context_view(
    image: Image.Image,
    bbox_xyxy: Tuple[int, int, int, int],
    expand_ratio: float = 0.25,
) -> Image.Image:
    """
    带上下文的区域 crop。
    """
    x1, y1, x2, y2 = bbox_xyxy
    w = x2 - x1
    h = y2 - y1

    ex = int(w * expand_ratio)
    ey = int(h * expand_ratio)

    W, H = image.size
    nx1 = max(0, x1 - ex)
    ny1 = max(0, y1 - ey)
    nx2 = min(W, x2 + ex)
    ny2 = min(H, y2 + ey)

    return image.crop((nx1, ny1, nx2, ny2))


def build_region_views(
    image: Image.Image,
    region: RegionCandidate,
    context_ratio: float = 0.25,
) -> Dict[str, Image.Image]:
    """
    给单个 region 构造三种视图。
    """
    return {
        "masked": make_masked_view(image, region.mask),
        "box": make_box_view(image, region.bbox_xyxy),
        "context": make_context_view(image, region.bbox_xyxy, expand_ratio=context_ratio),
    }