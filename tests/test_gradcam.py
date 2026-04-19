"""
GradCAM 可视化实验 v2:
- 筛选有实质文本的 query（跳过 "What is this?" 之类的短问题）
- 修复 GradCAM 梯度捕获
- 同时输出 GradCAM 和 PatchSim 对比

运行: python tests/test_gradcam.py

失败
"""

import sys
import os
import random
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class CLIPGradCAM:

    def __init__(self, model_name="openai/clip-vit-large-patch14", device="cuda"):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading CLIP: {model_name}")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        logger.info(f"CLIP loaded on {self.device}")

    def compute_heatmap(self, image: Image.Image, text: str) -> np.ndarray:
        """
        GradCAM: 对 image-text similarity 反向传播，
        用最后一层 ViT hidden state 的梯度生成热力图。
        """
        inputs = self.processor(
            text=[text], images=[image],
            return_tensors="pt", padding=True, truncation=True,
        ).to(self.device)

        # 需要梯度
        self.model.zero_grad()
        for param in self.model.parameters():
            param.requires_grad_(True)

        # Forward: 拿到 vision encoder 最后一层的 hidden states
        vision_outputs = self.model.vision_model(
            pixel_values=inputs["pixel_values"],
            output_hidden_states=True,
            return_dict=True,
        )

        # 最后一层 hidden state: (1, num_patches+1, hidden_dim)
        last_hidden = vision_outputs.last_hidden_state
        last_hidden.retain_grad()

        # 投影到 CLIP 空间
        pooled = vision_outputs.pooler_output  # (1, hidden_dim)
        image_embeds = self.model.visual_projection(pooled)
        image_embeds = F.normalize(image_embeds, dim=-1)

        # Text embedding
        text_outputs = self.model.text_model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
        text_embeds = self.model.text_projection(text_outputs.pooler_output)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # Similarity
        similarity = (image_embeds * text_embeds).sum()

        # Backward
        similarity.backward()

        # 读取梯度
        grad = last_hidden.grad  # (1, num_patches+1, hidden_dim)

        if grad is None:
            logger.warning("GradCAM: gradient is None, falling back to PatchSim")
            return self.compute_patch_similarities(image, text)

        # 去掉 CLS token
        act = last_hidden[:, 1:, :].detach()
        grad = grad[:, 1:, :].detach()

        # GradCAM 公式: weights = mean(grad), cam = ReLU(sum(weights * act))
        weights = grad.mean(dim=1, keepdim=True)  # (1, 1, dim)
        cam = (act * weights).sum(dim=-1)          # (1, num_patches)

        # 尝试不用 ReLU，看看原始值分布
        cam_raw = cam.clone()

        # ReLU
        cam = F.relu(cam)

        # 如果 ReLU 后全零，用原始绝对值
        if cam.max() == 0:
            logger.warning("GradCAM: all values <=0 after ReLU, using absolute values")
            cam = cam_raw.abs()

        # Reshape to grid
        num_patches = cam.shape[1]
        grid_size = int(num_patches ** 0.5)
        cam = cam.reshape(1, 1, grid_size, grid_size)

        # Upsample
        cam = F.interpolate(
            cam.float(),
            size=(image.size[1], image.size[0]),
            mode="bilinear", align_corners=False,
        )

        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam

    def compute_patch_similarities(self, image: Image.Image, text: str) -> np.ndarray:
        """直接计算 text 和每个 patch token 的 cosine similarity。"""
        inputs = self.processor(
            text=[text], images=[image],
            return_tensors="pt", padding=True, truncation=True,
        ).to(self.device)

        with torch.no_grad():
            vision_outputs = self.model.vision_model(
                pixel_values=inputs["pixel_values"],
                output_hidden_states=True,
            )
            patch_tokens = vision_outputs.last_hidden_state[:, 1:, :]
            patch_embeds = self.model.visual_projection(patch_tokens)
            patch_embeds = F.normalize(patch_embeds, dim=-1)

            text_outputs = self.model.text_model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            text_embeds = self.model.text_projection(text_outputs.pooler_output)
            text_embeds = F.normalize(text_embeds, dim=-1)

            sims = (patch_embeds * text_embeds.unsqueeze(1)).sum(dim=-1)

        num_patches = sims.shape[1]
        grid_size = int(num_patches ** 0.5)
        sims = sims.reshape(1, 1, grid_size, grid_size)

        sims = F.interpolate(
            sims.float(),
            size=(image.size[1], image.size[0]),
            mode="bilinear", align_corners=False,
        )
        sims = sims.squeeze().cpu().numpy()
        if sims.max() > sims.min():
            sims = (sims - sims.min()) / (sims.max() - sims.min())
        return sims


def save_heatmap_overlay(image, heatmap, title, save_path, method="GradCAM"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image)
    axes[0].set_title("Original image", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title(f"{method} heatmap", fontsize=11)
    axes[1].axis("off")

    img_array = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    heatmap_color = cm.jet(heatmap)[:, :, :3]
    overlay = 0.5 * img_array + 0.5 * heatmap_color
    axes[2].imshow(np.clip(overlay, 0, 1))
    axes[2].set_title("Overlay", fontsize=11)
    axes[2].axis("off")

    wrapped = "\n".join([title[i:i+90] for i in range(0, len(title), 90)])
    fig.suptitle(wrapped, fontsize=10, y=0.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"  Saved: {save_path}")


def main():
    print("=" * 60)
    print("GradCAM Visualization Experiment v2")
    print("=" * 60)

    output_dir = Path("results/gradcam_vis_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: 找一个有实质文本的 query ----
    print("\n[Step 1] Loading queries from MRMR Knowledge...")
    from datasets import load_dataset

    ds = load_dataset("MRMRbenchmark/knowledge", "query", split="test")

    # 筛选条件: 有图片 + Science/Medicine + 文本长度 > 50 字符
    candidates = []
    for i, row in enumerate(ds):
        qid = row["id"]
        text = row.get("text", "")
        has_image = any(
            row.get(k) is not None for k in ["image", "image 1", "image 2"]
        )
        is_target = any(
            d in qid.lower()
            for d in ["agriculture", "biology", "chemistry", "geography",
                       "clinical_medicine", "diagnostics", "basic_medical", "pharmacy"]
        )
        text_long_enough = len(text) > 50  # 跳过 "What is this?" 之类的

        if has_image and is_target and text_long_enough:
            candidates.append(i)

    print(f"  Found {len(candidates)} queries with images + substantial text")

    # 随机选 2 个跑
    random.seed(123)
    selected = random.sample(candidates, min(2, len(candidates)))

    for sel_idx, idx in enumerate(selected):
        row = ds[idx]
        qid = row["id"]
        query_text = row["text"]

        # 拿图片
        query_image = None
        for key in ["image", "image 1", "image 2"]:
            if row.get(key) is not None and isinstance(row[key], Image.Image):
                query_image = row[key].convert("RGB")
                break

        if query_image is None:
            continue

        print(f"\n{'='*60}")
        print(f"  Query {sel_idx+1}: {qid}")
        print(f"  Text: {query_text[:150]}...")
        print(f"  Image: {query_image.size}")

        query_image.save(output_dir / f"{qid}_original.png")

        # ---- Step 2: Decompose ----
        print(f"\n[Step 2] Decomposing query...")
        from decomposition.text_decompose import TextDecomposer
        decomposer = TextDecomposer(device="cuda")
        subqueries = decomposer.decompose_query(query_text)

        print(f"  {len(subqueries)} subqueries:")
        for i, sq in enumerate(subqueries):
            print(f"    [{i}] {sq}")

        # ---- Step 3: GradCAM + PatchSim ----
        print(f"\n[Step 3] Computing heatmaps...")
        gradcam = CLIPGradCAM(device="cuda")

        for i, sq in enumerate(subqueries):
            print(f"\n  --- SQ{i}: {sq[:70]}...")

            hm_gc = gradcam.compute_heatmap(query_image, sq)
            save_heatmap_overlay(
                query_image, hm_gc,
                title=f"[GradCAM] SQ{i}: {sq}",
                save_path=str(output_dir / f"{qid}_sq{i}_gradcam.png"),
                method="GradCAM",
            )

            hm_ps = gradcam.compute_patch_similarities(query_image, sq)
            save_heatmap_overlay(
                query_image, hm_ps,
                title=f"[PatchSim] SQ{i}: {sq}",
                save_path=str(output_dir / f"{qid}_sq{i}_patchsim.png"),
                method="PatchSim",
            )

        # Full query baseline
        hm_full = gradcam.compute_heatmap(query_image, query_text)
        save_heatmap_overlay(
            query_image, hm_full,
            title=f"[GradCAM] Full: {query_text[:80]}",
            save_path=str(output_dir / f"{qid}_full_gradcam.png"),
            method="GradCAM",
        )

    print(f"\n{'='*60}")
    print(f"All results saved to: {output_dir}/")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  {f.name}")
    print("=" * 60)


if __name__ == "__main__":
    main()