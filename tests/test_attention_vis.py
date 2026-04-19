"""
Token-level attention 可视化 + 多层对比实验:
- streaming 模式加载数据
- 对比不同 transformer 层的 weighted pooling 区分度
- 输出每层的 cosine similarity 统计

运行: python tests/test_attention_vis.py
"""

import sys
import random
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def find_document_with_image_streaming():
    from datasets import load_dataset

    logger.info("Streaming corpus to find a document with image + long text...")

    ds = load_dataset(
        "MRMRbenchmark/knowledge", "corpus",
        split="test", streaming=True,
    )

    candidates = []
    for i, row in enumerate(ds):
        text = row.get("text", "")
        has_image = False
        for key in ["image", "image 1", "image 2", "vision"]:
            val = row.get(key)
            if val is not None and isinstance(val, Image.Image):
                has_image = True
                break

        modality = row.get("modality", "")
        if has_image and "image" in modality and len(text) > 200:
            candidates.append(row)
            logger.info(f"  Found candidate {len(candidates)}: {row['id']} ({len(text)} chars)")

        if len(candidates) >= 5:
            break
        if i > 0 and i % 500 == 0:
            logger.info(f"  Scanned {i} docs, found {len(candidates)} candidates...")
        if i >= 3000:
            break

    if not candidates:
        logger.error("No suitable document found")
        return None

    random.seed(456)
    return random.choice(candidates)


def load_model(device="cuda"):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    model_name = "Qwen/Qwen3-VL-Embedding-2B"
    logger.info(f"Loading {model_name}...")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    logger.info("Model loaded.")
    return model, processor


def find_token_ranges(input_ids):
    """找到 visual token 和 text token 的位置范围。"""
    vision_start_id = 151652
    vision_end_id = 151653

    start_positions = (input_ids == vision_start_id).nonzero(as_tuple=True)[0]
    end_positions = (input_ids == vision_end_id).nonzero(as_tuple=True)[0]

    if len(start_positions) == 0 or len(end_positions) == 0:
        return None, None, None, None

    vis_start = start_positions[0].item() + 1
    vis_end = end_positions[0].item()
    text_start = vis_end + 1
    text_end = len(input_ids)

    return vis_start, vis_end, text_start, text_end


def run_model_once(model, processor, image, text):
    """
    对 (text + image) 跑一次 forward，返回 outputs 和 token 位置信息。
    输出包含所有层的 attention 和 hidden states。
    """
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": text},
    ]}]

    inputs = processor.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
        return_dict=True, return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
              for k, v in inputs.items()}

    input_ids = inputs["input_ids"][0]
    vis_start, vis_end, text_start, text_end = find_token_ranges(input_ids)

    if vis_start is None:
        return None, None

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )

    token_info = {
        "vis_start": vis_start,
        "vis_end": vis_end,
        "text_start": text_start,
        "text_end": text_end,
        "num_visual": vis_end - vis_start,
        "num_text": text_end - text_start,
    }

    return outputs, token_info


def extract_pooled_vectors(outputs, token_info, layer_idx=-1):
    """
    从指定层提取 weighted 和 uniform pooling 的视觉向量。

    Args:
        outputs: model forward 的输出
        token_info: token 位置信息
        layer_idx: 用哪层的 attention 和 hidden states

    Returns:
        weighted_vec, uniform_vec: 各 (hidden_dim,) tensor
    """
    vis_start = token_info["vis_start"]
    vis_end = token_info["vis_end"]
    text_start = token_info["text_start"]
    text_end = token_info["text_end"]

    # hidden states: 第 layer_idx 层的输出
    # outputs.hidden_states[0] 是 embedding 层, [1] 是第 1 层输出, ..., [-1] 是最后一层
    hidden = outputs.hidden_states[layer_idx]
    visual_vectors = hidden[0, vis_start:vis_end, :].float()  # (num_vis, hidden_dim)

    # Uniform pooling
    uniform_vec = visual_vectors.mean(dim=0).cpu()

    # Weighted pooling: 用同一层的 attention
    attn = outputs.attentions[layer_idx].float()  # (1, heads, seq, seq)
    text_to_vis = attn[:, :, text_start:text_end, vis_start:vis_end]
    weights = text_to_vis.mean(dim=(0, 1, 2))  # (num_vis,)
    weights = weights / weights.sum()

    weighted_vec = (weights.unsqueeze(-1) * visual_vectors).sum(dim=0).cpu()

    return weighted_vec, uniform_vec


def extract_attention_heatmap(outputs, token_info, image_size, layer_idx=-1):
    """从指定层提取 attention heatmap。"""
    vis_start = token_info["vis_start"]
    vis_end = token_info["vis_end"]
    text_start = token_info["text_start"]
    text_end = token_info["text_end"]
    num_visual = token_info["num_visual"]

    attn = outputs.attentions[layer_idx].float()
    text_to_vis = attn[:, :, text_start:text_end, vis_start:vis_end]
    attn_weights = text_to_vis.mean(dim=(0, 1, 2)).cpu().numpy()

    # 推算 grid 尺寸
    grid_size = int(num_visual ** 0.5)
    # 尝试找到最接近的 h x w
    best_h, best_w = grid_size, grid_size
    for h in range(1, num_visual + 1):
        if num_visual % h == 0:
            w = num_visual // h
            if abs(h / w - image_size[1] / image_size[0]) < abs(best_h / best_w - image_size[1] / image_size[0]):
                best_h, best_w = h, w

    if best_h * best_w != num_visual:
        attn_weights = attn_weights[:best_h * best_w]

    heatmap = attn_weights.reshape(best_h, best_w)

    hm = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).float()
    hm = F.interpolate(hm, size=(image_size[1], image_size[0]),
                       mode="bilinear", align_corners=False)
    hm = hm.squeeze().numpy()

    if hm.max() > hm.min():
        hm = (hm - hm.min()) / (hm.max() - hm.min())
    else:
        hm = np.zeros_like(hm)

    return hm


def save_individual(image, prop, heatmap, save_path, index, layer_idx):
    img_array = np.array(image.convert("RGB")).astype(np.float32) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].imshow(image)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title(f"Attention heatmap (layer {layer_idx})", fontsize=11)
    axes[1].axis("off")

    heatmap_color = cm.jet(heatmap)[:, :, :3]
    overlay = 0.5 * img_array + 0.5 * heatmap_color
    axes[2].imshow(np.clip(overlay, 0, 1))
    axes[2].set_title("Overlay", fontsize=11)
    axes[2].axis("off")

    wrapped = "\n".join([prop[j:j+90] for j in range(0, len(prop), 90)])
    fig.suptitle(f"Layer {layer_idx} | P{index}: {wrapped}", fontsize=9, y=0.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    print("=" * 60)
    print("Multi-Layer Attention Comparison")
    print("=" * 60)

    output_dir = Path("results/attention_vis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: 找文档
    row = find_document_with_image_streaming()
    if row is None:
        return

    doc_id = row["id"]
    doc_text = row["text"]

    doc_image = None
    for key in ["image", "image 1", "image 2", "vision"]:
        val = row.get(key)
        if val is not None and isinstance(val, Image.Image):
            doc_image = val.convert("RGB")
            break

    print(f"\n  Document: {doc_id}")
    print(f"  Text: {doc_text[:200]}...")
    print(f"  Image: {doc_image.size}")

    doc_image.save(output_dir / f"{doc_id}_original.png")

    # Step 2: 分解
    print(f"\n[Step 2] Decomposing...")
    from decomposition.text_decompose import TextDecomposer
    decomposer = TextDecomposer(device="cuda")
    propositions = decomposer.decompose_document(doc_text)

    print(f"  {len(propositions)} propositions:")
    for i, p in enumerate(propositions):
        print(f"    [P{i}] {p[:80]}{'...' if len(p) > 80 else ''}")

    del decomposer
    torch.cuda.empty_cache()

    # Step 3: 加载模型
    print(f"\n[Step 3] Loading model...")
    model, processor = load_model(device="cuda")

    # Step 4: 对每个 proposition 跑一次 forward，缓存 outputs
    print(f"\n[Step 4] Running forward passes...")
    all_outputs = []
    num_layers = None
    for i, prop in enumerate(propositions):
        outputs, token_info = run_model_once(model, processor, doc_image, prop)
        if outputs is not None:
            all_outputs.append((i, prop, outputs, token_info))
            if num_layers is None:
                num_layers = len(outputs.attentions)
        if i == 0:
            print(f"  Visual tokens: {token_info['num_visual']}, Text tokens: {token_info['num_text']}")
            print(f"  Total layers: {num_layers}")

    print(f"  Got {len(all_outputs)} valid outputs")

    # Step 5: 对比不同层
    print(f"\n[Step 5] Comparing layers...")
    print(f"  Testing layers: 0, 1, 2, 5, 13, {num_layers - 1} (last)")

    test_layers = [0, 1, 2, 5, 13, num_layers - 1]
    # 去掉超出范围的层
    test_layers = [l for l in test_layers if l < num_layers]

    layer_results = {}

    for layer_idx in test_layers:
        weighted_vecs = []
        uniform_vecs = []

        for i, prop, outputs, token_info in all_outputs:
            w_vec, u_vec = extract_pooled_vectors(outputs, token_info, layer_idx=layer_idx)
            weighted_vecs.append(w_vec)
            uniform_vecs.append(u_vec)

        n = len(weighted_vecs)
        w_off = []
        u_off = []
        for a in range(n):
            for b in range(a + 1, n):
                w_off.append(F.cosine_similarity(
                    weighted_vecs[a].unsqueeze(0),
                    weighted_vecs[b].unsqueeze(0),
                ).item())
                u_off.append(F.cosine_similarity(
                    uniform_vecs[a].unsqueeze(0),
                    uniform_vecs[b].unsqueeze(0),
                ).item())

        w_mean = np.mean(w_off)
        w_min = np.min(w_off)
        w_std = np.std(w_off)
        u_mean = np.mean(u_off)
        diff = u_mean - w_mean

        layer_results[layer_idx] = {
            "w_mean": w_mean, "w_min": w_min, "w_std": w_std,
            "u_mean": u_mean, "diff": diff,
        }

        print(f"\n  Layer {layer_idx:3d}:")
        print(f"    Uniform  - mean: {u_mean:.6f}")
        print(f"    Weighted - mean: {w_mean:.6f}, min: {w_min:.6f}, std: {w_std:.6f}")
        print(f"    Diff (uniform - weighted): {diff:.6f}")

    # Step 6: 总结
    print(f"\n{'='*60}")
    print(f"  Layer comparison summary")
    print(f"{'='*60}")
    print(f"  {'Layer':>8s} {'U_mean':>10s} {'W_mean':>10s} {'W_min':>10s} {'W_std':>10s} {'Diff':>10s}")
    print(f"  {'-'*58}")

    best_layer = None
    best_diff = -1

    for layer_idx in test_layers:
        r = layer_results[layer_idx]
        marker = ""
        if r["diff"] > best_diff:
            best_diff = r["diff"]
            best_layer = layer_idx
        print(f"  {layer_idx:>8d} {r['u_mean']:>10.6f} {r['w_mean']:>10.6f} "
              f"{r['w_min']:>10.6f} {r['w_std']:>10.6f} {r['diff']:>10.6f}")

    print(f"\n  Best layer: {best_layer} (diff = {best_diff:.6f})")

    if best_diff > 0.01:
        print(f"  → Weighted pooling 在 layer {best_layer} 产生了显著差异!")
    elif best_diff > 0.001:
        print(f"  → Layer {best_layer} 有一定差异，值得进一步测试")
    else:
        print(f"  → 所有层差异都很小，weighted pooling 方案可能需要调整")

    # Step 7: 保存最佳层的热力图
    print(f"\n[Step 7] Saving heatmaps for best layer ({best_layer})...")
    for i, prop, outputs, token_info in all_outputs:
        hm = extract_attention_heatmap(outputs, token_info, doc_image.size, layer_idx=best_layer)
        save_individual(doc_image, prop, hm,
                       output_dir / f"{doc_id}_L{best_layer}_prop{i}.png",
                       i, best_layer)

    print(f"\n{'='*60}")
    print(f"Results in: {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()