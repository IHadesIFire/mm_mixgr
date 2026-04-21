"""验证 GME-Qwen2-VL 的 value/hidden 在中间层是否保留视觉 token 的空间对齐性。

目的(对应 FOCUS 论文里 Neo et al. ICLR 2025 的核心前提):
    对 GME forward 一次 (image + "subiculum"),
    取某中间层 hidden_states,
    算 "subiculum" text token 和每个 visual token 的 cosine similarity,
    reshape 成 2D 热力图叠在原图上,
    肉眼看高亮是否落在图中 "Sub" 椭圆附近。

结论判据:
    - 高亮聚焦到 Sub 位置附近  → GME 保留空间对齐性, FOCUS 可以在 GME 上跑
    - 全图均匀糊 / 高亮在角落  → GME 的 contrastive 微调破坏了对齐, 需要换回原版 Qwen2-VL 做定位

运行方式(两种环境都支持):
    本地:  python -m tests.test_GMEattention
    Colab: !python test_GMEattention.py

依赖:
    transformers>=4.45, datasets, torch, matplotlib, pillow, qwen-vl-utils (可选)
    Colab 环境会在线下载 Alibaba-NLP/gme-Qwen2-VL-7B-Instruct (~15GB, 用 A100 / L4)
    图像从 MRMR corpus 里按 TARGET_DOC_ID 查出来, 无需本地文件
"""

from __future__ import annotations

import os

# HF 缓存路径: 通过环境变量覆盖; 本地走 D:\hf_cache, Colab 走默认 ~/.cache/huggingface
# 在 Colab 里启动前可选: os.environ["HF_HUB_CACHE"] = "/content/hf_cache/hub"
# 这里只在环境变量未设置时给本地默认值, 所以 Colab 不会被污染
if os.name == "nt":  # Windows 本地
    os.environ.setdefault("HF_DATASETS_CACHE", r"D:\hf_cache\datasets")
    os.environ.setdefault("HF_HUB_CACHE", r"D:\hf_cache\hub")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
# Colab / Linux 不设置 offline, 让它按需下载

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from PIL import Image


# ============================================================
# 配置区 - 改这里就够
# ============================================================
GME_MODEL_NAME = "Alibaba-NLP/gme-Qwen2-VL-7B-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 目标文档: subiculum 脑解剖示意图
TARGET_DOC_ID = "test_Diagnostics_and_Laboratory_Medicine_76_additional"

# 目标词: 会在 Qwen2-VL tokenizer 里被拆成 subword
TARGET_WORD = "subiculum"

# 选几层画热力图. Qwen2-VL-7B 有 28 层, FOCUS 论文推荐后 25-60% (~14-28)
# 画这 3 层做对比, 看后面层是不是更清晰
LAYERS_TO_PROBE = [14, 20, 27]  # 27 是最后一层 (0-indexed)

# 输出目录: 本地放项目内 results/gme_spatial_probe,
# 单独上传 Colab 时(没有 tests/ 父目录)退回到 cwd/results/gme_spatial_probe;
# 环境变量 GME_PROBE_OUT 优先级最高
_here = Path(__file__).resolve()
if _here.parent.name == "tests":
    _default_out = _here.parent.parent / "results" / "gme_spatial_probe"
else:
    _default_out = Path.cwd() / "results" / "gme_spatial_probe"
OUT_DIR = Path(os.environ.get("GME_PROBE_OUT", str(_default_out)))
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[init] 输出目录: {OUT_DIR}")


# ============================================================
# Step 1: 加载图像
# ============================================================
def load_subiculum_image() -> Image.Image:
    """从 MRMR corpus 找到 subiculum 那条并取出 image 列."""
    print(f"[1/6] 加载 MRMR corpus, 找 {TARGET_DOC_ID} ...")
    ds = load_dataset("MRMRbenchmark/knowledge", "corpus", split="test")
    for raw_idx in range(len(ds)):
        row = ds[raw_idx]
        if str(row.get("id")) == TARGET_DOC_ID:
            img = row.get("image")
            if img is None:
                raise RuntimeError("这条 row 的 image 列是空的")
            if not isinstance(img, Image.Image):
                # HF datasets 有时返回 dict {'bytes':..., 'path':...}
                from io import BytesIO
                img = Image.open(BytesIO(img["bytes"]))
            img = img.convert("RGB")
            print(f"    找到! raw_idx={raw_idx}, size={img.size}")
            return img
    raise RuntimeError(f"corpus 里没找到 id={TARGET_DOC_ID}")


# ============================================================
# Step 2: 加载 GME 模型 + processor
# ============================================================
def load_gme():
    """加载 GME. AutoModel 包了一层, 底层是 Qwen2VL."""
    print(f"[2/6] 加载 GME 模型 {GME_MODEL_NAME} 到 {DEVICE} ...")
    from transformers import AutoModel, AutoProcessor

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        GME_MODEL_NAME,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=DEVICE,
    )
    model.eval()

    # GME 的 processor 通常也要 trust_remote_code
    processor = AutoProcessor.from_pretrained(
        GME_MODEL_NAME,
        trust_remote_code=True,
    )
    print(f"    done. 模型类型: {type(model).__name__}")

    # 打印关键子模块, 便于定位底层 Qwen2VL
    print("    top-level children:")
    for name, _ in model.named_children():
        print(f"      - {name}")

    return model, processor


# ============================================================
# Step 3: 定位 "底层 Qwen2VL" 的 forward 入口
# ============================================================
def get_base_qwen2vl(gme_model):
    """
    GME wrap 了 Qwen2VL. 要拿 hidden_states 必须调底层 forward.
    常见路径(按优先级试):
      - gme_model.base
      - gme_model.model
      - gme_model itself (如果 GME 其实就是直接继承 Qwen2VLForConditionalGeneration)
    打印结构帮助定位.
    """
    print("[3/6] 定位底层 Qwen2VL forward 入口 ...")

    candidates = []
    for attr in ["base", "model", "qwen2_vl", "vlm"]:
        if hasattr(gme_model, attr):
            candidates.append((attr, getattr(gme_model, attr)))

    if not candidates:
        # GME 自己就是 Qwen2VL 的情况
        print("    没找到 wrap 属性, 假设 gme_model 自己就是 Qwen2VL")
        return gme_model

    for attr, sub in candidates:
        cls_name = type(sub).__name__
        print(f"    候选: gme_model.{attr} → {cls_name}")
        if "Qwen2VL" in cls_name or hasattr(sub, "model"):
            print(f"    选用: gme_model.{attr}")
            return sub

    # 兜底: 取第一个
    print(f"    兜底选第一个: gme_model.{candidates[0][0]}")
    return candidates[0][1]


# ============================================================
# Step 4: 构造输入 & forward 拿 hidden_states
# ============================================================
def forward_and_get_hidden_states(model, processor, image: Image.Image, query_text: str):
    """
    用 Qwen2-VL 的 chat template 构造 (image, text) 输入, forward 一次, 返回:
      - hidden_states: tuple of [1, seq_len, hidden_dim], 长度 = num_layers+1 (含 embedding)
      - input_ids: [1, seq_len]
      - image_grid_thw: [1, 3] (t, h, w) patches, 注意是 MERGE 前的 grid
      - visual_token_indices: visual token 在 seq 里的位置 list
      - target_token_indices: TARGET_WORD 对应的 subword token 位置 list
    """
    print(f"[4/6] 构造输入并 forward: query='{query_text}' ...")

    # Qwen2-VL chat template: messages list of dicts
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query_text},
            ],
        }
    ]

    # processor 会自动处理 image + chat template
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(DEVICE)

    input_ids = inputs["input_ids"]  # [1, seq_len]
    seq_len = input_ids.shape[1]
    print(f"    seq_len = {seq_len}")
    print(f"    image_grid_thw = {inputs.get('image_grid_thw')}")

    # 找 visual token 范围: 在 Qwen2-VL 里, image token id 是特殊 id
    # 用 processor.tokenizer 查 vision_start / vision_end / image_pad 的 id
    tok = processor.tokenizer
    vision_start_id = tok.convert_tokens_to_ids("<|vision_start|>")
    vision_end_id = tok.convert_tokens_to_ids("<|vision_end|>")
    image_pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
    print(f"    special ids: vision_start={vision_start_id}, vision_end={vision_end_id}, image_pad={image_pad_id}")

    ids_list = input_ids[0].tolist()

    # visual token 占位符是 image_pad_id, 中间连续一段
    visual_token_indices = [i for i, tid in enumerate(ids_list) if tid == image_pad_id]
    print(f"    visual_token 数量 = {len(visual_token_indices)}")
    if visual_token_indices:
        print(f"    visual_token 范围: [{visual_token_indices[0]}, {visual_token_indices[-1]}]")

    # 找 TARGET_WORD 对应的 subword token 位置
    # 先用 tokenizer 单独 encode target, 拿到它会被拆成哪几个 id
    target_ids = tok.encode(TARGET_WORD, add_special_tokens=False)
    # 也试试带前导空格的版本 (Qwen 的 BPE 对前导空格敏感)
    target_ids_space = tok.encode(" " + TARGET_WORD, add_special_tokens=False)
    print(f"    target='{TARGET_WORD}' → token ids: {target_ids}")
    print(f"    target=' {TARGET_WORD}' → token ids: {target_ids_space}")

    # 在 input_ids 里搜子序列
    def find_subseq(haystack: list[int], needle: list[int]) -> list[int]:
        """返回 needle 在 haystack 中的起始位置列表."""
        out = []
        n = len(needle)
        for i in range(len(haystack) - n + 1):
            if haystack[i : i + n] == needle:
                out.append(i)
        return out

    target_starts = find_subseq(ids_list, target_ids)
    if not target_starts:
        target_starts = find_subseq(ids_list, target_ids_space)
        if target_starts:
            target_ids = target_ids_space
    if not target_starts:
        raise RuntimeError(f"在 input_ids 里找不到 '{TARGET_WORD}' 的 token 序列")

    target_start = target_starts[0]
    target_token_indices = list(range(target_start, target_start + len(target_ids)))
    print(f"    target_token_indices = {target_token_indices}")
    # 解码回来确认
    decoded = tok.decode(input_ids[0, target_token_indices])
    print(f"    decoded target tokens: '{decoded}'")

    # Forward
    # 底层 Qwen2VL 的 forward 支持 output_hidden_states
    print("    forward...")
    with torch.inference_mode():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states  # tuple, len = num_layers + 1
    print(f"    num layers (含 embedding) = {len(hidden_states)}")
    print(f"    hidden dim = {hidden_states[0].shape[-1]}")

    return {
        "hidden_states": hidden_states,
        "input_ids": input_ids,
        "image_grid_thw": inputs.get("image_grid_thw"),
        "visual_token_indices": visual_token_indices,
        "target_token_indices": target_token_indices,
    }


# ============================================================
# Step 5: 算 cosine similarity 热力图
# ============================================================
def compute_relevance_map(
    hidden_state: torch.Tensor,
    visual_idx: list[int],
    target_idx: list[int],
) -> np.ndarray:
    """
    hidden_state: [1, seq_len, hidden_dim], 某一层的输出
    返回: [num_visual_tokens] 的 1D 数组, 每个 visual token 对 target 的平均 cosine

    策略: 对 target 的每个 subword 分别算一张 map, 再 element-wise 相乘 (FOCUS 论文做法)
    相乘能让"所有 subword 都指向的位置"脱颖而出.
    """
    h = hidden_state[0]  # [seq_len, hidden_dim]
    h = F.normalize(h.float(), dim=-1)  # L2 normalize, 保证 cosine 数值稳定

    visual_feats = h[visual_idx]  # [num_v, D]

    # 对每个 target subword 算 cosine, 归一化到 [0,1], 然后累乘
    rel_map = None
    for t_idx in target_idx:
        t_feat = h[t_idx]  # [D]
        sims = (visual_feats * t_feat).sum(dim=-1)  # [num_v], cosine
        # 归一化到 [0,1], 避免负数相乘翻号
        sims_min, sims_max = sims.min(), sims.max()
        if (sims_max - sims_min).item() < 1e-6:
            # 完全糊, 直接返回全 0
            return np.zeros(len(visual_idx), dtype=np.float32)
        sims_norm = (sims - sims_min) / (sims_max - sims_min)
        rel_map = sims_norm if rel_map is None else rel_map * sims_norm

    return rel_map.cpu().numpy().astype(np.float32)


def reshape_to_2d(rel_map: np.ndarray, image_grid_thw) -> np.ndarray:
    """
    Qwen2-VL: 图像先切 h×w patch, ViT 后面有 2×2 spatial merge,
    所以最终 visual token 数 = (h/2) × (w/2) (假设 t=1).
    image_grid_thw 存的是 [t, h, w] (merge 前的 patch grid).

    返回: [H, W] 的 2D 数组, H=h//2, W=w//2.
    """
    if image_grid_thw is None:
        # fallback: 假设是正方形
        n = len(rel_map)
        side = int(np.sqrt(n))
        return rel_map[: side * side].reshape(side, side)

    thw = image_grid_thw[0].tolist()  # [t, h, w]
    t, h, w = thw
    # Qwen2-VL 的 merge_size 通常是 2
    merge = 2
    H, W = h // merge, w // merge
    expected = t * H * W
    print(f"    reshape: grid (t,h,w)=({t},{h},{w}), merged (H,W)=({H},{W}), expected={expected}, actual={len(rel_map)}")
    if expected != len(rel_map):
        # merge_size 可能不是 2, 尝试直接 h*w
        if t * h * w == len(rel_map):
            print("    fallback: merge_size=1, 用原 grid")
            H, W = h, w
        else:
            raise RuntimeError(f"token 数对不上: expected {expected} or {t*h*w}, got {len(rel_map)}")
    # t=1 情况直接 reshape; t>1 就取第一帧
    rel_2d = rel_map[: H * W].reshape(H, W)
    return rel_2d


# ============================================================
# Step 6: 可视化叠在原图上
# ============================================================
def visualize_and_save(
    image: Image.Image,
    rel_2d: np.ndarray,
    layer_idx: int,
    save_path: Path,
):
    """左: 原图; 中: 热力图; 右: 叠加."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title(f"Original (size={image.size})")
    axes[0].axis("off")

    # 热力图本身
    axes[1].imshow(rel_2d, cmap="hot", interpolation="nearest")
    axes[1].set_title(f"Relevance map layer={layer_idx}\nshape={rel_2d.shape}")
    axes[1].axis("off")

    # 叠加: 把热力图 resize 到原图尺寸
    from PIL import Image as PILImage

    heat_img = PILImage.fromarray((rel_2d * 255).astype(np.uint8)).resize(
        image.size, resample=PILImage.BILINEAR
    )
    heat_arr = np.array(heat_img).astype(np.float32) / 255.0

    axes[2].imshow(image)
    axes[2].imshow(heat_arr, cmap="hot", alpha=0.5)
    axes[2].set_title(f"Overlay layer={layer_idx}\n'{TARGET_WORD}' → visual tokens")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"    保存: {save_path}")


# ============================================================
# 主流程
# ============================================================
def main():
    # 1. 拿图
    image = load_subiculum_image()
    image.save(OUT_DIR / "input_image.png")

    # 2. 加载 GME
    gme_model, processor = load_gme()

    # 3. 定位底层 forward (有的版本 GME 自己就暴露 forward, 直接试)
    base = get_base_qwen2vl(gme_model)

    # 4. Forward + 拿 hidden_states
    # 用 FOCUS 论文的 existence prompt 格式, 让 target 作为自然句里的一个 token 出现
    query_text = f"Is there a {TARGET_WORD} in the image?"
    data = forward_and_get_hidden_states(base, processor, image, query_text)

    hidden_states = data["hidden_states"]
    visual_idx = data["visual_token_indices"]
    target_idx = data["target_token_indices"]
    image_grid_thw = data["image_grid_thw"]

    if not visual_idx:
        raise RuntimeError("没找到 visual token, 检查 image_pad_id 是否正确")

    # 5. 对每层算 relevance map 并可视化
    print(f"[5/6] 为层 {LAYERS_TO_PROBE} 计算 relevance map ...")
    num_layers_with_embed = len(hidden_states)  # = num_layers + 1
    for layer_idx in LAYERS_TO_PROBE:
        # hidden_states[0] 是 embedding 输出, hidden_states[k] 是第 k 层 transformer 后
        hs_idx = layer_idx + 1  # +1 因为 hidden_states[0] 是 embedding
        if hs_idx >= num_layers_with_embed:
            print(f"    跳过 layer={layer_idx}, 超出范围")
            continue
        print(f"    layer {layer_idx} ...")
        rel_1d = compute_relevance_map(hidden_states[hs_idx], visual_idx, target_idx)
        rel_2d = reshape_to_2d(rel_1d, image_grid_thw)
        # 打印统计量, 判断是不是糊成一片
        print(
            f"      rel_2d stats: min={rel_2d.min():.3f}, max={rel_2d.max():.3f}, "
            f"mean={rel_2d.mean():.3f}, std={rel_2d.std():.3f}"
        )
        save_path = OUT_DIR / f"layer_{layer_idx:02d}.png"
        visualize_and_save(image, rel_2d, layer_idx, save_path)

    print("[6/6] 完成. 查看:", OUT_DIR)
    print()
    print("===== 怎么判读 =====")
    print("  1) 看 layer_20.png 和 layer_27.png 的 overlay 列")
    print("  2) 高亮若聚焦到图中 Sub 椭圆 (图中央偏左的那块) → GME 保留空间对齐, FOCUS 可用")
    print("  3) 高亮若均匀一片糊, 或者 std 很小 (<0.1) → 对齐破坏, 需要换原版 Qwen2-VL")
    print("  4) 不同层表现差别大时, 以后层为准 (FOCUS 论文: 后 25-60% 层最好)")


if __name__ == "__main__":
    main()
