"""多句子版 GME 空间对齐探针 (key_noun 策略).

目的: 验证 GME 的 relevance map 是否随**主题词**移到不同区域,
还是之前 subiculum 的成功只是"认图里的 'Sub' 字样".

策略:
- 4 对 (sentence, key_noun), 每句瞄准空间上能区分的解剖区域,
  key_noun 是人工挑的主题词, 刻意不含图里的文字标签 (EC/DG/Sub/CA1/CA3)
- 整句作为上下文送模型; target = key_noun 的 subword 位置
  (和 subiculum 实验同一套逻辑, FOCUS 式 subword consensus)
- 只探 layer 20 (上次验证最强)
- 输出 grid: N 行 × 2 列 (heatmap | overlay), 另外每句单独一张大图

运行:
    本地:  python -m tests.test_GMEattention_multi
    Colab: !cd /content/mm_mixgr && python -m tests.test_GMEattention_multi
    (Colab 前需挂 Drive + 设 HF_*_CACHE 到 Drive)

之前 last-content-token 策略失败的原因:
    句号/EOS token 在 causal LM 里是 attention sink, 承载的是
    "陈述句结束" 这种格式信号, 而不是句子具体语义. 所以 4 句产出几乎一模一样的 map.
    换回 subword-level 的 key_noun target, 即回到 subiculum 成功那套信号通道.
"""

from __future__ import annotations

import os

# HF 缓存路径 (同现有脚本逻辑, 只在 Windows 本地强制 offline)
if os.name == "nt":
    os.environ.setdefault("HF_DATASETS_CACHE", r"D:\hf_cache\datasets")
    os.environ.setdefault("HF_HUB_CACHE", r"D:\hf_cache\hub")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

# 复用现有脚本的工具函数
from tests.test_GMEattention import (
    DEVICE,
    compute_relevance_map,
    install_hidden_state_hooks,
    load_gme,
    load_subiculum_image,
    reshape_to_2d,
)


# ============================================================
# 配置区
# ============================================================
# 4 句测试文本 + 人工挑的 key noun (subword-level target).
# 策略: 整句送给模型提供上下文, 但 target 只用 key noun 的 subword 位置
# (和 subiculum 实验一样, FOCUS 式 subword map 相乘做 consensus).
# 每句瞄准一个空间上能区分的解剖区域,
# 刻意不含图中出现的标签词 (EC / DG / Sub / CA1 / CA3 / subiculum / hippocampus)
SENTENCES = [
    # S1 → 左侧内嗅皮层 (EC 那一大片分层结构)
    ("Stratified cortical layers with apical dendrites extending toward the pial surface.",
     "dendrites"),
    # S2 → 顶部齿状回 (DG 那个 V 形结构)
    ("Densely packed granule cells arranged in a V-shaped layer.",
     "granule"),
    # S3 → 右侧主海马 (CA1/CA3 的锥体神经元弧)
    ("Large pyramidal neurons aligned in a single dense row along a curved arc.",
     "pyramidal"),
    # S4 → 左下角插图
    ("A simplified schematic block diagram with labeled arrows between boxes.",
     "schematic"),
]

# 只探最强那层 (上次验证: layer 20 信号最干净)
LAYER_TO_PROBE = 20

# 输出目录
_here = Path(__file__).resolve()
if _here.parent.name == "tests":
    _default_out = _here.parent.parent / "results" / "gme_spatial_probe_multi"
else:
    _default_out = Path.cwd() / "results" / "gme_spatial_probe_multi"
OUT_DIR = Path(os.environ.get("GME_PROBE_OUT_MULTI", str(_default_out)))
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[init] 输出目录: {OUT_DIR}")


# ============================================================
# 单次 forward: 拿句子的 last-content-token 对 visual token 的 relevance
# ============================================================
def forward_for_sentence(
    gme_model,
    captured: dict,
    processor,
    image: Image.Image,
    sentence: str,
    key_noun: str,
):
    """
    对 (image, sentence) forward 一次, 返回:
      - rel_1d: [num_visual_tokens]  key_noun 的 subword hidden vs 每个 visual token 的 cosine
                  (多个 subword 做 FOCUS 式 element-wise 相乘 consensus)
      - image_grid_thw: 用于 reshape 回 2D

    sentence = 送给模型的完整上下文句子
    key_noun = 句子里人工挑的主题词, 只用它的 subword 位置做 target
    """
    # 构造 chat template 输入 (整句作为上下文)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": sentence},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    inputs = processor(
        text=[text], images=[image], padding=True, return_tensors="pt"
    ).to(DEVICE)

    input_ids = inputs["input_ids"]
    ids_list = input_ids[0].tolist()
    tok = processor.tokenizer

    # 定位视觉 token 范围
    image_pad_id = tok.convert_tokens_to_ids("<|image_pad|>")
    visual_idx = [i for i, t in enumerate(ids_list) if t == image_pad_id]
    if not visual_idx:
        raise RuntimeError("没找到 visual token")

    # 定位 key_noun 的 subword 位置 (BPE 对前导空格敏感, 两种都试)
    noun_ids = tok.encode(key_noun, add_special_tokens=False)
    noun_ids_space = tok.encode(" " + key_noun, add_special_tokens=False)

    def find_subseq(hay: list[int], needle: list[int]) -> int:
        n = len(needle)
        for i in range(len(hay) - n + 1):
            if hay[i : i + n] == needle:
                return i
        return -1

    start = find_subseq(ids_list, noun_ids)
    if start < 0:
        start = find_subseq(ids_list, noun_ids_space)
        if start >= 0:
            noun_ids = noun_ids_space
    if start < 0:
        raise RuntimeError(
            f"key_noun '{key_noun}' 在 input_ids 里找不到. "
            f"请确认它在句子 '{sentence}' 里确实出现."
        )

    target_indices = list(range(start, start + len(noun_ids)))
    decoded = tok.decode(input_ids[0, target_indices])
    print(
        f"    key_noun='{key_noun}' → token 位置 {target_indices}, "
        f"decoded='{decoded}' (visual 范围: {visual_idx[0]}~{visual_idx[-1]})"
    )

    # Forward (GME 正常跑, hook 抓中间层)
    captured.clear()
    with torch.inference_mode():
        _ = gme_model(**inputs)

    if LAYER_TO_PROBE not in captured:
        raise RuntimeError(f"layer {LAYER_TO_PROBE} 未被 hook 捕获")

    hs = captured[LAYER_TO_PROBE]  # [1, seq, hidden]
    # 多 subword target: compute_relevance_map 会逐个算 → 归一化 → 相乘 (FOCUS consensus)
    rel_1d = compute_relevance_map(hs, visual_idx, target_indices)

    return rel_1d, inputs.get("image_grid_thw")


# ============================================================
# 可视化: grid 图 + 每句单独大图
# ============================================================
def save_grid(image: Image.Image, results: list, save_path: Path):
    """N 行 × 2 列: heatmap | overlay. results 是 (sent, noun, rel_2d) 元组列表."""
    n = len(results)
    fig, axes = plt.subplots(n, 2, figsize=(14, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i, (sent, noun, rel_2d) in enumerate(results):
        # 左: 低分辨率热力图
        axes[i, 0].imshow(rel_2d, cmap="hot", interpolation="nearest")
        axes[i, 0].set_title(f"S{i+1} '{noun}' heatmap {rel_2d.shape}")
        axes[i, 0].axis("off")
        # 右: overlay
        heat_img = Image.fromarray((rel_2d * 255).astype(np.uint8)).resize(
            image.size, resample=Image.BILINEAR
        )
        heat_arr = np.array(heat_img).astype(np.float32) / 255.0
        axes[i, 1].imshow(image)
        axes[i, 1].imshow(heat_arr, cmap="hot", alpha=0.5)
        # 标题包含 key_noun + 句子 (截短)
        title = sent if len(sent) <= 80 else sent[:77] + "..."
        axes[i, 1].set_title(f"S{i+1} key='{noun}': {title}", fontsize=10)
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"    保存 grid: {save_path}")


def save_per_sentence(image: Image.Image, results: list, out_dir: Path, layer: int):
    """每句一张大图, 方便放大看. results 是 (sent, noun, rel_2d) 元组列表."""
    for i, (sent, noun, rel_2d) in enumerate(results):
        heat_img = Image.fromarray((rel_2d * 255).astype(np.uint8)).resize(
            image.size, resample=Image.BILINEAR
        )
        heat_arr = np.array(heat_img).astype(np.float32) / 255.0
        plt.figure(figsize=(10, 5))
        plt.imshow(image)
        plt.imshow(heat_arr, cmap="hot", alpha=0.5)
        plt.title(f"S{i+1} (layer={layer}, key='{noun}'): {sent}", fontsize=11)
        plt.axis("off")
        plt.tight_layout()
        p = out_dir / f"sentence_{i+1}_layer{layer:02d}.png"
        plt.savefig(p, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"    保存: {p}")


# ============================================================
# 主流程
# ============================================================
def main():
    # 1. 拿图 (同现有脚本, 走 MRMR 缓存)
    image = load_subiculum_image()
    image.save(OUT_DIR / "input_image.png")

    # 2. 加载 GME
    gme_model, processor = load_gme()

    # 3. 装 hook, 只装在目标层
    captured, cleanup = install_hidden_state_hooks(gme_model, [LAYER_TO_PROBE])

    # 4. 依次跑每个 (句子, key_noun)
    results = []  # list of (sentence, key_noun, rel_2d)
    try:
        for i, (sent, noun) in enumerate(SENTENCES):
            print(f"\n[句子 {i+1}/{len(SENTENCES)}] key_noun='{noun}'")
            print(f"    {sent}")
            rel_1d, grid_thw = forward_for_sentence(
                gme_model, captured, processor, image, sent, noun
            )
            rel_2d = reshape_to_2d(rel_1d, grid_thw)
            print(
                f"    rel_2d stats: min={rel_2d.min():.3f}, max={rel_2d.max():.3f}, "
                f"mean={rel_2d.mean():.3f}, std={rel_2d.std():.3f}"
            )
            results.append((sent, noun, rel_2d))
    finally:
        cleanup()

    # 5. 可视化
    grid_path = OUT_DIR / f"sentence_grid_layer{LAYER_TO_PROBE:02d}.png"
    save_grid(image, results, grid_path)
    save_per_sentence(image, results, OUT_DIR, LAYER_TO_PROBE)

    # 6. 打印判读提示
    print("\n" + "=" * 60)
    print("判读指南 (每句的 target token = 人工挑的 key_noun 的 subword)")
    print("=" * 60)
    print("  S1 key='dendrites' → 应亮图左侧分层皮层 (EC 区, 放射状树突)")
    print("  S2 key='granule'   → 应亮图顶部 V/U 形结构 (DG 颗粒细胞层)")
    print("  S3 key='pyramidal' → 应亮图右侧弧形排列神经元 (CA1/CA3)")
    print("  S4 key='schematic' → 应亮图左下角小插图")
    print()
    print("  ✅ 4 张热力图落在 4 个不同区域 → GME 对不同主题词激活不同区域")
    print("  ⚠️  都亮同一处 / 全糊           → 对齐性有限, 需要换策略")
    print()
    print(f"结果: {OUT_DIR}")


if __name__ == "__main__":
    main()
