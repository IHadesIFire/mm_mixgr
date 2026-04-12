# GME-only MRMR baseline (Colab)

这个版本只保留 **GME 模型 + MRMR Knowledge 检索**，把 MixGR / decomposition / alignment 相关功能都拿掉了，但保留了原来的项目目录结构。

## 1. 在 Colab 解压并进入目录

```python
!unzip -q /content/mm_mixgr_gme_only.zip -d /content/
%cd /content/mm_mixgr_gme_only
```

## 2. 安装依赖

```python
!pip install -q -r requirements-colab.txt
```

## 3. 先跑一个小样本 smoke test

```python
!python main.py baseline \
    --model Alibaba-NLP/gme-Qwen2-VL-2B-Instruct \
    --domains Science Medicine \
    --max_queries 20 \
    --max_corpus 200 \
    --batch_size 2
```

## 4. 跑正式版（7B）

```python
!python main.py baseline \
    --model Alibaba-NLP/gme-Qwen2-VL-7B-Instruct \
    --domains Science Medicine \
    --batch_size 2
```

全量四个大类：

```python
!python main.py baseline \
    --model Alibaba-NLP/gme-Qwen2-VL-7B-Instruct \
    --batch_size 2
```

## 5. 断点续跑

这个版本会把 cache 和 checkpoint 都放到：

- `cache/embeddings/<run_id>/...`
- `results/<run_id>/...`

所以同一套参数重跑时，会自动复用：

- `queries.npz`
- `corpus.npz`
- `checkpoints/corpus_checkpoint_*.npz`

如果你改了模型 / 域 / `max_corpus` / `max_length` / `max_image_tokens`，会自动生成新的 `run_id`，不会串 cache。

强制清空同一次实验的 cache：

```python
!python main.py baseline \
    --model Alibaba-NLP/gme-Qwen2-VL-7B-Instruct \
    --domains Science Medicine \
    --batch_size 2 \
    --clear_cache
```

## 6. 结果文件

每次运行都会保存：

- `results/<run_id>/predictions.json`
- `results/<run_id>/metrics.json`
- `results/<run_id>/run_config.json`

## 7. 说明

- 默认任务是 `MRMRbenchmark/knowledge`
- query instruction 用的是 MRMR task metadata 那句：`Retrieve relevant documents that help answer the question.`
- query / corpus 的 `id` 前缀、图像 resize、`pytrec_eval` 评测，都是按 MRMR 这条 baseline 路径对齐的
- 7B 版显存要求高；Colab 上建议优先用 A100/L4 这类较大显存 GPU。先用 2B 做 smoke test 更稳。
