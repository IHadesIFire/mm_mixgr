### 启动
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=6 python scripts/run_region_alignment.py \
  --domains Science Medicine \
  --local_dataset_path /data0_yjh/czz_data/MRMR/Knowledge \
  --sam_checkpoint /data0_yjh/czz/mm_mixgr/checkpoints/sam/sam_vit_h_4b8939.pth \
  --sam_model_type vit_h \
  --clip_model /data0_yjh/czz/mm_mixgr/checkpoints/clip-vit-large-patch14 \
  --device cuda \
  --output_dir ./results/region_alignment \
  --max_queries 10 \
  --max_docs_per_query 3 \
  --top_k_regions 5 \
  --only_use_cache \
  --text_mode both

#### ################################
  PYTHONPATH=. CUDA_VISIBLE_DEVICES=7 python scripts/test_query_sam_clip.py \
  --query_path /data0_yjh/czz_data/MRMR/Knowledge/query \
  --cache_dir /data0_yjh/czz/mm_mixgr/cache/decompositions \
  --sam_checkpoint /data0_yjh/czz/mm_mixgr/checkpoints/sam/sam_vit_h_4b8939.pth \
  --sam_model_type vit_h \
  --clip_model /data0_yjh/czz/mm_mixgr/checkpoints/clip-vit-large-patch14 \
  --device cuda \
  --domains Science Medicine \
  --num_samples 3 \
  --max_regions 30 \
  --output_dir ./results/test_query_sam_clip

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=6 python scripts/test_qd_sam_clip.py \
  --query_path /data0_yjh/czz_data/MRMR/Knowledge/query \
  --corpus_path /data0_yjh/czz_data/MRMR/Knowledge/corpus \
  --cache_dir /data0_yjh/czz/mm_mixgr/cache/decompositions \
  --sam_checkpoint /data0_yjh/czz/mm_mixgr/checkpoints/sam/sam_vit_b_01ec64.pth \
  --sam_model_type vit_b \
  --clip_model /data0_yjh/czz/mm_mixgr/checkpoints/clip-vit-large-patch14 \
  --device cuda \
  --domains Science Medicine \
  --num_query_samples 3 \
  --num_doc_samples 3 \
  --max_regions 30 \
  --output_dir ./results/test_qd_sam_clip \
  --query_indices 1 5 12 \
  --doc_indices 4 8 20

  PYTHONPATH=. CUDA_VISIBLE_DEVICES=6 python scripts/test_qd_sam_clip.py \
  --query_path /data0_yjh/czz_data/MRMR/Knowledge/query \
  --corpus_path /data0_yjh/czz_data/MRMR/Knowledge/corpus \
  --cache_dir /data0_yjh/czz/mm_mixgr/cache/decompositions \
  --sam_checkpoint /data0_yjh/czz/mm_mixgr/checkpoints/sam/sam_vit_b_01ec64.pth \
  --sam_model_type vit_b \
  --clip_model /data0_yjh/czz/mm_mixgr/checkpoints/siglip-base-patch16-224 \
  --model_family siglip \
  --device cuda \
  --domains Science Medicine \
  --num_query_samples 3 \
  --num_doc_samples 3 \
  --query_indices 1 5 12 \
  --doc_indices 4 8 20 \
  --max_regions 30 \
  --output_dir ./results/test_qd_sam_siglip