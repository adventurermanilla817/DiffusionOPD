torchrun --nproc_per_node=8 --master_port=29502 scripts/evaluation.py \
    --checkpoint_path CHECKPOINT_PATH \
    --model_type sd3 \
    --dataset drawbench \
    --guidance_scale 4.5 \
    --mixed_precision fp16 \
    --save_images \
    --output_dir YOUR_OUTPUT_DIR