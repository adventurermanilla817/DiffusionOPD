export WANDB_API_KEY=xxx
export WANDB_ENTITY=xxx

python -m accelerate.commands.launch \
    --config_file scripts/accelerate_configs/multi_gpu.yaml \
    --num_machines=1 \
    --num_processes=8 \
    --main_process_port 29500 \
    -m scripts.train_sd3_opd \
    --config config/opd.py:sopd_pickscore
