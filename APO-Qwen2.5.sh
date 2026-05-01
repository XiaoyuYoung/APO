nproc_per_node=4

OMP_NUM_THREADS=32 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=$nproc_per_node \
MASTER_PORT=29502 \
swift rlhf \
    --rlhf_type dpo \
    --model /output-Qwen2.5-VL-7B-Instruct/checkpoint-10000-merged \
    --dataset APO.json \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-7 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps $(expr 16 / $nproc_per_node) \
    --eval_steps 100 \
    --save_steps 100 \
    --save_total_limit 5 \
    --deepspeed zero2 \
    --logging_steps 5 \
    --max_length 4096 \
    --output_dir output-Qwen2.5-VL-7B-Instruct-MIMIC-APO \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8