MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-severinsimmler/xlm-roberta-longformer-base-16384}
LONGFORMER_CKPT=${LONGFORMER_CKPT:-longformer_ckpt/xlm-roberta-longformer-base-16384.bin}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
DEVICES=${DEVICES:-1}
PRECISION=${PRECISION:-bf16-mixed}
STRATEGY=${STRATEGY:-auto}
export CUDA_VISIBLE_DEVICES

# Single-GPU (e.g., RTX 4080) pretraining defaults. Override any variable above as needed.
python lightning_pretrain.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --longformer_ckpt ${LONGFORMER_CKPT} \
    --train_file pretrain_data/train.json \
    --dev_file pretrain_data/dev.json \
    --item_attr_file pretrain_data/meta_data.json \
    --output_dir result/recformer_pretraining \
    --num_train_epochs ${NUM_TRAIN_EPOCHS:-32} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS:-12} \
    --preprocessing_num_workers ${PREPROCESSING_NUM_WORKERS:-8} \
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS:-8} \
    --batch_size ${BATCH_SIZE:-3} \
    --learning_rate ${LEARNING_RATE:-5e-5} \
    --temp 0.05 \
    --devices ${DEVICES} \
    --precision ${PRECISION} \
    --strategy ${STRATEGY} \
    --fix_word_embedding
