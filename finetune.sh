MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-severinsimmler/xlm-roberta-longformer-base-16384}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
DEVICE=${DEVICE:-0}
PRECISION=${PRECISION:-bf16}
export CUDA_VISIBLE_DEVICES

python finetune.py \
    --pretrain_ckpt pretrain_ckpt/seqrec_pretrain_ckpt.bin \
    --data_path finetune_data/Scientific \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS:-128} \
    --batch_size ${BATCH_SIZE:-12} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS:-6} \
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS:-8} \
    --device ${DEVICE} \
    --precision ${PRECISION} \
    --finetune_negative_sample_size -1
