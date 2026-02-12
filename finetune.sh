MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-schen/longformer-chinese-base-4096}

python finetune.py \
    --pretrain_ckpt pretrain_ckpt/seqrec_pretrain_ckpt.bin \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_path finetune_data/Scientific \
    --num_train_epochs 128 \
    --batch_size 16 \
    --device 3 \
    --fp16 \
    --finetune_negative_sample_size -1
