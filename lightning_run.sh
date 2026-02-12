# Use distributed data parallel
MODEL_NAME_OR_PATH=${MODEL_NAME_OR_PATH:-schen/longformer-chinese-base-4096}
ACCELERATOR=${ACCELERATOR:-auto}
DEVICES=${DEVICES:-auto}
STRATEGY=${STRATEGY:-auto}
PRECISION=${PRECISION:-32}

CUDA_VISIBLE_DEVICES=1,4,6,7 python lightning_pretrain.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --train_file pretrain_data/train.json \
    --dev_file pretrain_data/dev.json \
    --item_attr_file pretrain_data/meta_data.json \
    --output_dir result/recformer_pretraining \
    --num_train_epochs 32 \
    --gradient_accumulation_steps 8 \
    --preprocessing_num_workers 8 \
    --dataloader_num_workers 8  \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --temp 0.05 \
    --accelerator ${ACCELERATOR} \
    --devices ${DEVICES} \
    --strategy ${STRATEGY} \
    --precision ${PRECISION} \
    --device 4 \
    --fp16 \
    --fix_word_embedding
