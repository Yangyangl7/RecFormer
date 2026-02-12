import argparse
import os

import torch
from transformers import AutoModelForMaskedLM

from recformer import RecformerConfig, RecformerForPretraining


DEFAULT_MODEL_NAME = 'schen/longformer-chinese-base-4096'
DEFAULT_OUTPUT_CKPT = 'longformer_ckpt/longformer-chinese-base-4096.bin'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument('--output_ckpt_path', type=str, default=DEFAULT_OUTPUT_CKPT)
    return parser.parse_args()


def main():
    args = parse_args()

    backbone = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)

    config = RecformerConfig.from_pretrained(args.model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 51
    config.attention_window = [64] * config.num_hidden_layers
    model = RecformerForPretraining(config)

    backbone_state_dict = backbone.state_dict()
    recformer_state_dict = model.state_dict()

    for name, param in backbone_state_dict.items():
        if name not in recformer_state_dict:
            print('missing name', name)
            continue
        if recformer_state_dict[name].size() != param.size():
            print('wrong size', name, recformer_state_dict[name].size(), param.size())
            continue
        recformer_state_dict[name].copy_(param)

    output_dir = os.path.dirname(args.output_ckpt_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(recformer_state_dict, args.output_ckpt_path)
    print(f'Saved Recformer init checkpoint to {args.output_ckpt_path}')


if __name__ == '__main__':
    main()
