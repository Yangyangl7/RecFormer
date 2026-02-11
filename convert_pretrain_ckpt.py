import argparse
from collections import OrderedDict

import torch

from recformer import RecformerModel, RecformerConfig, RecformerForSeqRec


DEFAULT_MODEL_NAME = 'schen/longformer-chinese-base-4096'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_ckpt_path', type=str, default='pretrain_ckpt/pytorch_model.bin')
    parser.add_argument('--longformer_ckpt_path', type=str, default='longformer_ckpt/longformer-chinese-base-4096.bin')
    parser.add_argument('--model_name_or_path', type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument('--recformer_output_path', type=str, default='pretrain_ckpt/recformer_pretrain_ckpt.bin')
    parser.add_argument('--seqrec_output_path', type=str, default='pretrain_ckpt/seqrec_pretrain_ckpt.bin')
    return parser.parse_args()


def build_config(model_name_or_path):
    config = RecformerConfig.from_pretrained(model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 51
    config.attention_window = [64] * config.num_hidden_layers
    return config


def main():
    args = parse_args()

    state_dict = torch.load(args.pretrained_ckpt_path)
    longformer_state_dict = torch.load(args.longformer_ckpt_path)

    state_dict['_forward_module.model.longformer.embeddings.word_embeddings.weight'] = longformer_state_dict[
        'longformer.embeddings.word_embeddings.weight'
    ]

    recformer_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('_forward_module.model.longformer.'):
            new_key = key[len('_forward_module.model.longformer.'):]
            recformer_state_dict[new_key] = value

    recformer_model = RecformerModel(build_config(args.model_name_or_path))
    recformer_model.load_state_dict(recformer_state_dict)
    torch.save(recformer_state_dict, args.recformer_output_path)
    print(f'Converted RecformerModel checkpoint: {args.recformer_output_path}')

    seqrec_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('_forward_module.model.'):
            new_key = key[len('_forward_module.model.'):]
            seqrec_state_dict[new_key] = value

    seqrec_model = RecformerForSeqRec(build_config(args.model_name_or_path))
    seqrec_model.load_state_dict(seqrec_state_dict, strict=False)
    torch.save(seqrec_state_dict, args.seqrec_output_path)
    print(f'Converted RecformerForSeqRec checkpoint: {args.seqrec_output_path}')


if __name__ == '__main__':
    main()
