# Learning Language Representations for Sequential Recommendation

This repository contains the replication of the paper **"Text Is All You Need: Learning Language Representations for Sequential Recommendation"**, a model learns natural language representations for sequential recommendation.

The KDD 2023 paper [Text Is All You Need: Learning Language Representations for Sequential Recommendation](https://arxiv.org/abs/2305.13731).

## Quick Links

- [Overview](#overview)
- [Dependencies](#dependencies)
- [Pretraining](#pretraining)
- [Pretrained Model](#pretrained-model)
- [Finetuning](#finetuning)
- [Contact](#contact)
- [Citation](#citation)

## Overview

In this paper, the authors propose to model user preferences and item features as language representations that can be generalized to new items and datasets. To this end, the authors present a novel framework, named Recformer, which effectively learns language representations for sequential recommendation. Specifically, the authors propose to formulate an item as a "sentence" (word sequence) by flattening item key-value attributes described by text so that an item sequence for a user becomes a sequence of sentences. For recommendation, Recformer is trained to understand the "sentence" sequence and retrieve the next "sentence". To encode item sequences, the authors design a bi-directional Transformer similar to the model Longformer but with different embedding layers for sequential recommendation. For effective representation learning, the authors propose novel pretraining and finetuning methods which combine language understanding and recommendation tasks. Therefore, Recformer can effectively recommend the next item based on language representations.

## Dependencies

Train and test the model using the following main dependencies:
- Python 3.10.10
- PyTorch 2.0.0
- PyTorch Lightning 2.0.0
- Transformers 4.28.0
- Deepspeed 0.9.0

## Pretraining
### Dataset
8 categories in [Amazon dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) for pretraining:

Training:
- `Automotive`
- `Cell Phones and Accessories`
- `Clothing, Shoes and Jewelry`
- `Electronics`
- `Grocery and Gourmet Food`
- `Home and Kitchen`
- `Movies and TV`

Validation:
- `CDs and Vinyl`

You can process these data using the provided scripts `pretrain_data/meta_data_process.py` and `pretrain_data/interaction_data_process.py`. You need to set meta data path `META_ROOT` and interaction data path `SEQ_ROOT` in the two files. Then run the following commands:
```bash
cd pretrain_data
python meta_data_process.py
python interaction_data_process.py
```
Or, you can download the processed data from [here](https://drive.google.com/file/d/11wTD3jMoP_Fb5SlHfKr28NIMCnG_jOpy/view?usp=sharing).

### Training

The pretraining code is based on the framework [Pytorch-Lightning](https://lightning.ai/docs/pytorch/stable/). The default backbone model is `severinsimmler/xlm-roberta-longformer-base-16384` with different `token type embedding` and `item position embedding`.

First, you need to adjust pretrained Longformer checkpoint to the model. You can run the following command:
```bash
python save_longformer_ckpt.py
```
This code will automatically download the backbone from Hugging Face and adjust it to Recformer format. The default output path is `longformer_ckpt/xlm-roberta-longformer-base-16384.bin`.
You can also pass explicit arguments:
```bash
python save_longformer_ckpt.py \
  --model_name_or_path severinsimmler/xlm-roberta-longformer-base-16384 \
  --output_ckpt_path longformer_ckpt/xlm-roberta-longformer-base-16384.bin
```

Then, you can pretrain your own model with the default settings by running the following command:
```bash
bash lightning_run.sh
```
The script is tuned for single-GPU training by default (`CUDA_VISIBLE_DEVICES=0`, `--devices 1`, `--precision bf16-mixed`, `--strategy auto`), which is suitable for RTX 4080.
You can override resource-related settings from the shell, for example:
```bash
CUDA_VISIBLE_DEVICES=0 BATCH_SIZE=8 GRADIENT_ACCUMULATION_STEPS=8 PRECISION=bf16-mixed bash lightning_run.sh
```
For a 12GB RTX 4080 on Windows, a safer pretraining start point from dry-run is:
- `BATCH_SIZE=3`
- `GRADIENT_ACCUMULATION_STEPS=12`
If you see `CUDA launch timed out`, reduce `BATCH_SIZE` further or adjust Windows TDR settings.
If you use the training strategy `deepspeed_stage_2`, you need to first convert zero checkpoint to lightning checkpoint by running `zero_to_fp32.py` (automatically generated to checkpoint folder from pytorch-lightning):
```bash
python zero_to_fp32.py . pytorch_model.bin
```
Finally, please convert the lightning checkpoint to pytorch checkpoint (they have different model parameter names) by running `convert_pretrain_ckpt.py`:
```bash
python convert_pretrain_ckpt.py \
  --pretrained_ckpt_path pretrain_ckpt/pytorch_model.bin \
  --longformer_ckpt_path longformer_ckpt/xlm-roberta-longformer-base-16384.bin \
  --model_name_or_path severinsimmler/xlm-roberta-longformer-base-16384 \
  --recformer_output_path pretrain_ckpt/recformer_pretrain_ckpt.bin \
  --seqrec_output_path pretrain_ckpt/seqrec_pretrain_ckpt.bin
```
All paths can be passed from command-line arguments.

## Pretrained Model

We reproduce pretrained checkpoints for `RecformerModel` and `RecformerForSeqRec` used in the KDD paper.
|              Model              |
|:-------------------------------|
|[RecformerModel](https://drive.google.com/file/d/1aWsPLLgBaO51mPqzZrNdPmlBkMEZ-naR/view?usp=sharing)|
|[RecformerForSeqRec](https://drive.google.com/file/d/1BEboY3NxAUOBe6YwYZ_RsQ4BR6IIbl0-/view?usp=sharing)|

You can load the pretrained model by running the following code:
```python
import torch
from recformer import RecformerModel, RecformerConfig, RecformerForSeqRec

config = RecformerConfig.from_pretrained('severinsimmler/xlm-roberta-longformer-base-16384')
config.max_attr_num = 3  # max number of attributes for each item
config.max_attr_length = 32 # max number of tokens for each attribute
config.max_item_embeddings = 51 # max number of items in a sequence +1 for cls token
config.attention_window = [64] * config.num_hidden_layers # attention window for each layer

model = RecformerModel(config)
model.load_state_dict(torch.load('recformer_ckpt.bin'))

model = RecformerForSeqRec(config)
model.load_state_dict(torch.load('recformer_seqrec_ckpt.bin'), strict=False)
# strict=False because RecformerForSeqRec doesn't have lm_head
```

## Finetuning
### Dataset
We use 6 categories in [Amazon dataset](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) to evaluate our model:

- `Industrial and Scientific`
- `Musical Instruments`
- `Arts, Crafts and Sewing`
- `Office Products`
- `Video Games`
- `Pet Supplies`

You can process these data using our provided scripts `finetune_data/process.py`. You need to set meta data path `--meta_file_path`, interaction data path `--file_path` and output path `--output_path` to run the following commands:
```bash
cd finetune_data
python process.py --meta_file_path META_PATH --file_path SEQ_PATH --output_path OUTPUT_FOLDER
```

We also provide all processed data like this paper [here](https://drive.google.com/file/d/123AHjsvZFTeT_Mhfb81eMHvnE8fbsFi3/view?usp=sharing).

### Training
We train `RecformerForSeqRec` with two-stage finetuning like the KDD paper to conduct the sequential recommendation with Recformer. A sample script is provided for finetuning:
```bash
bash finetune.sh
```
The finetuning script defaults to single-GPU bf16 (`CUDA_VISIBLE_DEVICES=0`, `--precision bf16`) for better RTX 4080 utilization.
For a 12GB RTX 4080, dry-run suggests starting from:
- `BATCH_SIZE=12`
- `GRADIENT_ACCUMULATION_STEPS=6`
Our code will train and evaluate the model for the sequential recommendation task and return all metrics reported in that KDD paper.

<strong>Note</strong>: from our empirical results, you can set a smaller maximum length (512 or 256, our model is default to 1024) of Recformer `e.g., config.max_token_num = 512` to obtain more efficient finetuning and inference without obvious performance decay (128 has an obvious decay).

## Contact

If you have any questions related to the code or the paper, feel free to create an issue or email Jiacheng Li (`j9li@ucsd.edu`), the corresponding author of the KDD paper. Thanks!

## Citation

Please cite the paper if you use Recformer in your work:

```bibtex
@article{Li2023TextIA,
  title={Text Is All You Need: Learning Language Representations for Sequential Recommendation},
  author={Jiacheng Li and Ming Wang and Jin Li and Jinmiao Fu and Xin Shen and Jingbo Shang and Julian McAuley},
  journal={Proceedings of the 29th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  year={2023}
}
```
