from torch.utils.tensorboard import SummaryWriter
import logging
from torch.utils.data import DataLoader
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
import json
import argparse
import ast
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from recformer import RecformerForPretraining, RecformerTokenizer, RecformerConfig, LitWrapper
from collator import PretrainDataCollatorWithPadding
from lightning_dataloader import ClickDataset

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default='schen/longformer-chinese-base-4096')
parser.add_argument('--temp', type=float, default=0.05, help="Temperature for softmax.")
parser.add_argument('--preprocessing_num_workers', type=int, default=8, help="The number of processes to use for the preprocessing.")
parser.add_argument('--train_file', type=str, required=True)
parser.add_argument('--dev_file', type=str, required=True)
parser.add_argument('--item_attr_file', type=str, required=True)
parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--num_train_epochs', type=int, default=10)
parser.add_argument('--gradient_accumulation_steps', type=int, default=8)
parser.add_argument('--dataloader_num_workers', type=int, default=2)
parser.add_argument('--mlm_probability', type=float, default=0.15)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=5e-5)
parser.add_argument('--valid_step', type=int, default=2000)
parser.add_argument('--log_step', type=int, default=2000)
parser.add_argument('--device', type=int, default=0, help='Legacy single GPU index when --devices is not set.')
parser.add_argument('--devices', type=str, default=None, help='Trainer devices, e.g. 1, "[0]", "0,1", or "auto".')
parser.add_argument('--accelerator', type=str, default='auto', choices=['auto', 'cpu', 'gpu'], help='Trainer accelerator.')
parser.add_argument('--strategy', type=str, default='auto', help='Trainer strategy, e.g. auto, deepspeed_stage_2, ddp.')
parser.add_argument('--precision', type=str, default='32', help='Trainer precision, e.g. 32, 16-mixed, bf16-mixed.')
parser.add_argument('--fp16', action='store_true', help='Deprecated shortcut; if set, overrides --precision to 16-mixed.')
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--longformer_ckpt', type=str, default='longformer_ckpt/longformer-chinese-base-4096.bin')
parser.add_argument('--fix_word_embedding', action='store_true')



tokenizer_glb: RecformerTokenizer = None


def _init_tokenizer_worker(model_name_or_path, config_dict):
    global tokenizer_glb
    worker_config = RecformerConfig.from_pretrained(model_name_or_path)
    for k, v in config_dict.items():
        setattr(worker_config, k, v)
    tokenizer_glb = RecformerTokenizer.from_pretrained(model_name_or_path, worker_config)


def _par_tokenize_doc(doc):
    if tokenizer_glb is None:
        raise RuntimeError('tokenizer_glb is None. tokenizer worker is not initialized correctly.')
    
    item_id, item_attr = doc

    input_ids, token_type_ids = tokenizer_glb.encode_item(item_attr)

    return item_id, input_ids, token_type_ids


def resolve_trainer_runtime(args):
    precision = '16-mixed' if args.fp16 else args.precision

    if args.devices is None:
        devices = 1 if args.accelerator == 'cpu' else [args.device]
    else:
        if args.devices == 'auto':
            devices = 'auto'
        elif args.devices.startswith('['):
            devices = ast.literal_eval(args.devices)
        elif ',' in args.devices:
            devices = [int(x.strip()) for x in args.devices.split(',') if x.strip()]
        else:
            try:
                devices = int(args.devices)
            except ValueError:
                devices = args.devices

    accelerator = args.accelerator
    if accelerator == 'gpu' and not torch.cuda.is_available():
        print('[TrainerConfig] CUDA unavailable, fallback to CPU.')
        accelerator = 'cpu'
        devices = 1

    strategy = args.strategy
    if accelerator != 'gpu' and strategy.startswith('deepspeed'):
        print('[TrainerConfig] Deepspeed requires GPU, fallback strategy=auto.')
        strategy = 'auto'

    if accelerator == 'gpu' and isinstance(devices, list) and len(devices) <= 1 and strategy.startswith('deepspeed'):
        print('[TrainerConfig] Single GPU detected, fallback strategy=auto for better compatibility.')
        strategy = 'auto'

    if accelerator == 'cpu' and precision != '32':
        print('[TrainerConfig] CPU training uses precision=32 for compatibility.')
        precision = '32'

    print(f'[TrainerConfig] accelerator={accelerator}, devices={devices}, strategy={strategy}, precision={precision}')
    return accelerator, devices, strategy, precision


def main():
    
    args = parser.parse_args()
    print(args)
    seed_everything(42)

    config = RecformerConfig.from_pretrained(args.model_name_or_path)
    config.max_attr_num = 3
    config.max_attr_length = 32
    config.max_item_embeddings = 51  # 50 item and 1 for cls
    config.attention_window = [64] * config.num_hidden_layers
    config.max_token_num = 1024
    tokenizer = RecformerTokenizer.from_pretrained(args.model_name_or_path, config)

    global tokenizer_glb
    tokenizer_glb = tokenizer

    # preprocess corpus
    path_corpus = Path(args.item_attr_file)
    dir_corpus = path_corpus.parent
    dir_preprocess = dir_corpus / 'preprocess'
    dir_preprocess.mkdir(exist_ok=True)

    path_tokenized_items = dir_preprocess / f'tokenized_items_{path_corpus.name}'

    if path_tokenized_items.exists():
        print(f'[Preprocessor] Use cache: {path_tokenized_items}')
    else:
        print(f'Loading attribute data {path_corpus}')
        item_attrs = json.load(open(path_corpus))
        if args.preprocessing_num_workers <= 1:
            doc_tuples = [_par_tokenize_doc(doc) for doc in tqdm(item_attrs.items(), total=len(item_attrs), ncols=100, desc=f'[Tokenize] {path_corpus}')]
        else:
            config_dict = {
                'max_attr_num': config.max_attr_num,
                'max_attr_length': config.max_attr_length,
                'max_item_embeddings': config.max_item_embeddings,
                'attention_window': config.attention_window,
                'max_token_num': config.max_token_num,
            }
            with Pool(
                processes=args.preprocessing_num_workers,
                initializer=_init_tokenizer_worker,
                initargs=(args.model_name_or_path, config_dict),
            ) as pool:
                pool_func = pool.imap(func=_par_tokenize_doc, iterable=item_attrs.items())
                doc_tuples = list(tqdm(pool_func, total=len(item_attrs), ncols=100, desc=f'[Tokenize] {path_corpus}'))

        tokenized_items = {item_id: [input_ids, token_type_ids] for item_id, input_ids, token_type_ids in doc_tuples}

        json.dump(tokenized_items, open(path_tokenized_items, 'w'))

    tokenized_items = json.load(open(path_tokenized_items))#dir_preprocess / f'attr_small.json'))#
    print(f'Successfully load {len(tokenized_items)} tokenized items.')

    data_collator = PretrainDataCollatorWithPadding(tokenizer, tokenized_items, mlm_probability=args.mlm_probability)
    train_data = ClickDataset(json.load(open(args.train_file)), data_collator)
    dev_data = ClickDataset(json.load(open(args.dev_file)), data_collator)

    train_loader = DataLoader(train_data, 
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              collate_fn=train_data.collate_fn,
                              num_workers=args.dataloader_num_workers)
    dev_loader = DataLoader(dev_data, 
                            batch_size=args.batch_size, 
                            collate_fn=dev_data.collate_fn,
                            num_workers=args.dataloader_num_workers)
    
    pytorch_model = RecformerForPretraining(config)
    pytorch_model.load_state_dict(torch.load(args.longformer_ckpt))

    if args.fix_word_embedding:
        print('Fix word embeddings.')
        for param in pytorch_model.longformer.embeddings.word_embeddings.parameters():
            param.requires_grad = False

    model = LitWrapper(pytorch_model, learning_rate=args.learning_rate)

    checkpoint_callback = ModelCheckpoint(save_top_k=5, monitor="accuracy", mode="max", filename="{epoch}-{accuracy:.4f}")
    
    accelerator, devices, strategy, precision = resolve_trainer_runtime(args)

    trainer = Trainer(accelerator=accelerator,
                     max_epochs=args.num_train_epochs,
                     devices=devices,
                     accumulate_grad_batches=args.gradient_accumulation_steps,
                     val_check_interval=args.valid_step,
                     default_root_dir=args.output_dir,
                     gradient_clip_val=1.0,
                     log_every_n_steps=args.log_step,
                     precision=precision,
                     strategy=strategy,
                     callbacks=[checkpoint_callback]
                     )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=dev_loader, ckpt_path=args.ckpt)



if __name__ == "__main__":
    main()
