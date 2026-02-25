from torch.utils.tensorboard import SummaryWriter
import logging
from torch.utils.data import DataLoader
from multiprocessing import Pool
from pathlib import Path
from tqdm import tqdm
import json
import argparse
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from recformer import RecformerForPretraining, RecformerTokenizer, RecformerConfig, LitWrapper
from collator import PretrainDataCollatorWithPadding
from lightning_dataloader import ClickDataset

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default='severinsimmler/xlm-roberta-longformer-base-16384')
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
parser.add_argument('--device', type=int, default=None, help='Deprecated alias for --devices')
parser.add_argument('--devices', type=int, default=1, help='Number of visible GPUs to use')
parser.add_argument('--strategy', type=str, default='auto', help='Trainer strategy, e.g. auto, ddp, deepspeed_stage_2')
parser.add_argument('--precision', type=str, default='bf16-mixed', choices=['32', '16-mixed', 'bf16-mixed'])
parser.add_argument('--fp16', action='store_true', help='Deprecated alias to set --precision 16-mixed')
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--longformer_ckpt', type=str, default='longformer_ckpt/xlm-roberta-longformer-base-16384.bin')
parser.add_argument('--fix_word_embedding', action='store_true')
parser.add_argument('--disable_tf32', action='store_true')



tokenizer_glb: RecformerTokenizer = None
def _par_tokenize_doc(doc):
    
    item_id, item_attr = doc

    input_ids, token_type_ids = tokenizer_glb.encode_item(item_attr)

    return item_id, input_ids, token_type_ids


def setup_gpu_runtime(disable_tf32: bool):
    if not torch.cuda.is_available():
        return False

    torch.backends.cudnn.benchmark = True
    if not disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    return True


def main():
    
    args = parser.parse_args()
    print(args)
    seed_everything(42)
    has_cuda = setup_gpu_runtime(args.disable_tf32)

    if args.fp16:
        args.precision = '16-mixed'
    if args.device is not None:
        args.devices = args.device
    if not has_cuda:
        args.precision = '32'

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
        pool = Pool(processes=args.preprocessing_num_workers)
        pool_func = pool.imap(func=_par_tokenize_doc, iterable=item_attrs.items())
        doc_tuples = list(tqdm(pool_func, total=len(item_attrs), ncols=100, desc=f'[Tokenize] {path_corpus}'))
        tokenized_items = {item_id: [input_ids, token_type_ids] for item_id, input_ids, token_type_ids in doc_tuples}
        pool.close()
        pool.join()

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
                              num_workers=args.dataloader_num_workers,
                              pin_memory=has_cuda,
                              persistent_workers=args.dataloader_num_workers > 0)
    dev_loader = DataLoader(dev_data, 
                            batch_size=args.batch_size, 
                            collate_fn=dev_data.collate_fn,
                            num_workers=args.dataloader_num_workers,
                            pin_memory=has_cuda,
                            persistent_workers=args.dataloader_num_workers > 0)
    
    pytorch_model = RecformerForPretraining(config)
    pytorch_model.load_state_dict(torch.load(args.longformer_ckpt))

    if args.fix_word_embedding:
        print('Fix word embeddings.')
        for param in pytorch_model.longformer.embeddings.word_embeddings.parameters():
            param.requires_grad = False

    model = LitWrapper(pytorch_model, learning_rate=args.learning_rate)

    checkpoint_callback = ModelCheckpoint(save_top_k=5, monitor="accuracy", mode="max", filename="{epoch}-{accuracy:.4f}")
    trainer_strategy = args.strategy
    if args.devices <= 1 and args.strategy.startswith('deepspeed'):
        trainer_strategy = 'auto'
    
    trainer = Trainer(accelerator="gpu" if has_cuda else "cpu",
                     max_epochs=args.num_train_epochs,
                     devices=args.devices if has_cuda else 1,
                     accumulate_grad_batches=args.gradient_accumulation_steps,
                     val_check_interval=args.valid_step,
                     default_root_dir=args.output_dir,
                     gradient_clip_val=1.0,
                     log_every_n_steps=args.log_step,
                     precision=args.precision,
                     strategy=trainer_strategy,
                     callbacks=[checkpoint_callback]
                     )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=dev_loader, ckpt_path=args.ckpt)



if __name__ == "__main__":
    main()
