# RecFormer 替换 Longformer 为 longformer-chinese 方案

本文给出将当前 `RecFormer` 仓库从英文骨干 `allenai/longformer-base-4096` 迁移到你本地 `longformer-chinese` 的可执行方案。

## 1. 代码库结构与 Longformer 依赖点

仓库主目录可分为三层：

- **模型定义层**：`recformer/models.py`、`recformer/tokenization.py`
- **训练脚本层**：`lightning_pretrain.py`、`finetune.py`、`lightning_run.sh`
- **权重转换层**：`save_longformer_ckpt.py`、`convert_pretrain_ckpt.py`

其中 Longformer 的强耦合点：

1. `RecformerConfig` 直接继承 `LongformerConfig`，编码器使用 `LongformerEncoder`，预训练头使用 `LongformerLMHead`。因此只要 `longformer-chinese` 与 HuggingFace Longformer 结构兼容，就不需要改模型主干实现。  
2. `RecformerTokenizer` 继承 `LongformerTokenizer`，分词器和词表完全依赖 backbone 的 tokenizer。  
3. 脚本中多处硬编码 `allenai/longformer-base-4096` 作为 `model_name_or_path` 或 `LONGFORMER_TYPE`。这些位置改为你本地 longformer-chinese 路径即可。

## 2. 迁移前提检查（必须先确认）

请先确认你的 `longformer-chinese` 满足：

- 包含 `config.json` 且 `model_type`/结构与 Longformer 兼容。
- 包含 tokenizer 文件（如 `vocab.json` + `merges.txt` 或对应 fast tokenizer 文件）。
- 最大位置长度满足你的序列长度（`max_position_embeddings`，通常 >= 1024）。
- 若你要直接复用当前 RecFormer 旧 checkpoint：**词表和 embedding 维度必须一致**，否则会在 `load_state_dict` 时报 shape mismatch。

## 3. 推荐迁移路径

### 路径 A（推荐）：仅替换 backbone 初始化来源，重新预训练

适合：你希望得到真正中文语义能力，不强求复用英文权重。

步骤：

1. 把所有 `from_pretrained('allenai/longformer-base-4096')` 改成你的本地路径，比如：
   - `--model_name_or_path /path/to/longformer-chinese`
   - `LONGFORMER_TYPE = '/path/to/longformer-chinese'`
2. 用中文 backbone 重新生成 RecFormer 初始化权重：
   - 运行 `save_longformer_ckpt.py`（建议先参数化，见第 4 节）。
3. 走原 pretrain 流程训练，再用 `convert_pretrain_ckpt.py` 转换。
4. 再做 two-stage finetune。

优点：稳定、兼容当前代码范式。  
风险：需要重新预训练成本。

### 路径 B：尽量复用旧 RecFormer checkpoint（不推荐）

适合：只做快速实验。

前提非常苛刻：

- 中文 longformer 与英文 longformer **hidden size / layer 数 / attention 头数完全一致**。
- tokenizer/词表大小一致（通常不成立）。

如果词表大小不一致，只能：

- `strict=False` 局部加载；
- 对 `word_embeddings` 和 `lm_head.decoder` 做 resize/重初始化；
- 再进行较长 warmup 训练。

该路径通常不如路径 A 稳定。

## 4. 建议最小代码改造（提高可维护性）

为了避免后续反复改硬编码，建议把几个脚本参数化：

1. `save_longformer_ckpt.py`
   - 新增 `--model_name_or_path`（默认可保留旧值）
   - 新增 `--output_ckpt_path`
2. `convert_pretrain_ckpt.py`
   - 把 `LONGFORMER_TYPE` 改为命令行参数 `--model_name_or_path`
   - 把输入输出路径都改成参数
3. `lightning_run.sh` / `finetune.sh`
   - 通过环境变量注入 `MODEL_NAME_OR_PATH`

这样你就可以统一切换：

```bash
MODEL_NAME_OR_PATH=/path/to/longformer-chinese
```

## 5. 关键兼容点与排障清单

### 5.1 tokenizer 类型兼容

当前 `RecformerTokenizer` 继承 `LongformerTokenizer`。若你的中文模型发布的是 `BertTokenizer` 或 `RobertaTokenizer` 而非 Longformer tokenizer，可能触发 `from_pretrained` 类型不匹配。

处理建议：

- 首选使用与 Longformer 兼容发布格式的中文模型；
- 如果不兼容，改造 `RecformerTokenizer` 为组合模式（内部持有 `AutoTokenizer`）而不是继承 `LongformerTokenizer`。

### 5.2 位置编码长度

`config.max_token_num`、`max_item_embeddings` 与 backbone `max_position_embeddings` 需联动检查，避免超长导致越界。

### 5.3 attention_window 层数

当前代码会校验 `len(attention_window) == num_hidden_layers`。若中文 backbone 层数变化，记得动态生成：

```python
config.attention_window = [64] * config.num_hidden_layers
```

### 5.4 checkpoint 键名映射

`convert_pretrain_ckpt.py` 依赖固定前缀裁剪逻辑。如果你调整了训练封装层（例如 lightning module 名称），要同步改前缀。

## 6. 验证流程（建议按此顺序）

1. **静态加载验证**
   - `RecformerConfig.from_pretrained(chinese_path)`
   - `RecformerTokenizer.from_pretrained(chinese_path, config)`
   - `RecformerForPretraining(config)` 前向一小批数据
2. **初始化权重验证**
   - 运行 `save_longformer_ckpt.py` 并检查是否有大量 `missing name` / `wrong size`
3. **小规模预训练冒烟**
   - 跑 100~500 step，确认 loss 下降
4. **小规模 finetune 冒烟**
   - 单数据集 1 epoch，确认指标可计算且无 NaN

## 7. 一套可直接执行的迁移实施建议

- 第 1 周：完成参数化改造 + 加载冒烟。  
- 第 2 周：中文 backbone 初始化预训练（短周期）+ 验证。  
- 第 3 周：全量预训练 + two-stage finetune。  
- 第 4 周：A/B 对比英文 backbone 与中文 backbone 的指标与线上延迟。

---

如果你愿意，我可以下一步直接给你一版 **可提交的代码补丁**（把上述脚本全部参数化，默认支持 `longformer-chinese` 路径），并附上最小 smoke test 命令。
