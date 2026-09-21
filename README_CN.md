# BiomedCoOp 中文说明

## 当前训练主线

本仓库使用冻结的 BiomedCLIP 进行医学图像少样本提示学习。当前主线为 CoOp + Visual Deep Prompt + Text Deep Prompt + Original-style TCP，并可启用 Competitive Visual Prompt（CVP）；Confusion 和 Expert MoE 代码、配置与测试已移除。数据集、few-shot sampling、augmentation、优化器、scheduler、learning rate、epoch、batch size 和 seed 约束沿用原设置。

训练集使用 K-shot 采样，验证集使用原有完整划分；统一训练 100 轮，按完整验证集 accuracy 最优选择模型，在完整测试集上评估。训练、验证和测试的 batch size 固定为 32，num_workers 固定为 8。主实验 seed 为 1、2、3，补充实验使用 seed 4、5。

## 安装

在仓库根目录执行：

```bash
conda activate /mnt/nas1/disk09/yuejianwu/.conda/envs/biocoop
pip install -r requirements.txt
pip install -e ./Dassl.pytorch
```

## 单次训练

DermaMNIST 4-shot、seed 1 的 TCP + CVP 训练命令：

```bash
python train.py \
  --root /mnt/nas1/disk09/yuejianwu/biomedcoop/data \
  --output-dir output/tcp/dermamnist/shots_4/seed1 \
  --seed 1 \
  --trainer CoOpVPT_BiomedCLIP \
  --dataset-config-file configs/datasets/dermamnist.yaml \
  --config-file configs/trainers/CoOp/dermamnist_native_vpt_tcp.yaml \
  DATASET.NUM_SHOTS 4
```

学习率、优化器、训练轮数和提示结构由 YAML 配置提供。当前 TCP 配置文件为 `configs/trainers/CoOp/dermamnist_native_vpt_tcp.yaml`。

## TCP：共享 TKE 实现

`models/original_style_tcp.py` 是当前唯一的 TCP 实现。每条 biomedical description 独立经过冻结 BiomedCLIP Text Encoder，得到 projected bank：

```text
[C, 50, D_proj]
```

每个类别独立计算 Mean-50 prototype：

```python
w_c = F.normalize(description_bank[c].mean(dim=0), dim=-1)
```

所有类别共用一个 TKE，不建立类别专属 MLP：

```text
D_proj → D_proj / 4 → QuickGELU → 4 × hidden_dim
512   → 128       →          → 3072
```

TKE 输出 reshape 为 `[C, 4, hidden_dim]`，默认 hidden size 为 768，因此每个类别得到 4 个 class-aware prompt tokens。TKE 参数量为 461,952。

Text Encoder 的 block 0–7 使用正常 CoOp/Text Deep Prompt。历史直接替换模式（`FUSION=False`）在进入 block 8 前，将 CLS 后的 4 个 prompt slots 一次性替换为对应类别的 TKE tokens；block 9–11 直接使用上一层 hidden states，不再重新生成或覆盖 TCP tokens。

当前默认 `TRAINER.TCP.FUSION=True`，block 8 使用加权融合：

```text
T = TextMLP(prototype)                         # [C,4,768]
P8 = 独立可学习 Deep Prompt                     # [4,768]，类别间共享
tokens = alpha * T + (1 - alpha) * P8          # [C,4,768]
```

`FUSION_ALPHA=0.5` 为固定系数，不参与训练。P8 使用标准差 0.02 的正态初始化，增加 3,072 个训练参数；初始化时保留原随机数状态，不改变其他参数和训练采样。融合不涉及上一层 prompt hidden states，后续层也不重复注入。

当前实现不包含 5×10 grouping、LayerBasis、XProto/B+Delta、跨类别 centering、norm matching、layer gate 或多层 TCP 重注入。description bank 和 class prototype 均为 frozen buffer；BiomedCLIP backbone 也保持冻结。训练参数只有 CoOp context、Visual Deep Prompt、注入前 Text Deep Prompt 和共享 TKE。TCP prompt bundle 共 486,528 个可训练参数（TCP 开启时）。

`TRAINER.TCP.ENABLED=False` 时保留普通 Text Deep Prompt，冻结 TKE 参数并跳过 TCP 注入及融合。主要配置为：

```yaml
TRAINER:
  TCP:
    ENABLED: True
    DESCRIPTION_CACHE: ""
    INSERT_LAYER: 8
    FUSION: True
    FUSION_ALPHA: 0.5
```

## Competitive Visual Prompt

`models/competitive_visual_prompt.py` 复用 TCP 的 frozen Mean-50 class prototype `mu: [C, 512]`。视觉端与文本端对称：不再计算 Base logits、competitor 或 prototype 差值，而是将 `mu_c` 直接送入独立的共享 Visual TKE：

```text
512 → 128 → QuickGELU → 4 × 768
```

Visual TKE 输出 `class_visual_prompts: [C, 4, 768]`，随后只在 batch 维扩展为 `[B, C, 4, 768]`；不同样本共享同一类别原型生成的初始视觉 token。Visual TKE 参数量为 461,952，参数不与 TCP Text TKE 共享。视觉 forward 的输入输出和层级路径如下：

```text
image [B,3,H,W]
  → patch embedding + VPT block 0–7（每张图只运行一次）
  → shared_state [B,L,768]
  → mu [C,512] → Visual TKE → class_visual_prompts [C,4,768]
  → shared_state 按类别扩展为 [B*C,L,768]
  → prompts 扩展为 [B*C,4,768]
  → 仅在 block 8 前替换 prompt slots：[CLS, VisualPrototypePrompt_init, patches]
  → block 9–11：[CLS, 上一层 VisualPrototypePrompt, 上一层 patches]
  → final CLS → visual norm/projection → visual_features [B,C,512]
  → 与对应 TCP text_features [C,512] 做逐类别 cosine logit
  → logits [B,C] → CE
```

以上为保留的并行路径。当前默认采用串行；`TRAINER.COMPETITIVE_VISUAL_PROMPT.SERIAL=True` 改为：

```text
prototype [C,512] → TextMLP → T [C,4,768]
T → 逐 token 共享 VisualMLP (768 → 128 → QuickGELU → 768) → [C,4,768]
```

串行 VisualMLP 共 197,504 个参数；输入使用融合前的 T，不经过池化或 detach。每次 forward 只计算一次 TextMLP，文本注入与视觉映射共享输出，因此视觉 loss 可以反向更新 TextMLP。视觉端在 block 8 使用下述加权融合。

当前视觉融合：`TRAINER.COMPETITIVE_VISUAL_PROMPT.FUSION=True` 时，在视觉 block 8 使用 `alpha * VisualMLP(T) + (1 - alpha) * VisualDeepPrompt_8`，`FUSION_ALPHA=0.5` 固定不训练。独立的 `VisualDeepPrompt_8: [4,768]` 在类别和样本间共享，按原 VPT 的均匀分布初始化，增加 3,072 个参数；初始化保留随机数状态，不改变公共参数。只融合一次，CLS、patch tokens 和 block 9–11 的传播方式不变。当前默认开启；加载历史直接替换 checkpoint 时需显式关闭融合。

关闭视觉融合时，视觉端使用单次 replacement 语义：启用时只创建 block 0–7 的 Visual Deep Prompt 参数；进入 block 8 前，Visual TKE 生成的 4 个 token 替换原视觉 prompt slots，但不替换 CLS 或任何 patch token；block 9–11 不再注入或替换 VPT，直接传播 block 8 更新后的视觉原型 prompt 和 patches。最后删除 prompt slots 后仍由原 CLS/global pooling 产生分类特征，不对视觉原型 prompt 做 pooling。Visual TKE 与 CoOp、block 0–7 Visual/Text Deep Prompt、TCP Text TKE 联合训练；ViT/BERT backbone、visual projection、Mean-50 prototype 和 logit scale 均冻结，loss 只有 CE。关闭视觉原型 prompt 时仍创建并使用完整 12 层 Visual Deep Prompt Base。

```yaml
TRAINER:
  COMPETITIVE_VISUAL_PROMPT:
    ENABLED: True
    INSERT_LAYER: 8
    NUM_TOKENS: 4
    BOTTLENECK_DIM: 128
    SERIAL: True
    FUSION: True
    FUSION_ALPHA: 0.5
```

`TRAINER.COMPETITIVE_VISUAL_PROMPT.ENABLED=False` 时不构建 Visual TKE，并直接走修改前的 Original-style TCP Base forward、checkpoint protocol 和 prompt bundle。

## 批量运行

可以在服务器 Bash 中循环调用 `train.py`：

```bash
DATA_ROOT=/mnt/nas1/disk09/yuejianwu/biomedcoop/data
OUTPUT_ROOT=output/tcp
TRAINER_CONFIG=configs/trainers/CoOp/dermamnist_native_vpt_tcp.yaml

for SHOTS in 4 8 16 32; do
  for SEED in 1 2 3; do
    python train.py \
      --root "$DATA_ROOT" \
      --output-dir "$OUTPUT_ROOT/dermamnist/shots_${SHOTS}/seed${SEED}" \
      --seed "$SEED" \
      --trainer CoOpVPT_BiomedCLIP \
      --dataset-config-file configs/datasets/dermamnist.yaml \
      --config-file "$TRAINER_CONFIG" \
      DATASET.NUM_SHOTS "$SHOTS" || exit 1
  done
done
```

其他现有方法仍可使用各自 Trainer 和配置文件，例如：

| 方法 | Trainer | 配置文件 |
|---|---|---|
| 原生 CoOp | `CoOp_BiomedCLIP` | `configs/trainers/CoOp/dermamnist_native.yaml` |
| CoOp + Visual/Text VPT + TCP + CVP | `CoOpVPT_BiomedCLIP` | `configs/trainers/CoOp/dermamnist_native_vpt_tcp.yaml` |
| BiomedCoOp | `BiomedCoOp_BiomedCLIP` | `configs/trainers/BiomedCoOp/few_shot/dermamnist.yaml` |

## Checkpoint 与验证

训练 checkpoint 保存 `prompt_parameters` bundle、optimizer、scheduler、AMP scaler 及 TCP 结构元数据。恢复时严格校验 TKE 维度、层数、插入层、token 数和 description bank 元数据；当前代码不提供已删除 MultiText TCP、Confusion 或 MoE 实现的加载入口，并保留当前 TKE checkpoint 的固定 protocol 标识。

融合 checkpoint 额外记录固定 alpha，串行 checkpoint 记录文本 token 输入及映射维度；显式关闭相应融合及串行配置时仍可加载原替换／并行 checkpoint。

当前使用 `TEST.FINAL_MODEL="best_val"`、`TEST.BEST_METRIC="accuracy"`、`TEST.SAVE_BEST_METRICS=["accuracy"]`。每次训练 100 轮，每轮在原有完整验证集评估，accuracy 严格提高时保存最优模型，同分保留较早轮次。最终加载该模型评估完整测试集。

### 当前两种融合方法全量实验

```bash
conda activate /mnt/nas1/disk09/yuejianwu/.conda/envs/biocoop
python -u -m tools.experiments batch \
  --output output/prompt_fusion_fullval_best --gpus 0 2
```

脚本运行两种融合方法，训练及选模配置统一使用 `configs/trainers/CoOp/dermamnist_native_vpt_tcp.yaml`。运行主机为 `10.154.63.11`，GPU 0、2 各运行一个任务；每 15 分钟记录一次进度摘要，任务完成即接续下一组。

| 方法 | 当前结构 |
|---|---|
| CoOp + 双端 Deep Prompt + Class Text Token | 文本 block 8 固定 0.5 类别 token／独立 Deep Prompt 融合，关闭 Class Visual Token |
| 再加 Class Visual Token | 串行 TextMLP→VisualMLP，文本和视觉 block 8 均固定 0.5 融合 |

DermaMNIST、Kvasir、CHMNIST × 4/8/16/32-shot × seed 1/2/3 × 两方法，共 72 组。普通 Deep Prompt 的逐层替换机制保持原实现；融合针对类别 token 在 block 8 的注入。串行输入为融合前的 TextMLP 输出，梯度不断开。

全部从头训练，不复用旧选模结果。模型构建后统一重设训练 seed；固定 description bank 沿用 `caches/three_stage_banks/`。未完成任务重新执行时归档后从头训练，已完成任务跳过。

输出包括 `protocol.json`、每组的 `results.json`（`best_val`，含所选轮次和验证 accuracy）、`per_seed.csv`、`summary.csv`、`overall.csv` 和 `summary.md`。汇总报告逐 seed 测试 accuracy、均值和标准差；整体 accuracy 先对 seed 求均值，再对数据集和 shot 等权平均。

单组训练使用 `python -m tools.experiments single --output <目录> --dataset dermamnist --method visual --shots 4 --seed 4`。批量模式可用 `--tasks <tasks.json>` 接续已有实验清单及其输出目录中的 `protocol.json`。调度、训练和汇总统一放在 `tools/experiments.py`。

补充实验：DermaMNIST、4-shot、串行＋双端 0.5 融合另跑 seed 4、5，沿用 100 轮训练和完整验证集 accuracy 最优选模。当前 72 组队列完成后在 GPU 0、2 启动，独立保存至 `output/prompt_fusion_fullval_best_seed45/`，不混入原 seed 1/2/3 主表。

运行测试：

```bash
python -m pytest tests -q
```

测试使用小型 BERT 和 ViT，不需要下载真实 BiomedCLIP 权重；真实权重集成测试需要显式设置 `RUN_BIOMEDCLIP_INTEGRATION=1`。

## 测试集混淆计数矩阵

`build_test_confusion_count_matrix.py` 仅遍历 dataset 的 `test` split，使用冻结的
BiomedCLIP 与现有 Mean-50 biomedical description 类别原型，以 cosine logits 的
`argmax` 作为预测类别，累计整数矩阵 `M[真实类别, 预测类别]`：

```bash
CUDA_VISIBLE_DEVICES=0 python build_test_confusion_count_matrix.py \
  --root /mnt/nas1/disk09/yuejianwu/biomedcoop/data \
  --dataset-config-file configs/datasets/dermamnist.yaml \
  --output output/test_confusion_counts/DermaMNIST
```

输出为 `test_confusion_count_matrix.pt` 和
`test_confusion_count_matrix.png`。矩阵是 CPU `torch.long` 计数张量；图片直接
标注每个真实类别与预测类别组合的测试图片数量，不做归一化，也不使用
train/validation 图片。
