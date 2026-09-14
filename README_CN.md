# BiomedCoOp 中文说明

## 当前训练主线

本仓库使用冻结的 BiomedCLIP 进行医学图像少样本提示学习。当前 CoOp + Visual/Text VPT 训练器只保留一条共享 TKE TCP 路径；Confusion 和 Expert MoE 代码、配置与测试已移除。数据集、few-shot sampling、augmentation、优化器、scheduler、learning rate、epoch、batch size 和 seed 约束沿用原设置。

训练集使用 K-shot 采样，验证集和测试集保持官方划分。训练、验证和测试的 batch size 固定为 32，num_workers 固定为 8。主实验 seed 为 1、2、3，补充实验可使用 4。

## 安装

在仓库根目录执行：

```bash
conda activate /mnt/nas1/disk09/yuejianwu/.conda/envs/biocoop
pip install -r requirements.txt
pip install -e ./Dassl.pytorch
```

## 单次训练

DermaMNIST 4-shot、seed 1 的 TCP 训练命令：

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

Text Encoder 的 block 0–7 使用正常 CoOp/Text Deep Prompt。进入 block 8 前，CLS 后的 4 个 prompt slots 一次性替换为对应类别的 TKE tokens；block 9–11 直接使用上一层 hidden states，不再重新生成或覆盖 TCP tokens。

当前实现不包含 5×10 grouping、LayerBasis、XProto/B+Delta、跨类别 centering、norm matching、layer gate 或多层 TCP 重注入。description bank 和 class prototype 均为 frozen buffer；BiomedCLIP backbone 也保持冻结。训练参数只有 CoOp context、Visual Deep Prompt、注入前 Text Deep Prompt 和共享 TKE。TCP prompt bundle 共 486,528 个可训练参数（TCP 开启时）。

`TRAINER.TCP.ENABLED=False` 时保留普通 Text Deep Prompt，冻结 TKE 参数并跳过 TCP replacement。TCP 没有实现模式选择项，配置只包含：

```yaml
TRAINER:
  TCP:
    ENABLED: True
    DESCRIPTION_CACHE: ""
    INSERT_LAYER: 8
```

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
| CoOp + Visual/Text VPT + TCP | `CoOpVPT_BiomedCLIP` | `configs/trainers/CoOp/dermamnist_native_vpt_tcp.yaml` |
| BiomedCoOp | `BiomedCoOp_BiomedCLIP` | `configs/trainers/BiomedCoOp/few_shot/dermamnist.yaml` |

## Checkpoint 与验证

训练 checkpoint 保存 `prompt_parameters` bundle、optimizer、scheduler、AMP scaler 及 TCP 结构元数据。恢复时严格校验 TKE 维度、层数、插入层、token 数和 description bank 元数据；当前代码不提供旧 TCP、Confusion 或 MoE 实现的加载入口。

运行测试：

```bash
python -m pytest tests -q
```

测试使用小型 BERT 和 ViT，不需要下载真实 BiomedCLIP 权重；真实权重集成测试需要显式设置 `RUN_BIOMEDCLIP_INTEGRATION=1`。
