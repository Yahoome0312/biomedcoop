# BiomedCoOp 中文说明

## 方法说明

本仓库基于 BiomedCLIP 实现医学图像少样本提示学习。当前 DermaMNIST 主线使用 CoOp、Visual/Text VPT、可选 TCP 和可选 Full Confusion 从头联合训练，TCP 与 Full Confusion 均默认开启。Full Confusion 仅根据当前图像的在线预测选择混淆类别对，并使用 LLM 给出的有向类别对描述生成 Semantic 特征，不再直接计算类别文本特征差。BiomedCLIP 主干保持冻结，仅更新已启用的提示和 Full Confusion 参数；K-shot 采样只作用于训练集，验证集和测试集保持官方完整划分。

同一次实验中的可训练提示参数共用一套优化器和配置中的 `OPTIM.LR`，各提示分支不再设置独立学习率。训练、验证和测试的 `batch_size` 固定为 `32`，`num_workers` 固定为 `8`。实验 seed 只允许 `1、2、3`，每条训练命令运行其中一个 seed。cuDNN 使用 PyTorch 默认状态。

## 安装

在仓库根目录执行：

```bash
pip install -r requirements.txt
pip install -e ./Dassl.pytorch
```

## 单次训练

直接运行 DermaMNIST 4-shot、seed 1 的 Full Confusion + TCP，无需预生成 confusion bank：

```bash
python train.py \
  --root /mnt/nas1/disk09/yuejianwu/biomedcoop/data \
  --output-dir output/full_confusion/tcp_on/shots_4/seed1 \
  --seed 1 \
  --trainer CoOpVPT_BiomedCLIP \
  --dataset-config-file configs/datasets/dermamnist.yaml \
  --config-file configs/trainers/CoOp/dermamnist_native_vpt_multitext_tcp.yaml \
  DATASET.NUM_SHOTS 4
```

学习率、优化器、训练轮数及提示结构均以 YAML 配置为准，命令行不重复传入这些参数。

## TCP 消融

TCP 默认开启，因此上面的命令就是 TCP-on 对照组。TCP-off 保留相同的 Full Confusion 和 VPT 中已有的 Text Deep Prompt，只将 TCP 类别描述残差固定为零，并冻结 TCP 投影和门控参数；不会增加第二套 Text Prompt。CoOp、Visual/Text VPT、Full Confusion、margin loss、优化器、`OPTIM.LR`、batch、workers、shot 和 seed 均保持不变。

运行 TCP-off 时，在同一条训练命令末尾增加：

```bash
TRAINER.TCP.ENABLED False
```

例如 DermaMNIST 4-shot、seed 1：

```bash
python train.py \
  --root /mnt/nas1/disk09/yuejianwu/biomedcoop/data \
  --output-dir output/tcp_ablation/without_tcp/shots_4/seed1 \
  --seed 1 \
  --trainer CoOpVPT_BiomedCLIP \
  --dataset-config-file configs/datasets/dermamnist.yaml \
  --config-file configs/trainers/CoOp/dermamnist_native_vpt_multitext_tcp.yaml \
  DATASET.NUM_SHOTS 4 \
  TRAINER.TCP.ENABLED False
```

TCP-on 与 TCP-off 必须使用不同输出目录；两者的 checkpoint 会记录 TCP 状态，不能交叉恢复或加载。

## Confusion Aware 消融

Confusion Aware 默认开启。关闭时在训练命令末尾增加：

```bash
TRAINER.CONFUSION_AWARE.ENABLED False
```

关闭后代码会直接使用基础分类 logits 和交叉熵，并跳过 `confuse_pair/<dataset>.txt`、Confusion Adapter、margin loss、Confusion 分析记录及对应梯度检查。`GAMMA=0` 和 `LAMBDA_CONF=0` 也不再作为关闭方式。

五组消融使用下面的 Trainer 和开关组合：

| 方法 | Trainer | 命令末尾配置 |
|---|---|---|
| CoOp | `CoOp_BiomedCLIP` | 无 |
| CoOp + Deep Prompt | `CoOpVPT_BiomedCLIP` | `TRAINER.TCP.ENABLED False TRAINER.CONFUSION_AWARE.ENABLED False` |
| CoOp + Deep Prompt + MT-TCP | `CoOpVPT_BiomedCLIP` | `TRAINER.CONFUSION_AWARE.ENABLED False` |
| CoOp + Deep Prompt + Confusion Aware | `CoOpVPT_BiomedCLIP` | `TRAINER.TCP.ENABLED False` |
| CoOp + Deep Prompt + MT-TCP + Confusion Aware | `CoOpVPT_BiomedCLIP` | 无 |

不同组合必须使用不同输出目录。checkpoint 会同时记录 TCP 和 Confusion Aware 状态，不能在不同组合之间交叉恢复或加载。

## 批量运行 shots 和 seeds

旧的批量启动文件已删除。需要批量实验时，直接在服务器 Bash 中循环调用 `train.py`：

```bash
DATA_ROOT=/mnt/nas1/disk09/yuejianwu/biomedcoop/data
OUTPUT_ROOT=output/full_confusion/tcp_on
TRAINER=CoOpVPT_BiomedCLIP
TRAINER_CONFIG=configs/trainers/CoOp/dermamnist_native_vpt_multitext_tcp.yaml

for SHOTS in 1 2 4 8 16 32; do
  for SEED in 1 2 3; do
    OUTPUT_DIR="${OUTPUT_ROOT}/shots_${SHOTS}/seed${SEED}"
    if ! python train.py \
      --root "$DATA_ROOT" \
      --output-dir "$OUTPUT_DIR" \
      --seed "$SEED" \
      --trainer "$TRAINER" \
      --dataset-config-file configs/datasets/dermamnist.yaml \
      --config-file "$TRAINER_CONFIG" \
      DATASET.NUM_SHOTS "$SHOTS"; then
      echo "训练失败：shots=${SHOTS}, seed=${SEED}" >&2
      exit 1
    fi
  done
done
```

其他现有方法也使用同一条命令，只需替换 Trainer 和配置文件：

| 方法 | Trainer | 配置文件 |
|---|---|---|
| BiomedCoOp | `BiomedCoOp_BiomedCLIP` | `configs/trainers/BiomedCoOp/few_shot/dermamnist.yaml` |
| 原生 CoOp | `CoOp_BiomedCLIP` | `configs/trainers/CoOp/dermamnist_native.yaml` |
| CoOp + Visual/Text VPT + MT-TCP + Full Confusion | `CoOpVPT_BiomedCLIP` | `configs/trainers/CoOp/dermamnist_native_vpt_multitext_tcp.yaml` |

## Full Confusion

当前方法只保留在线 confusion，不读取 support 图像生成的离线概率矩阵。已移除构建脚本及 `BANK_ROOT`、`PRIOR_ALPHA` 配置。

设基础 logits 为 $z\in\mathbb{R}^{B\times C}$，在线概率为 $p=\operatorname{softmax}(\operatorname{stopgrad}(z))$。训练时锚点 $a=y$；验证和测试时 $a=\arg\max_c z_c$，不输入真实标签。困难负类为 $b=\arg\max_{c\ne a}p_c$，随当前图像和模型预测变化；不累计跨批次 bank，也不使用离线先验加权。

Adapter 输入为全局图像特征 `[B,512]`、patch tokens `[B,N,768]`、类别文本特征 `[C,512]`、基础 logits `[B,C]`、logit scale 和锚点 `[B]`。类别对描述经冻结文本编码器平均、归一化后形成语义特征表 `[C,C,512]`，在模型初始化时生成；它提供类别对语义，不包含 support 图像混淆概率。

选中类别对的描述特征经 Semantic projector 生成语义向量，分别引导全局特征门控和 patch 注意力，再通过全局/局部门控及融合层得到 confusion 特征 $h$。最终图像特征为 $\hat v=\operatorname{normalize}(\operatorname{normalize}(v)+\gamma\operatorname{normalize}(h))$，使用原类别文本特征计算最终 logits，输出 `[B,C]` 及配对、在线概率、门控权重等分析信息。

训练目标为 $L=L_{CE}+\lambda_{conf}\operatorname{mean}(\operatorname{softplus}(z^{final}_b-z^{final}_y))$。离散配对选择不反向传播，语义/视觉融合分支保留梯度。在线版本采用新的 checkpoint protocol，旧离线版本 checkpoint 不支持直接恢复；关闭 Confusion 的 protocol 保持不变。

Semantic 特征来自数据集对应的有向类别对文件：DermaMNIST、Kvasir 和 CHMNIST 分别使用 `confuse_pair/DermaMNIST.txt`、`confuse_pair/Kvasir.txt` 和 `confuse_pair/CHMNIST.txt`。文件必须是下面的 JSON 结构：

```json
{
  "class a": {
    "class b": [
      "Compared with class a, class b differs in ...",
      "Another directed distinction ..."
    ]
  },
  "class b": {
    "class a": [
      "Compared with class b, class a differs in ..."
    ]
  }
}
```

外层类别必须与数据集类别完全对应；每个类别必须包含指向其他所有类别的描述，不能包含自身；每个有向类别对可以有不同数量的描述，但列表不能为空。代码会动态读取类别数量，将同一有向类别对的全部描述通过冻结 BiomedCLIP 编码后平均，因此其他数据集不需要修改模型代码，只需增加同格式文件。

## 本次 no-bank 实验

本次运行 DermaMNIST、Kvasir、CHMNIST 的 4/8/16/32-shot，seed 1/2/3，共 36 次；TCP 关闭，在线 Confusion 开启。其余参数使用主线 YAML（100 epoch）。输出位于 `output/no_bank_confusion_3datasets_4_32shot/<dataset>/tcp_off/shots_<K>/seed<S>/`；`evaluation/accuracy` 和 `evaluation/balanced_accuracy` 分别保存按对应验证指标选出的 checkpoint 的测试结果，`_manager/status.json` 记录调度进度。
