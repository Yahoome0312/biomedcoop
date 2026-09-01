# BiomedCoOp 中文说明

## 方法说明

本仓库基于 BiomedCLIP 实现医学图像少样本提示学习。当前 DermaMNIST 主线使用 CoOp、Visual/Text VPT、可选 TCP 和固定的 Full Confusion 从头联合训练。Full Confusion 使用训练集 support 样本构建的 Soft Bank，不再提供其他 Confusion 变体。BiomedCLIP 主干保持冻结，仅更新提示及 Full Confusion 参数；K-shot 采样只作用于训练集，验证集和测试集保持官方完整划分。

同一次实验中的可训练提示参数共用一套优化器和配置中的 `OPTIM.LR`，各提示分支不再设置独立学习率。训练、验证和测试的 `batch_size` 固定为 `32`，`num_workers` 固定为 `8`。实验 seed 只允许 `1、2、3`，每条训练命令运行其中一个 seed。cuDNN 使用 PyTorch 默认状态。

## 安装

在仓库根目录执行：

```powershell
pip install -r requirements.txt
pip install -e .\Dassl.pytorch
```

## 单次训练

先为 DermaMNIST 4-shot 的 seed 1、2、3 构建 Soft Bank：

```powershell
python scripts/coopvpt/build_confusion_prior.py `
  --data-root D:\Data\dermamnist `
  --output-root output\soft_confusion_banks `
  --shots 4
```

然后运行 DermaMNIST 4-shot、seed 1 的 Full Confusion + TCP：

```powershell
python train.py `
  --root D:\Data\dermamnist `
  --output-dir output\full_confusion\tcp_on\shots_4\seed1 `
  --seed 1 `
  --trainer CoOpVPT_BiomedCLIP `
  --dataset-config-file configs/datasets/dermamnist.yaml `
  --config-file configs/trainers/CoOp/dermamnist_native_vpt_multitext_tcp.yaml `
  DATASET.NUM_SHOTS 4
```

学习率、优化器、训练轮数及提示结构均以 YAML 配置为准，命令行不重复传入这些参数。

## TCP 消融

TCP 默认开启，因此上面的命令就是 TCP-on 对照组。TCP-off 保留相同的 Full Confusion 和 VPT 中已有的 Text Deep Prompt，只将 TCP 类别描述残差固定为零，并冻结 TCP 投影和门控参数；不会增加第二套 Text Prompt。CoOp、Visual/Text VPT、Full Confusion、margin loss、优化器、`OPTIM.LR`、batch、workers、shot 和 seed 均保持不变。

运行 TCP-off 时，在同一条训练命令末尾增加：

```powershell
TRAINER.TCP.ENABLED False
```

例如 DermaMNIST 4-shot、seed 1：

```powershell
python train.py `
  --root D:\Data\dermamnist `
  --output-dir output\tcp_ablation\without_tcp\shots_4\seed1 `
  --seed 1 `
  --trainer CoOpVPT_BiomedCLIP `
  --dataset-config-file configs/datasets/dermamnist.yaml `
  --config-file configs/trainers/CoOp/dermamnist_native_vpt_multitext_tcp.yaml `
  DATASET.NUM_SHOTS 4 `
  TRAINER.TCP.ENABLED False
```

TCP-on 与 TCP-off 必须使用不同输出目录；两者的 checkpoint 会记录 TCP 状态，不能交叉恢复或加载。

## 批量运行 shots 和 seeds

旧的批量启动文件已删除。需要批量实验时，直接在 PowerShell 中循环调用 `train.py`：

```powershell
$dataRoot = 'D:\Data\dermamnist'
$outputRoot = 'output\full_confusion\tcp_on'
$trainer = 'CoOpVPT_BiomedCLIP'
$trainerConfig = 'configs/trainers/CoOp/dermamnist_native_vpt_multitext_tcp.yaml'
$shotsList = @(1, 2, 4, 8, 16, 32)
$seedList = @(1, 2, 3)

foreach ($shots in $shotsList) {
  foreach ($seed in $seedList) {
    $outputDir = Join-Path $outputRoot "shots_$shots\seed$seed"
    python train.py `
      --root $dataRoot `
      --output-dir $outputDir `
      --seed $seed `
      --trainer $trainer `
      --dataset-config-file configs/datasets/dermamnist.yaml `
      --config-file $trainerConfig `
      DATASET.NUM_SHOTS $shots

    if ($LASTEXITCODE -ne 0) {
      throw "训练失败：shots=$shots, seed=$seed"
    }
  }
}
```

其他现有方法也使用同一条命令，只需替换 Trainer 和配置文件：

| 方法 | Trainer | 配置文件 |
|---|---|---|
| BiomedCoOp | `BiomedCoOp_BiomedCLIP` | `configs/trainers/BiomedCoOp/few_shot/dermamnist.yaml` |
| 原生 CoOp | `CoOp_BiomedCLIP` | `configs/trainers/CoOp/dermamnist_native.yaml` |
| CoOp + Visual/Text VPT + MT-TCP + Full Confusion | `CoOpVPT_BiomedCLIP` | `configs/trainers/CoOp/dermamnist_native_vpt_multitext_tcp.yaml` |

## Full Confusion

批量实验前，为全部 shots 和固定 seed 1、2、3 构建 Soft Bank：

```powershell
python scripts/coopvpt/build_confusion_prior.py `
  --data-root D:\Data\dermamnist `
  --output-root output\soft_confusion_banks `
  --shots 1 2 4 8 16 32
```

Soft Bank 路径已在 YAML 中固定为 `output/soft_confusion_banks`。训练阶段使用真实标签选择 Bank 行和困难负类；验证、测试阶段没有标签输入，使用基础 logits 的 top-1 选择 Bank 行。训练损失固定为交叉熵加 confusion margin loss。
