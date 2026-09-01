# BiomedCoOp 中文说明

## 方法说明

本仓库基于 BiomedCLIP 实现医学图像少样本提示学习。当前 DermaMNIST 主线使用 CoOp、Visual VPT-Deep 和 MT-TCP 从头联合训练，并提供基于训练集 support 样本构建的 Soft Bank 混淆感知分支。BiomedCLIP 主干保持冻结，仅更新提示相关参数；K-shot 采样只作用于训练集，验证集和测试集保持官方完整划分。

同一次实验中的可训练提示参数共用一套优化器和配置中的 `OPTIM.LR`，各提示分支不再设置独立学习率。训练、验证和测试的 `batch_size` 固定为 `32`，`num_workers` 固定为 `8`。实验 seed 只允许 `1、2、3`，每条训练命令运行其中一个 seed。cuDNN 使用 PyTorch 默认状态。

## 安装

在仓库根目录执行：

```powershell
pip install -r requirements.txt
pip install -e .\Dassl.pytorch
```

## 单次训练

下面命令运行 DermaMNIST 4-shot、seed 1 的 MT-TCP 基线（`b0`）：

```powershell
python train.py `
  --root D:\Data\dermamnist `
  --output-dir output\mt_tcp\b0\shots_4\seed1 `
  --seed 1 `
  --trainer CoOpVPT_BiomedCLIP `
  --dataset-config-file configs/datasets/dermamnist.yaml `
  --config-file configs/trainers/CoOp/dermamnist_native_vpt_multitext_tcp.yaml `
  DATASET.NUM_SHOTS 4
```

学习率、优化器、训练轮数及提示结构均以 YAML 配置为准，命令行不重复传入这些参数。

## TCP 消融

TCP 默认开启，因此上面的命令就是 TCP-on 对照组。TCP-off 保留 VPT 中已有的 Text Deep Prompt 及其可学习 token，只将 TCP 类别描述残差固定为零，并冻结 TCP 投影和门控参数；不会增加第二套 Text Prompt，也没有新增 Text Prompt 配置。CoOp、Visual/Text VPT、优化器、`OPTIM.LR`、batch、workers、shot 和 seed 均保持不变。为避免同时改变两个机制，TCP-off 仅允许使用不含 Confusion 模块的 `b0`。

运行 TCP-off 时，在同一条训练命令末尾增加：

```powershell
TRAINER.TCP.ENABLED False `
TRAINER.CONFUSION_AWARE.VARIANT b0
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
  TRAINER.TCP.ENABLED False `
  TRAINER.CONFUSION_AWARE.VARIANT b0
```

TCP-on 与 TCP-off 必须使用不同输出目录；两者的 checkpoint 会记录 TCP 状态，不能交叉恢复或加载。

## 批量运行 shots 和 seeds

旧的批量启动文件已删除。需要批量实验时，直接在 PowerShell 中循环调用 `train.py`：

```powershell
$dataRoot = 'D:\Data\dermamnist'
$outputRoot = 'output\mt_tcp\b0'
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
| CoOp + Visual VPT-Deep + MT-TCP | `CoOpVPT_BiomedCLIP` | `configs/trainers/CoOp/dermamnist_native_vpt_multitext_tcp.yaml` |

## 混淆感知分支

先为需要的 shot 和 seed 构建 Soft Bank：

```powershell
python scripts/coopvpt/build_confusion_prior.py `
  --data-root D:\Data\dermamnist `
  --output-root output\soft_confusion_banks `
  --shots 1 2 4 8 16 32
```

然后在单次训练命令末尾追加所需变体和 Bank 路径，例如：

```powershell
TRAINER.CONFUSION_AWARE.VARIANT full `
TRAINER.CONFUSION_AWARE.BANK_ROOT output/soft_confusion_banks
```

可用变体为 `b0`、`pair`、`semantic`、`semantic_global`、`semantic_local`、`global_local` 和 `full`；其中 `b0` 不使用 Soft Bank。
