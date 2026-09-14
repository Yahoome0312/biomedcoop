# BiomedCoOp 中文说明

## 方法说明

本仓库基于 BiomedCLIP 实现医学图像少样本提示学习。当前 DermaMNIST 主线使用 CoOp、Visual/Text VPT、可选 TCP 和可选 Full Confusion 从头联合训练，TCP 与 Full Confusion 均默认开启。Full Confusion 仅根据当前图像的在线预测选择混淆类别对，并使用 LLM 给出的有向类别对描述生成 Semantic 特征，不再直接计算类别文本特征差。BiomedCLIP 主干保持冻结，仅更新已启用的提示和 Full Confusion 参数；K-shot 采样只作用于训练集，验证集和测试集保持官方完整划分。

同一次实验中的可训练提示参数共用一套优化器和配置中的 `OPTIM.LR`，各提示分支不再设置独立学习率。训练、验证和测试的 `batch_size` 固定为 `32`，`num_workers` 固定为 `8`。主实验 seed 使用 `1、2、3`，补充实验允许使用 `4`，每条训练命令运行其中一个 seed。cuDNN 使用 PyTorch 默认状态。

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

设基础 logits 为 $z\in\mathbb{R}^{B\times C}$，在线概率为 $p=\operatorname{softmax}(\operatorname{stopgrad}(z))$。训练、验证和测试统一使用预测路由：$a=\arg\max_c z_c$，$b=\arg\max_{c\ne a}z_c$，即 `pair_first` 和 `pair_second` 分别为 base top-1 和 top-2。GT 不参与语义类别对选择，因此允许 `pair_second == label`。

Adapter 输入为全局图像特征 `[B,512]`、patch tokens `[B,N,768]`、类别文本特征 `[C,512]`、基础 logits `[B,C]`、logit scale 和锚点 `[B]`。类别对描述经冻结文本编码器平均、归一化后形成语义特征表 `[C,C,512]`，在模型初始化时生成；它提供类别对语义，不包含 support 图像混淆概率。

选中类别对的描述特征经 Semantic projector 生成语义向量，分别引导全局特征门控和 patch 注意力，再通过全局/局部门控及融合层得到 confusion 特征 $h$。最终图像特征为 $\hat v=\operatorname{normalize}(\operatorname{normalize}(v)+\gamma\operatorname{normalize}(h))$，使用原类别文本特征计算最终 logits，输出 `[B,C]` 及配对、在线概率、门控权重等分析信息。

margin loss 的 hard negative 与语义路由独立：先复制并 detach 基础 logits，将 GT 位置置为 $-\infty$，再取 $q=\arg\max_{c\ne y}z_c$，始终保证 $q\ne y$。训练目标为 $L=L_{CE}+\lambda_{conf}\operatorname{mean}(\operatorname{softplus}(z^{final}_q-z^{final}_y))$。CE、$\lambda_{conf}$ 和其他训练参数不变。离散选择不反向传播，语义/视觉融合分支保留梯度。

confusion details 复用现有字典并记录 `pair_first`、`pair_second`、`competitor`、`base_prediction` 和 `final_prediction`。训练 epoch 记录、验证结果和测试结果增加 `base_top2_accuracy`、`base_top5_accuracy`；逐样本评估记录同时保存两个命中标记。当前预测路由版本使用独立 checkpoint protocol，不能恢复 GT 路由实验的 checkpoint；关闭 Confusion 的 protocol 保持不变。

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

## 预测路由第一阶段实验

第一阶段只把训练语义类别对从 GT 路由替换为 base top-1/top-2，并把 margin competitor 替换为排除 GT 后的 base hard negative。对照设置保持 TCP 关闭，原模型、Prompt、ConfusionAwareAdapter、优化与数据配置均不变。实验覆盖 DermaMNIST、Kvasir、CHMNIST 的 4/8/16/32-shot 和 seed 1/2/3，共 36 次；GPU 1–6 各自串行执行完整实验。

输出位于 `output/predicted_routing_stage1_3datasets_4_32shot/<dataset>/tcp_off/shots_<K>/seed<S>/`。查看队列用 `tmux attach -t predicted-routing-s1` 或读取 `_manager/status.json`；汇总结果写入 `_summary/results_detailed.csv` 和 `_summary/results_summary.csv`。

## 原始 TCP + Confusion 联合重跑

使用撤销safe correction后的原始模型：TCP文本编码得到特征和logits，Confusion使用这些表示与预测top-1/top-2进行融合。最终由Confusion输出预测，loss为原有CE+LAMBDA_CONF×L_conf，无额外gate、preservation或梯度/RNG隔离。

三数据集×4/8/16/32-shot×seeds1/2/3，共36次，先8-shot再4/16/32-shot，不设置性能筛选。配置沿用原任务YAML，显式TCP.ENABLED=True与CONFUSION_AWARE.ENABLED=True；模型和训练超参数不改。

队列：`output/original_tcp_confusion_joint_3datasets_4_32shot/_manager/run.py`；tmux会话`original-joint-36`。输出为`<dataset>/both/shots_<K>/seed<S>/`，`evaluation/accuracy`和`evaluation/balanced_accuracy`保存两种验证选模的测试结果。`_manager/status.json`记录进度，`_summary`保存逐seed和均值/标准差CSV及历史对照。对话在确认首任务训练后结束，tmux继续运行。


## 冻结双专家 Linear MoE（最小验证版）

入口为 `TRAINER.EXPERT_MOE.ENABLED=True`，自动选择 `ExpertMoE_BiomedCLIP`。
`TCP_CHECKPOINT` 与 `CONF_CHECKPOINT` 接受历史 `prompt_parameters/model-best.pth.tar` 文件路径；必须来自相同 dataset、classnames、shots、seed 和模型配置。沿用原 checkpoint 的严格结构、TCP bank 元数据与 Confusion prediction-routing protocol 校验，不兼容旧 GT-routing checkpoint。

两个专家分别通过原 CoOpVPT 构建器恢复独立的 CoOp、Visual Deep Prompt、Text Deep Prompt 和对应 adapter。TCP 专家关闭 Confusion；Confusion 专家将 TCP residual scale 设为 0，保留原文本聚合与自己的 pre-confusion prediction routing。历史 checkpoint 保存的是完整可学习 Prompt bundle，冻结 BiomedCLIP 和 description/pair banks 仍由原加载路径重建、校验。第一版使用两份独立 backbone，便于保证全部参数对象独立。

两个专家 `requires_grad=False`，始终 eval，并在 `no_grad` 中前向；外层 `.train()` 只使 Router 进入训练状态。输入 image `[B,3,H,W]`，专家 logits `[B,C]`。Router 输入为 `[softmax(tcp_logits), softmax(conf_logits), abs(p_tcp-p_conf)]`，形状 `[B,3C]`；唯一可学习层为零初始化 `Linear(3C,2,bias=True)`，共 `6C+2` 个参数（8 类为 50）。softmax 后输出 `[B,2]`，两列分别为 TCP/Confusion 权重，初始均为 0.5、每行和为 1。

概率融合为 `p_final=w_tcp*p_tcp+w_conf*p_conf`，MoE forward 返回 `log(p_final.clamp_min(1e-8))`；唯一训练目标为该输出的 `F.nll_loss(output,label)`。Router 与概率运算使用 fp32。optimizer/checkpoint 仅包含 Router；没有专家辅助 loss 或专家更新。`return_weights=True` 可在 MoE 模式返回 `(log_probs, weights)`。

继续使用原全部 few-shot train dataset、sampler、augmentation、seed、batch size、optimizer、LR、scheduler 与 MAX_EPOCH。只在 MoE DataLoader 保留最后不足一个 batch 的样本，不新建 split。每个 epoch 只用既有 validation accuracy 选择 `router/model-best.pth.tar`；测试不参与训练或选模。原 YAML 的 `SKIP_FINAL_TEST=True` 保留，可显式 eval 最佳 Router。`MODE` 支持 `linear_moe`、`tcp_only`、`conf_only`，两个单专家模式只用于 `--eval-only`，直接返回原 logits、不需要 Router checkpoint。

推荐命令（占位路径替换为同一实验的实际路径，GPU_ID 选一张可用卡；shots 可为 4/8/16/32）：

```bash
conda activate /mnt/nas1/disk09/yuejianwu/.conda/envs/biocoop
GPU_ID=0
TCP_CKPT=/absolute/path/to/tcp_run/prompt_parameters/model-best.pth.tar
CONF_CKPT=/absolute/path/to/conf_run/prompt_parameters/model-best.pth.tar
COMMON=(--root /absolute/path/to/data --dataset-config-file configs/datasets/chmnist.yaml
  --config-file configs/trainers/CoOp/dermamnist_native_vpt_multitext_tcp.yaml --seed 1)
MOE=(TRAINER.EXPERT_MOE.ENABLED True DATASET.NUM_SHOTS 4
  TRAINER.EXPERT_MOE.TCP_CHECKPOINT "$TCP_CKPT"
  TRAINER.EXPERT_MOE.CONF_CHECKPOINT "$CONF_CKPT")
CUDA_VISIBLE_DEVICES=$GPU_ID python train.py "${COMMON[@]}" \
  --output-dir output/linear_moe/chmnist/shots_4/seed1 "${MOE[@]}"
CUDA_VISIBLE_DEVICES=$GPU_ID python train.py "${COMMON[@]}" --eval-only \
  --model-dir output/linear_moe/chmnist/shots_4/seed1 \
  --output-dir output/linear_moe/chmnist/shots_4/seed1/test "${MOE[@]}" TEST.SPLIT test
for MODE in tcp_only conf_only; do
  CUDA_VISIBLE_DEVICES=$GPU_ID python train.py "${COMMON[@]}" --eval-only \
    --output-dir output/linear_moe/chmnist/shots_4/seed1/$MODE \
    "${MOE[@]}" TRAINER.EXPERT_MOE.MODE "$MODE" TEST.SPLIT test
done
```

`tests/test_expert_moe.py` 使用微型 tower 运行原 `CustomCLIP.forward`，检查专家独立与冻结、梯度隔离、优化一步前后专家参数逐值相等、初始及训练后权重归一化、单专家旁路、Confusion 自身预测选 pair 且不接收 GT，以及加载配置隔离。真实数据上的性能是否超过单专家仍需完成 Router 训练和对照评估。

历史专家可分别配置 `TRAINER.EXPERT_MOE.TCP_DESCRIPTION_CACHE`、`TCP_LAYER_DESCRIPTION_CACHE`、`CONF_DESCRIPTION_CACHE`、`CONF_LAYER_DESCRIPTION_CACHE`，以恢复各自训练时的 bank；为空时沿用 `TRAINER.TCP` 的缓存设置。优先使用与历史 checkpoint 校验一致的现存缓存，默认严格校验全部元数据。首组实验的 TCP 缓存来自 `output/selective_confusion_v1/confusion_predictions/_cache/dermamnist/`。

当原 checkpoint 未持久化 bank、只能从冻结 BiomedCLIP 与原描述重建时，可显式设置 `TRAINER.EXPERT_MOE.REBUILD_BANKS=True`。此模式仅允许数值 prior/pair feature 字节指纹不同并打印提示；不修改历史文件或文本聚合，仍严格校验参数结构、类别/描述/模型/方法/协议，并从 checkpoint 所在运行目录的 `initialization_manifest.json` 校验原 Confusion pair 描述来源。原单分支默认校验行为保持不变。本次实验在 Router 训练前对照历史 validation accuracy，且逐样本核对 Confusion 的 final prediction 与 pair_first/pair_second；test 不用于恢复核对。

首组实测（GPU4，DermaMNIST 4-shot seed1，100 epoch）：validation选出的epoch1 Router在test上accuracy为63.79%，TCP为62.04%、Confusion为63.34%；对应balanced accuracy为42.71%、43.94%、34.82%。仅有单组accuracy小幅收益，尚不足以证明全面改进。结果及专家恢复核对记录位于`output/linear_moe/dermamnist/shots_4/seed1/`。

全部36组Linear MoE队列入口为 `output/linear_moe/_manager/run_all.py`，固定GPU4串行执行，复用已经完成的实验。`--check-only`检查所需历史文件。队列先完成Kvasir/CHMNIST 32-shot真实优化步预运行（DermaMNIST已有完整结果），再训练剩余配置；每组仍运行100epoch，按validation accuracy选模并测试三个模式。汇总只关注accuracy，输出 `output/linear_moe/accuracy_summary.csv` 和 `accuracy_summary.md`；进度见 `_manager/status.json`，运行中15分钟心跳、任务退出立即接续。

截至2026-09-14 07:54，Linear MoE的36/36组实验全部完成，无失败/重试。按三个seed均值比较，12个dataset×shot设置有11个优于最佳单专家，唯一例外是DermaMNIST32-shot（-0.07个百分点）。逐seed对比为26胜、5平、5负，平均较同seed最佳单专家提升0.65个百分点。36组等权test accuracy为TCP74.53%、Confusion75.66%、MoE76.88%。完整均值±样本标准差、逐seed结果、选模epoch及耗时见[完整实验报告](output/linear_moe/full_experiment_report.md)。19/36组最佳Router为epoch1；本次没有固定50/50融合对照，因此尚不能单独归因于动态路由学习。

TCP基线来源澄清（2026-09-14更新）：上述MoE使用`paper_3datasets_4methods/coop_deep_prompt_mttcp`，36组TCP重评test accuracy与该批原始记录逐seed一致。用户旧表三个数据集全部12个均值/标准差已定位到`coop_deep_prompt_mttcp_gpu2_4_32shot`的best_validation_accuracy.json，即最佳验证集成绩，不是测试集成绩；对应36个checkpoint均存在。两批均为grouped10/LayerBasis版本，不能把旧表当成mean50测试结果，也不能将旧表与MoE测试结果直接比较。补测入口为`output/historical_tcp_test_comparison/evaluate.py`，使用GPU4，仅加载validation accuracy最佳checkpoint并验证恢复后评估test，不训练。

六方法统一结果见[核对表](output/linear_moe/six_method_accuracy_verified.md)：从216条原始test记录重算，使用validation accuracy选模和3个seed样本标准差；包含CoOp、CoOp+视觉/文本Deep Prompt、TCP、预测路由Confusion、联合TCP+Confusion与冻结专家Linear MoE。TCP明确为paper批次grouped10/LayerBasis版本，不混入旧TCP验证集表或mean50结果；逐seed原始文件路径保存在six_method_accuracy_detailed.csv。

旧TCP批次补评已完成36/36组，每组val accuracy均复现。两批TCP的聚合元数据均为grouped10_layer_residual，不是mean50；旧批test与paper批test在12个设置中5高7低，整体等权均值分别74.52%和74.53%。旧表CHMNIST32-shot90.57±1.38是val，实际test为88.65±0.78。详见[旧批TCP测试集对比](output/historical_tcp_test_comparison/comparison.md)，逐seed checkpoint路径和成绩见同目录detailed.csv。恢复允许bank数值指纹差异，不宣称bank逐位相同。


### Original-style Biomedical TCP（结构消融）

在现有运行命令末尾设置 `TRAINER.TCP.MODE original_style`；默认 `multitext` 保留原 LayerBasis + XProto 路径及旧 checkpoint。`TRAINER.TCP.INSERT_LAYER` 默认 8（从 0 编号）。数据、采样、CoOp、Visual Deep Prompt、优化器和训练日程不变。

新模块 `models/original_style_tcp.py` 使用冻结 BiomedCLIP 对每条 description 独立编码后的最终投影特征 `[C,50,D_proj]`，按类别计算 `w_c = normalize(mean_i(t_ci))`，注册为不可训练 buffer。它复用 projected description cache，不构建中间层 description bank。共享 TKE 为 `Linear(D_proj,D_proj//4) → QuickGELU → Linear(D_proj//4,4*hidden_dim)`，直接 reshape 为 `[C,4,hidden_dim]`；默认维度为 `512→128→3072→[C,4,768]`。

block 0–7 沿用 CoOp + Text Deep Prompt，block 8 输入处把 CLS 后的位置 1–4 替换为该类别的 TKE tokens。block 9–11 不调用 prompt replacement，所有 hidden states 自然传播。Original-style 不含 5×10 grouping、LayerBasis、XProto 残差组合、B+Delta、跨类别 centering、token norm matching、layer gate 或多层 TCP 重注入。Mean-50 prototype 的归一化仍保留。

backbone 与 description bank 均冻结，仅训练原 CoOp context、Visual Deep Prompt、注入前必要的 Text Deep Prompt 和共享 TKE。TKE 含 bias 共 461,952 参数；默认 8 层 Text Deep Prompt 共 24,576 参数；新 TCP 参数包合计 486,528，相比现有 MultiText 包 531,589 减少 45,061。CoOp/视觉/Confusion 参数量不变。`TCP.ENABLED=False` 时新模式使用完整普通 Text Deep Prompt 路径并冻结 TKE。

Original-style 仅改变 TCP 结构，沿用现有分类目标；启用 Confusion 时保留原有 Confusion loss 和路由，不增加额外知识一致性损失。新旧 TCP 参数包使用不同 metadata 校验，禁止交叉加载；现有 MultiText checkpoint 字段不变。

局部验证：`python -m pytest tests/test_original_style_tcp.py tests/test_multitext_tcp.py tests/test_coop_vpt_biomedclip.py tests/test_confusion_aware.py tests/test_expert_moe.py tests/test_text_vpt.py tests/test_dual_best_checkpoints.py -q`。测试使用小型 BERT，无需训练或下载 backbone。
