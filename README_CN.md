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

默认插入位置为 layer 7（0起始编号）。Text Encoder 的 block 0–6 使用正常 CoOp/Text Deep Prompt。进入 block 7 前，CLS 后的 4 个 prompt slots 一次性替换为对应类别的 TKE tokens；block 8–11 直接使用上一层 hidden states，不再重新生成或覆盖 TCP tokens。设 `FUSION_WEIGHT=0.5` 时，在block7前与该层普通Text Deep Prompt各0.5融合。历史层8checkpoint训练或测试时须显式指定 `INSERT_LAYER=8`；历史层8融合入口 `run_fusion_experiments.py` 已显式保持层8。

当前实现不包含 5×10 grouping、LayerBasis、XProto/B+Delta、跨类别 centering、norm matching、layer gate 或多层 TCP 重注入。description bank 和 class prototype 均为 frozen buffer；BiomedCLIP backbone 也保持冻结。训练参数只有 CoOp context、Visual Deep Prompt、注入前 Text Deep Prompt 和共享 TKE。默认层7直接替换TCP prompt bundle共483,456个可训练参数；层7融合为486,528。

### 第 8 层 0.5 融合对照

`TRAINER.TCP.FUSION_WEIGHT` 默认 1.0，保持原先在 block 8 完全替换 4 个 prompt slots 的行为。设为 0.5 时，block 8 额外学习一组普通 Text Deep Prompt，输入该 block 前将每个槽位设为 `0.5 × 原 Text Deep Prompt + 0.5 × Mean-50 Class Text Token`；block 0–7 仍使用普通 Text Deep Prompt，block 9–11 不再覆盖这 4 个槽位。融合参数与 TKE、CoOp context 和 Visual Deep Prompt 一起从头训练；此次对照仍使用 `ALPHA=0`，不启用图像引导。`scripts/run_fusion_experiments.py` 运行三数据集×4/8/16/32-shot×seed1/2/3共36组，输出至 `output/class_text_token_fusion_0p5_seed123`，与上文完整替换版的独立测试结果比较。

`TRAINER.TCP.ENABLED=False` 时保留普通 Text Deep Prompt，冻结 TKE 参数并跳过 TCP replacement。TCP 没有实现模式选择项，配置只包含：

```yaml
TRAINER:
  TCP:
    ENABLED: True
    DESCRIPTION_CACHE: ""
    INSERT_LAYER: 7
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

后半段插入位置实验入口为 `scripts/run_fusion_layers.py`：BERT block 采用0起始编号，后半段为6–11，补跑6/7/9/10/11共180组，层8复用已完成的36组融合结果。输出目录 `output/class_text_token_fusion_0p5_layers6_11_seed123` 按 `layer_<L>/<dataset>/shots_<K>/seed<S>` 保存，各层沿用0.5融合、100 epoch与验证accuracy选模。测试入口增加 `--insert-layer`，与训练配置一致；汇总表含层8历史对照。不同插入位置会改变普通Text Deep Prompt的层数和参数量，其余设置保持原配置。

## Checkpoint 与验证

训练 checkpoint 保存 `prompt_parameters` bundle、optimizer、scheduler、AMP scaler 及 TCP 结构元数据。恢复时严格校验 TKE 维度、层数、插入层、token 数和 description bank 元数据；当前代码不提供旧 TCP、Confusion 或 MoE 实现的加载入口。

运行测试：

```bash
python -m pytest tests -q
```

测试使用小型 BERT 和 ViT，不需要下载真实 BiomedCLIP 权重；真实权重集成测试需要显式设置 `RUN_BIOMEDCLIP_INTEGRATION=1`。

## Text-Guided Visual Semantic Distillation

最终固定参数为 `WEIGHT=0.1`、`TEMPERATURE=0.5`，总损失 `L=CE+0.1 L_sem`。`TRAINER.SEMANTIC_DISTILL.ENABLED` 默认 False，保留 Original CE baseline；固定实验入口显式启用蒸馏。模型结构、文本特征和测试分类前向保持原样，不向视觉 Transformer 插入文本 token。

同次前向得到归一化图像 `v:[B,512]`、类别文本 `t:[C,512]`。训练计算 `R=detach(t) @ detach(t).T:[C,C]`，选 GT 行 `R[y]:[B,C]`，teacher 为 `softmax(R[y]/0.5).detach()`；student 为 `v @ detach(t).T:[B,C]`（不乘 logit_scale）。`L_sem=KL(teacher || softmax(student/0.5))`，使用 batchmean，无温度平方因子。semantic loss 只更新 Visual Deep Prompt；CE 继续更新 Visual/Text Deep Prompt、CoOp 和共享 TKE。冻结 BiomedCLIP backbone 无参数梯度。

默认 `GRAD_NORM_INTERVAL=0`，只在首个训练batch通过 autograd.grad 记录 semantic_grad_norm，并单独 loss_sem.backward 验证所有非视觉提示参数无梯度，清空后正常联合更新；后续batch不为监控额外反传。设为1恢复历史逐batch监控，设为N>1每N步采样（从此次训练/恢复启动首批计数）。loss_ce、loss_sem、total_loss正常记录，semantic_grad_norm仅在实际采样步直接写入TensorBoard，不把旧值重复记成新值；TensorBoard使用epoch×num_batches+batch_idx的全局步数，与loss曲线及续训对齐。semantic_gradient_audit.json仍保存六种张量形状及首次梯度隔离结果。

测试只执行标准归一化 cosine logits，不计算文本关系、teacher、student 或 KL，不需要 GT。checkpoint 保存蒸馏配置，续训仅校验训练核心开关/权重/温度，监控频率不参与兼容性限制；旧checkpoint没有GRAD_NORM_INTERVAL也可恢复。测试仍加载正常prompt bundle。evaluator从实际加载的checkpoint["semantic_distill"]记录semantic_enabled、semantic_weight、semantic_temperature，来源标记checkpoint；旧checkpoint缺少元数据时写null并标记unavailable，绝不以当前evaluation config的默认值冒充训练参数。此修复不改变分类预测。

`scripts/run_semantic_distill.py` 是唯一固定实验队列：DermaMNIST/Kvasir/CHMNIST × 4/8/16/32-shot × seed1/2/3，共36组、12个setting，layer7、FUSION_WEIGHT=1.0、λ=0.1、τ=0.5。使用 GPU0/1/2/6/7，每卡两任务；不使用GPU4。运行命令：

```bash
conda activate /mnt/nas1/disk09/yuejianwu/.conda/envs/biocoop
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. python -m scripts.run_semantic_distill
```

输出 `output/text_guided_visual_semantic_distill_lambda0p1_tau0p5_seed123`，按 validation accuracy 选 checkpoint 后独立 test。Original 对照复用 `output/class_text_token_replacement_layer7_seed123/class_text_token` 的同配置结果。汇总均值±样本std（ddof=1）、Δ、12-setting等权平均、win/tie/loss、最大下降及损失/梯度CSV/PNG曲线。已有完成结果跳过，未完成训练沿用现有checkpoint续训。

`scripts/validate_semantic_distill.py` 保留真实三数据集一步训练、梯度隔离、关闭开关输出精确比较及checkpoint保存/加载验证；小模型回归测试验证固定λ/τ损失公式及semantic/CE梯度。网格搜索、pilot接续和GPU交接脚本已移除。

历史网格324组结果保留在 `output/text_guided_visual_semantic_distill_grid_seed123`。本次选择固定λ=0.1、τ=0.5，12-setting等权test为74.5975%，比Original74.1650%提升0.4325pp，8升/0平/4降，最大单setting下降0.8313pp。三个数据集各自四个shot的等权平均均提升：DermaMNIST+0.5528pp、Kvasir+0.3125pp、CHMNIST+0.4322pp。该选择以三个数据集平均均不下降为依据，仍有单setting下降。选定36组历史checkpoint和曲线位于 `output/text_guided_visual_semantic_distill_grid_seed123/lambda_0.1_tau_0.5`，完整历史统计见all_results_report.md、all_results_detailed.csv、all_settings_summary.csv。

## Class-conditioned Visual Prompt（CVP）

CVP 是独立的视觉类别条件实验，`TRAINER.CVP.ENABLED` 默认 False，与 Semantic Distill 同时开启会报错。现有普通视觉 forward、Original TKE 和 Semantic KL 分支保持不变。新增 `ClassConditionedVisualPrompt` 复用文本编码器的冻结 Mean-50 `class_prior b_c∈R^512`，输入 detach，计算 `P_c^V=g_v(b_c)∈R^(4×768)`；Image MLP 为 Linear(512,128)→QuickGELU→Linear(128,3072)，共有461,952个新参数，不新增 Text MLP。

`TimmViTVisualPromptEncoder.forward_with_class_prompt(image, class_prompt_tokens, insert_layer=7, fusion_weight=0.5)` 接收 `[B,3,H,W]` 与 `[B,4,768]`。代码block0–6继续原 Visual Deep Prompt；block7（论文/图中第8个Block）执行前计算 `Q_c^V=0.5·P_7^V+0.5·g_v(b_c)`，用融合结果替换4个prompt slots。其中 `P_7^V` 是原Visual Deep Prompt在layer7的可训练参数，不是layer6输出的隐藏状态；`b_c` 是与文本MLP共享的冻结Mean-50类别prior。block8–11不再次覆盖，最后删除slots并沿用原pooling/projection。普通Visual Deep Prompt的layer0–7均接收梯度，layer8–11的参数切片梯度为零。文本端同样为 `Q_c^T=0.5·P_7^T+0.5·g_t(b_c)`；两端MLP各自独立。

`CVPCustomCLIP` 训练、验证和测试均遍历所有候选类别，输出 `v:[B,C,512]`；文本端仍用现有Original TKE得到 `t:[C,512]`。`z_ic=exp(s)·normalize(v_ic)^T normalize(t_c)`，以 `einsum("bcd,cd->bc")` 得到 `[B,C]`，GT仅用于 `cross_entropy(z,y)`，不添加任何额外loss。CoOp context、Visual/Text Deep Prompt、Original TKE和Image MLP共同训练；backbone、description bank、class prior和logit_scale冻结。

默认配置为 `CVP.ENABLED=False, INSERT_LAYER=7, NUM_TOKENS=4, BOTTLENECK_DIM=128, FUSION_WEIGHT=0.5`。文本TCP与视觉CVP的配置、模块构造和CVP运行/评估入口默认fusion均为0.5；显式传入1.0仍支持历史直接替换。当前独立训练/测试入口显式设置CVP/TCP开启、Semantic关闭、两端layer7、两端 `FUSION_WEIGHT=0.5`。checkpoint仅开启时含 `cvp.*` 参数；独立 `cvp_enabled/cvp_insert_layer/cvp_num_tokens/cvp_bottleneck_dim/cvp_fusion_weight` 元数据用于拒绝CVP/non-CVP及视觉融合权重错配。历史无CVP元数据的checkpoint按non-CVP加载；历史CVP checkpoint缺少 `cvp_fusion_weight` 时按1.0处理，不能加载到当前0.5融合实验。

运行 `PYTHONPATH=. HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=1 /mnt/nas1/disk09/yuejianwu/.conda/envs/biocoop/bin/python -m scripts.validate_cvp`，依次检查三数据集真实小batch梯度、特征形状、labels独立性、checkpoint往返及正式batch32单步显存和耗时。验证完成后运行 `PYTHONPATH=. /mnt/nas1/disk09/yuejianwu/.conda/envs/biocoop/bin/python -m scripts.run_cvp`。

正式队列GPU1/2/6/7每卡最多两个独立任务（8个槽位），默认 `--jobs-per-gpu=1`，可显式设为2，DermaMNIST/Kvasir/CHMNIST × shots4/8/16/32 × seeds1/2/3共36组，沿用100epoch、batch32与原优化器/数据设置。输出 `output/class_conditioned_visual_prompt_text0p5_visual0p5_seed123`；按validation accuracy选模后独立test，对照 `output/class_text_token_fusion_0p5_layers6_11_seed123/layer_7`。逐组CSV保留实际checkpoint的CVP/fusion设置；队列对同卡任务按真实batch32峰值显存加2GiB余量逐个预留，并扣除其他用户的显存占用，不足时每15min检查一次，不干扰其他任务。完成后读取实际best checkpoint核对epoch和配置。汇总均值±样本std、12-setting等权平均、Δ和win/tie/loss。显存不足时不自动改变batch或精度，停止正式启动并报告实测情况。

历史视觉直接替换版本在2026-09-29通过真实BiomedCLIP验证：三数据集均完成batch2前向/反向、全部训练分支梯度检查、labels独立性、checkpoint往返和正式batch32单步；总训练参数988416。DermaMNIST/Kvasir/CHMNIST的batch32峰值分别17836.01/20251.29/20250.50MiB，单步分别0.624/0.715/0.708s。完整记录在 `output/class_conditioned_visual_prompt_fusion0p5_validation/validation_summary.json`。该版正式实验已按用户指令终止，结果目录 `output/class_conditioned_visual_prompt_fusion0p5_seed123` 已删除；上述验证数值仅对应历史直接替换版本。当前双端融合验证记录使用 `output/class_conditioned_visual_prompt_text0p5_visual0p5_validation/validation_summary.json`；正式管理日志为新实验输出目录的 `_manager/manager.log`，最终结果以 `comparison_report.md` 与 `_manager/final_validation.json` 为准。

支持 `--adopt-manager <PID>` 无重启交接：暂停指定旧CVP manager派发，读取其直接训练/测试子进程，通过pidfd等待各进程自然结束；原任务在原GPU槽位接续测试，待办队列排除活动case和已完成结果。所有接手进程结束后清理旧manager，新队列最后自动核对36份实际checkpoint。交接记录保存 `_manager/handoff.json`。

历史直接替换实验的单槽和双槽manager均已终止。当前双端0.5融合使用新队列、新结果目录，命令 `python -m scripts.run_cvp --jobs-per-gpu 2`，固定GPU1/2/6/7共8槽。

2026-09-29双端0.5融合版本：项目回归69 passed/1 skipped，真实BiomedCLIP三数据集检查全部通过（参数总量988416）。batch32峰值显存/单步耗时：DermaMNIST 17836.15MiB/0.647s；Kvasir 20251.43MiB/0.721s；CHMNIST 20250.64MiB/0.718s。新队列tmux `cvp-text0p5-visual0p5-two`，GPU1/2/6/7每卡双槽，日志 `output/class_conditioned_visual_prompt_text0p5_visual0p5_seed123/_manager/manager.log`。

当前调度（用户最新设置）：GPU1/2/6/7每卡1程序、共4槽，运行 `python -m scripts.run_cvp --jobs-per-gpu 1`。tmux `cvp-text0p5-visual0p5-one`，日志 `_manager/single_jobs_manager.log`，切换记录 `_manager/single_slot_transition.json`；两端layer7/fusion0.5及其余实验参数保持原设置。此前双槽manager已暂停派发，保留的4个训练由新单槽队列接续，其余任务排队。

CVP双端融合36组seed1/2/3实验全部完成，真实checkpoint选模和配置核对通过。12-setting等权平均CVP75.43%、文本fusion0.5 layer7 baseline75.61%，差值−0.18pp，win/tie/loss=7/0/5。五个下降setting的seed4/5补充共10组完成，其补跑代码按用户要求撤回；结果和用户指定五选三报告保留于 `output/class_conditioned_visual_prompt_text0p5_visual0p5_seed45`，原固定三seed结果保留于 `output/class_conditioned_visual_prompt_text0p5_visual0p5_seed123`。五选三与固定三seed统计口径分开记录。
