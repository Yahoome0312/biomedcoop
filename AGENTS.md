# AGENTS.md

版本：v0.1  
适用范围：本文件所在的仓库根目录及其全部子目录。若子目录中出现更具体的 `AGENTS.md`，仅该子目录优先采用更具体的规则。

## 1. 用户与沟通背景

- 用户是生物医学工程专业博士一年级学生，研究方向是医学图像与人工智能。
- 用户主要使用 Python，代码能力处于一般水平。解释实现时不得假定用户熟悉复杂工程惯例；首次出现关键缩写、损失项或训练技巧时，用一句话说明其作用。
- 默认使用中文沟通；代码标识符、命令、指标名和论文中的标准术语保留英文。
- 当前工作的学术目标是完成用户的第一篇论文。任何实验设计和代码修改都要优先考虑可复现性、公平对比、消融完整性和论文可解释性。

## 2. 当前最高优先级项目

当前最高优先级是本仓库 `BiomedCoOp`：研究医学图像分类中的 confusion-aware few-shot learning，并以可发表的方法和实验结果为目标。

根据当前 `README_CN.md`、主配置和测试，现阶段主线为：

- 数据集主线：DermaMNIST。
- 基础模型：BiomedCLIP。
- 基础方法：CoOp + Visual/Text Deep Prompt。
- 已实现模块一：MT-TCP，对类别描述进行聚合并注入文本深层提示。
- 已实现模块二：Full Confusion / Confusion Aware，使用 support 样本生成的 Soft Bank、类别对语义描述、Confusion Adapter 和 confusion margin loss。
- 当前目标：验证两个模块是否在严格匹配的 few-shot 协议下稳定提升，并争取达到或超过可比工作的 SOTA。

### 2.1 已实现的主线方法模块

以下内容来自当前代码调用关系，不根据输出目录名称推断。主入口是 `train.py`，主训练器是 `trainers/CoOp/coop_vpt_biomedclip.py` 中的 `CoOpVPT_BiomedCLIP`。

#### A. 冻结的 BiomedCLIP 主干

- `models/biomedclip_loader.py` 从 `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` 加载预训练模型。
- 加载时校验视觉端必须是 OpenCLIP `TimmModel` 包装的 timm `VisionTransformer`，文本端沿用 BiomedCLIP 的 BERT 文本塔。
- 联合训练开始前冻结 BiomedCLIP 主干；优化器只接收显式启用的 prompt 和 adapter 参数。
- 这部分提供固定的图像—文本表征空间，不属于当前方法的可训练 backbone 微调。

#### B. CoOp 连续文本上下文

- `trainers/CoOp/coop_biomedclip.py` 中的 `PromptLearner` 为每个类别构造连续 prompt embedding。
- 当前主配置使用 4 个 CoOp context tokens，初始化文本为 `a photo of a`，类别 token 位于末尾。
- CoOp 输出作为 BERT 文本编码器的基础输入，与 Visual/Text Deep Prompt、MT-TCP 和 Confusion Aware 共同训练。

#### C. Visual Deep Prompt

- `models/vpt.py` 中的 `VisualPromptParameters` 为视觉 Transformer 每层保存独立的可训练 prompt；`TimmViTVisualPromptEncoder` 负责插入和逐层替换。
- 第 0 层在视觉前缀 token 与 patch tokens 之间插入 prompt；后续层替换上一层 prompt，序列中的真实 patch tokens 保持不变。
- 编码结束后移除 prompt，再按原 BiomedCLIP 路径完成 pooling 和 projection。
- `return_tokens=True` 时同时返回全局图像特征和 patch tokens，后者供 Full Confusion 的局部分支使用。
- 当前主配置为 deep 模式、每层 4 个视觉 prompt tokens、dropout 0。

#### D. Text Deep Prompt

- `models/text_vpt.py` 为 BERT 每一层保存独立的可训练文本 prompt，并在每层编码前替换 prompt slots。
- prompt 插在 CLS token 后；代码会预留位置并检查是否会截断有效文本。若有效 token 会被截断，直接报错。
- BiomedCLIP BERT、pooler 和 projection 保持冻结，只有 `TextPromptParameters` 参与训练。
- 在主线中，Text Deep Prompt 由 `MultiTextTCPBertTextEncoder` 内部统一承载；关闭 TCP 时仍保留并训练这套 Text Deep Prompt，而不是切换到另一套文本编码器。

#### E. MT-TCP

- `models/multitext_tcp.py` 实现当前保留的 `LayerBasis + XProto` 路径；代码将方法身份固定为 `grouped10_layer_residual` 与 `late_centered_norm_residual`。
- 每个类别必须有 50 条 `BIOMEDCOOP_TEMPLATES` 描述。代码构建两套冻结表示：BiomedCLIP projected text space 中的描述 Bank，以及 BERT 第 8 个 block 之前的 layer-description Bank。
- 50 条 layer descriptions 按固定顺序分成 5 组、每组 10 条并求均值；前 4 组加上全组均值得到 4 个 layer-aligned class tokens。
- 类别描述均值经过 `down_projection -> QuickGELU -> up_projection` 生成可学习修正，并用可学习 residual 系数加入 4 个 class tokens。`up_projection` 从零初始化，因此初始修正为零。
- BERT 第 0–7 层只使用共享 Text Deep Prompt；从 zero-based 第 8 层开始，将 class tokens 做跨类别中心化和 norm matching，再通过逐层 sigmoid gate 残差注入 4 个 Text Deep Prompt tokens。
- 当前固定配置为 4 个文本 tokens、bottleneck 128、insert layer 8、gate 初值 0.05。
- TCP-off 将 MT-TCP residual scale 精确设为 0，并冻结 TCP 投影和 gate；Text Deep Prompt 继续训练。测试验证 TCP-off 输出能精确退化为 Text Deep Prompt baseline。
- 描述缓存会校验类别顺序、描述数量、描述 fingerprint、模型身份、tensor shape、归一化和 Bank fingerprint；不匹配时直接报错。

#### F. Support-only Soft Confusion Bank

- `scripts/coopvpt/build_confusion_prior.py` 使用冻结的原始 BiomedCLIP 和模板 `a photo of a {}.` 编码 K-shot 训练 support 样本，不使用 Visual/Text Prompt、TCP 或 Confusion Adapter。
- 脚本先检查 support 与 validation/test 图像路径无重叠，再保存每个 support 样本的类别概率、预测、置信度和样本身份。
- `models/confusion_aware.py` 中的 `compute_soft_confusion_prior` 按真实类别对所有 support 概率向量求均值，将对角线清零，但不重新归一化非对角元素。
- hard confusion counts 只作为诊断信息保存，不参与 Soft Bank 计算或训练。
- Bank 按 dataset、shots 和 seed 分目录保存，并记录 class order、support fingerprint、预处理 fingerprint、模型身份和 Bank fingerprint。
- Soft Bank 是固定的类别条件先验，不检索 support 图像、不进入 autograd，也不使用 validation/test 图像构建。

#### G. 有向类别对语义 Bank

- `models/confusion_aware.py` 从 `confuse_pair/<dataset>.txt` 读取 LLM 生成的有向类别对描述，例如 `class A -> class B` 与 `class B -> class A` 是两个独立条目。
- 文件必须覆盖当前数据集所有类别，以及每个类别指向其他所有类别的有向 pair；允许不同 pair 拥有不同数量的描述，但列表不能为空。
- 每条描述经冻结 BiomedCLIP 编码；同一有向 pair 的多条描述求均值并归一化，形成 `[num_classes, num_classes, feature_dim]` 的语义 Bank。
- 类别数和每个 pair 的描述数动态读取，不写死为 DermaMNIST；代码同时保存描述文本和编码特征 fingerprint。

#### H. Full Confusion / Confusion Aware Adapter

- `ConfusionAwareAdapter` 先从基础 logits 中为每个 anchor 选择一个困难负类。候选分数为当前样本预测概率乘以 `1 + PRIOR_ALPHA * Soft Bank 对应值`，并排除 anchor 自身；用于选择的概率已经 detach。
- 训练阶段 anchor 使用真实标签；验证和测试没有标签输入时，anchor 使用基础 logits 的 top-1 预测。代码显式检查训练 anchor 必须等于真实标签。
- Semantic 分支对选中有向类别对的 LLM feature 做可学习投影。
- Global 分支用 semantic feature 对当前图像的全局特征逐维门控并归一化。
- Local 分支将 semantic feature 投影为 query，对当前图像 patch tokens 做 attention，再把加权局部视觉特征投影到图文共同空间。
- Global/Local gate 为每张图像产生两个 softmax 权重，融合全局和局部 confusion evidence；随后与 semantic feature 共同经过 final fusion。
- 融合结果作为强度为 `GAMMA` 的归一化残差加入原始全局图像特征，再与同一套文本特征计算最终 logits。该实现只有一个最终分类器和一份最终 logits，不是双分类器 ensemble。
- 当前主配置为 `PRIOR_ALPHA=1.0`、`GAMMA=0.2`。

#### I. 联合目标与消融路径

- Full Confusion 开启时，训练目标为 `cross_entropy(final_logits, label) + LAMBDA_CONF * confusion_margin_loss`。
- `confusion_margin_loss` 使用 `softplus(logit_competitor - logit_true)`，直接约束真实类别分数高于当前选择的困难负类；当前 `LAMBDA_CONF=1.0`。
- Confusion Aware 关闭时，跳过 Soft Bank、类别对文件、Confusion Adapter、patch-token 分支、margin loss 和 confusion 分析，只用基础 logits 的交叉熵。
- TCP 与 Confusion Aware 是独立开关，代码支持 TCP-on/off 与 Confusion-on/off 的组合；不同状态的 checkpoint 不能交叉加载。

### 2.2 已实现的训练、评估与复现功能

- 联合训练：CoOp、Visual Deep Prompt、Text Deep Prompt、已启用的 MT-TCP 和已启用的 Confusion Adapter 从 epoch 1 开始使用同一个 AdamW 优化器和同一学习率联合训练；没有分阶段切换分类器。
- 参数审计：启动时检查实际可训练参数集合与预期模块完全一致，并记录各分支 trainable parameter count 和 frozen parameter count。
- 梯度审计：第一次反向传播后检查每个启用分支的梯度 norm 必须大于 0，同时检查冻结 backbone 不得收到梯度。
- 初始化清单：在输出目录写入 `initialization_manifest.json`，记录 protocol、seed、shots、模块开关、核心初始化 fingerprint、参数列表和参数量。
- Confusion 分析：每个训练 epoch 保存类别 pair 选择计数和 Global/Local 平均权重；验证和测试按样本保存 anchor、competitor、prior、selection score 和融合权重的 gzip JSON。
- 指标：当前 evaluator 计算 accuracy、error rate、macro-F1、balanced accuracy 和 multiclass AUC，并可输出逐类结果与 confusion matrix。
- 双指标 checkpoint：当前主配置以 accuracy 作为默认 `BEST_METRIC`，同时分别跟踪并保存 accuracy-best 与 balanced-accuracy-best checkpoint 和对应完整验证指标记录。
- checkpoint 完整性：保存 prompt bundle、optimizer、scheduler、AMP scaler、protocol、TCP/Confusion 开关及 Bank fingerprints；加载或恢复时对 protocol、模块状态、Bank 和 TCP 方法身份做严格一致性校验。
- 运行记录：支持 fp16、fp32 和 AMP 路径，记录训练 loss 分量、accuracy、learning rate、confusion 融合权重和 peak CUDA memory。

### 2.3 论文中应如何界定这些内容

- 可作为主要方法模块讨论：MT-TCP 与 Full Confusion / Confusion Aware。
- 应作为共同基础或 baseline 组件说明：冻结 BiomedCLIP、CoOp、Visual Deep Prompt、Text Deep Prompt。
- 应作为 Full Confusion 的组成部分说明：Support-only Soft Confusion Bank、有向类别对语义 Bank、困难 pair 选择、Semantic/Global/Local 融合和 confusion margin loss。
- 应放在实验设置或复现性部分说明：参数/梯度审计、fingerprint、初始化清单、checkpoint 一致性、双指标保存和 confusion 分析日志。
- 除非消融结果证明独立贡献，不得把一个工程校验、缓存机制或日志功能包装成新的论文创新点。

## 3. 仓库目录与职责

| 路径 | 职责 | 操作规则 |
|---|---|---|
| `train.py` | 训练与评估入口 | 修改命令行参数或默认配置前必须说明影响 |
| `trainers/` | 训练器、损失组合和训练流程 | 主线实现重点检查 `trainers/CoOp/coop_vpt_biomedclip.py` |
| `models/` | 模型模块 | MT-TCP、Confusion Aware、Text VPT 等核心实现放在这里 |
| `configs/datasets/` | 数据集配置 | 不得未经确认改变数据集、路径语义或划分规则 |
| `configs/trainers/` | 优化器、学习率、epoch、prompt 和模块开关 | 任何参数改动均按第 6 节执行 |
| `datasets/`、`data/` | 数据集定义、加载和采样 | 不得让 K-shot 采样影响验证集或测试集 |
| `confuse_pair/` | 各数据集的有向类别对描述 | 修改描述内容或类别映射视为实验变量，必须记录版本和理由 |
| `scripts/coopvpt/` | Soft Bank 构建、实验运行、聚合和诊断脚本 | 新脚本必须对应已确认的实验需求，不得顺手扩展实验矩阵 |
| `tests/` | 单元测试和回归测试 | 功能变更优先运行直接相关测试，再决定是否扩大测试范围 |
| `analysis/` | 诊断和结果分析 | 分析代码不得修改训练产物或筛选规则 |
| `output/` | checkpoint、缓存、日志和实验结果 | 视为实验记录；不得手工改结果、覆盖既有实验或未经同意删除 |
| `README_CN.md` | 当前中文方法与运行协议 | 行为或协议变化后，在同一已确认任务中同步更新 |
| `Dassl.pytorch/`、`open_clip/`、`clip/` | 框架或第三方基础代码 | 能在本项目代码中完成时不修改这些目录；确需修改时单独说明原因 |

读取信息时，优先级依次为：用户本轮明确说明、实际代码与配置、测试、`README_CN.md`、历史输出目录名称。目录名称不能单独证明某个实验结论。

## 4. 默认协作流程

每个代码修改或实验任务必须按以下顺序进行：

1. 先确认目标：用自己的话复述要解决的问题、期望产物和成功指标。
2. 先做只读检查：读取相关代码、配置、测试、已有输出和 `git diff`，找出真实入口与现状。
3. 先给计划：列出拟修改文件、方法、参数变化、验证方式和预计运行成本。
4. 等待确认：用户确认计划后，才修改代码、配置或实验协议。若用户在同一条消息中已经明确要求按具体方案直接实现，可视为该方案已确认，但仍要先展示执行计划。
5. 最小范围执行：只实现已确认内容，不顺手重构、不增加新功能、不扩大实验矩阵。
6. 验证：先运行与改动直接相关的测试；涉及训练逻辑时，再做能够发现梯度、shape、checkpoint 或配置错误的最小验证。
7. 汇报：先讲方法和指标，再列文件、测试、风险及尚未完成事项。

只读搜索、读取文件、查看 Git 状态和分析已有结果不需要另行确认。

## 5. 已授权操作

下列操作在服务于用户已提出且计划中已列明的任务时可以执行，不需要再次单独请求许可：

- 安装所需依赖。
- 联网检索论文、文档、代码或公开基准。
- 运行长时间训练、评估或参数搜索。
- 执行 Git 操作。
- 访问完成任务所需的外部服务。

这些授权不允许扩大任务范围。执行前仍须在计划中说明目的；长任务须说明命令、数据集、shots、seeds、epoch、输出目录和预期耗时，执行后须报告实际状态。任何上述操作如果会删除文件，仍受第 6 节的删除规则约束。

## 6. 必须遵守的修改边界

### 6.1 删除

- 删除任何文件或目录前，必须取得用户明确同意。
- 请求同意时必须列出准确目标、删除原因、是否可恢复以及对实验复现的影响。
- 不得用覆盖空文件、Git 清理、移动到不可发现位置等方式绕过删除确认。

### 6.2 新功能与兜底代码

- 不得加入用户未要求、计划未确认的新功能。
- 不得擅自加入 fallback、静默重试、异常吞掉、默认替代路径、缺失数据自动补齐或“失败后继续”的逻辑。
- 遇到输入、checkpoint、缓存、类别描述或配置不匹配时，默认做法是明确报错并说明根因；只有用户确认后才能加入特定恢复策略。
- 不得以“提高鲁棒性”或“顺便兼容”为理由扩展已确认范围。

### 6.3 参数与实验协议

修改任何训练、模型、数据或评估参数前，必须在计划中列出：

- 参数所在文件或命令。
- 旧值与拟议新值。
- 修改假设和预期影响。
- 哪些对照组必须保持相同。
- 是否会导致已有 checkpoint 或结果不可比较。

获得确认后才能修改。参数包括但不限于 learning rate、optimizer、weight decay、epoch、batch size、workers、seed、shots、prompt 长度、模块开关、loss 权重、gate、margin、模型选择指标和数据划分。

当前 `README_CN.md` 记录的主协议包括：训练集使用 K-shot，验证集和测试集使用官方完整划分；实验 seed 为 1、2、3；batch size 为 32；workers 为 8。除非用户确认新实验协议，否则不得改变这些约束。

### 6.4 禁止隐晦修改

- 可能会改变行为的一两行修改必须在计划和结果汇报中单独解释：改了什么、为什么能解决问题、如何验证。
- 不得只说“修复细节”“小优化”或“清理代码”。
- 不得通过调整常数、条件分支、默认值、随机种子、选择指标或输出解析方式，制造未声明的指标提升。
- 如果无法给出修改与问题之间的因果解释，先停下并与用户讨论，不得提交该修改。

## 7. 实验与论文证据规则

开始实验前必须明确：研究假设、对照组、唯一实验变量、数据划分、shots、seeds、训练轮数、配置文件、checkpoint 选择规则、指标和输出目录。

实验执行遵守以下规则：

- 不同方法、模块开关、参数组、shots 和 seeds 使用不同输出目录，不覆盖既有结果。
- 不在测试集上选择超参数或 checkpoint；模型选择依据必须预先写明并来自验证集。
- 对照实验只改变计划中声明的变量，其余配置保持一致。
- TCP 和 Confusion Aware 状态不同的 checkpoint 不交叉恢复或加载。
- 结果至少报告各 seed 的原始值及 mean ± std；只有一个 seed 时明确标注为单 seed，不写成稳定提升。
- 当前 evaluator 计算 accuracy、error rate、macro-F1、balanced accuracy 和 AUC；主配置分别保存 accuracy-best 与 balanced-accuracy-best checkpoint。汇报时给出相应 checkpoint 的完整指标，AUC 不可计算时明确标注 `NaN`，不得估算。
- 类别不平衡任务不能只用 accuracy 下结论；至少同时检查 balanced accuracy 和逐类结果或 confusion matrix。
- 所有提升都注明相对哪个 baseline、绝对提升多少个百分点，以及是否使用相同协议。
- 不挑选有利 seed，不隐藏失败运行，不把搜索集上的最佳值当作最终复现实验。

## 8. 测试与质量要求

- 修改 `models/confusion_aware.py` 时，至少运行 `tests/test_confusion_aware.py`。
- 修改 `models/multitext_tcp.py` 时，至少运行 `tests/test_multitext_tcp.py`。
- 修改 CoOpVPT 集成、checkpoint 或 Text VPT 时，分别检查 `tests/test_coop_vpt_biomedclip.py`、`tests/test_dual_best_checkpoints.py`、`tests/test_text_vpt.py` 中直接相关测试。
- 测试失败时报告失败命令、首个根因和影响范围；不得加入兜底逻辑使测试表面通过。
- 无法运行测试时，明确写出缺失依赖、硬件或数据，并提供已完成的静态检查结果。
- 不以格式化或大范围重构混入功能修改；必要重构需作为独立计划项。

## 9. 输出与汇报格式

### 9.1 讨论或方案

按以下顺序输出：

1. 结论或建议。
2. 方法原理及它为什么可能有效。
3. 支持证据，包括代码位置、已有实验或论文依据。
4. 风险、混杂变量和仍需用户决定的事项。

明确区分“代码中已确认的事实”“根据证据作出的推断”和“尚未验证的提议”。

### 9.2 代码完成汇报

按以下顺序输出：

1. 完成了什么，以及对应的已确认目标。
2. 方法和关键实现逻辑。
3. 修改文件及每个功能性改动的理由，包含一两行改动。
4. 执行的测试、结果和未覆盖部分。
5. 是否改变参数、实验协议、输出格式或 checkpoint 兼容性。

### 9.3 实验汇报

优先使用表格，至少包含：

| Method | Shots | Seeds | Selection rule | ACC | Average ACC |Balanced ACC | Macro-F1 | AUC | Δ vs. baseline |
|---|---:|---|---|---:|---:|---:|---:|---:|

- 多 seed 指标写 `mean ± std`，Average ACC在3个seed的ACC的基础上计算。
- 先解释方法差异，再解释指标；不需要罗列输出日志。
- 给出配置文件、输出目录和结果文件路径，使结果可复查。
- 失败、中断、缺失或不可比的结果必须显式标注。

## 10. 停止并询问用户的条件

遇到以下任一情况，停止修改并向用户说明选项：

- 需要删除文件或目录。
- 需求会引入计划外功能或兜底逻辑。
- 需要修改未获确认的参数、数据划分、指标或模型选择规则。
- 现有代码、README、配置和实验结果互相矛盾，且不同解释会影响论文结论。
- 无法说明某个行为修改的直接原因或验证方式。
- 发现数据泄漏、测试集调参、对照不公平、结果不可复现或疑似指标解析错误。

## 11. 每次任务结束前自检

1. 我知道用户是谁吗？——生物医学工程专业博士一年级学生，研究医学图像与人工智能，主要使用 Python，正在准备第一篇论文。
2. 我知道当前最重要的项目吗？——本仓库的 confusion-aware few-shot 医学图像分类项目；当前主线是 DermaMNIST、BiomedCLIP、MT-TCP 和 Full Confusion / Confusion Aware。
3. 哪些事不能擅自做？——不能删除文件，不能增加计划外功能或兜底代码，不能未经确认修改参数或实验协议，不能做无理由且未说明的一两行行为修改。
4. 我是否先给出了计划，并只执行了用户确认的范围？
5. 我是否优先汇报了方法、指标、对照和证据，而不是只汇报改了哪些文件？
