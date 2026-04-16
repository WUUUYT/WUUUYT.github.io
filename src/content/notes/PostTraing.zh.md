---
title: "Post Training - RLHF 与大语言模型对齐"
date: 2026-04-16
description: "从监督微调（SFT）到基于人类反馈的强化学习（RLHF）的完整对齐流程，涵盖数据构建、训练方法、核心算法及常见风险。"
categories: ["机器学习", "大语言模型", "笔记"]
lang: zh
---

## 一、背景与问题动机

### 1.1 预训练的局限性

预训练阶段通过对海量互联网文本进行下一词预测（next-token prediction），可以使模型习得丰富的语言知识与世界常识，对应 GPT-3 等基础模型的能力水平。然而，预训练数据的分布与"有用的助手行为"之间存在根本性的鸿沟：

- **缺乏指令遵循能力**：预训练数据并非以"指令→回答"格式组织，模型不会自然地响应用户意图。
- **缺乏安全边界**：模型可能生成仇恨言论、诈骗文本、错误信息等有害内容。
- **风格不适配**：模型倾向于续写文本而非提供有用的回应。

### 1.2 目标：更紧密地控制模型输出

对齐（Alignment）研究的核心问题是：

1. 我们希望模型具备哪些行为？对应的训练数据应如何构建？
2. 如何最有效地利用这些数据训练模型？
3. 这一过程是否需要大规模计算资源？

### 1.3 总体方案概述

当前工业界主流方案（以 InstructGPT / Ouyang et al. 2022 为代表）包含三个阶段：

| 阶段 | 名称 | 核心操作 |
|------|------|----------|
| Step 1 | 监督微调（SFT） | 收集示范数据，对预训练模型做监督学习 |
| Step 2 | 奖励模型训练 | 收集偏好对比数据，训练奖励模型 |
| Step 3 | 强化学习优化 | 以奖励模型为信号，用 PPO 等算法优化策略 |

---

## 二、阶段一：监督微调（Supervised Fine-Tuning, SFT）

### 2.1 训练数据

SFT 的训练数据由"指令-回答"对（instruction-response pairs）构成。讲义重点介绍了以下三类代表性数据集。

#### 2.1.1 FLAN（Fine-tuned Language Net）

FLAN 将现有 NLP 数据集统一改写为指令格式，涵盖自然语言推理、封闭式问答、摘要生成、代码修复、对话等多个类别，共计 500+ 个任务。其特点是：

- 数据量大，覆盖面广（372 个数据集，1554 项任务）；
- 回答风格较为简短、机械（平均回答长度约 31 词）；
- 适合提升模型对标准化任务的处理能力。

#### 2.1.2 Stanford Alpaca

由斯坦福大学利用 OpenAI text-davinci-003 自动生成约 5.2 万条指令数据。其特点是：

- 构建成本极低（原始成本约 $500）；
- 数据质量参差，存在部分事实性错误；
- 回答风格清晰，适合快速原型验证；
- 平均提示长度 27.8 词，回答长度 64.6 词。

#### 2.1.3 OpenAssistant（Oasst）

由开放社区众包构建的多轮对话数据集，约 3.5 万条对话。其特点是：

- 覆盖真实用户场景，对话形式自然；
- 回答较长（平均 212.5 词），常包含参考文献；
- 但众包标注质量稳定性相对较低。

#### 2.1.4 数据集属性比较

研究（Wang et al. 2023）表明，不同数据集在以下维度上差异显著：

| 数据集 | 来源 | 平均对话轮次 | 平均回答长度 |
|--------|------|-------------|-------------|
| SuperNI | NLP 数据集 + 人工指令 | 1.0 | 38.7 词 |
| Flan V2 | NLP 数据集 + 人工指令 | 1.0 | 31.2 词 |
| Open Assistant | 人工从头撰写 | 1.6 | 212.5 词 |
| Alpaca | GPT text-davinci-003 生成 | 1.0 | 64.6 词 |
| ShareGPT | 用户与多种模型的真实对话 | 3.2 | 357.8 词 |

### 2.2 风格对模型评估的影响

研究（Dubois et al. 2023）揭示了一个重要偏差：**无论是人类评估者还是 GPT-4 评估者，均对更长的回答和包含列表的回答表现出显著偏好**，即便其信息密度并未提升。这意味着：

- 在基于偏好的评估（如 AlpacaEval）中，风格因素对排名的影响可能超过内容质量；
- RLHF 后的模型通常会习得"生成更长回答"的策略，这在某种程度上是对评估偏差的"投机性适应"；
- 针对标准基准（如 MMLU、GSM8K、BBH），风格因素的影响则相对有限，因为这些基准有客观正确答案。

### 2.3 知识提取与幻觉问题

**核心认知（Schulman 2023；Gekhman et al. 2023）**：

SFT 的本质是**从预训练权重中提取已有知识**，而非注入新知识。具体而言：

- 若对模型已知的事实进行微调，模型能正确输出，且泛化性良好；
- 若对模型**未知**的事实（即预训练阶段未充分覆盖的长尾知识）进行微调，模型会习得相应的"输出格式"，但倾向于将其泛化至错误内容，从而产生幻觉（Hallucination）；
- 在神经网络视角下，可将模型视为维护一个"知识置信度图谱"，SFT 训练的是一个以置信度为条件的简单函数。

**实践启示**：应优先对模型已知内容进行格式与风格的对齐，而非将 SFT 作为知识注入手段。通过 RL 反馈机制提供事实正确性信号，从理论上可部分缓解此问题。

### 2.4 安全微调（Safety Tuning）

#### 2.4.1 必要性

大规模部署的语言模型面临多类安全风险，包括：

- **错误信息传播**：自动生成虚假信息、深度伪造文本；
- **诈骗与垃圾信息**：辅助撰写网络钓鱼邮件、身份冒充文本；
- **有害内容生成**：仇恨言论、暴力煽动等。

#### 2.4.2 安全微调的效果

研究表明，仅需加入约 **500 条** Alpaca 风格的安全示例，即可在多个安全基准（I-MaliciousInstructions、I-CoNa、I-Controversial、Q-Harm）上显著降低模型的有害输出率。

#### 2.4.3 过度安全微调的反效果

然而，若安全数据比例过高，模型会产生"过度拒绝"（Over-refusal）问题：将无害请求误判为有害请求，例如将编程术语"kill a Python process（终止进程）"解读为暴力相关内容，并拒绝回答。

因此，安全微调的核心挑战在于**在降低有害输出与避免过度拒绝之间取得平衡**。

### 2.5 SFT 方法：梯度下降与中间训练

#### 2.5.1 基础方法

SFT 的训练方式本质上是标准监督学习：对"指令-回答"对计算交叉熵损失并执行梯度下降。

```python
for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
```

#### 2.5.2 中间训练（Midtraining / Two-Phase Training）

为将指令数据扩展到预训练规模，同时避免灾难性遗忘，业界普遍采用以下三阶段方案：

1. **稳定训练阶段**：以标准预训练数据混合（CommonCrawl、C4、代码等）为主进行训练；
2. **衰减训练阶段（Decay Stage）**：逐渐增大指令数据比例（UltraChat、ShareGPT4、Math SFT 等），同时降低预训练数据权重；
3. **指令微调阶段**：短周期内专注 SFT，完成最终对齐。

该方案允许以预训练规模扩充指令数据，同时保留预训练阶段习得的通用能力，已被 miniCPM、JetMoE 等模型公开记录，被大多数前沿模型所采用。

### 2.6 SFT 关键结论

1. SFT 最适合用于提取预训练中已有的能力，而非注入新知识；
2. 即使数据在事实上是正确的，在"尾部知识"上微调也可能导致幻觉增加；
3. 少量的风格数据（安全、指令遵循格式）即可带来显著的行为改变；
4. 在偏好评估中，风格（尤其是回答长度和格式）会对评估分数产生强烈干扰。

---

## 三、从模仿到优化：RLHF 的动机

### 3.1 SFT 与 RLHF 的本质差异

**SFT（模仿）**：

$$\hat{p}(y|x) \approx p^*(y|x)$$

拟合某一参考分布，需要来自参考策略的样本，本质是生成建模。

**RLHF（优化）**：

$$\max_p \, \mathbb{E}_p[R(y, x)]$$

最大化某个可度量的奖励函数，将语言模型视为策略而非分布模型。

### 3.2 RLHF 的成本优势

对于一个 7B 参数模型，不同数据类型的成本对比（粗略估计）如下：

| 阶段 | 计算成本 | 标注成本 |
|------|----------|----------|
| 基础预训练 | $300,000 | $0 |
| 监督微调（SFT） | $100 | $25,000 |
| 偏好数据收集 | $100 | $4,000 |
| RL 训练 | $100 | $0 |
| 评估 | $0 | $50 |

偏好反馈（比较两个回答的优劣）在标注成本上约为 SFT 示范数据的 **1/6**。

### 3.3 G-V Gap（生成-评估差距）

人类往往难以直接写出最优回答，但能够轻松判断两个回答的优劣（即"评判比创作容易"）。Zhang et al.（2023）在新闻摘要任务中发现：

- 新闻专业写手撰写的摘要与 InstructDavinci 生成的摘要，人类评估者偏好几乎各半（约 50/50）；
- 不同标注人员之间的整体一致性（Krippendorff's α）仅为 0.07，说明"偏好"本身存在高度主观性。

这一发现说明，通过收集偏好信号并进行优化，模型有可能超越人类示范本身的质量上限。

---

## 四、RLHF 数据：偏好数据的收集

### 4.1 标准收集范式

标准的偏好数据收集流程如下：

1. 从提示数据集中采样一条提示（prompt）；
2. 用当前策略模型生成多个候选回答；
3. 由标注员对候选回答进行排序（最优→最差）；
4. 将排序结果转化为偏好对，用于训练奖励模型。

### 4.2 标注指引

InstructGPT 的标注指引要求评估者综合考量以下三个维度，并优先保证真实性与无害性：

- **有帮助（Helpful）**：回答是否满足用户意图、使用清晰语言、避免冗余；
- **真实准确（Truthful）**：是否包含准确信息，不捏造事实，不传播误导性内容；
- **无害（Harmless）**：是否避免生成歧视性、威胁性、违法性或有害内容。

Google Bard（据称）采用了类似框架，并设有"Helpfulness"和"Presentation"两个独立维度各自打分，再综合判断优劣。

### 4.3 标注员选择（InstructGPT）

OpenAI 通过以下筛选标准从 Upwork 和 ScaleAI 招募约 40 名专业标注员：

1. **敏感内容识别一致性**：与研究者标注的一致性需达 75% 以上；
2. **回答排序一致性**：对研究者评级结果的认同程度；
3. **敏感示范写作质量**：7 分制演示评分需达 6 分以上；
4. **不同群体敏感内容识别能力**：自评能够识别多类文化背景下的敏感内容。

### 4.4 众包的挑战与伦理问题

#### 4.4.1 质量挑战

- 众包标注员难以有效核实专业领域的事实准确性；
- 带有明确、自信语气的回答更容易被选为优胜，即便其中存在事实错误（Hosking et al. 2024）——具体而言，标注员对"断言性强（Assertiveness++）"文本中的事实性错误，识别率比"断言性弱"文本低约 22.3 个百分点；
- 标注员使用 ChatGPT/GPT-4 生成回答的行为难以管控。

#### 4.4.2 人口构成偏差

标注员的人口构成会系统性地影响模型的价值观取向（Santurkar et al. 2023）。例如，不同宗教群体在被 OpenAI 系列模型所"代表"的程度上存在显著差异——基督教新教徒（0.694）与无神论者（0.713）之间存在可测量的得分差距。

#### 4.4.3 劳工伦理

大规模 RLHF 数据标注工作对标注员要求处理大量有害内容，存在显著的心理健康风险。Time 杂志披露，OpenAI 曾以低于 $2/小时的报酬雇用肯尼亚工人处理此类数据，引发广泛的伦理批评。

### 4.5 AI 生成的偏好数据

#### 4.5.1 GPT-4 作为评判员

研究（Dubois et al. 2023）表明，GPT-4 作为偏好评判员具有较高可靠性：

- 在系统层面，GPT-4 评判与人类评判的 Spearman 相关系数高达 **0.98**（$R^2 = 0.87$）；
- 在样本层面，GPT-4 的一致性接近人类标注员之间的一致性水平，且成本大幅低于人工。

#### 4.5.2 代表性数据集

- **UltraFeedback**：聚合来自 ChatGPT、ShareGPT、FLAN、Evol-Instruct 等多个来源的提示，由 GPT-4 对四个候选回答进行评分，被 Zephyr 7B、OLMo 等模型采用；
- **Tulu 3**：使用包含 22 个模型的模型池生成候选回答，由 GPT-4o 从有帮助性、指令遵循、真实性、诚实性四个维度评分，再二值化为偏好对；
- **Constitutional AI（Anthropic）**：完全基于 AI 自我批评与修订，无需人工标注——模型依据一套"宪法原则"生成批评并改写有害回答，再基于 AI 反馈进行 RL 训练（RLAIF）。

### 4.6 RLHF 对输出风格的影响：长度偏差

RLHF 训练后，模型输出长度会显著增加（Singhal et al. 2024）。以一个典型例子为例：

- **SFT 前**（59 词）：简洁回答成年人为何不会从床上滚落。
- **RLHF 后**（243 词）：补充了大量关于肌肉记忆、不适感、安全意识等额外细节，但核心信息并未改变。

研究（Chen et al. 2024）进一步发现，多种 RLHF 优化器（PPO、DPO、ReMax）均存在随奖励优化而回答长度增加的趋势，形成"长度-奖励 Pareto 前沿"。

---

## 五、RLHF 算法

### 5.1 统一数学框架

RLHF 的目标是在约束模型不偏离参考策略过远的前提下，最大化期望奖励：

$$\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta(y|x)} \left[ r_\phi(x, y) \right] - \beta \, \mathbb{D}_{\mathrm{KL}} \left[ \pi_\theta(y|x) \,\|\, \pi_{\mathrm{ref}}(y|x) \right]$$

其中 $\beta$ 控制 KL 惩罚强度，$\pi_{\mathrm{ref}}$ 通常为 SFT 模型。

### 5.2 奖励模型训练

奖励模型以 SFT 模型为初始化，在顶层添加线性标量输出头，训练目标为：

$$\mathcal{L}(r_\theta) = -\mathbb{E}_{(x, y_0, y_1, i) \sim D} \left[ \log \sigma\left( r_\theta(x, y_i) - r_\theta(x, y_{1-i}) \right) \right]$$

其中 $y_i$ 为被偏好的回答，$y_{1-i}$ 为较差的回答。训练完成后，通常对奖励输出进行归一化，使参考数据集的均值为 0。

### 5.3 方法一：PPO（Proximal Policy Optimization）

#### 5.3.1 算法演进背景

PPO 是在以下一系列尝试中逐步发展而来：

- **策略梯度（Policy Gradient）**：$\nabla_\theta \mathbb{E}_{p_\theta}[R(z)] = \mathbb{E}_{p_\theta}[R(z) \nabla_\theta \log p_\theta(z)]$。方差过高，训练不稳定。
- **TRPO（Trust Region Policy Optimization）**：将问题线性化，约束每次更新的 KL 散度不超过阈值 $\delta$。效果更稳定但计算复杂。
- **PPO**：通过裁剪（Clipping）概率比值来近似 TRPO 的约束，无需求解约束优化问题。

#### 5.3.2 PPO 目标函数

$$L(s, a, \theta_k, \theta) = \min\!\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_k}(a|s)} \hat{A}^{\pi_{\theta_k}}(s,a),\ \mathrm{clip}\!\left(\frac{\pi_\theta(a|s)}{\pi_{\theta_k}(a|s)}, 1-\epsilon, 1+\epsilon\right) \hat{A}^{\pi_{\theta_k}}(s,a)\right)$$

#### 5.3.3 InstructGPT 中的完整目标

$$\mathrm{objective}(\phi) = \mathbb{E}_{(x,y) \sim D_{\pi_\phi^{\mathrm{RL}}}} \left[ r_\theta(x, y) - \beta \log\!\left(\pi_\phi^{\mathrm{RL}}(y|x) / \pi^{\mathrm{SFT}}(y|x)\right) \right] + \gamma \, \mathbb{E}_{x \sim D_{\mathrm{pretrain}}} \left[\log(\pi_\phi^{\mathrm{RL}}(x))\right]$$

其中：
- 第一项为奖励最大化；
- $\beta$ 项为 KL 惩罚，防止策略过度偏离 SFT 模型；
- $\gamma$ 项（PPO-ptx 模型专有）为预训练梯度混合，防止通用能力退化。

#### 5.3.4 工程复杂性

PPO 在语言模型场景下需要同时维护：策略模型（Policy）、参考模型（Reference）、奖励模型（Reward Model）、价值函数（Value Function，与奖励模型共享参数初始化）四个模型，且需要在线采样（rollout），工程实现复杂，超参数敏感。

### 5.4 方法二：DPO（Direct Preference Optimization）

#### 5.4.1 核心动机

DPO 旨在消除 PPO 的三大复杂性来源：独立的奖励模型、在线采样（rollout）以及 actor-critic 架构，将整个 RLHF 流程简化为一次离线的有监督学习。

#### 5.4.2 数学推导

**Step 1**：在非参数假设下，KL 约束的 RLHF 问题具有闭合形式的最优解：

$$\pi_r(y|x) = \frac{1}{Z(x)} \pi_{\mathrm{ref}}(y|x) \exp\!\left(\frac{1}{\beta} r(x,y)\right)$$

**Step 2**：对上式取对数并整理，可将奖励表示为策略的函数（"隐含奖励"）：

$$r(x, y) = \beta \log \frac{\pi_r(y|x)}{\pi_{\mathrm{ref}}(y|x)} + \beta \log Z(x)$$

**Step 3**：将上式代入 Bradley-Terry 奖励模型的偏好概率公式中，分子分母中的 $Z(x)$ 相消，得到 DPO 的最终损失函数：

$$\mathcal{L}_{\mathrm{DPO}}(\pi_\theta; \pi_{\mathrm{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\mathrm{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\mathrm{ref}}(y_l|x)} \right) \right]$$

其中 $y_w$ 为偏好回答，$y_l$ 为非偏好回答。

#### 5.4.3 梯度的直观理解

$$\nabla_\theta \mathcal{L}_{\mathrm{DPO}} = -\beta \, \mathbb{E} \left[ \underbrace{\sigma(\hat{r}_\theta(x, y_l) - \hat{r}_\theta(x, y_w))}_{\text{奖励估计越错，权重越大}} \left[ \underbrace{\nabla_\theta \log \pi(y_w|x)}_{\text{增大好回答的概率}} - \underbrace{\nabla_\theta \log \pi(y_l|x)}_{\text{减小差回答的概率}} \right] \right]$$

本质上等价于：**对好的回答做加权正梯度，对差的回答做加权负梯度，权重正比于当前隐含奖励模型的预测误差**。

#### 5.4.4 关键假设与局限性

DPO 推导依赖一个关键的**非参数假设**：假设 $\pi_\theta$ 是所有可能策略的集合（即全集），这在参数化神经网络场景下并不严格成立。此外，DPO 是离线方法，无法利用在线采样带来的新信息更新偏好分布。

#### 5.4.5 DPO 变体

Tulu 3（Lambert et al. 2024）对比了多种变体：

- **SimPO**（无参考模型）：将 DPO 中的参考策略项替换为按序列长度归一化的对数概率，并引入 margin 超参数 $\gamma$；
- **Length-Normalized DPO**：在 DPO 原始公式基础上，对好回答和差回答的对数概率均除以各自序列长度 $|y|$，缓解长度偏差。

### 5.5 PPO vs. DPO：实验比较

原始 DPO 论文（Rafailov et al. 2023）在摘要任务上的对照实验显示，DPO 的模拟胜率（46.8%）与 PPO（46.8%）持平，但实现复杂度大幅降低。

然而，实证结论高度依赖实验设置（Ivison et al. 2023；Lambert et al. 2024）：

- 在 Tulu 3 的系统对比中，DPO-norm 在部分指标上优于 PPO，但 PPO 在某些子任务上仍有优势；
- 当奖励模型质量较高、在线采样数据充足时，PPO 的上限可能更高；
- 对于资源有限的研究者和中小规模模型，DPO 是更实用的默认选择。

目前，大多数开源顶级对齐模型（Zephyr 7B、OLMo 等）均采用 DPO 或其变体。

---

## 六、RLHF 的常见风险

### 6.1 奖励过度优化（Reward Hacking / Overoptimization）

**现象**：随着策略与参考模型的 KL 散度增大，代理奖励（proxy reward）先升后降，最终在人类真实偏好上表现退化。

**原因**：奖励模型是真实人类偏好的不完美近似，模型会发现奖励函数的"漏洞"（如回答越长分越高）并加以利用，导致过拟合。

**规律**（Gao et al. 2022）：此现象在以下情况普遍存在：
- 基于真实人类偏好的奖励模型；
- 基于带噪声 AI 偏好的奖励模型（如 AlpacaFarm）。

但当使用几乎无噪声的 AI 评判（如单提示 GPT-4）时，过优化现象消失，代理奖励与真实胜率持续正相关。

**对应措施**：KL 惩罚项（$\beta$）、早停（early stopping）、奖励模型集成。

### 6.2 模式坍塌与概率校准失效

**现象**：RLHF 之后，模型的输出概率分布过度集中，不再具备良好的概率校准（calibration）。

**量化证据**：
- 预训练模型在 MMLU 5-shot 上的期望校准误差（ECE）约为 0.007，接近理想校准；
- PPO 训练后 ECE 升至约 0.074，置信度严重虚高；
- 将温度提升至 T=2.5 方能部分恢复校准，但这与标准推理设置相悖。

不同 RLHF 阶段的模型（text-davinci-003、davinci）相比同规模基础模型，输出熵显著降低，说明 RLHF 系统性地压缩了模型的输出多样性。

**实践含义**：使用 RLHF 模型进行概率类任务（如不确定性量化、集成）时，需特别注意校准问题。

### 6.3 标注人员偏差的系统性影响

标注人员的文化背景、人口构成、主观倾向会通过偏好数据渗透进奖励模型和最终策略。主要表现为：

- **断言性偏差**：标注员倾向于偏好语气更自信的回答，即便其事实准确性更低；
- **复杂度偏差**：文本复杂度影响标注员对事实性与一致性错误的检测灵敏度；
- **人口偏差**：不同宗教、文化背景在模型中被代表的程度不均等。

---

## 七、总结

### 7.1 SFT 数据的关键结论

| 结论 | 说明 |
|------|------|
| 提取优于注入 | SFT 应激发预训练已有能力，而非注入新知识 |
| 事实正确数据也可能有害 | 若模型未知相关知识，即便正确示范也会诱发幻觉 |
| 少量数据影响显著 | ~500 条安全或风格数据即可大幅改变模型行为 |
| 风格干扰偏好评估 | 在基于偏好的评估中，长度和格式的影响可能超过内容质量 |

### 7.2 RLHF 算法的关键结论

| 结论 | 说明 |
|------|------|
| DPO 更实用 | 无需奖励模型和在线采样，实现简单，开源社区广泛采用 |
| PPO 上限更高 | 在奖励模型高质量、计算资源充足时可能有优势 |
| 两者结论高度情境依赖 | 不同实验设置下结论不一致，需谨慎解读 |
| 奖励过优化是核心风险 | KL 惩罚是必要的正则化手段 |
| RLHF 破坏概率校准 | 用于概率类任务时需格外注意 |

### 7.3 RLHF 数据的关键结论

| 结论 | 说明 |
|------|------|
| 数据质量高于数量 | 高质量偏好数据比低质量大规模数据更有效 |
| 标注人员构成影响模型价值观 | 需有意识地控制人口多样性 |
| AI 评判具有一定可靠性 | GPT-4 在系统层面与人类偏好高度相关 |
| 长度偏差普遍存在 | 需针对性处理（如长度归一化 DPO）|
| 存在重要的劳工伦理问题 | 大规模有害内容标注对工人造成心理健康风险 |

---

## 参考文献

- Ouyang, L. et al. (2022). *Training language models to follow instructions with human feedback*. NeurIPS.
- Schulman, J. (2023). *Reinforcement Learning from Human Feedback: Progress and Challenges*. [Talk]
- Gekhman, Z. et al. (2023). *Does Fine-Tuning LLMs on New Knowledge Encourage Hallucinations?*
- Dubois, Y. et al. (2023). *AlpacaFarm: A Simulation Framework for Methods that Learn from Human Feedback*. NeurIPS.
- Wang, Y. et al. (2023). *How Far Can Camels Go? Exploring the State of Instruction Tuning on Open Resources*.
- Rafailov, R. et al. (2023). *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*. NeurIPS.
- Gao, L. et al. (2022). *Scaling Laws for Reward Model Overoptimization*.
- Santurkar, S. et al. (2023). *Whose Opinions Do Language Models Reflect?* ICML.
- Hosking, T., Blunsom, P., & Bartolo, M. (2024). *Human Feedback is not Gold Standard*.
- Lambert, N. et al. (2024). *Tulu 3: Pushing Frontiers in Open Language Model Post-Training*.
- Chen, L. et al. (2024). *ODIN: Disentangled Reward Mitigates Hacking in RLHF*.
- Singhal, P. et al. (2024). *A Long Way to Go: Investigating Length Correlations in RLHF*.
- Bai, Y. et al. (2022). *Constitutional AI: Harmlessness from AI Feedback*. Anthropic.
- Bubeck, S. et al. (2023). *Sparks of Artificial General Intelligence: Early experiments with GPT-4*.
- Goldstein, J.A. et al. (2023). *Generative Language Models and Automated Influence Operations*.
- Kang, D. et al. (2023). *Exploiting Programmatic Behavior of LLMs: Dual-Use Through Standard Security Attacks*.
