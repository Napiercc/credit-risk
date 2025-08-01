---

# 💡 项目实验规划文档

本实验旨在探索多角色语言模型（Multi-Agent LLM）在信贷风险预测任务中的表现，涵盖模型对比、变体消融与解释分析多个维度。

---

## 一、基本设置

### ✅ 评估指标

* **ROC-AUC**：Receiver Operating Characteristic 曲线下面积
* **PR-AUC**：Precision-Recall 曲线下面积
* **KS statistic**：Kolmogorov–Smirnov 统计量
* **H-measure**：基于决策成本的评估指标
* **F1-score**：综合考虑 Precision 和 Recall 的调和平均数

### 📊 数据集

* **Dataset A**：实际业务背景下的结构化+文本数据（如银行信贷数据）
* **Dataset B**：行业公开的异构风险数据集（如LendingClub）

---

## 二、主实验设置

### 🔮 Prompt LLM Model（选择2-3个）

| 模型名称       | 说明              |
| ---------- | --------------- |
| `GPT-4`    | 高质量生成，推理能力强     |
| `GPT-3.5`  | 成本较低，适用于大规模实验   |
| `DeepSeek` | 国内替代方案，兼具性能与开放性 |

### 📎 Text Encoder（选择2-3个）

| 编码器        | 特点说明           |
| ---------- | -------------- |
| `BERT`     | 预训练语言理解模型，表现稳定 |
| `LLaMA`    | 轻量化大语言模型，适配性强  |
| `FastText` | 高效词向量工具，适合冷启动  |

### 🧪 比较方法

#### 🔹 传统表格方法（Pure Table）

* `Logistic Regression (LR)`
* `XGBoost`
* `GBDT`
* `MLP`（全连接神经网络）

#### 🔹 LLM 增强方法

* `MLP + Single Agent`
* `MLP + Multi Agent（文本拼接）`
* `MLP + Multi Agent（MOE架构）`

---

## 三、消融实验设计

### 1️⃣ 模型组件分析

* Prompt LLM Model：仅保留 `GPT-4`
* Text Encoder：仅保留 `BERT`

### 2️⃣ MOE结构变体（聚合策略对比）

* `Average Pooling`：三个Embedding平均
* `Max Pooling`：取最大值聚合
* `Expert Dropout`：随机屏蔽某个expert

### 3️⃣ Expert移除实验（组件重要性）

* `w/o E1`：去掉解释者1
* `w/o E2`：去掉解释者2
* `w/o E3`：去掉解释者3

### 4️⃣ Expert 扰动实验（鲁棒性分析）

* 人为干预一个 Agent 的解释输出（如语义污染）
* 所有 Agent 使用同一个解释文本（减少多样性）

---

## 四、解释能力分析实验

### 🎯 解释质量实验

| 模式                   | 描述                  |
| -------------------- | ------------------- |
| Pure Text Prediction | 仅用 LLM 对文本部分进行分类预测  |
| E1 / E2 / E3         | 三个不同解释 Agent 独立效果分析 |
| 三个文本拼接               | 将三段解释拼接后作为输入        |

### 🔍 解释内容分析

* 从样本中抽取典型 `credit case`，可视化三个解释 Agent 输出
* 使用关键词分析（或 Lime/SHAP）对解释进行结构化挖掘
* Case Study：

  * 不同风险样本激活了哪些 Experts？
  * 是否存在某种 Agent 更擅长处理特定类型样本？
* 可视化工具建议：

  * Attention Map
  * Expert Routing/Gating 权重分布图

---
