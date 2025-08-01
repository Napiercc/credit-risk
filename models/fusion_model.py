import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict  # 新增：用于初始化权重记录容器


class BERTEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token特征


class Expert(nn.Module):
    """专家网络：专注于处理特定类型的文本特征"""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.fc(x)


class GatingNetwork(nn.Module):
    """门控网络：动态计算专家权重"""

    def __init__(self, input_dim, num_experts):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        weights = F.softmax(self.fc(x), dim=1)
        return weights


class CreditRiskPredictor(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_encoder = BERTEncoder(config.bert_model_name)

        # MoE模块
        self.num_experts = config.num_experts
        self.experts = nn.ModuleList([
            Expert(config.text_dim, config.expert_hidden_dim)
            for _ in range(self.num_experts)
        ])
        self.gate = GatingNetwork(config.text_dim * 3, self.num_experts)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(config.text_dim + config.structured_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1)
        )

        # 新增：初始化权重记录容器（用defaultdict存储不同类型样本的权重）
        self.expert_weights_store = defaultdict(list)

    def forward(self, input_ids1, attention_mask1,
                input_ids2, attention_mask2,
                input_ids3, attention_mask3,
                structured_features,
                record_weights=False,
                sample_type=None):
        # 编码3个文本通道
        feat1 = self.text_encoder(input_ids1, attention_mask1)
        feat2 = self.text_encoder(input_ids2, attention_mask2)
        feat3 = self.text_encoder(input_ids3, attention_mask3)

        # MoE融合
        text_concat = torch.cat([feat1, feat2, feat3], dim=1)
        gate_weights = self.gate(text_concat)  # 门控权重 [batch_size, num_experts]
        avg_text_feat = (feat1 + feat2 + feat3) / 3
        expert_outputs = torch.stack([expert(avg_text_feat) for expert in self.experts], dim=1)
        moe_feat = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)

        # 最终融合
        combined = torch.cat([moe_feat, structured_features], dim=1)
        outputs = self.fusion(combined)

        # 记录权重（调用修改后的方法）
        if record_weights and sample_type is not None:
            self._record_weights(gate_weights, sample_type)

        return outputs

    # 修正：权重记录方法（改名并使用正确的存储容器）
    def _record_weights(self, gate_weights, sample_type):
        """将权重存入expert_weights_store"""
        self.expert_weights_store[sample_type].extend(gate_weights.cpu().detach().numpy())

    # 修正：打印权重统计（使用存储容器）
    def print_weight_stats(self):
        for sample_type, weights_list in self.expert_weights_store.items():
            if not weights_list:
                continue
            weights = np.array(weights_list)
            avg_weights = weights.mean(axis=0)
            print(f"\n{sample_type}的专家平均权重:")
            for i, avg in enumerate(avg_weights):
                print(f"专家{i}: {avg:.4f}")
            print(f"最活跃专家: 专家{np.argmax(avg_weights)}")

    # 修正：绘制权重分布（使用存储容器）
    def plot_weight_distribution(self, save_path):
        plt.figure(figsize=(12, 6))
        bar_width = 0.25
        sample_types = list(self.expert_weights_store.keys())

        for i, sample_type in enumerate(sample_types):
            weights_list = self.expert_weights_store[sample_type]
            if not weights_list:
                continue
            weights = np.array(weights_list)
            avg_weights = weights.mean(axis=0)
            plt.bar(
                np.arange(self.num_experts) + i * bar_width,
                avg_weights,
                width=bar_width,
                label=sample_type
            )

        plt.xlabel("专家编号")
        plt.ylabel("平均权重")
        plt.title("不同类型样本的专家权重分布")
        plt.xticks(
            np.arange(self.num_experts) + bar_width * (len(sample_types) - 1) / 2,
            [f"专家{i}" for i in range(self.num_experts)]
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()