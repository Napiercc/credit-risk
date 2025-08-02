import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score, confusion_matrix
)
from sklearn.utils import resample
from tabulate import tabulate
from hmeasure import h_score
from models.fusion_model import CreditRiskPredictor
from utils.data_loader import prepare_dataloaders
from config import Config

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams['axes.unicode_minus'] = False


# 计算置信区间
def calculate_confidence_interval(y_true, y_score, metric_func, n_bootstraps=1000, confidence_level=0.95):
    scores = []
    for _ in range(n_bootstraps):
        indices = resample(range(len(y_true)))
        if len(np.unique(y_true[indices])) < 2:
            continue
        score = metric_func(y_true[indices], y_score[indices])
        scores.append(score)
    lower = np.percentile(scores, (1 - confidence_level) / 2 * 100)
    upper = np.percentile(scores, (1 + confidence_level) / 2 * 100)
    return lower, upper


# 绘制ROC曲线
def plot_roc_curve(y_true, y_score, save_path=None):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('假正例率 (FPR)')
    plt.ylabel('真正例率 (TPR)')
    plt.title('受试者工作特征曲线')
    plt.legend(loc="lower right")
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# 绘制PR曲线
def plot_precision_recall_curve(y_true, y_score, save_path=None):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    avg_precision = average_precision_score(y_true, y_score)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR曲线 (AP = {avg_precision:.4f})')
    plt.xlabel('召回率 (Recall)')
    plt.ylabel('精确率 (Precision)')
    plt.title('精确率-召回率曲线')
    plt.legend(loc="upper right")
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# 绘制概率分布
def plot_probability_distribution(pos_probs, neg_probs, ks_threshold, save_path=None):
    plt.figure(figsize=(10, 6))
    sns.histplot(pos_probs, label='正类（违约）', color='red', alpha=0.5, kde=True)
    sns.histplot(neg_probs, label='负类（不违约）', color='blue', alpha=0.5, kde=True)
    plt.axvline(x=ks_threshold, color='green', linestyle='--', label=f'KS最佳阈值: {ks_threshold:.4f}')
    plt.legend()
    plt.title('预测概率分布')
    plt.xlabel('预测概率')
    plt.ylabel('样本数')
    plt.grid(axis='y', alpha=0.7)
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# 打印或保存混淆矩阵和主要分类指标
def format_confusion_and_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0
    recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall_pos) / (precision + recall_pos) if (precision + recall_pos) > 0 else 0

    cm_str = tabulate(cm, headers=['负类', '正类'], tablefmt='grid')
    metrics = [
        ['准确率', f'{accuracy:.4f}'],
        ['正类精确率', f'{precision:.4f}'],
        ['正类召回率', f'{recall_pos:.4f}'],
        ['负类召回率', f'{recall_neg:.4f}'],
        ['正类F1分数', f'{f1:.4f}'],
    ]
    metrics_str = tabulate(metrics, headers=['指标', '值'], tablefmt='fancy_grid')
    return cm_str, metrics_str


# 模型评估
def evaluate():
    image_save_dir = r"data\re\10000_170_moe_4"
    os.makedirs(image_save_dir, exist_ok=True)
    report_path = os.path.join(image_save_dir, 'evaluation_report.txt')

    config = Config()
    _, _, test_loader, _ = prepare_dataloaders(config)
    model = CreditRiskPredictor(config).to(config.device)
    model.load_state_dict(torch.load(config.model_save_path))
    model.eval()

    # # 初始化专家权重记录（清空历史记录）
    # model.expert_weights_record.clear()

    test_labels, test_probs = [], []
    with torch.no_grad():
        for batch in test_loader:
            # 三通道输入处理
            input_ids1 = batch['input_ids1'].to(config.device)
            attention_mask1 = batch['attention_mask1'].to(config.device)
            input_ids2 = batch['input_ids2'].to(config.device)
            attention_mask2 = batch['attention_mask2'].to(config.device)
            input_ids3 = batch['input_ids3'].to(config.device)
            attention_mask3 = batch['attention_mask3'].to(config.device)

            structured_features = batch['structured_features'].to(config.device)
            labels = batch['labels'].cpu().numpy()

            # 前向传播时记录专家权重（按真实标签分类）
            batch_size = labels.shape[0]
            for i in range(batch_size):
                # 提取单个样本
                single_inputs = (
                    input_ids1[i:i + 1], attention_mask1[i:i + 1],
                    input_ids2[i:i + 1], attention_mask2[i:i + 1],
                    input_ids3[i:i + 1], attention_mask3[i:i + 1],
                    structured_features[i:i + 1]
                )

                # 记录权重：按真实标签分类（"违约样本"/"非违约样本"）
                sample_type = "违约样本" if labels[i] == 1 else "非违约样本"
                model(*single_inputs, record_weights=True, sample_type=sample_type)

            # 批量计算预测概率（用于评估指标）
            outputs = model(
                input_ids1, attention_mask1,
                input_ids2, attention_mask2,
                input_ids3, attention_mask3,
                structured_features
            ).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()

            test_labels.extend(labels)
            test_probs.extend(probs)

    test_labels = np.array(test_labels)
    test_probs = np.array(test_probs)

    # 计算KS及阈值
    fpr, tpr, thresholds = roc_curve(test_labels, test_probs)
    ks = max(tpr - fpr)
    ks_threshold = thresholds[np.argmax(tpr - fpr)]
    preds = (test_probs >= ks_threshold).astype(int)

    auc = roc_auc_score(test_labels, test_probs)
    prauc = average_precision_score(test_labels, test_probs)
    h = h_score(test_labels, preds)
    auc_ci_lower, auc_ci_upper = calculate_confidence_interval(test_labels, test_probs, roc_auc_score)

    # 构建报告文本
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('模型评估报告\n')
        f.write('=' * 50 + '\n')
        metrics_lines = [
            ['AUC', f'{auc:.4f}', f'({auc_ci_lower:.4f}, {auc_ci_upper:.4f})'],
            ['KS统计量', f'{ks:.4f}', f'阈值: {ks_threshold:.4f}'],
            ['PR-AUC', f'{prauc:.4f}', ''],
            ['H-measure', f'{h:.4f}', '']
        ]
        table = tabulate([['指标', '值', '备注']] + metrics_lines, headers='firstrow', tablefmt='fancy_grid')
        f.write(table + '\n\n')

        cm_str, metrics_str = format_confusion_and_metrics(test_labels, preds)
        f.write('混淆矩阵:\n')
        f.write(cm_str + '\n')
        f.write('主要分类指标:\n')
        f.write(metrics_str + '\n')

    # 生成并保存原有评估图像
    plot_roc_curve(test_labels, test_probs, save_path=os.path.join(image_save_dir, "roc_curve.png"))
    plot_precision_recall_curve(test_labels, test_probs, save_path=os.path.join(image_save_dir, "pr_curve.png"))
    pos_probs = test_probs[test_labels == 1]
    neg_probs = test_probs[test_labels == 0]
    plot_probability_distribution(pos_probs, neg_probs, ks_threshold,
                                  save_path=os.path.join(image_save_dir, "dist.png"))

    # 新增：生成专家权重分析结果
    print("\n===== 专家权重分析结果 =====")
    model.print_weight_stats()
    model.plot_weight_distribution(os.path.join(image_save_dir, "moe_expert_weights.png"))

    print(f"已将评估报告保存到: {report_path}")


if __name__ == '__main__':
    evaluate()
