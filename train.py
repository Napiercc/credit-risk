import torch
import random
import numpy as np
from matplotlib import font_manager
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.fusion_model import CreditRiskPredictor
from utils.data_loader import prepare_dataloaders
from config import Config

# 设置随机种子以保证实验可重复性
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# 初始化种子
set_seed(42)

# 设置图片清晰度
plt.rcParams['figure.dpi'] = 300

# 最安全的字体设置：直接指定 SimHei 路径
simhei_path = r"C:\Windows\Fonts\simhei.ttf"          # Windows 黑体
try:
    simhei_prop = font_manager.FontProperties(fname=simhei_path)
    plt.rcParams['font.family'] = simhei_prop.get_name()
except Exception:
    plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False  # 确保负号正常显示

class EarlyStopping:
    """早停机制：当 monitored metric 在 patience 个 epoch 内不再提升，则停止训练。"""

    def __init__(self, patience=3, mode='max', min_delta=1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best = None
        self.num_bad = 0
        if mode == 'min':
            self._is_better = lambda cur, best: cur < best - min_delta
        else:
            self._is_better = lambda cur, best: cur > best + min_delta

    def step(self, current):
        if self.best is None:
            self.best = current
            return False
        if self._is_better(current, self.best):
            self.best = current
            self.num_bad = 0
            return False
        else:
            self.num_bad += 1
            return self.num_bad >= self.patience


def train():
    config = Config()
    train_loader, val_loader, _, _ = prepare_dataloaders(config)
    model = CreditRiskPredictor(config).to(config.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # 替换原criterion
    # criterion = torch.nn.BCELoss()
    criterion = torch.nn.BCEWithLogitsLoss()  # 内置Sigmoid，与模型输出logits匹配

    # pos_weight = torch.tensor(3.0).to(config.device)
    # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    early_stopper = EarlyStopping(patience=3, mode='max', min_delta=1e-4)

    best_auc = 0.0
    best_model_state = None
    train_losses = []
    val_aucs = []

    for epoch in range(config.epochs):
        model.train()
        total_loss = 0

        # 三通道训练阶段
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Train]'):
            # 文本通道1
            input_ids1 = batch['input_ids1'].to(config.device)
            attention_mask1 = batch['attention_mask1'].to(config.device)
            # 文本通道2
            input_ids2 = batch['input_ids2'].to(config.device)
            attention_mask2 = batch['attention_mask2'].to(config.device)
            # 文本通道3
            input_ids3 = batch['input_ids3'].to(config.device)
            attention_mask3 = batch['attention_mask3'].to(config.device)

            structured_features = batch['structured_features'].to(config.device)
            labels = batch['labels'].to(config.device).float()

            outputs = model(
                input_ids1, attention_mask1,
                input_ids2, attention_mask2,
                input_ids3, attention_mask3,
                structured_features
            ).squeeze()
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        val_labels, val_preds = [], []
        # 三通道验证阶段
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{config.epochs} [Val]'):
                input_ids1 = batch['input_ids1'].to(config.device)
                attention_mask1 = batch['attention_mask1'].to(config.device)
                input_ids2 = batch['input_ids2'].to(config.device)
                attention_mask2 = batch['attention_mask2'].to(config.device)
                input_ids3 = batch['input_ids3'].to(config.device)
                attention_mask3 = batch['attention_mask3'].to(config.device)

                structured_features = batch['structured_features'].to(config.device)
                labels = batch['labels'].cpu().numpy()

                outputs = model(
                    input_ids1, attention_mask1,
                    input_ids2, attention_mask2,
                    input_ids3, attention_mask3,
                    structured_features
                ).squeeze()
                preds = torch.sigmoid(outputs).cpu().numpy()

                val_labels.extend(labels)
                val_preds.extend(preds)

        val_auc = roc_auc_score(val_labels, val_preds)
        val_aucs.append(val_auc)
        print(f'Epoch {epoch+1}/{config.epochs} | Train Loss: {avg_train_loss:.4f} | Val AUC: {val_auc:.4f}')

        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()

        if early_stopper.step(val_auc):
            print(f'Early stopping triggered at epoch {epoch+1}')
            break

        torch.cuda.empty_cache()

    if best_model_state is not None:
        torch.save(best_model_state, config.model_save_path)
        print(f'Model saved at {config.model_save_path}')

    # 绘制并保存训练曲线
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.plot(range(1,len(train_losses)+1), train_losses, label='训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练损失曲线')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(range(1,len(val_aucs)+1), val_aucs, label='验证集AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('验证集AUC曲线')
    plt.legend()

    plt.tight_layout()
    plt.savefig('./data/result/train_curve/training_plots.png')
    plt.close()

if __name__ == '__main__':
    train()
