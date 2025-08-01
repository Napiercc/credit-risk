import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from utils.woe_encoder import WOEEncoder


class CreditDataset(Dataset):
    def __init__(self, role_texts, structured_features, labels, tokenizer, max_length):
        # role_texts格式: [(text1, text2, text3), ...]
        self.role_texts = role_texts
        self.structured_features = structured_features
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.role_texts)

    def __getitem__(self, idx):
        text1, text2, text3 = self.role_texts[idx]

        # 分别编码三个角色文本（不压缩，使用原始长度，截断至max_length）
        encoding1 = self.tokenizer.encode_plus(
            text1, add_special_tokens=True, max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        encoding2 = self.tokenizer.encode_plus(
            text2, add_special_tokens=True, max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        encoding3 = self.tokenizer.encode_plus(
            text3, add_special_tokens=True, max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )

        return {
            'input_ids1': encoding1['input_ids'].flatten(),
            'attention_mask1': encoding1['attention_mask'].flatten(),
            'input_ids2': encoding2['input_ids'].flatten(),
            'attention_mask2': encoding2['attention_mask'].flatten(),
            'input_ids3': encoding3['input_ids'].flatten(),
            'attention_mask3': encoding3['attention_mask'].flatten(),
            'structured_features': torch.tensor(self.structured_features[idx], dtype=torch.float32),
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32)
        }


def load_data(config):
    # 加载数据（不拼接文本，保留原始三列）
    df = pd.read_csv(config.data_path)

    # 分割数据集
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df[config.label_column])
    val_df, test_df = train_test_split(test_df, test_size=0.33, random_state=42, stratify=test_df[config.label_column])

    return train_df, val_df, test_df


def prepare_dataloaders(config):
    train_df, val_df, test_df = load_data(config)
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name)

    # WOE编码结构化特征
    woe_encoder = WOEEncoder()
    woe_encoder.fit(train_df[config.structured_columns], train_df[config.label_column])

    # 转换结构化特征
    train_struct = woe_encoder.transform(train_df[config.structured_columns]).values
    val_struct = woe_encoder.transform(val_df[config.structured_columns]).values
    test_struct = woe_encoder.transform(test_df[config.structured_columns]).values

    # 构造角色文本列表（不压缩，直接使用原始文本）
    train_role_texts = list(zip(
        train_df['risk_analyst_eval'].fillna(""),
        train_df['industry_specialist_eval'].fillna(""),
        train_df['legal_advisor_eval'].fillna("")
    ))
    val_role_texts = list(zip(
        val_df['risk_analyst_eval'].fillna(""),
        val_df['industry_specialist_eval'].fillna(""),
        val_df['legal_advisor_eval'].fillna("")
    ))
    test_role_texts = list(zip(
        test_df['risk_analyst_eval'].fillna(""),
        test_df['industry_specialist_eval'].fillna(""),
        test_df['legal_advisor_eval'].fillna("")
    ))

    # 创建数据集
    train_dataset = CreditDataset(
        train_role_texts, train_struct, train_df[config.label_column].values,
        tokenizer, config.max_length
    )
    val_dataset = CreditDataset(
        val_role_texts, val_struct, val_df[config.label_column].values,
        tokenizer, config.max_length
    )
    test_dataset = CreditDataset(
        test_role_texts, test_struct, test_df[config.label_column].values,
        tokenizer, config.max_length
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    return train_loader, val_loader, test_loader, tokenizer