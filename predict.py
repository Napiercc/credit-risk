import torch
from transformers import AutoTokenizer
from models.fusion_model import CreditRiskPredictor
from utils.woe_encoder import WOEEncoder
from config import Config
import pandas as pd
from utils.data_loader import load_data


def predict(risk_text, industry_text, legal_text, structured_features):
    config = Config()
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name)

    # 初始化模型并加载权重
    model = CreditRiskPredictor(config).to(config.device)
    model.load_state_dict(torch.load(config.model_save_path))
    model.eval()

    # 处理三个角色的文本输入（不压缩，直接编码）
    encoding1 = tokenizer.encode_plus(
        risk_text, add_special_tokens=True, max_length=config.max_length,
        padding='max_length', truncation=True, return_tensors='pt'
    )
    encoding2 = tokenizer.encode_plus(
        industry_text, add_special_tokens=True, max_length=config.max_length,
        padding='max_length', truncation=True, return_tensors='pt'
    )
    encoding3 = tokenizer.encode_plus(
        legal_text, add_special_tokens=True, max_length=config.max_length,
        padding='max_length', truncation=True, return_tensors='pt'
    )

    # 处理结构化特征（WOE编码）
    woe_encoder = WOEEncoder()
    train_df, _, _ = load_data(config)
    woe_encoder.fit(train_df[config.structured_columns], train_df[config.label_column])

    feat_df = pd.DataFrame([structured_features], columns=config.structured_columns)
    structured_features_woe = woe_encoder.transform(feat_df).values[0]
    structured_tensor = torch.tensor(structured_features_woe, dtype=torch.float32).unsqueeze(0).to(config.device)

    # 模型预测
    with torch.no_grad():
        input_ids1 = encoding1['input_ids'].to(config.device)
        attention_mask1 = encoding1['attention_mask'].to(config.device)
        input_ids2 = encoding2['input_ids'].to(config.device)
        attention_mask2 = encoding2['attention_mask'].to(config.device)
        input_ids3 = encoding3['input_ids'].to(config.device)
        attention_mask3 = encoding3['attention_mask'].to(config.device)

        output = model(
            input_ids1, attention_mask1,
            input_ids2, attention_mask2,
            input_ids3, attention_mask3,
            structured_tensor
        )
        probability = output.item()

    return {
        'probability': probability,
        'prediction': 1 if probability >= 0.5 else 0,
        'risk_level': 'High Risk' if probability >= 0.7 else ('Medium Risk' if probability >= 0.3 else 'Low Risk')
    }


# 示例使用
if __name__ == '__main__':
    # 三个角色的原始文本（不压缩）
    risk_text = "风险分析师评估内容..."  # 原始完整文本
    industry_text = "行业专家评估内容..."  # 原始完整文本
    legal_text = "法律顾问评估内容..."  # 原始完整文本

    # 结构化特征（与原格式一致）
    structured_features = [22, 59000, 'RENT', 123.0, 'PERSONAL', 'D', 35000, 16.02, 1, 0.59, 'Y', 3]

    result = predict(risk_text, industry_text, legal_text, structured_features)
    print(f"违约概率: {result['probability']:.4f}")
    print(f"预测结果: {'违约' if result['prediction'] == 1 else '不违约'}")
    print(f"风险等级: {result['risk_level']}")