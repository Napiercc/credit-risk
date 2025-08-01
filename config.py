import torch


class Config:
    # 数据配置
    data_path = r"D:\credit_risk\credit-risk-3channel\data\raw\500_sample\credit_risk_dataset_500_with_170_multirole.csv"
    label_column = 'label'  # 标签列名
    structured_columns = [ # 结构化特征
        'person_age','person_income','person_home_ownership','person_emp_length','loan_intent','loan_grade',
        'loan_amnt','loan_int_rate','loan_status','loan_percent_income','cb_person_default_on_file',
        'cb_person_cred_hist_length'
    ]
    role_columns = [  # 三个角色文本列（新增）
        "risk_analyst_eval",
        "industry_specialist_eval",
        "legal_advisor_eval"
    ]

    # 模型配置
    bert_model_name = r"D:\credit_risk\credit-risk-3channel\models\bert-base-chinese"
    text_dim = 768  # BERT输出维度
    structured_dim = 12  # 结构化特征维度
    hidden_dim = 128  # 隐藏层维度
    dropout = 0.2  # Dropout率

    # MoE新增参数
    num_experts = 4  # 专家数量（建议4-8，根据数据量调整）
    expert_hidden_dim = 512  # 专家网络隐藏层维度

    # 训练配置
    max_length = 512  # 单个文本最大长度（不压缩，保留原始长度）
    batch_size = 8  # 多通道输入需减小batch_size避免显存溢出
    learning_rate = 3e-5  # 学习率
    epochs = 10  # 训练轮数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 保存路径
    model_save_path = 'models/checkpoints/best_model_500_multichannel_170_moe_4_3.0.pt'