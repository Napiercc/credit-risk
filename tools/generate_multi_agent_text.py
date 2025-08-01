import os
import sys
import time
import json
import pandas as pd
import requests
from tqdm import tqdm

# ——— 加载 API key ———
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)
api_key = config["api_key"]

# ——— GPT API 配置（港科大）———
url = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# ——— 读取数据 ———
input_path = r"D:\credit_risk\credit-risk - 副本 (2)\data\raw\500_sample\credit_risk_dataset_500_with_label.csv"
df = pd.read_csv(input_path, encoding="utf-8-sig")

# ——— 字段检查 ———
required_cols = [
    'person_age',
    'person_income',
    'person_home_ownership',
    'person_emp_length',
    'loan_intent',
    'loan_grade',
    'loan_amnt',
    'loan_int_rate',
    'loan_status',
    'loan_percent_income',
    'cb_person_default_on_file',
    'cb_person_cred_hist_length'
]
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"缺少字段：{col}")

# ——— 角色定义 ———
roles = {
    "risk_analyst_eval": (
        "你是一名银行贷款风险分析师，请阅读以下贷款申请内容，"
        "从借款人的信用记录、收入来源、还款能力、负债水平等角度，"
        "分析其是否存在信用风险或违约可能，并写出你的专业判断。"
        "请以条列形式输出分析逻辑，最后给出一个总体风险评级："
        "低风险 / 中风险 / 高风险。"
    ),
    "industry_specialist_eval": (
        "你是一名贷款材料审核专员，请阅读以下文本，检查是否存在以下问题："
        "1）关键信息是否缺失（如收入来源、负债情况、经营时间等）；"
        "2）描述是否含糊、逻辑是否清晰；"
        "3）是否存在自相矛盾或难以判断的表述。"
        "请列出发现的问题，并指出还需补充哪些信息。"
    ),
    "legal_advisor_eval": (
        "你是一名还款意愿分析师，请阅读以下贷款文本，"
        "从主观动机、个人责任感、诚信程度、目标规划等方面评估借款人是否具有"
        "积极的还款意愿。请指出文中体现出积极意愿或潜在逃避动机的表达，"
        "并作出总结判断：明确意愿 / 意愿不明 / 可疑态度。"
    )
}

# ——— 构造提示信息 ———
def generate_credit_info_prompt(row):
    return (
        "以下是贷款申请人的基本信息：\n"
        f"- 年龄：{row['person_age']} 岁\n"
        f"- 年收入：{row['person_income']} 元\n"
        f"- 住房情况：{row['person_home_ownership']}\n"
        f"- 工作年限：{row['person_emp_length']} 月\n"
        f"- 借款用途：{row['loan_intent']}\n"
        f"- 信贷评分等级：{row['loan_grade']}\n"
        f"- 贷款金额：{row['loan_amnt']} 元\n"
        f"- 利率：{row['loan_int_rate']}%\n"
        f"- 贷款状态：{row['loan_status']}\n"
        f"- 贷款金额占收入比例：{row['loan_percent_income']}\n"
        f"- 是否有违约记录：{row['cb_person_default_on_file']}\n"
        f"- 信用历史时长：{row['cb_person_cred_hist_length']} 年\n"
    )

# ——— GPT 调用函数（保持 max_chars 字数控制）———
def call_gpt_strict_len(prompt: str,
                        max_chars: int = 512,
                        temperature: float = 0.7,
                        max_retries: int = 3) -> str:
    system_msg = (
        "你是金融领域专家。在回答时请务必严格控制输出"
        f"在不超过{max_chars}个中文字符之内，涵盖所有关键信息，"
        "不要事后截断。"
    )
    user_msg = prompt + f"\n\n请用不超过{max_chars}个中文字符作答。"

    data = {
        "model": "gpt-4o-2024-08-06",
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ],
        "temperature": temperature,
        "max_tokens": 200
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=15)
            response.raise_for_status()
            result = response.json()
            output = result["choices"][0]["message"]["content"].strip()
            return output[:max_chars]
        except Exception as e:
            print(f"[Error attempt {attempt+1}] {e}")
            time.sleep(3)
    return "生成失败"

# ——— 主循环：按角色生成文本 ———
for role_field, role_prompt in roles.items():
    outputs = []
    print(f"⏳ 正在生成：{role_field} …")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        info_prompt = generate_credit_info_prompt(row)
        full_prompt = role_prompt + "\n\n" + info_prompt
        out = call_gpt_strict_len(full_prompt, max_chars=170)
        outputs.append(out)
    df[role_field] = outputs

# ——— 保存结果 ———
save_path = r"D:\credit_risk\credit-risk\data\raw\credit_risk_dataset_500_with_170_multirole.csv"
df.to_csv(save_path, index=False, encoding="utf-8-sig")
print(f"\n✅ 已保存新数据集：{save_path}")
