import pandas as pd
import time
import json
import requests
from tqdm import tqdm

# 读取本地 API key
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

api_key = config["api_key"]

# GPT API 配置
url = "https://gpt-api.hkust-gz.edu.cn/v1/chat/completions"
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

# 三个角色的系统设定
roles = {
    "risk_analyst_eval": "你是一名银行贷款风险分析师，请阅读以下贷款申请内容，从借款人的信用记录、收入来源、还款能力、负债水平等角度，分析其是否存在信用风险或违约可能，并写出你的专业判断。请以条列形式输出分析逻辑，最后给出一个总体风险评级（低风险 / 中风险 / 高风险）。",

    "industry_specialist_eval": "你是一名贷款材料审核专员，请阅读以下文本，检查是否存在以下问题：1）关键信息是否缺失（如收入来源、负债情况、经营时间等）；2）描述是否含糊、逻辑是否清晰；3）是否存在自相矛盾或难以判断的表述。请列出发现的问题，并指出还需补充哪些信息。",

    "legal_advisor_eval": "你是一名还款意愿分析师，请阅读以下贷款文本，从主观动机、个人责任感、诚信程度、目标规划等方面评估借款人是否具有积极的还款意愿。请指出文中体现出积极意愿或潜在逃避动机的表达，并作出总结判断：“明确意愿 / 意愿不明 / 可疑态度”。"
}

# 构造自然语言贷款信息
def generate_credit_info_prompt(row):
    return f"""以下是贷款申请人的基本信息：
- 年龄：{row['person_age']} 岁
- 年收入：{row['person_income']} 元
- 住房情况：{row['person_home_ownership']}
- 工作年限：{row['person_emp_length']} 月
- 借款用途：{row['loan_intent']}
- 信贷评分等级：{row['loan_grade']}
- 贷款金额：{row['loan_amnt']} 元
- 利率：{row['loan_int_rate']}%
- 贷款金额占收入比例：{row['loan_percent_income']}
- 是否有违约记录：{row['cb_person_default_on_file']}
- 信用历史时长：{row['cb_person_cred_hist_length']} 年
"""

# GPT 调用函数（港科大接口）
def call_gpt(prompt, temperature=0.7, max_retries=3):
    data = {
        "model": "gpt-4o-2024-08-06",
        "messages": [
            {"role": "system", "content": "你是一名金融领域专家。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": 170
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=15)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"[Error @Attempt {attempt + 1}] {e}")
            time.sleep(3)

    return "生成失败"

# 主处理流程
df = pd.read_csv(r"D:\credit_risk\credit-risk\data\raw\credit_risk_dataset_500_with_new_label.csv")

for role_field, role_prompt in roles.items():
    generated_texts = []
    print(f"⏳ 正在生成角色文本：{role_field} ...")
    for _, row in tqdm(df.iterrows(), total=len(df)):
        credit_info = generate_credit_info_prompt(row)
        full_prompt = role_prompt + "\n\n" + credit_info
        gpt_output = call_gpt(full_prompt)
        generated_texts.append(gpt_output)
    df[role_field] = generated_texts

# 保存最终结果
save_path = r"D:\credit_risk\credit-risk\data\raw\credit_risk_dataset_500_with_new_label.csv"
df.to_csv(save_path, index=False)
print(f"\n✅ 已保存新数据集：{save_path}")
