"""
compress_extractive.py

对 CSV 中三列文本做抽取式压缩，保留关键信息，不改变原意。
"""

import os
import re
import sys
import pandas as pd
import jieba
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

def preprocess(text: str) -> str:
    """
    去掉常见的模板化前缀，比如“1. 主观动机：”或“以下是对...评估：”等。
    """
    # 删除行首序号和标签
    text = re.sub(r"^\s*\d+\s*[\.\、]?\s*[^：]{1,10}：", "", text)
    # 删除“以下是…评估：”这类
    text = re.sub(r"^.*?评估：", "", text)
    return text.strip()

def extract_summary(text: str, sentence_count: int = 3) -> str:
    """
    基于 TextRank 的抽取式摘要：
    1. 先预处理去掉模板前缀，
    2. 再提取最重要的 sentence_count 句。
    """
    text = preprocess(text)
    if not text:
        return ""
    # 用 sumy 构建文档 parser
    parser = PlaintextParser.from_string(text, SumyTokenizer("chinese"))
    summarizer = TextRankSummarizer()
    top_sents = summarizer(parser.document, sentence_count)
    # 合并成段落
    return "。".join(str(s).strip() for s in top_sents) + "。"

def main():
    # —— 用户配置 ——
    input_csv   = r"D:\credit_risk\credit-risk\data\raw\500_sample\credit_risk_dataset_500_3role.csv"
    output_csv  = r"D:\credit_risk\credit-risk\data\raw\500_sample\credit_risk_dataset_500_3role_compressed.csv"
    cols        = [
        "risk_analyst_eval",
        "industry_specialist_eval",
        "legal_advisor_eval"
    ]
    sentence_count = 2  # 每条抽取几句，可根据压缩率调整

    # —— 读取数据 ——
    if not os.path.isfile(input_csv):
        print(f"[ERROR] 找不到输入文件：{input_csv}")
        sys.exit(1)
    df = pd.read_csv(input_csv, encoding="utf-8-sig")

    # —— 批量抽取 ——
    for col in cols:
        if col not in df.columns:
            raise KeyError(f"缺少列：{col}")
        print(f"[INFO] 压缩列 `{col}`，抽取前 {sentence_count} 句 …")
        df[col] = df[col].fillna("").astype(str).map(
            lambda txt: extract_summary(txt, sentence_count)
        )

    # —— 写回文件 ——
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[SUCCESS] 压缩完成，已保存到：{output_csv}")

if __name__ == "__main__":
    main()
