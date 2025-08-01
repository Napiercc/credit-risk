#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
summary_replace_noprompt.py

混合式摘要（先抽取、再生成），并用摘要内容替换原评估列：
1. 用 TextRank 抽取最重要的 3 句
2. 用 BART‑large‑chinese 精炼为不超过 120 字的摘要（不再使用 Prompt，把短文本直接传给模型）
3. 用摘要结果覆盖原列，保留文件中其他列不变
"""

import os
import sys
import pandas as pd
import torch
import jieba
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as SumyTokenizer
from sumy.summarizers.text_rank import TextRankSummarizer
from transformers import AutoTokenizer, BartForConditionalGeneration

def extract_top_sentences(text: str, k: int = 3) -> str:
    """用 TextRank 抽取最重要的 k 句，合并成短文本。"""
    if not text or not text.strip():
        return ""
    parser     = PlaintextParser.from_string(text, SumyTokenizer("chinese"))
    summarizer = TextRankSummarizer()
    top_sents  = summarizer(parser.document, k)
    return "。".join(str(s).strip() for s in top_sents) + "。"

def refine_with_bart(
    short_text: str,
    tokenizer: AutoTokenizer,
    model: BartForConditionalGeneration,
    device: torch.device,
    max_len: int = 170,
    min_len: int = 20
) -> str:
    """
    对抽取后的短文本直接做抽象式精炼摘要，
    不带任何 Prompt，避免前缀出现在输出里。
    """
    # 直接使用短文本编码
    batch = tokenizer(
        short_text,
        truncation=True,
        padding="longest",
        max_length=256,
        return_tensors="pt"
    ).to(device)

    ids = model.generate(
        batch.input_ids,
        attention_mask=batch.attention_mask,
        max_length=max_len,
        min_length=min_len,
        num_beams=6,
        length_penalty=2.5,
        no_repeat_ngram_size=3,
        early_stopping=True
    )
    summary = tokenizer.decode(ids[0], skip_special_tokens=True)
    # 去掉多余空格
    return "".join(summary.split())

def main():
    # ——— 配置 ———
    input_csv   = r"D:\credit_risk\credit-risk\data\raw\500_sample\10.csv"
    output_csv  = r"D:\credit_risk\credit-risk\data\raw\500_sample\10_compressed.csv"
    target_cols = [
        "risk_analyst_eval",
        "industry_specialist_eval",
        "legal_advisor_eval"
    ]

    # ——— 读取数据 ———
    if not os.path.isfile(input_csv):
        print(f"[ERROR] 找不到输入文件：{input_csv}")
        sys.exit(1)
    df = pd.read_csv(input_csv, encoding="utf-8-sig")

    # ——— 初始化模型 ———
    model_id  = "fnlp/bart-large-chinese"
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] 加载模型 `{model_id}` 到设备 {device}…")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    model     = BartForConditionalGeneration.from_pretrained(model_id).to(device)

    # ——— 生成并替换列 ———
    for col in target_cols:
        if col not in df.columns:
            raise KeyError(f"缺少列：{col}")
        print(f"[INFO] 处理列 `{col}`…")
        new_vals = []
        for text in df[col].fillna("").astype(str):
            short   = extract_top_sentences(text, k=3)                   # 抽取
            summary = refine_with_bart(short, tokenizer, model, device)  # 精炼
            new_vals.append(summary)
        df[col] = new_vals  # 原地替换

    # ——— 保存结果 ———
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"[SUCCESS] 已用摘要覆盖原列并保存到：{output_csv}")

if __name__ == "__main__":
    main()
