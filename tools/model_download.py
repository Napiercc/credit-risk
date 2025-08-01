# from transformers import BertTokenizer, BertModel
#
# model_name = "bert-base-chinese"
#
# save_directory = "./models/bert-base-chinese"
#
# # 下载并保存分词器
# tokenizer = BertTokenizer.from_pretrained(model_name)
# tokenizer.save_pretrained(save_directory)
#
# # 下载并保存模型
# model = BertModel.from_pretrained(model_name)
# model.save_pretrained(save_directory)



from huggingface_hub import snapshot_download

# 只要在脚本里执行一次，就会把整个仓库（包括 LFS 管理的 spiece.model）下载到本地
snapshot_download(
    repo_id="IDEA-CCNL/Randeng-Pegasus-238M-Summary-Chinese",
    local_dir="D:/credit_risk/credit-risk/models/Summary-Chinese",
    resume_download=True,
    use_auth_token=False    # 如果是私有模型，设置为 True 并先登录：huggingface-cli login
)
print("✅ 模型文件已下载完成。")
