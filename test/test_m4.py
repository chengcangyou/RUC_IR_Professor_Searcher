# test_bge.py
from sentence_transformers import SentenceTransformer
import torch

# 1. 检测 M4 的 MPS 加速
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"正在使用设备: {device} (如果是 mps 说明 M4 加速已开启)")

# 2. 加载模型 (首次运行会自动下载约 1GB 模型文件)
print("正在下载/加载 BGE-M3 模型...")
model = SentenceTransformer('BAAI/bge-m3', device=device)

# 3. 模拟你的场景：中文 Query，英文 Doc
query = "信息检索"
docs = [
    "Information Retrieval is the science of searching for information in a document.", # 相关
    "Deep Learning focuses on neural networks.", # 不相关
    "Professor X works on Web Search and IR systems." # 相关
]

# 4. 编码 (生成向量)
query_embedding = model.encode(query, normalize_embeddings=True)
doc_embeddings = model.encode(docs, normalize_embeddings=True)

# 5. 计算相似度 (点积)
similarities = query_embedding @ doc_embeddings.T

# 6. 打印结果
for doc, score in zip(docs, similarities):
    print(f"分数: {score:.4f} -> {doc}")