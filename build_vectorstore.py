"""
向量数据库构建模块

类比：向量数据库就像一个"语义索引"——传统数据库用 B+ 树按数值/字母排序，
而向量数据库按"语义相似度"排序。你把所有研报文本块存进去，
之后用问题去检索，它会返回语义上最接近的 K 个文本块。

整个流程：
  PDF 页面 → 文本切块 → TF-IDF 向量化 → 存入 Chroma → 持久化到磁盘

为什么用 TF-IDF 而不是深度学习模型？
  因为 tokenizers Rust 库在我的 Windows 环境下与 transformers 存在底层兼容 bug。
  TF-IDF 是纯 Python 实现，零外部依赖，中文友好（基于字符 n-gram）。
  类比：就像使用 Pandas 做数据分析时先用 describe() 快速看全貌，
  TF-IDF 是快速验证 RAG 全链路的第一步。后续可换更强的语义模型。
"""
import os
import pickle
import shutil
import numpy as np
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_chroma import Chroma

from load_and_split import load_and_split
from config import (
    DATA_PATH, PERSIST_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP
)

VECTORIZER_PATH = os.path.join(PERSIST_DIR, "tfidf_vectorizer.pkl")


# ── TF-IDF Embedding 包装器 ─────────────────────────────
# TF-IDF 原理类比：
#   我在淘宝数据里做过的漏斗分析——统计每个步骤的转化率。
#   TF-IDF 也是"统计"：TF 统计词在文档中出现频率，IDF 统计词在整个语料中的稀有度。
#   两者相乘 = 这个词对这篇文档的"代表性"得分。
class TfidfEmbeddings:
    """
    基于 sklearn TfidfVectorizer 的 embedding 类。
    实现 embed_documents / embed_query 即可被 LangChain Chroma 使用（鸭子类型）。

    关键设计：首次构建时在所有文档上 fit，然后把 vectorizer 序列化到磁盘。
    后续加载时从磁盘恢复，保证查询向量维度和库中向量一致。
    """

    def __init__(self, vectorizer=None):
        if vectorizer is not None:
            self.vectorizer = vectorizer
            self._fitted = True
        else:
            self.vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(1, 3),
                max_features=1000,
            )
            self._fitted = False

    def _fit_if_needed(self, texts: List[str]):
        """首次调用时在全部文档上 fit TF-IDF"""
        if not self._fitted:
            print("   🔧 正在构建 TF-IDF 词汇表（仅首次需要）...")
            self.vectorizer.fit(texts)
            self._fitted = True
            print(f"   ✅ 词汇表大小: {len(self.vectorizer.get_feature_names_out())} 个特征")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量编码文档（存入 Chroma 时调用）"""
        self._fit_if_needed(texts)
        matrix = self.vectorizer.transform(texts)
        return matrix.toarray().tolist()

    def embed_query(self, text: str) -> List[float]:
        """编码单个查询（检索时调用）"""
        if not self._fitted:
            raise RuntimeError("TF-IDF vectorizer 尚未拟合，无法编码查询")
        matrix = self.vectorizer.transform([text])
        return matrix.toarray().tolist()[0]


def get_embeddings(force_rebuild=False):
    """
    获取 TF-IDF embedding 实例。

    参数:
        force_rebuild: 是否强制重建（忽略磁盘缓存的 vectorizer）
    """
    if not force_rebuild and os.path.exists(VECTORIZER_PATH):
        print("   📦 从磁盘加载 TF-IDF 词汇表...")
        with open(VECTORIZER_PATH, "rb") as f:
            vectorizer = pickle.load(f)
        print(f"   ✅ 词汇表加载完成: {len(vectorizer.get_feature_names_out())} 个特征")
        return TfidfEmbeddings(vectorizer=vectorizer)

    print("   🧠 初始化 TF-IDF 向量化器（纯 Python，零网络依赖）...")
    return TfidfEmbeddings()


def get_vectorstore():
    """
    获取 Chroma 向量数据库实例。

    逻辑：
    - 如果 chroma_db/ 目录已存在且有 vectorizer 缓存 → 从磁盘加载
    - 如果不存在 → 从头构建：加载 PDF → 切分 → TF-IDF 向量化 → 持久化

    返回:
        vectorstore: Chroma 向量数据库实例
    """
    # ── 步骤 1: 检查已有向量库 ──
    if os.path.exists(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        print(f"📦 检测到已有向量数据库: {PERSIST_DIR}")
        print("   正在加载...（如果不需要旧数据，请删除该目录后重新运行）")

        embeddings = get_embeddings()

        vectorstore = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
        )
        print(f"✅ 向量数据库加载完成")
        return vectorstore

    # ── 步骤 2: 从零构建 ──
    print("🔨 首次运行，开始构建向量数据库...\n")

    chunks = load_and_split(DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP)

    if not chunks:
        print("\n❌ 错误: data 目录中没有 PDF 文件或无法提取文字。")
        print("   请将 PDF 研报放入 data/ 文件夹后重新运行。")
        exit(1)

    embeddings = get_embeddings(force_rebuild=True)

    print(f"\n💾 正在将 {len(chunks)} 个文本块向量化并存入 Chroma...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )

    # 保存拟合好的 TF-IDF vectorizer，下次加载时复用
    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(embeddings.vectorizer, f)
    print(f"   💾 TF-IDF 词汇表已缓存至 {VECTORIZER_PATH}")

    print(f"✅ 向量数据库构建完成！已保存到 {PERSIST_DIR}/")
    print(f"   共收录 {len(chunks)} 个文本块，来自 {DATA_PATH}/ 中的 PDF。")
    return vectorstore


# ── 独立运行入口 ──
if __name__ == "__main__":
    if os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
        print(f"🗑️  已删除旧的向量库: {PERSIST_DIR}")
    get_vectorstore()
