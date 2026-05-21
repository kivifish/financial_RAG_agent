"""
项目全局配置
这里集中管理所有可调参数，方便一处修改、全局生效。
"""
import os
from dotenv import load_dotenv

# 加载 .env 文件到系统环境变量
load_dotenv()

# ── HuggingFace 国内镜像配置 ────────────────────────────
# 国内网络无法直连 huggingface.co，使用 hf-mirror.com 镜像
# 类比：就像用清华镜像 pip install，这里用 HuggingFace 镜像下载模型
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ── 路径配置 ────────────────────────────────────────────
# PDF 研报存放的文件夹（相对路径，从项目根目录算起）
DATA_PATH = "./data"

# Chroma 向量数据库的持久化目录（相当于 SQLite 的 .db 文件存储位置）
PERSIST_DIR = "./chroma_db"

# ── 文本切分配置 ────────────────────────────────────────
# 每个 chunk 的最大字符数（500 个字符 ≈ 中文约 250 字，相当于一段话的长度）
CHUNK_SIZE = 500
# 相邻 chunk 之间的重叠字符数（防止关键信息被切断在边界上）
CHUNK_OVERLAP = 50

# ── 检索配置 ────────────────────────────────────────────
# 每次检索返回最相似的 K 个文本块（类似推荐系统中 Top-K 召回）
K_RETRIEVALS = 4

# ── Embedding 配置 ──────────────────────────────────────
# 直接用 DeepSeek API 做 embedding，彻底避免本地 tokenizers 兼容问题（真的搞了很久！）
# DeepSeek 兼容 OpenAI embeddings 接口
EMBEDDING_MODEL = "deepseek-chat"  # DeepSeek embedding 模型名

# ── DeepSeek API 配置 ───────────────────────────────────
# DeepSeek 兼容 OpenAI 接口格式，所以可以用 openai 库调用
# 类比：就像调用 PPO 时的 predict 方法一样，我们通过 URL + API Key 远程调用模型
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
# 从环境变量安全读取 API Key（绝不硬编码在代码中！）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
