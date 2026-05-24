# 📊 金融研报 RAG 问答助手

基于 **RAG（检索增强生成）** 技术的金融研报智能问答系统。将 PDF 研报/年报处理后，用自然语言提问，系统自动检索相关内容并用 DeepSeek 大模型生成准确答案。

## 🧠 核心原理

```
用户问题 → Agent 思考 → 需要检索？→ 调用 RAG 工具检索研报
              ↑              ↓ 不需要              ↓ 观察结果
              └——— 信息不足则继续 ←——— 信息充足 → 输出答案（附引用来源）
```

**第一版**是普通 RAG：检索一次 → 直接回答（固定流水线）。  
**当前版本**升级为 Agent 模式：LLM 自己判断是否需要检索、检索几次、什么时候停。简单问题一轮过，复杂问题自动拆多步。

## 🚀 快速开始

### 1. 环境准备

```bash
# 创建虚拟环境（Python 3.9+ 推荐）
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# 安装依赖（使用清华镜像加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 2. 配置 API Key

编辑 `.env` 文件，填入你的 DeepSeek API Key：

```
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> 💡 获取 API Key：访问 [DeepSeek 开放平台](https://platform.deepseek.com/)，注册后在 API Keys 页面创建。

### 3. 放置 PDF 文件

将需要分析的年报或券商研报 PDF 文件放入 `data/` 文件夹。

### 4. 启动问答

```bash
python app.py
```

首次运行会自动提取 PDF 文本并构建 TF-IDF 向量数据库，请耐心等待。

## 📝 示例问题

- "茅台2024年的营收是多少？"
- "宁德时代在研发投入方面有哪些表述？"
- "分析报告中提到的核心风险有哪些？"
- "该公司的核心竞争力是什么？"

## 🗂️ 项目结构

```
rag_research_helper/
├── .env                  # API Key 配置（不提交到 Git）
├── .gitignore            # Git 忽略规则
├── requirements.txt      # Python 依赖列表
├── config.py             # 全局配置（路径、参数等）
├── load_and_split.py     # PDF 加载与文本切分
├── build_vectorstore.py  # 向量数据库构建（TF-IDF + Chroma）
├── agent.py              # Agent 模块（ReAct 循环 + 工具调用）
├── qa_chain.py           # [旧版] 普通 RAG 问答链（保留参考）
├── app.py                # 主程序入口（交互式终端）
├── data/                 # 存放 PDF 研报
│   └── README.md
└── chroma_db/            # 向量数据库持久化目录（自动生成）
```

## ❓ 常见问题

| 问题 | 解决方法 |
|------|---------|
| `ModuleNotFoundError` | 确保已激活虚拟环境并运行 `pip install -r requirements.txt` |
| API 连接失败 | 检查网络、API Key 是否正确、余额是否充足 |
| 首次运行很慢 | 正在提取 PDF 文本并构建 TF-IDF + 向量库，仅首次需要 |
| PDF 无法解析 | 确保 PDF 是文字型（非扫描图片），pdfplumber 无法提取图片中的文字 |
| 回答不准确 | 尝试调整 `config.py` 中的 `CHUNK_SIZE` 和 `K_RETRIEVALS` |

## 🔧 技术栈

- **PDF 解析**: pdfplumber
- **文本切分**: LangChain RecursiveCharacterTextSplitter
- **向量化**: TF-IDF（sklearn TfidfVectorizer，纯 Python，零网络依赖）
- **向量数据库**: Chroma（持久化到本地）
- **大模型**: DeepSeek API（兼容 OpenAI 接口）
- **Agent 框架**: LangChain AgentExecutor + Tool Calling（ReAct 模式）
- **编排框架**: LangChain

---

📧 有任何问题欢迎交流！
