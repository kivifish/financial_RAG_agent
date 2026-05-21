"""
问答链模块 —— RAG 的核心组装

这里把「检索器」和「大模型」串联成一个问答链路：

  用户问题 → 检索器（从向量库找 Top-K 相关文本块）
           → 拼接到 prompt 模板 → 发给 DeepSeek → 返回答案

类比强化学习项目：
  - 检索器 ≈ 从经验池中采样最相关的 transition
  - 大模型 ≈ PPO 的 actor 网络，根据输入（状态+经验）输出行为（答案）

整个链路 = 采样 + 推理，这就是 RAG 的本质！
"""
import json
from langchain_classic.chains import RetrievalQA
from langchain_classic.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from config import K_RETRIEVALS, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, DEEPSEEK_API_KEY

# ── Monkey-patch: 修复 OpenAI JSON 序列化中的 surrogate 字符问题 ──
# PDF 文本可能含无效 Unicode 代理字符（lone surrogates），Python 的 UTF-8
# 编码器拒绝处理这类字符。在 json.dumps 后、.encode() 前清理一遍。
_original_openapi_dumps = None


def _install_openai_json_fix():
    """替换 openai 的 openapi_dumps，在 .encode() 前清除 surrogate 字符"""
    global _original_openapi_dumps
    try:
        from openai._utils._json import openapi_dumps as _original, _CustomEncoder
        _original_openapi_dumps = _original

        def safe_openapi_dumps(obj):
            json_str = json.dumps(
                obj,
                cls=_CustomEncoder,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            )
            cleaned = json_str.encode("utf-8", errors="replace").decode("utf-8")
            return cleaned.encode()

        import openai._utils._json
        openai._utils._json.openapi_dumps = safe_openapi_dumps
        import openai._base_client
        if hasattr(openai._base_client, "openapi_dumps"):
            openai._base_client.openapi_dumps = safe_openapi_dumps
    except ImportError:
        pass


_install_openai_json_fix()


# ── 自定义 Prompt 模板 ──────────────────────────────────
# 这个模板告诉 DeepSeek：「你是金融助手，只能根据我给的资料回答，别瞎编」
# {context} 和 {question} 是占位符，LangChain 会在运行时自动填入
RAG_PROMPT_TEMPLATE = """
你是一个金融研报问答助手。只根据以下提供的上下文回答问题。
如果上下文里没有答案，请直接说"根据给定研报无法回答该问题"。不要编造信息。
回答时尽量引用来源（文件名和页码）。

上下文：
{context}

问题：{question}

回答：
"""

RAG_PROMPT = PromptTemplate(
    template=RAG_PROMPT_TEMPLATE,
    input_variables=["context", "question"],
)


def create_qa_chain(vectorstore, llm=None):
    """
    创建 RAG 问答链。

    参数:
        vectorstore: Chroma 向量数据库实例（用作检索器）
        llm: 大模型实例，如果不传则自动创建 ChatOpenAI

    返回:
        qa_chain: RetrievalQA 链，可以直接调用 run() 回答问题
    """
    # ── 如果没有传入 LLM，则创建一个 ──
    # temperature=0 意味着模型每次输出几乎一致（确定性），
    # 这对问答场景很重要——我们不想要"创意"，我们要"准确"
    if llm is None:
        llm = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            openai_api_key=DEEPSEEK_API_KEY,
            openai_api_base=DEEPSEEK_BASE_URL,  # 指向 DeepSeek 而非 OpenAI
            temperature=0,
        )

    # ── 构建 RetrievalQA 链 ──
    # 类比：把「检索引擎」和「LLM 推理」焊在一起
    # 1. retriever: 负责从向量库中检索最相关的 K 个文本块
    # 2. chain_type="stuff": 把检索到的 K 个块"塞"到 prompt 里一起发给 LLM
    # 3. return_source_documents=True: 返回引用来源，方便追溯
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",   # "stuff" 策略：把所有相关文本块拼起来塞进 context
        retriever=vectorstore.as_retriever(
            search_kwargs={"k": K_RETRIEVALS}  # 每次检索返回 4 个最相似的 chunk
        ),
        return_source_documents=True,  # 关键！这样我们能看到答案的来源
        chain_type_kwargs={
            "prompt": RAG_PROMPT  # 使用我们自定义的系统提示
        },
    )

    return qa_chain


def answer_question(chain, question: str) -> dict:
    """
    使用问答链回答单个问题。

    参数:
        chain: RetrievalQA 链实例
        question: 用户输入的自然语言问题

    返回:
        dict: {"answer": 答案文本, "sources": 来源信息列表}
    """
    # chain.invoke() 的内部流程：
    # ① 把 question 向量化
    # ② 在向量库中检索 Top-4 相似 chunk
    # ③ 填充 RAG_PROMPT_TEMPLATE
    # ④ 发送给 DeepSeek 生成答案
    # ⑤ 返回答案 + 来源文档
    result = chain.invoke({"query": question})

    # 从返回结果中提取答案文本
    answer = result["result"]

    # 提取引用来源（去重、排序）
    sources = []
    seen = set()
    for doc in result["source_documents"]:
        source_key = f"{doc.metadata['source']} 第{doc.metadata['page']}页"
        if source_key not in seen:
            sources.append(source_key)
            seen.add(source_key)

    return {
        "answer": answer,
        "sources": sources,
    }
