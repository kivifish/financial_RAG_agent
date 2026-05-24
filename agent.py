"""
Agent 模块 —— ReAct 模式的「思考→行动→观察→再思考」循环

架构变化：
  旧：用户问题 → 检索一次 → LLM 回答（固定流水线）
  新：用户问题 → LLM 思考 → 决定是否检索/检索什么 → 执行 → 观察结果
              ↑                                                 |
              └──────────── 信息不足则继续 ─────────────────────┘
              ↓ 信息充足
            输出最终答案

技术：LangChain create_react_agent + AgentExecutor
LLM 按固定文本格式输出 Thought/Action/Action Input，系统解析后执行工具
"""
import json
import re
from langchain_classic.agents import create_react_agent, AgentExecutor
from langchain_classic.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from config import (
    K_RETRIEVALS, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL, DEEPSEEK_API_KEY,
    MAX_AGENT_STEPS, AGENT_VERBOSE,
)


# ── Monkey-patch: 修复 OpenAI JSON 序列化中的 surrogate 字符问题 ──
def _install_openai_json_fix():
    try:
        from openai._utils._json import openapi_dumps as _original, _CustomEncoder

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


# ── Agent 系统提示词（ReAct 文本格式）──────────────────
AGENT_SYSTEM_PROMPT = """你是一个金融研报问答助手。你可以使用以下工具来检索研报中的相关信息：

{tools}

严格按以下格式回答（关键词 Action/Action Input/Final Answer 必须用英文）：

Question: 用户的问题
Thought: 分析当前需要检索什么信息
Action: 要使用的工具名（[{tool_names}] 之一）
Action Input: 传给工具的查询关键词
Observation: 工具返回的结果
... (Thought/Action/Action Input/Observation 可重复多次)
Thought: 信息充足，可以给出最终答案了
Final Answer: 基于检索结果用中文给出最终答案

重要规则：
- Action 和 Action Input 关键词必须用英文，不能翻译成中文
- 回答必须基于检索到的实际内容，绝对不要编造数据
- 多次检索后仍找不到，直接说"根据给定研报无法回答该问题"
- 回答时注明引用来源（文件名和页码），方便用户追溯
- 闲聊类问题（打招呼、问能力等）直接回答即可，不需要检索

Question: {input}
{agent_scratchpad}"""


def create_agent(vectorstore, llm=None):
    """
    创建 AgentExecutor 实例。

    参数:
        vectorstore: Chroma 向量数据库实例
        llm: 大模型实例，不传则自动创建 ChatOpenAI（指向 DeepSeek）

    返回:
        AgentExecutor 实例
    """
    if llm is None:
        llm = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            openai_api_key=DEEPSEEK_API_KEY,
            openai_api_base=DEEPSEEK_BASE_URL,
            temperature=0,
        )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": K_RETRIEVALS}
    )

    # ── 注册工具：RAG 检索器 ──
    @tool
    def search_research_reports(query: str) -> str:
        """在金融研报中搜索信息。输入自然语言查询（如"茅台2024年营收"），返回相关段落的全文和来源（文件名+页码）。适用于：查找财务数据、公司业务信息、行业分析、风险提示等任何需要从研报中获取的内容。"""
        docs = retriever.invoke(query)
        if not docs:
            return "未找到与「{}」相关的信息，请尝试更换关键词。".format(query)
        parts = []
        for doc in docs:
            src = doc.metadata.get("source", "未知")
            pg = doc.metadata.get("page", "未知")
            parts.append("[来源: {} 第{}页]\n{}".format(src, pg, doc.page_content))
        return "\n\n---\n\n".join(parts)

    tools = [search_research_reports]

    # ── 组装 Prompt 模板 ──
    # 使用字符串模板（非 ChatPromptTemplate），ReAct 格式的
    # agent_scratchpad 是字符串变量，由 AgentExecutor 自动填充
    prompt = PromptTemplate.from_template(AGENT_SYSTEM_PROMPT)

    # ── 创建 Agent + Executor ──
    agent = create_react_agent(llm, tools, prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=MAX_AGENT_STEPS,
        verbose=AGENT_VERBOSE,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )

    return executor


def _parse_sources(intermediate_steps) -> list:
    """从 Agent 中间步骤中提取引用来源"""
    sources = []
    seen = set()
    for action, observation in intermediate_steps:
        if action.tool == "search_research_reports":
            matches = re.findall(r"\[来源: (.+?) 第(\d+)页\]", observation)
            for src, pg in matches:
                key = "{} 第{}页".format(src, pg)
                if key not in seen:
                    sources.append(key)
                    seen.add(key)
    return sources


def query_agent(executor, question: str) -> dict:
    """
    使用 Agent 回答一个问题。

    AgentExecutor 内部自动完成 ReAct 循环：
      LLM 输出 Thought → 如果是工具调用 → 执行工具 → 把结果喂回 LLM → 继续
                      → 如果是最终答案 → 结束循环，返回给用户

    参数:
        executor: AgentExecutor 实例
        question: 用户输入的自然语言问题

    返回:
        dict: {"answer": 答案文本, "sources": 来源列表}
    """
    result = executor.invoke({"input": question})
    return {
        "answer": result["output"],
        "sources": _parse_sources(result.get("intermediate_steps", [])),
    }
