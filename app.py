"""
主程序入口 —— 命令行交互式 RAG 问答助手

启动方式：在项目根目录（rag_research_helper/）下运行：
    python app.py

整个流程：
  PDF → 文本块 → 向量化 → 存入 Chroma → 接收问题 → 检索 → DeepSeek 回答

类比：就像你开发的游戏 AI ——
  初始化环境 → 加载模型 → 进入循环（观察 → 推理 → 行动）→ 输出结果
"""
import os
import sys

# ── Windows 终端 UTF-8 编码修复 ──
# Windows 默认用 GBK 编码，遇到 emoji 会报 UnicodeEncodeError
# 这里强制 stdout 使用 UTF-8，类似于你在 Pandas 里设置 encoding="utf-8"
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 加载环境变量（必须在导入 config 之前，确保 API Key 可用）
from dotenv import load_dotenv
load_dotenv()

from config import DEEPSEEK_API_KEY
from build_vectorstore import get_vectorstore
from agent import create_agent, query_agent


def check_env():
    """检查必要的环境配置是否就绪"""
    if not DEEPSEEK_API_KEY or DEEPSEEK_API_KEY == "your_api_key_here":
        print("❌ 错误: 未设置 DEEPSEEK_API_KEY")
        print("   请在 .env 文件中填入你的 DeepSeek API Key，格式：")
        print("   DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx")
        print("\n   💡 获取方式: 访问 https://platform.deepseek.com/ 注册并获取 API Key")
        sys.exit(1)


def sanitize_text(text: str) -> str:
    """清除文本中的无效 UTF-8 代理字符，防止打印时报错"""
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def print_welcome():
    """打印欢迎信息和项目简介"""
    print("=" * 60)
    print("   📊 金融研报 RAG 问答助手")
    print("=" * 60)
    print("   功能: 上传研报 PDF → 智能检索 → AI 回答")
    print("   数据: data/ 文件夹中的 PDF")
    print("   模型: DeepSeek + TF-IDF（Agent ReAct 模式）")
    print()
    print("   💡 输入问题开始对话，输入 exit 退出")
    print("=" * 60)
    print()


def main():
    """主函数：初始化系统 → 进入交互问答循环"""
    # ── 第 0 步：环境检查 ──
    check_env()

    # ── 第 1 步：初始化向量数据库 ──
    # 这一步会加载现有向量库，或从 PDF 重新构建
    # 类比：加载游戏环境 env = Breakout()
    print("🔧 正在初始化向量数据库...")
    try:
        vectorstore = get_vectorstore()
    except Exception as e:
        print(f"❌ 向量数据库初始化失败: {e}")
        sys.exit(1)

    # ── 第 2 步：创建 Agent ──
    # Agent 模式：LLM 拿到问题后自己决定检索几次、什么时候停
    # 类比：从固定策略升级到了让 AI 自己决策
    print("\n🤖 正在连接 DeepSeek 大模型...")
    try:
        agent = create_agent(vectorstore)
        print("✅ Agent 就绪！\n")
    except Exception as e:
        print(f"❌ 创建问答链失败: {e}")
        print("   请检查: 1) 网络连接  2) API Key 是否正确  3) API 余额是否充足")
        sys.exit(1)

    # ── 第 3 步：进入交互循环 ──
    # 类比训练循环中的 for step in range(num_steps):
    print_welcome()

    while True:
        try:
            # 等待用户输入问题
            question = input("🔍 请输入您的问题: ").strip()

            if not question:
                continue

            if question.lower() in ["exit", "quit", "退出", "q"]:
                print("👋 再见！")
                break

            # 调用问答链获取答案
            print("   ⏳ 正在检索相关资料并生成答案...\n")
            result = query_agent(agent, question)

            # 打印回答（sanitize 防止 PDF 中的脏字符导致打印崩溃）
            print("📝 回答:")
            print("-" * 50)
            print(sanitize_text(result["answer"]))
            print("-" * 50)

            # 打印引用来源
            if result["sources"]:
                print("\n📚 引用来源（基于以下研报内容）：")
                for source in result["sources"]:
                    print(f"   · {sanitize_text(source)}")
            else:
                print("\n⚠️  未能找到明确来源")

            print()

        except EOFError:
            # 非交互式环境（如管道输入、后台运行），正常退出
            print("\n👋 检测到非交互式环境，程序退出。")
            break
        except KeyboardInterrupt:
            print("\n\n👋 按 Ctrl+C 退出，再见！")
            break
        except Exception as e:
            print(f"\n⚠️  出错了: {e}")
            print("   请重新输入问题，或输入 exit 退出。\n")


if __name__ == "__main__":
    main()
