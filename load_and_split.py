"""
PDF 加载与文本切分模块

类比：就像用 Pandas 的 read_csv() 把 CSV 文件读成 DataFrame，
这里用 pdfplumber 把 PDF 每一页读成一段文本，再用 LangChain 的
文本切分器切成适合 RAG 检索的小块（chunks）。
"""
import os
import pdfplumber
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def sanitize_text(text: str) -> str:
    """清除 PDF 中常见的无效 UTF-8 代理字符，防止 JSON 序列化/打印时报错"""
    return text.encode("utf-8", errors="replace").decode("utf-8", errors="replace")


def load_pdfs_from_directory(directory_path: str) -> list[Document]:
    """
    遍历指定目录下所有 PDF 文件，逐页提取文本，
    每一页生成一个 LangChain Document 对象。

    参数:
        directory_path: PDF 文件夹路径

    返回:
        documents: 包含所有页面的 Document 列表
                  每个 Document 的 metadata 存储了来源文件名和页码
    """
    documents = []

    # 列出目录中所有以 .pdf 结尾的文件（不区分大小写）
    pdf_files = [
        f for f in os.listdir(directory_path)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        print(f"⚠️  {directory_path} 目录下没有找到 PDF 文件！")
        print("   请将研报或年报 PDF 放入该文件夹后重试。")
        return documents

    print(f"📂 在 {directory_path} 中发现 {len(pdf_files)} 个 PDF 文件")

    for pdf_file in pdf_files:
        file_path = os.path.join(directory_path, pdf_file)
        print(f"   📄 正在解析: {pdf_file}")

        # 类比 Pandas 的 read_csv：pdfplumber.open() 打开 PDF，
        # 然后用 .pages 遍历每一页，类似遍历 DataFrame 的每一行
        try:
            with pdfplumber.open(file_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    # 提取该页的文本内容，清理无效字符
                    raw_text = page.extract_text() or ""
                    text = sanitize_text(raw_text)

                    # 跳过空页（比如全是图片的页面，pdfplumber 提取不到文字）
                    if not text.strip():
                        continue

                    # 每个页面创建一个 Document，
                    # metadata 存储来源信息——这在后续回答时会用来引用来源
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": pdf_file,   # 来源文件名
                            "page": page_num       # 页码
                        }
                    )
                    documents.append(doc)

            print(f"      ✓ 已提取 {len(pdf.pages)} 页，"
                  f"其中 {sum(1 for d in documents if d.metadata['source'] == pdf_file)} 页有文字内容")

        except Exception as e:
            print(f"      ✗ 解析 {pdf_file} 时出错: {e}")
            continue

    print(f"✅ 共生成 {len(documents)} 个 Document（每页一个）")
    return documents


def split_documents(
    docs: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> list[Document]:
    """
    使用 LangChain 的 RecursiveCharacterTextSplitter 将长文档切分为小块。

    为什么需要切分？
    1. 大模型一次能处理的上下文（context window）有限，不能把整本 PDF 塞进去。
    2. 小块文本的语义更集中，向量检索时更精准——就像淘宝搜索"蓝牙耳机"，
       你不会返回整个店铺页面，而是返回最相关的商品卡片。

    切分器的工作原理（递归字符切分）：
    - 先尝试用段落分隔符（\n\n）切分
    - 如果某段还是太长，再用句号、空格等更细粒度的分隔符继续切
    - 最终确保每个 chunk 的长度 ≤ chunk_size

    参数:
        docs: 切分前的 Document 列表
        chunk_size: 每个 chunk 的最大长度（字符数）
        chunk_overlap: 相邻 chunk 之间的重叠长度

    返回:
        切分后的 Document 列表（metadata 包含来源和页码）
    """
    # 初始化切分器——RecursiveCharacterTextSplitter 是 LangChain 中最常用的切分器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # 分隔符优先级：段落 → 句子 → 短句（中文兼容）
        separators=["\n\n", "\n", "。", "；", "，", " ", ""]
    )

    print(f"🔪 正在切分 {len(docs)} 个 Document...")
    print(f"   chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    # split_documents 会自动保留每个 Document 的 metadata，
    # 所以切分后的 chunk 仍然知道来自哪个 PDF 的第几页
    chunks = text_splitter.split_documents(docs)

    print(f"✅ 切分完成：{len(docs)} 个页面 → {len(chunks)} 个文本块 (chunk)")
    return chunks


# ── 快捷入口：一次性完成「加载 + 切分」────────────────
def load_and_split(directory_path: str,
                   chunk_size: int = 500,
                   chunk_overlap: int = 50) -> list[Document]:
    """
    加载 PDF 并切分——把两个步骤打包成一个函数，方便 build_vectorstore 调用。

    类比：就像做数据预处理 pipeline 中的 load_and_preprocess()，
    封装了多个数据清洗步骤。
    """
    docs = load_pdfs_from_directory(directory_path)
    if not docs:
        return []
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    return chunks
