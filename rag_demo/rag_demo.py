"""RAG Demo — 体验 LLM + 检索增强生成（RAG）的效果。

知识库：自动加载 .qoder/skills/ 下所有设备文档（SKILL.md、protocols.md 等）
向量库：ChromaDB（内存）
LLM：复用项目 .env 中的 DeepSeek API

Embedding 方案（三选一）：
  • semantic（默认）: BAAI/bge-small-zh-v1.5，512 维，中文优化，需 torch
  • onnx           : all-MiniLM-L6-v2，384 维，纯 onnxruntime，无需 torch
  • tfidf          : TF-IDF 稀疏向量，零依赖

用法：
    python rag_demo/rag_demo.py              # 语义 Embedding（默认，BGE 中文模型）
    python rag_demo/rag_demo.py --onnx       # ONNX（轻量，无需 torch）
    python rag_demo/rag_demo.py --tfidf      # TF-IDF（对比用）

交互命令：
    直接输入问题      → RAG 问答（检索 + LLM）
    /raw <问题>       → 仅用 LLM 回答（无检索，对比用）
    /docs             → 查看已加载的文档数
    /q 或 quit       → 退出
"""

import os
import sys
import time
import re

# ── HuggingFace 国内镜像（避免网络不可达）──────────────
if not os.getenv("HF_ENDPOINT"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ── 项目根目录 & .env ──────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


# ══════════════════════════════════════════════════════
#  Embedding 方案 A：ONNX 语义 Embedding（无需 torch）
# ══════════════════════════════════════════════════════

class OnnxEmbedding:
    """纯 ONNX 推理的语义 Embedding，不依赖 torch / sentence-transformers。

    使用 ChromaDB 已下载的 all-MiniLM-L6-v2 ONNX 模型，
    通过 onnxruntime + tokenizers 直接做 tokenize → 推理 → mean pooling。

    依赖：onnxruntime + tokenizers（轻量，无 torch）
    """

    # ChromaDB 自动下载的模型路径
    MODEL_DIR = os.path.expanduser(
        "~/.cache/chroma/onnx_models/all-MiniLM-L6-v2/onnx"
    )

    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        # 1. 加载 tokenizer（HuggingFace tokenizers 库，纯 Rust 实现）
        tokenizer_path = os.path.join(self.MODEL_DIR, "tokenizer.json")
        self._tokenizer = Tokenizer.from_file(tokenizer_path)
        self._tokenizer.enable_padding()
        self._tokenizer.enable_truncation(max_length=512)

        # 2. 加载 ONNX 模型（onnxruntime，C++ 推理引擎）
        model_path = os.path.join(self.MODEL_DIR, "model.onnx")
        self._session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],  # 纯 CPU，无需 CUDA
        )
        self._dim = 384  # all-MiniLM-L6-v2 输出维度

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        """批量编码文本 → 384 维归一化向量。"""
        # ① Tokenize（Rust 实现，极快）
        encodings = self._tokenizer.encode_batch(texts)

        # ② 构造 ONNX 输入
        input_ids = [e.ids for e in encodings]
        attention_mask = [e.attention_mask for e in encodings]
        token_type_ids = [[0] * len(ids) for ids in input_ids]  # BERT 标准输入，全 0

        # ③ ONNX 推理（C++ 引擎，无需 torch）
        outputs = self._session.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )
        # outputs[0] shape: (batch, seq_len, hidden_dim=384)
        token_embeddings = outputs[0]

        # ④ Mean Pooling：对 attention_mask=1 的 token 取平均
        results = []
        for i, mask in enumerate(attention_mask):
            mask_sum = sum(mask)
            if mask_sum == 0:
                results.append([0.0] * self._dim)
                continue
            pooled = [0.0] * self._dim
            for d in range(self._dim):
                s = 0.0
                for j, m in enumerate(mask):
                    if m:
                        s += float(token_embeddings[i][j][d])  # np.float32 → Python float
                pooled[d] = s / mask_sum
            # ⑤ L2 归一化
            norm = sum(v * v for v in pooled) ** 0.5
            if norm > 0:
                pooled = [v / norm for v in pooled]
            results.append(pooled)

        return results

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode_batch(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode_batch([text])[0]


# ══════════════════════════════════════════════════════
#  Embedding 方案 B：语义 Embedding（BGE，中文优化）
# ══════════════════════════════════════════════════════

class SemanticEmbedding:
    """语义 Embedding：BAAI/bge-small-zh-v1.5（512 维稠密向量）。

    专为中文优化的 Sentence-BERT 模型，理解语义而非仅匹配关键词。
    同义词、近义词、上下位词都能正确匹配。

    依赖：sentence-transformers + torch（约 2GB）
    """

    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        # bge 模型推荐 query 加前缀 "为这个句子生成表示以检索相关文档："
        return self._model.encode(
            [f"为这个句子生成表示以检索相关文档：{text}"],
            normalize_embeddings=True,
        ).tolist()[0]


# ══════════════════════════════════════════════════════
#  Embedding 方案 C：TF-IDF（稀疏向量，关键词匹配）
# ══════════════════════════════════════════════════════

class TfidfEmbedding:
    """轻量 TF-IDF embedding，纯本地计算，零依赖。"""

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._dim: int = 0

    def fit(self, texts: list[str]):
        import math
        all_tokens = [self._tokenize(t) for t in texts]
        token_freq: dict[str, int] = {}
        for tokens in all_tokens:
            for tok in set(tokens):
                token_freq[tok] = token_freq.get(tok, 0) + 1
        n = len(texts)
        idx = 0
        for token, df in sorted(token_freq.items()):
            if 2 <= df <= n * 0.8:
                self._vocab[token] = idx
                self._idf[token] = math.log(n / (1 + df))
                idx += 1
        self._dim = idx

    def embed(self, text: str) -> list[float]:
        import math
        vec = [0.0] * self._dim
        tokens = self._tokenize(text)
        if not tokens:
            return vec
        tf: dict[str, int] = {}
        for tok in tokens:
            tf[tok] = tf.get(tok, 0) + 1
        total = len(tokens)
        for tok, count in tf.items():
            if tok in self._vocab:
                vec[self._vocab[tok]] = (count / total) * self._idf[tok]
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r'[\u4e00-\u9fff]|[a-zA-Z0-9]+', text.lower())


# ══════════════════════════════════════════════════════
#  知识库构建 & 向量存储
# ══════════════════════════════════════════════════════

def load_documents(skills_dir: str) -> list:
    """从 skills 目录加载所有 .md 文件。"""
    from langchain_core.documents import Document

    docs = []
    for root, dirs, files in os.walk(skills_dir):
        dirs[:] = [d for d in dirs if d not in ("app", "__pycache__")]
        for f in files:
            if f.endswith(".md"):
                filepath = os.path.join(root, f)
                with open(filepath, "r", encoding="utf-8") as fh:
                    content = fh.read()
                rel_path = os.path.relpath(filepath, skills_dir)
                docs.append(Document(
                    page_content=content,
                    metadata={"source": rel_path, "filename": f},
                ))
    return docs


def split_documents(docs: list, chunk_size=500, overlap=50) -> list:
    """将文档切分为更小的 chunk。"""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n## ", "\n### ", "\n---", "\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def build_vectorstore(chunks: list, mode="semantic", verbose=True):
    """构建 ChromaDB 向量存储。

    Args:
        mode: "semantic" — BAAI/bge-small-zh-v1.5 语义 embedding（512维，中文优化）
              "onnx"     — all-MiniLM-L6-v2 ONNX（384维，无需 torch）
              "tfidf"    — TF-IDF 稀疏向量（关键词匹配）
    """
    from langchain_chroma import Chroma
    import chromadb

    texts = [c.page_content for c in chunks]

    # 选择 embedding 方案
    if mode == "onnx":
        embedding = OnnxEmbedding()
        if verbose:
            print("   Embedding: all-MiniLM-L6-v2 ONNX (384 维, 纯 onnxruntime 推理)")
    elif mode == "semantic":
        embedding = SemanticEmbedding()
        if verbose:
            print("   Embedding: BAAI/bge-small-zh-v1.5 (512 维中文语义向量)")
    else:
        embedding = TfidfEmbedding()
        embedding.fit(texts)
        if verbose:
            print(f"   Embedding: TF-IDF ({embedding._dim} 维稀疏向量)")

    if verbose:
        print(f"   向量化 {len(texts)} 个 chunks...")

    # 写入 ChromaDB
    client = chromadb.Client()
    collection = client.create_collection(
        name="lab_equipment",
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        documents=texts,
        metadatas=[c.metadata for c in chunks],
        ids=[f"doc_{i}" for i in range(len(chunks))],
        embeddings=embedding.embed_documents(texts),
    )

    # LangChain 包装
    return Chroma(
        client=client,
        collection_name="lab_equipment",
        embedding_function=embedding,
    )


# ══════════════════════════════════════════════════════
#  RAG 核心
# ══════════════════════════════════════════════════════

def rag_answer(question: str, vectorstore, llm, k=4):
    """RAG 问答：检索相关文档 → 拼接上下文 → LLM 生成回答。"""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    # 1. 检索
    t0 = time.time()
    retrieved = vectorstore.similarity_search(question, k=k)
    search_time = time.time() - t0

    # 2. 构建上下文
    context_parts = []
    sources = []
    for i, doc in enumerate(retrieved, 1):
        source = doc.metadata.get("source", "unknown")
        snippet = doc.page_content[:600]
        context_parts.append(f"--- 来源 [{i}]: {source} ---\n{snippet}")
        sources.append(source)

    context = "\n\n".join(context_parts) if context_parts else "(未找到相关文档)"

    # 3. LLM 生成
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "你是一个实验室设备知识助手。请**仅基于以下检索到的参考文档**回答问题。\n"
         "如果参考文档中没有相关信息，请明确说明「参考文档中未找到相关信息」。\n"
         "不要编造不在文档中的内容。\n\n"
         "## 参考文档\n{context}"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})
    return answer, sources, search_time


def raw_llm_answer(question: str, llm):
    """纯 LLM 回答（无 RAG），用于对比效果。"""
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个实验室设备知识助手。请根据你的知识回答以下问题。"),
        ("human", "{question}"),
    ])

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})


# ══════════════════════════════════════════════════════
#  主程序
# ══════════════════════════════════════════════════════

def main():
    from langchain_openai import ChatOpenAI

    # 解析模式参数
    if "--onnx" in sys.argv:
        mode = "onnx"
    elif "--tfidf" in sys.argv:
        mode = "tfidf"
    else:
        mode = "semantic"

    mode_labels = {
        "semantic": "语义 Embedding (BAAI/bge-small-zh-v1.5, 512维)",
        "onnx": "ONNX 语义 Embedding (all-MiniLM-L6-v2, 384维)",
        "tfidf": "TF-IDF (稀疏向量)",
    }
    mode_label = mode_labels[mode]

    print("=" * 60)
    print(f"  🔬 RAG Demo — 实验室设备知识库问答")
    print(f"  Embedding: {mode_label}")
    print("=" * 60)

    # ── 1. 初始化 LLM ──
    print("\n[1/4] 初始化 LLM...")
    llm = ChatOpenAI(
        model=os.getenv("XDL_LLM_MODEL"),
        api_key=os.getenv("XDL_LLM_API_KEY"),
        openai_api_base=os.getenv("XDL_LLM_API_BASE"),
        temperature=0.3,
    )
    print(f"   模型: {os.getenv('XDL_LLM_MODEL')}")

    # ── 2. 加载文档 ──
    skills_dir = os.path.join(PROJECT_ROOT, ".qoder", "skills")
    print(f"\n[2/4] 加载知识库: {skills_dir}")
    docs = load_documents(skills_dir)
    print(f"   加载了 {len(docs)} 个文档文件")

    # ── 3. 切分文档 ──
    print(f"\n[3/4] 切分文档...")
    chunks = split_documents(docs)
    print(f"   切分为 {len(chunks)} 个 chunks (每块约 500 字符)")

    # ── 4. 构建向量索引 ──
    print(f"\n[4/4] 构建向量索引...")
    t0 = time.time()
    vectorstore = build_vectorstore(chunks, mode=mode)
    build_time = time.time() - t0
    print(f"   索引构建耗时: {build_time:.2f}s")

    # ── 交互式问答 ──
    print(f"\n{'=' * 60}")
    print("  ✅ RAG 系统就绪！")
    print(f"{'=' * 60}")
    print("  命令:")
    print("    直接输入问题     → RAG 问答（检索 + LLM）")
    print("    /raw <问题>      → 仅 LLM 回答（无检索，对比效果）")
    print("    /docs            → 查看已加载文档")
    print("    /q 或 quit       → 退出")
    print(f"{'=' * 60}\n")

    while True:
        try:
            question = input("🧑 你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue
        if question.lower() in ("/q", "quit", "exit"):
            print("再见！")
            break

        if question == "/docs":
            print(f"\n📚 已加载 {len(docs)} 个文档，{len(chunks)} 个 chunks:")
            for doc in docs:
                print(f"   - {doc.metadata['source']}  ({len(doc.page_content)} 字符)")
            print()
            continue

        if question.startswith("/raw "):
            raw_q = question[5:].strip()
            print(f"\n🤖 [纯 LLM，无检索]")
            t0 = time.time()
            answer = raw_llm_answer(raw_q, llm)
            print(f"   {answer}")
            print(f"\n   ⏱ 耗时: {time.time() - t0:.1f}s\n")
            continue

        # ── RAG 问答 ──
        print(f"\n🔍 检索中...")
        t0 = time.time()
        answer, sources, search_time = rag_answer(question, vectorstore, llm)
        total_time = time.time() - t0

        print(f"📄 检索到 {len(sources)} 个相关片段 ({search_time:.3f}s):")
        for i, src in enumerate(sources, 1):
            print(f"   [{i}] {src}")

        print(f"\n🤖 回答:")
        print(f"   {answer}")
        print(f"\n   ⏱ 检索: {search_time:.3f}s | 总计: {total_time:.1f}s")
        print()


if __name__ == "__main__":
    main()
