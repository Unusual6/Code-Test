"""
PagedAttention 示例代码
=======================
使用 vLLM（官方实现）演示 PagedAttention 的核心用法。

安装依赖（需要 GPU + CUDA）:
    pip install torch vllm

如果只需要理解原理，可以只看 Part 2（纯 PyTorch 手动实现）。

参考论文: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
         (Kwon et al., SOSP 2023)
"""

# ============================================================
# Part 1: 使用 vLLM 官方包运行 PagedAttention（需要 GPU）
# ============================================================

def demo_vllm_paged_attention():
    """
    使用 vLLM 的 LLM 接口进行文本生成。
    vLLM 内部自动使用 PagedAttention 管理 KV Cache。
    """
    from vllm import LLM, SamplingParams

    # 使用小模型方便演示，也可以换成更大的模型
    model_name = "facebook/opt-125m"

    # 创建 LLM 实例 —— vLLM 会自动管理 PagedAttention 的 block 分配
    llm = LLM(
        model=model_name,
        tensor_parallel_size=1,       # GPU 并行数
        gpu_memory_utilization=0.7,   # GPU 显存使用比例
        max_model_len=512,            # 最大序列长度
        block_size=16,                # 每个 PagedAttention block 的 token 数
        dtype="float16",
    )

    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=64,
    )

    # 准备 prompts（支持批量，vLLM 会用 PagedAttention 高效调度）
    prompts = [
        "The future of artificial intelligence is",
        "In a world where robots and humans coexist,",
        "PagedAttention improves LLM serving by",
    ]

    # 批量生成 —— 内部使用 PagedAttention 管理 KV Cache
    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        print(f"\n--- Prompt {i+1} ---")
        print(f"输入: {output.prompt}")
        print(f"生成: {output.outputs[0].text}")


# ============================================================
# Part 2: 纯 PyTorch 手动实现 PagedAttention 核心逻辑（可学习原理）
# ============================================================

def demo_paged_attention_from_scratch():
    """
    用纯 PyTorch 手动模拟 PagedAttention 的核心机制：
    1. 将 KV Cache 分成固定大小的 block
    2. 通过 block table 映射逻辑位置到物理位置
    3. 在计算 attention 时，按 block 读取 KV Cache
    """
    import torch
    import math

    torch.manual_seed(42)

    # ---------- 超参数 ----------
    batch_size = 2          # 批量大小
    num_heads = 4           # 注意力头数
    head_dim = 16           # 每个头的维度
    block_size = 4          # 每个 block 存储的 token 数
    num_blocks = 8          # 物理 block 总数
    seq_len_q = 3           # query 序列长度
    seq_len_kv = 10         # key/value 序列长度（KV Cache 中已有的）

    # ---------- 模拟 Q, K, V ----------
    Q = torch.randn(batch_size, num_heads, seq_len_q, head_dim)
    # K, V 存储在物理 block 中
    # 物理 block 形状: (num_blocks, block_size, num_heads, head_dim)
    K_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)
    V_cache = torch.randn(num_blocks, block_size, num_heads, head_dim)

    # ---------- Block Table（逻辑→物理映射）----------
    # block_table[b][i] = 第 b 个请求的第 i 个逻辑 block 对应的物理 block 编号
    # 每个请求需要 ceil(seq_len_kv / block_size) 个 block
    num_kv_blocks = math.ceil(seq_len_kv / block_size)  # = 3
    block_table = torch.tensor([
        [0, 2, 5],  # batch 0: 逻辑 block 0→物理0, 1→物理2, 2→物理5
        [1, 3, 7],  # batch 1: 逻辑 block 0→物理1, 1→物理3, 2→物理7
    ])

    # ---------- PagedAttention 计算 ----------
    scale = 1.0 / math.sqrt(head_dim)
    outputs = torch.zeros(batch_size, num_heads, seq_len_q, head_dim)

    for b in range(batch_size):
        # 从物理 block 中收集该请求的完整 K, V
        k_list, v_list = [], []
        for block_idx in range(num_kv_blocks):
            phys_block = block_table[b, block_idx].item()
            k_list.append(K_cache[phys_block])    # (block_size, num_heads, head_dim)
            v_list.append(V_cache[phys_block])

        # 拼接成完整的 KV Cache: (seq_len_padded, num_heads, head_dim)
        K_full = torch.cat(k_list, dim=0)  # (num_kv_blocks * block_size, num_heads, head_dim)
        V_full = torch.cat(v_list, dim=0)

        # 截取实际长度（最后一个 block 可能未填满）
        K_full = K_full[:seq_len_kv]  # (seq_len_kv, num_heads, head_dim)
        V_full = V_full[:seq_len_kv]

        # 计算 Attention
        for h in range(num_heads):
            k_h = K_full[:, h, :]  # (seq_len_kv, head_dim)
            v_h = V_full[:, h, :]  # (seq_len_kv, head_dim)
            q_h = Q[b, h]          # (seq_len_q, head_dim)

            # Scaled Dot-Product Attention
            scores = torch.matmul(q_h, k_h.T) * scale  # (seq_len_q, seq_len_kv)
            attn_weights = torch.softmax(scores, dim=-1)
            outputs[b, h] = torch.matmul(attn_weights, v_h)  # (seq_len_q, head_dim)

    print("=" * 60)
    print("PagedAttention 手动计算结果")
    print("=" * 60)
    print(f"Q shape:              {Q.shape}")
    print(f"K_cache block shape:  {K_cache.shape}")
    print(f"Block table:          {block_table.tolist()}")
    print(f"Output shape:         {outputs.shape}")
    print(f"Output[0,0,0,:5]:     {outputs[0, 0, 0, :5].tolist()}")
    print()
    print("核心思想：KV Cache 不必连续存储，通过 block table 将")
    print("逻辑上连续的 token 映射到物理上分散的 block 中，")
    print("从而消除显存碎片，大幅提升 GPU 显存利用率。")


# ============================================================
# Part 3: vLLM PagedAttention 底层 kernel 直接调用（高级用法）
# ============================================================

def demo_vllm_paged_attention_kernel():
    """
    直接调用 vLLM 的 PagedAttention CUDA kernel。
    这是最底层的 API，适合需要深入理解实现的场景。
    需要 GPU + vLLM 安装。
    """
    import torch
    from vllm._custom_ops import paged_attention_v1

    torch.manual_seed(0)

    # 参数设置
    batch_size = 2
    num_heads = 8
    head_dim = 64
    block_size = 16
    max_num_blocks_per_seq = 4
    num_blocks = 16

    # Q: (batch_size, num_heads, head_dim) —— 每个请求只算最后一个 token
    query = torch.randn(batch_size, num_heads, head_dim, device="cuda", dtype=torch.float16)

    # KV Cache: (num_blocks, block_size, num_kv_heads, head_dim)
    key_cache = torch.randn(num_blocks, block_size, num_heads, head_dim, device="cuda", dtype=torch.float16)
    val_cache = torch.randn(num_blocks, block_size, num_heads, head_dim, device="cuda", dtype=torch.float16)

    # Block table: (batch_size, max_num_blocks_per_seq)
    block_tables = torch.tensor(
        [[0, 1, 2, 3],
         [4, 5, 6, 7]],
        device="cuda", dtype=torch.int32,
    )

    # 每个请求的实际 context 长度
    context_lens = torch.tensor([32, 48], device="cuda", dtype=torch.int32)

    # 调用 PagedAttention kernel
    scale = 1.0 / (head_dim ** 0.5)
    output = torch.empty_like(query)

    paged_attention_v1(
        output,
        query,
        key_cache,
        val_cache,
        num_heads,
        head_dim,
        block_size,
        block_tables,
        context_lens,
        scale,
        1,           # num_kv_heads (GQA 时用)
        "auto",      # kv_cache_dtype
        1.0,         # k_scale
        1.0,         # v_scale
    )

    print("=" * 60)
    print("vLLM PagedAttention Kernel 直接调用结果")
    print("=" * 60)
    print(f"Query shape:  {query.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output[0,:5]: {output[0, 0, :5].tolist()}")


if __name__ == "__main__":
    import sys

    if "--vllm" in sys.argv:
        print(">>> 运行 vLLM PagedAttention 示例（需要 GPU）...")
        demo_vllm_paged_attention()
    elif "--kernel" in sys.argv:
        print(">>> 运行 vLLM PagedAttention Kernel 示例（需要 GPU）...")
        demo_vllm_paged_attention_kernel()
    else:
        print(">>> 运行 PagedAttention 手动实现示例（CPU 可运行）...")
        demo_paged_attention_from_scratch()
        print("\n提示: 使用 --vllm 参数运行 vLLM 版本, --kernel 运行底层 kernel 版本")
