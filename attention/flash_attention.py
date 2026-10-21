"""
FlashAttention 完整实现
========================
手搓版 + 官方库调用版，通过参数选择运行模式。

核心思想（参考 Tri Dao et al., 2022）:
  1. Tiling        : 将 Q, K, V 分块，每块大小适配 SRAM
  2. Online Softmax: 单遍扫描逐块更新 softmax 统计量（running max + running sum）
  3. 中间矩阵 S_ij, P_ij 始终留在 SRAM，永不写出到 HBM

用法:
    python flash_attention.py                        # 手搓版：逐步演示 + 正确性验证
    python flash_attention.py --mode scratch          # 同上
    python flash_attention.py --mode scratch --causal  # 手搓版 + causal mask
    python flash_attention.py --mode scratch --bench   # 手搓版速度对比
    python flash_attention.py --mode sdpa              # PyTorch 2.0+ SDPA（需 GPU）
    python flash_attention.py --mode flash-attn        # flash-attn 官方包（需 GPU + 编译）
    python flash_attention.py --mode official          # 依次运行所有官方方式
    python flash_attention.py --mode bench             # 官方方式性能对比（需 GPU）

安装依赖:
    pip install torch>=2.0                        # 基础（手搓版无需 GPU）
    pip install flash-attn --no-build-isolation   # flash-attn 方式（需 CUDA + 编译）
    pip install xformers                          # xFormers 方式
"""

from __future__ import annotations

import argparse
import math
import time

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    # 占位类，让 class 定义在无 torch 时不报错
    class _DummyModule:
        pass
    class _DummyNN:
        Module = _DummyModule
        Linear = _DummyModule
    nn = _DummyNN()


# ============================================================
# Part 1: 手搓 FlashAttention（纯 PyTorch，无需 GPU）
# ============================================================

def flash_attention_forward(
    Q: torch.Tensor,          # (batch, heads, N, d)
    K: torch.Tensor,          # (batch, heads, N, d)
    V: torch.Tensor,          # (batch, heads, N, d)
    block_size: int = 64,     # 每个块包含的 token 数（对应 SRAM 容量）
    causal: bool = False,     # 是否使用 causal（autoregressive）mask
) -> torch.Tensor:
    """
    FlashAttention 前向传播（纯 PyTorch 实现）。

    与标准 Attention 的区别：
      标准: S = QK^T (N×N) → 写 HBM → softmax → P (N×N) → 写 HBM → PV
      Flash: 逐块计算 S_ij, P_ij，全部在 SRAM 完成，不写出 N×N 矩阵
    """
    B, H, N, d = Q.shape
    assert K.shape == (B, H, N, d), f"K shape 不匹配: {K.shape}"
    assert V.shape == (B, H, N, d), f"V shape 不匹配: {V.shape}"

    num_blocks = math.ceil(N / block_size)
    scale = 1.0 / math.sqrt(d)
    O = torch.zeros_like(Q)

    # ================================================================
    # 外层循环：遍历 Q 的每个块 Q_i
    # Q_i 加载到 SRAM 后，在整个内循环期间常驻
    # ================================================================
    for i in range(num_blocks):
        i_start = i * block_size
        i_end = min(i_start + block_size, N)

        # Q 块加载到 SRAM：(B, H, B_i, d)
        Q_i = Q[:, :, i_start:i_end, :]

        # 该 Q 块的局部统计量初始化
        m_i = torch.full((B, H, i_end - i_start, 1), float('-inf'), dtype=Q.dtype, device=Q.device)
        l_i = torch.zeros((B, H, i_end - i_start, 1), dtype=Q.dtype, device=Q.device)
        O_i = torch.zeros((B, H, i_end - i_start, d), dtype=Q.dtype, device=Q.device)

        # ============================================================
        # 内层循环：遍历 K, V 的每个块 K_j, V_j
        # K_j, V_j 每轮加载到 SRAM，用完后立即丢弃
        # ============================================================
        for j in range(num_blocks):
            j_start = j * block_size
            j_end = min(j_start + block_size, N)

            K_j = K[:, :, j_start:j_end, :]
            V_j = V[:, :, j_start:j_end, :]

            # Step 1: 局部注意力分数 S_ij = Q_i @ K_j^T / sqrt(d)
            S_ij = torch.matmul(Q_i, K_j.transpose(-2, -1)) * scale

            # Step 2: Causal Mask
            if causal:
                row_idx = torch.arange(i_start, i_end, device=Q.device).unsqueeze(-1)
                col_idx = torch.arange(j_start, j_end, device=Q.device).unsqueeze(0)
                mask = col_idx > row_idx
                S_ij = S_ij.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

            # Step 3: Online Softmax 更新
            #   m_new = max(m_old, rowmax(S_ij))
            #   correction = exp(m_old - m_new)
            #   l_new = correction * l_old + rowsum(exp(S_ij - m_new))
            #   O_new = correction * O_old + exp(S_ij - m_new) @ V_j
            m_ij = S_ij.max(dim=-1, keepdim=True).values
            m_new = torch.maximum(m_i, m_ij)
            correction = torch.exp(m_i - m_new)
            P_ij = torch.exp(S_ij - m_new)
            l_new = correction * l_i + P_ij.sum(dim=-1, keepdim=True)

            # Step 4: 更新输出累加器
            O_i = correction * O_i + torch.matmul(P_ij, V_j)

            m_i = m_new
            l_i = l_new
            # K_j, V_j, S_ij, P_ij 全部丢弃（释放 SRAM）

        # 外循环结束：归一化，写回 HBM
        O[:, :, i_start:i_end, :] = O_i / l_i

    return O


def standard_attention_forward(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:
    """标准 Scaled Dot-Product Attention（完整 N×N 矩阵，IO 瓶颈所在）"""
    B, H, N, d = Q.shape
    scale = 1.0 / math.sqrt(d)
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    if causal:
        mask = torch.triu(torch.ones(N, N, device=Q.device, dtype=torch.bool), diagonal=1)
        S = S.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    P = torch.softmax(S, dim=-1)
    return torch.matmul(P, V)


# ============================================================
# Part 2: 官方库调用（需 GPU）
# ============================================================

class FlashAttentionSDPA(nn.Module):
    """
    方式 1: PyTorch 2.0+ scaled_dot_product_attention。
    内部自动选择 FlashAttention / Memory-Efficient / Math 后端。
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0, causal: bool = False):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        self.dropout = dropout
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        # ★ 核心：自动使用 FlashAttention 后端
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=self.causal,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, C)
        return self.out_proj(attn_out)


class FlashAttentionOfficial(nn.Module):
    """
    方式 2: flash-attn 官方包（Tri Dao 实现，性能最优）。
    支持 MHA / GQA（Grouped Query Attention）。
    """

    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: int | None = None, causal: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = embed_dim // num_heads
        self.causal = causal
        kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, kv_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from flash_attn import flash_attn_func
        B, N, C = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim)
        attn_out = flash_attn_func(
            q, k, v, dropout_p=0.0,
            softmax_scale=1.0 / math.sqrt(self.head_dim),
            causal=self.causal,
        )
        return self.out_proj(attn_out.reshape(B, N, -1))


# ============================================================
# Demo: 手搓版
# ============================================================

def demo_algorithm_walkthrough():
    """用极小矩阵逐步演示 FlashAttention 的计算流程"""
    torch.manual_seed(7)
    N, d, block_size = 6, 4, 3
    Q = torch.randn(1, 1, N, d)
    K = torch.randn(1, 1, N, d)
    V = torch.randn(1, 1, N, d)

    print("=" * 65)
    print("  FlashAttention 逐步计算演示")
    print(f"  N={N}, d={d}, block_size={block_size} → 共 {math.ceil(N/block_size)} 块")
    print("=" * 65)

    scale = 1.0 / math.sqrt(d)
    num_blocks = math.ceil(N / block_size)

    print(f"\n  Q shape: {Q.shape}（1×1×{N}×{d}，已省略 batch/head 维度）")
    print(f"  第 1 块 token: [0, 1, 2]")
    print(f"  第 2 块 token: [3, 4, 5]")

    print("\n" + "─" * 50)
    print("  外循环 i=0：Q 块 = Q[0:3]，加载到 SRAM，常驻")
    print("─" * 50)

    m_i = torch.full((1, 1, block_size, 1), float('-inf'))
    l_i = torch.zeros((1, 1, block_size, 1))
    O_i = torch.zeros((1, 1, block_size, d))
    Q_0 = Q[:, :, 0:block_size, :]

    for j in range(num_blocks):
        j_start, j_end = j * block_size, min((j+1) * block_size, N)
        K_j = K[:, :, j_start:j_end, :]
        V_j = V[:, :, j_start:j_end, :]

        S_ij = torch.matmul(Q_0, K_j.transpose(-2, -1)) * scale
        m_ij = S_ij.max(dim=-1, keepdim=True).values
        m_new = torch.maximum(m_i, m_ij)
        correction = torch.exp(m_i - m_new)
        P_ij = torch.exp(S_ij - m_new)
        l_new = correction * l_i + P_ij.sum(dim=-1, keepdim=True)
        O_i = correction * O_i + torch.matmul(P_ij, V_j)

        print(f"\n  内循环 j={j}：K 块 = K[{j_start}:{j_end}]，V 块 = V[{j_start}:{j_end}]")
        print(f"    S_0{j} shape  : {tuple(S_ij.shape)}  (B_i×B_j，留在 SRAM)")
        print(f"    m_ij (局部max): {m_ij.squeeze().tolist()}")
        print(f"    m_new (全局max): {m_new.squeeze().tolist()}")
        print(f"    correction    : {correction.squeeze().tolist()}")
        print(f"    l_new (分母)  : {l_new.squeeze().tolist()}")
        m_i, l_i = m_new, l_new

    O_0 = O_i / l_i
    print(f"\n  外循环 i=0 结束，O[0:3] = O_i / l_i（写回 HBM）")
    print(f"    O[0] = {O_0[0,0,0].tolist()}")

    O_std = standard_attention_forward(Q, K, V)
    print(f"\n  标准 Attention O[0] = {O_std[0,0,0].tolist()}")
    diff = (O_0 - O_std[:, :, 0:block_size, :]).abs().max().item()
    print(f"  最大误差            = {diff:.2e} ✓\n")


def demo_correctness(causal: bool = False):
    """验证 FlashAttention 与标准 Attention 的数值一致性"""
    torch.manual_seed(42)
    batch, heads, N, d, block_size = 2, 4, 128, 64, 32
    Q = torch.randn(batch, heads, N, d, dtype=torch.float32)
    K = torch.randn(batch, heads, N, d, dtype=torch.float32)
    V = torch.randn(batch, heads, N, d, dtype=torch.float32)

    print("=" * 65)
    print("  FlashAttention vs Standard Attention 正确性验证")
    print(f"  N={N}, H={heads}, d={d}, B={block_size}, causal={causal}")
    print("=" * 65)

    O_std = standard_attention_forward(Q, K, V, causal=causal)
    O_flash = flash_attention_forward(Q, K, V, block_size=block_size, causal=causal)

    max_diff = (O_std - O_flash).abs().max().item()
    mean_diff = (O_std - O_flash).abs().mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(O_std.flatten(), O_flash.flatten(), dim=0).item()

    print(f"\n  最大绝对误差  : {max_diff:.2e}")
    print(f"  平均绝对误差  : {mean_diff:.2e}")
    print(f"  余弦相似度    : {cos_sim:.10f}")
    print(f"  {'✓ 数值等价验证通过' if max_diff < 1e-4 else '✗ 数值差异较大'}")
    print(f"\n  Standard  [0,0,0,:5]: {O_std[0,0,0,:5].tolist()}")
    print(f"  Flash     [0,0,0,:5]: {O_flash[0,0,0,:5].tolist()}\n")


def demo_block_size_effect():
    """演示不同块大小对数值精度的影响"""
    torch.manual_seed(0)
    batch, heads, N, d = 1, 2, 256, 64
    Q = torch.randn(batch, heads, N, d, dtype=torch.float32)
    K = torch.randn(batch, heads, N, d, dtype=torch.float32)
    V = torch.randn(batch, heads, N, d, dtype=torch.float32)
    O_std = standard_attention_forward(Q, K, V)

    print("=" * 65)
    print("  块大小（Block Size）对精度的影响")
    print("=" * 65)
    print(f"  {'Block Size':<12} {'Max Error':<14} {'Mean Error':<14} {'Blocks':<8}")
    print("-" * 50)

    for bs in [8, 16, 32, 64, 128, 256]:
        O_flash = flash_attention_forward(Q, K, V, block_size=bs)
        max_err = (O_std - O_flash).abs().max().item()
        mean_err = (O_std - O_flash).abs().mean().item()
        print(f"  {bs:<12} {max_err:<14.2e} {mean_err:<14.2e} {math.ceil(N/bs):<8}")

    print("\n  结论：所有块大小下误差均在 FP32 精度范围内（< 1e-4）。\n")


def demo_scratch_speed():
    """手搓版速度对比（纯 PyTorch，无法体现 SRAM IO 优化）"""
    torch.manual_seed(123)
    batch, heads, d, block_size = 1, 4, 64, 64

    print("=" * 65)
    print("  手搓版 Flash vs Standard 耗时对比（纯 PyTorch）")
    print("  注意：真正加速需 CUDA kernel，此处仅展示算法结构")
    print("=" * 65)
    print(f"  {'N':<8} {'Standard (ms)':<18} {'Flash (ms)':<18} {'Flash/Std':<10}")
    print("-" * 60)

    for N in [128, 256, 512, 1024]:
        Q = torch.randn(batch, heads, N, d)
        K = torch.randn(batch, heads, N, d)
        V = torch.randn(batch, heads, N, d)

        t0 = time.perf_counter()
        for _ in range(3):
            standard_attention_forward(Q, K, V)
        t_std = (time.perf_counter() - t0) / 3 * 1000

        t0 = time.perf_counter()
        for _ in range(3):
            flash_attention_forward(Q, K, V, block_size=block_size)
        t_flash = (time.perf_counter() - t0) / 3 * 1000

        print(f"  {N:<8} {t_std:<18.2f} {t_flash:<18.2f} {t_flash/t_std:<10.2f}x")

    print("\n  纯 PyTorch 版 Flash 更慢（Python 循环开销 > IO 节省），")
    print("  真正加速来自 CUDA kernel（官方 flash-attn 可达 2-5x）。\n")


# ============================================================
# Demo: 官方库调用版
# ============================================================

def demo_sdpa():
    """演示 PyTorch 2.0+ SDPA（自动调用 FlashAttention 后端）"""
    print("=" * 65)
    print("  方式 1：PyTorch 2.0+ scaled_dot_product_attention")
    print("=" * 65)

    version = torch.__version__
    major, minor = map(int, version.split('.')[:2])
    if not ((major > 2) or (major == 2 and minor >= 0)):
        print(f"  ✗ PyTorch {version} 不支持 SDPA，需要 >= 2.0\n")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"  PyTorch: {version}, 设备: {device}, 精度: {dtype}\n")

    embed_dim, num_heads, seq_len, batch_size = 512, 8, 1024, 2
    model = FlashAttentionSDPA(embed_dim, num_heads, causal=True).to(device, dtype)
    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

    with torch.no_grad():
        out = model(x)

    print(f"  输入 shape : {x.shape}")
    print(f"  输出 shape : {out.shape}")
    print(f"  参数量     : {sum(p.numel() for p in model.parameters()):,}")

    if device == "cuda":
        print(f"\n  FlashAttention 后端可用: {torch.backends.cuda.flash_sdp_enabled()}")
        print(f"  Memory-Efficient 可用  : {torch.backends.cuda.mem_efficient_sdp_enabled()}")
        print(f"  Math 回退可用          : {torch.backends.cuda.math_sdp_enabled()}")
    print()


def demo_flash_attn():
    """演示 flash-attn 官方包（需要安装 + GPU）"""
    print("=" * 65)
    print("  方式 2：flash-attn 官方包（Tri Dao 实现）")
    print("=" * 65)

    try:
        import flash_attn
        print(f"  flash-attn 版本: {flash_attn.__version__}")
    except ImportError:
        print("  ✗ 未安装。安装: pip install flash-attn --no-build-isolation\n")
        return

    if not torch.cuda.is_available():
        print("  ✗ 需要 CUDA GPU\n")
        return

    device, dtype = "cuda", torch.bfloat16
    embed_dim, num_heads, seq_len, batch_size = 1024, 16, 2048, 2
    x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

    print(f"\n  [MHA] embed_dim={embed_dim}, heads={num_heads}, N={seq_len}")
    model = FlashAttentionOfficial(embed_dim, num_heads, causal=True).to(device, dtype)
    with torch.no_grad():
        print(f"  输出 shape: {model(x).shape}")

    num_kv_heads = 4
    print(f"\n  [GQA] Q heads={num_heads}, KV heads={num_kv_heads}")
    model_gqa = FlashAttentionOfficial(embed_dim, num_heads, num_kv_heads=num_kv_heads, causal=True).to(device, dtype)
    with torch.no_grad():
        print(f"  输出 shape: {model_gqa(x).shape}")
        print(f"  GQA 参数量比 MHA 节省: {(1 - num_kv_heads/num_heads)*100:.0f}%（K+V 投影）")
    print()


def demo_xformers():
    """演示 Meta 的 xFormers（memory_efficient_attention）"""
    print("=" * 65)
    print("  方式 3：xFormers（Meta memory-efficient attention）")
    print("=" * 65)

    try:
        import xformers.ops as xops
    except ImportError:
        print("  ✗ 未安装。安装: pip install xformers\n")
        return

    if not torch.cuda.is_available():
        print("  ✗ 需要 CUDA GPU\n")
        return

    device, dtype = "cuda", torch.float16
    batch, heads, N, d = 2, 8, 1024, 64
    q = torch.randn(batch, N, heads, d, device=device, dtype=dtype)
    k = torch.randn(batch, N, heads, d, device=device, dtype=dtype)
    v = torch.randn(batch, N, heads, d, device=device, dtype=dtype)

    out = xops.memory_efficient_attention(q, k, v, attn_bias=xops.LowerTriangularMask())
    print(f"  输入 Q shape: {q.shape}  (batch, N, heads, head_dim)")
    print(f"  输出 shape  : {out.shape}\n")


def demo_official_bench():
    """官方方式性能对比（需 GPU）"""
    print("=" * 65)
    print("  性能对比：Standard vs SDPA vs flash-attn（需 CUDA GPU）")
    print("=" * 65)

    if not torch.cuda.is_available():
        print("  ✗ 需要 CUDA GPU\n")
        return

    device, dtype = "cuda", torch.float16
    configs = [
        (512,   8,  1024,  2),
        (1024, 16,  2048,  2),
        (1024, 16,  4096,  1),
        (2048, 32,  8192,  1),
    ]

    print(f"\n  {'Config (d,H,N,B)':<25} {'Standard':>12} {'SDPA':>12} {'flash-attn':>12}")
    print("  " + "-" * 65)

    for d, H, N, B in configs:
        results = {}

        # Standard
        try:
            q = torch.randn(B, H, N, d, device=device, dtype=dtype)
            k, v = q.clone(), q.clone()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                S = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
                P = torch.softmax(S, dim=-1)
                _ = torch.matmul(P, v)
            torch.cuda.synchronize()
            results['Standard'] = f"{(time.perf_counter()-t0)/5*1000:.1f}ms"
        except RuntimeError:
            results['Standard'] = "OOM"

        # SDPA
        try:
            q2 = torch.randn(B, H, N, d, device=device, dtype=dtype)
            k2, v2 = q2.clone(), q2.clone()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                _ = torch.nn.functional.scaled_dot_product_attention(q2, k2, v2, is_causal=True)
            torch.cuda.synchronize()
            results['SDPA'] = f"{(time.perf_counter()-t0)/5*1000:.1f}ms"
        except RuntimeError:
            results['SDPA'] = "OOM"

        # flash-attn
        try:
            from flash_attn import flash_attn_func
            q3 = torch.randn(B, N, H, d, device=device, dtype=dtype)
            k3, v3 = q3.clone(), q3.clone()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(5):
                _ = flash_attn_func(q3, k3, v3, causal=True)
            torch.cuda.synchronize()
            results['flash-attn'] = f"{(time.perf_counter()-t0)/5*1000:.1f}ms"
        except ImportError:
            results['flash-attn'] = "未安装"
        except RuntimeError:
            results['flash-attn'] = "OOM"

        cfg = f"({d},{H},{N},{B})"
        print(f"  {cfg:<25} "
              f"{results.get('Standard','-'):>12} "
              f"{results.get('SDPA','-'):>12} "
              f"{results.get('flash-attn','-'):>12}")

    print("\n  预期：SDPA/flash-attn 快 2-5x，Standard 长序列易 OOM。\n")


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="FlashAttention 完整实现（手搓版 + 官方库调用版）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        模式说明:
        scratch      手搓版：逐步演示 + 正确性验证 + 块大小影响（默认，无需 GPU）
        sdpa         PyTorch 2.0+ SDPA（需 GPU）
        flash-attn   flash-attn 官方包（需 GPU + 编译安装）
        official     依次运行所有官方方式（sdpa + flash-attn + xformers）
        bench        官方方式性能对比（需 GPU）

        可选参数:
        --causal     仅 scratch 模式有效，启用 causal mask
        --bench      仅 scratch 模式有效，运行速度对比
                """,
    )
    parser.add_argument(
        "--mode", type=str, default="scratch",
        choices=["scratch", "sdpa", "flash-attn", "official", "bench"],
        help="运行模式（默认 scratch）",
    )
    parser.add_argument("--causal", action="store_true", help="scratch 模式下启用 causal mask")
    parser.add_argument("--bench", action="store_true", help="scratch 模式下运行速度对比（而非默认演示）")
    args = parser.parse_args()

    if torch is None:
        print("✗ 未安装 torch，请先安装: pip install torch")
        return

    if args.mode == "scratch":
        if args.bench:
            demo_scratch_speed()
        else:
            demo_algorithm_walkthrough()
            demo_correctness(causal=args.causal)
            demo_block_size_effect()
            print("提示: --causal 启用 causal mask, --bench 运行速度对比")

    elif args.mode == "sdpa":
        demo_sdpa()

    elif args.mode == "flash-attn":
        demo_flash_attn()

    elif args.mode == "official":
        print(">>> 依次运行所有官方方式...\n")
        demo_sdpa()
        demo_flash_attn()
        demo_xformers()

    elif args.mode == "bench":
        demo_official_bench()


if __name__ == "__main__":
    main()
