"""
MLA (Multi-Head Latent Attention) 示例代码
==========================================
DeepSeek-V2/V3 的核心注意力机制。

核心思想：
  1. 将 KV 压缩到低秩潜向量 c（压缩比可达 10x+）
  2. 推理时只存 c，按需解压为 K、V
  3. Q 也做低秩压缩（decoupled RoPE 保留位置信息）

对比：
  MHA: KV Cache = 2 × heads × head_dim per layer
  MLA: KV Cache = latent_dim + rope_dim per layer  (小 10~50 倍)

用法:
    python mla_attention.py               # 基本演示 + 正确性对比
    python mla_attention.py --bench       # MLA vs MHA 速度/显存对比（需 GPU）
"""

from __future__ import annotations

import argparse
import math
import time

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    torch = None
    class _Dummy:
        Module = type('Module', (), {'__init__': lambda s: None})
        Linear = _Dummy if '_Dummy' in dir() else type('Linear', (), {})
    nn = type('nn', (), {'Module': type('Module', (), {}), 'Linear': type('Linear', (), {})})()


# ============================================================
# RoPE (Rotary Position Embedding)
# ============================================================

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    对 x 施加旋转位置编码。
    x: (B, heads, N, d)
    cos, sin: (1, 1, N, d)
    """
    d = x.shape[-1]
    x1 = x[..., :d // 2]
    x2 = x[..., d // 2:]
    rotated = torch.cat([-x2, x1], dim=-1)
    return x * cos + rotated * sin


def precompute_rope(seq_len: int, dim: int, device, dtype) -> tuple:
    """预计算 RoPE 的 cos/sin 表"""
    theta = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, theta)  # (seq_len, dim/2)
    freqs = torch.cat([freqs, freqs], dim=-1)  # (seq_len, dim)
    cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, N, dim)
    sin = freqs.sin().unsqueeze(0).unsqueeze(0)
    return cos.to(dtype), sin.to(dtype)


# ============================================================
# MLA (Multi-Head Latent Attention)
# ============================================================

class MultiHeadLatentAttention(nn.Module):
    """
    Multi-Head Latent Attention (DeepSeek-V2 风格)。

    关键创新：
      - Q 压缩: x → c_q (低秩) → q_no_rope，再单独加 q_rope（保留位置信息）
      - KV 压缩: x → c_kv (低秩) → k_no_rope + v，再单独加 k_rope
      - 推理时 KV Cache 只存 c_kv + k_rope，体积远小于完整 K, V

    参数说明：
      d_model:       模型隐藏维度（如 512）
      num_heads:     Q 的头数（如 8）
      qk_nope_dim:   Q/K 中不带 RoPE 的部分维度（每头，如 64）
      qk_rope_dim:   Q/K 中带 RoPE 的部分维度（每头，如 16）
      v_dim:         V 的每头维度（如 64）
      latent_dim:    潜向量 c 的维度（如 128，远小于 num_heads × head_dim）
    """

    def __init__(
        self,
        d_model: int = 512,
        num_heads: int = 8,
        qk_nope_dim: int = 64,     # Q/K 非 RoPE 部分（每头）
        qk_rope_dim: int = 16,     # Q/K RoPE 部分（每头）
        v_dim: int = 64,           # V 每头维度
        latent_dim: int = 128,     # 潜向量维度（压缩瓶颈）
    ):
        super().__init__()
        self.num_heads = num_heads
        self.qk_nope_dim = qk_nope_dim
        self.qk_rope_dim = qk_rope_dim
        self.v_dim = v_dim
        self.latent_dim = latent_dim
        self.scale = 1.0 / math.sqrt(qk_nope_dim + qk_rope_dim)

        # ──────────────────────────────────────────────
        # Q 路径：x → c_q → q_nope + q_rope
        # ──────────────────────────────────────────────
        # Q 下投影：压缩到低秩
        self.W_dq = nn.Linear(d_model, latent_dim, bias=False)
        # Q 上投影（nope 部分）：c_q → 每头的 q_nope
        self.W_uq = nn.Linear(latent_dim, num_heads * qk_nope_dim, bias=False)
        # Q 的 RoPE 部分：直接从 x 生成（不经过压缩，保留位置信息）
        self.W_qr = nn.Linear(d_model, num_heads * qk_rope_dim, bias=False)

        # ──────────────────────────────────────────────
        # KV 路径：x → c_kv → k_nope + v（压缩存储）
        # ──────────────────────────────────────────────
        # KV 下投影：压缩到潜向量 c_kv（★ 推理时只存这个）
        self.W_dkv = nn.Linear(d_model, latent_dim, bias=False)
        # K 上投影（nope 部分）：c_kv → 每头的 k_nope
        self.W_uk = nn.Linear(latent_dim, num_heads * qk_nope_dim, bias=False)
        # V 上投影：c_kv → 每头的 v
        self.W_uv = nn.Linear(latent_dim, num_heads * v_dim, bias=False)
        # K 的 RoPE 部分：直接从 x 生成
        self.W_kr = nn.Linear(d_model, qk_rope_dim, bias=False)  # 注意：只有 1 份，不乘 heads

        # 输出投影
        self.W_o = nn.Linear(num_heads * v_dim, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,           # (B, N, d_model)
        cos: torch.Tensor,         # (1, 1, N, rope_dim)
        sin: torch.Tensor,         # (1, 1, N, rope_dim)
    ) -> torch.Tensor:
        B, N, _ = x.shape

        # ==============================================================
        # Q 路径
        # ==============================================================
        # Step 1: Q 压缩 x → c_q
        c_q = self.W_dq(x)   # (B, N, latent_dim)  ← 低秩中间表示

        # Step 2: Q 上投影 c_q → q_nope（多头）
        q_nope = self.W_uq(c_q)  # (B, N, heads * qk_nope_dim)
        q_nope = q_nope.view(B, N, self.num_heads, self.qk_nope_dim)
        # q_nope: (B, N, heads, qk_nope_dim)

        # Step 3: Q 的 RoPE 部分（直接从 x，不经过压缩）
        q_rope = self.W_qr(x)  # (B, N, heads * qk_rope_dim)
        q_rope = q_rope.view(B, N, self.num_heads, self.qk_rope_dim)
        q_rope = apply_rope(q_rope, cos, sin)  # 施加旋转位置编码

        # Step 4: 拼接 q_nope + q_rope → 完整 Q
        q = torch.cat([q_nope, q_rope], dim=-1)
        # q: (B, N, heads, qk_nope_dim + qk_rope_dim)
        q = q.transpose(1, 2)  # (B, heads, N, qk_dim)

        # ==============================================================
        # KV 路径（★ MLA 核心：压缩存储）
        # ==============================================================
        # Step 1: KV 压缩 x → c_kv
        c_kv = self.W_dkv(x)   # (B, N, latent_dim)  ← ★ 推理时存这个！

        # Step 2: K 上投影 c_kv → k_nope（多头）
        k_nope = self.W_uk(c_kv)  # (B, N, heads * qk_nope_dim)
        k_nope = k_nope.view(B, N, self.num_heads, self.qk_nope_dim)

        # Step 3: V 上投影 c_kv → v（多头）
        v = self.W_uv(c_kv)  # (B, N, heads * v_dim)
        v = v.view(B, N, self.num_heads, self.v_dim)
        v = v.transpose(1, 2)  # (B, heads, N, v_dim)

        # Step 4: K 的 RoPE 部分（★ 只有 1 份，所有头共享）
        k_rope = self.W_kr(x)  # (B, N, qk_rope_dim)  ← 注意不是 num_heads × rope_dim
        k_rope = apply_rope(k_rope, cos, sin)
        k_rope = k_rope.unsqueeze(2).expand(-1, -1, self.num_heads, -1)
        # k_rope: (B, N, heads, qk_rope_dim)  ← 广播到所有头

        # Step 5: 拼接 k_nope + k_rope → 完整 K
        k = torch.cat([k_nope, k_rope], dim=-1)
        k = k.transpose(1, 2)  # (B, heads, N, qk_dim)

        # ==============================================================
        # Attention 计算
        # ==============================================================
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, heads, N, N)

        # Causal mask
        mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # (B, heads, N, v_dim)

        # 合并多头 + 输出投影
        attn_output = attn_output.transpose(1, 2).contiguous()  # (B, N, heads, v_dim)
        attn_output = attn_output.view(B, N, -1)  # (B, N, heads * v_dim)
        return self.W_o(attn_output)  # (B, N, d_model)

    def kv_cache_size(self, batch_size: int, seq_len: int) -> dict:
        """计算 KV Cache 大小（推理时每 token 的存储量）"""
        # MLA: 存 c_kv + k_rope
        mla_per_token = self.latent_dim + self.qk_rope_dim  # c_kv + k_rope
        mla_total = batch_size * seq_len * mla_per_token

        # 等价 MHA: 存完整 K + V
        k_dim = self.qk_nope_dim + self.qk_rope_dim
        mha_per_token = self.num_heads * (k_dim + self.v_dim) * 2  # K + V
        mha_total = batch_size * seq_len * mha_per_token

        ratio = mha_total / mla_total
        return {
            'MLA_per_token': mla_per_token,
            'MLA_total': mla_total,
            'MHA_per_token': mha_per_token,
            'MHA_total': mha_total,
            'compression_ratio': ratio,
        }


# ============================================================
# 等效标准 MHA（用于对比验证）
# ============================================================

class StandardMHA(nn.Module):
    """标准 Multi-Head Attention，用于与 MLA 对比"""

    def __init__(self, d_model: int, num_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = 1.0 / math.sqrt(head_dim)
        self.q_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(num_heads * head_dim, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = torch.triu(torch.ones(N, N, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, N, -1)
        return self.out_proj(out)


# ============================================================
# 演示
# ============================================================

def demo_mla():
    """演示 MLA 的结构和 KV Cache 压缩效果"""
    torch.manual_seed(42)

    d_model = 512
    num_heads = 8
    qk_nope_dim = 64
    qk_rope_dim = 16
    v_dim = 64
    latent_dim = 128

    mla = MultiHeadLatentAttention(
        d_model=d_model,
        num_heads=num_heads,
        qk_nope_dim=qk_nope_dim,
        qk_rope_dim=qk_rope_dim,
        v_dim=v_dim,
        latent_dim=latent_dim,
    )

    total_params = sum(p.numel() for p in mla.parameters())
    print("=" * 65)
    print("  MLA (Multi-Head Latent Attention) 结构演示")
    print("=" * 65)
    print(f"  d_model={d_model}, heads={num_heads}, latent_dim={latent_dim}")
    print(f"  qk_nope_dim={qk_nope_dim}, qk_rope_dim={qk_rope_dim}, v_dim={v_dim}")
    print(f"  总参数量: {total_params:,}")

    # KV Cache 对比
    cache_info = mla.kv_cache_size(batch_size=1, seq_len=4096)
    print(f"\n  KV Cache 对比 (batch=1, seq_len=4096):")
    print(f"  ──────────────────────────────────────────")
    print(f"  MLA: 每 token 存 {cache_info['MLA_per_token']} floats "
          f"(c_kv={latent_dim} + k_rope={qk_rope_dim})")
    print(f"  MHA: 每 token 存 {cache_info['MHA_per_token']} floats "
          f"(K+V 完整存储)")
    print(f"  压缩比: {cache_info['compression_ratio']:.1f}x")

    # 前向传播
    B, N = 2, 128
    x = torch.randn(B, N, d_model)
    cos, sin = precompute_rope(N, qk_rope_dim, x.device, x.dtype)
    out = mla(x, cos, sin)

    print(f"\n  前向传播:")
    print(f"    输入: {x.shape} → 输出: {out.shape}")

    # 展示 Q 路径和 KV 路径的维度变化
    print(f"\n  数据流维度变化:")
    print(f"  ──────────────────────────────────────────")
    print(f"  Q 路径:")
    print(f"    x ({d_model}) → c_q ({latent_dim}) → q_nope ({num_heads}×{qk_nope_dim}={num_heads*qk_nope_dim})")
    print(f"    x ({d_model}) → q_rope ({num_heads}×{qk_rope_dim}={num_heads*qk_rope_dim}) [+RoPE]")
    print(f"    拼接: q = [{qk_nope_dim} + {qk_rope_dim}] = {qk_nope_dim+qk_rope_dim} per head")
    print(f"  KV 路径:")
    print(f"    x ({d_model}) → c_kv ({latent_dim}) ★ 存储这个")
    print(f"    c_kv ({latent_dim}) → k_nope ({num_heads}×{qk_nope_dim}={num_heads*qk_nope_dim})")
    print(f"    c_kv ({latent_dim}) → v ({num_heads}×{v_dim}={num_heads*v_dim})")
    print(f"    x ({d_model}) → k_rope ({qk_rope_dim}) [+RoPE, 共享给所有头]")
    print()


def demo_mla_vs_mha_params():
    """对比 MLA 和 MHA 的参数量和 KV Cache"""
    print("=" * 65)
    print("  MLA vs MHA 参数量和 KV Cache 对比")
    print("=" * 65)

    configs = [
        # (d_model, heads, qk_nope, qk_rope, v_dim, latent_dim, name)
        (512,  8,  64,  16,  64,  128, "小型模型"),
        (1024, 16, 64,  32,  64,  256, "中型模型"),
        (2048, 32, 128, 64,  128, 512, "大型模型"),
    ]

    for d_model, heads, qk_nope, qk_rope, v_dim, latent, name in configs:
        mla = MultiHeadLatentAttention(d_model, heads, qk_nope, qk_rope, v_dim, latent)
        mha = StandardMHA(d_model, heads, qk_nope + qk_rope)

        mla_params = sum(p.numel() for p in mla.parameters())
        mha_params = sum(p.numel() for p in mha.parameters())

        cache = mla.kv_cache_size(1, 4096)

        print(f"\n  [{name}] d_model={d_model}, heads={heads}")
        print(f"    MLA 参数: {mla_params:>10,}  |  KV Cache: {cache['MLA_per_token']:>6} floats/token")
        print(f"    MHA 参数: {mha_params:>10,}  |  KV Cache: {cache['MHA_per_token']:>6} floats/token")
        print(f"    KV Cache 压缩比: {cache['compression_ratio']:.1f}x")

    print()


def demo_data_flow():
    """逐步展示 MLA 的完整数据流"""
    torch.manual_seed(0)

    d_model, heads = 256, 4
    qk_nope, qk_rope, v_dim, latent = 32, 8, 32, 64

    mla = MultiHeadLatentAttention(d_model, heads, qk_nope, qk_rope, v_dim, latent)
    B, N = 1, 16
    x = torch.randn(B, N, d_model)
    cos, sin = precompute_rope(N, qk_rope, x.device, x.dtype)

    print("=" * 65)
    print("  MLA 逐步数据流")
    print(f"  d_model={d_model}, heads={heads}, latent={latent}")
    print(f"  qk_nope={qk_nope}, qk_rope={qk_rope}, v_dim={v_dim}")
    print("=" * 65)

    # Q 路径
    c_q = mla.W_dq(x)
    print(f"\n  [Q 路径]")
    print(f"    x:           {x.shape}")
    print(f"    c_q = W_dq(x):  {c_q.shape}  ← 低秩压缩 (d={d_model} → {latent})")

    q_nope = mla.W_uq(c_q).view(B, N, heads, qk_nope)
    print(f"    q_nope = W_uq(c_q): {q_nope.shape}")

    q_rope = mla.W_qr(x).view(B, N, heads, qk_rope)
    q_rope = apply_rope(q_rope, cos, sin)
    print(f"    q_rope = W_qr(x):   {q_rope.shape}  [+RoPE]")

    q = torch.cat([q_nope, q_rope], dim=-1).transpose(1, 2)
    print(f"    q = concat:         {q.shape}")

    # KV 路径
    c_kv = mla.W_dkv(x)
    print(f"\n  [KV 路径 — ★ 推理只存 c_kv + k_rope]")
    print(f"    c_kv = W_dkv(x):    {c_kv.shape}  ← ★ 存这个 ({latent} floats/token)")

    k_nope = mla.W_uk(c_kv).view(B, N, heads, qk_nope)
    print(f"    k_nope = W_uk(c_kv): {k_nope.shape}  ← 从 c_kv 解压")

    v = mla.W_uv(c_kv).view(B, N, heads, v_dim).transpose(1, 2)
    print(f"    v = W_uv(c_kv):      {v.shape}  ← 从 c_kv 解压")

    k_rope = mla.W_kr(x)
    k_rope = apply_rope(k_rope, cos, sin)
    print(f"    k_rope = W_kr(x):    {k_rope.shape}  ← ★ 存这个 ({qk_rope} floats/token)")
    print(f"    总共存: {latent} + {qk_rope} = {latent + qk_rope} floats/token")

    k_rope_expanded = k_rope.unsqueeze(2).expand(-1, -1, heads, -1)
    k = torch.cat([k_nope, k_rope_expanded], dim=-1).transpose(1, 2)
    print(f"    k = concat:          {k.shape}")

    # Attention
    scores = torch.matmul(q, k.transpose(-2, -1)) * mla.scale
    mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
    scores = scores.masked_fill(mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, N, -1)
    out = mla.W_o(out)
    print(f"\n  [Attention]")
    print(f"    scores:  {scores.shape}  (heads, N, N)")
    print(f"    output:  {out.shape}")
    print()


# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="MLA (Multi-Head Latent Attention) 示例代码",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bench", action="store_true", help="运行 MLA vs MHA 对比")
    args = parser.parse_args()

    if torch is None:
        print("✗ 未安装 torch，请先安装: pip install torch")
        return

    demo_data_flow()
    demo_mla()
    demo_mla_vs_mha_params()


if __name__ == "__main__":
    main()
