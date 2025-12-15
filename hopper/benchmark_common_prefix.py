"""
Benchmark for common prefix optimization in FlashAttention.

This optimization is for scenarios where multiple sequences share a common prefix 
(e.g., system prompt in LLM serving).

Data layout:
- Baseline: [Seq0, Seq1, ...] where Seq_i = [Common_copy_i, Unique_i]
- Common prefix: [Common, Unique0, Unique1, ...]

Benefits:
- Memory savings: (batch_size - 1) * common_len tokens
- Compute savings: common² / 2 computed once instead of batch_size times
- Bandwidth savings: common K/V loaded once, reused via L2 cache

Usage:
    python benchmark_common_prefix.py
"""

import argparse
import torch
import time
from flash_attn_interface import flash_attn_varlen_func


def create_baseline_data(batch_size, common_len, unique_lens, num_heads, head_dim, device, dtype):
    """Create baseline data where each sequence has its own copy of common prefix."""
    full_lens = [common_len + ul for ul in unique_lens]
    total_len = sum(full_lens)
    
    Q = torch.randn(total_len, num_heads, head_dim, device=device, dtype=dtype)
    K = torch.randn(total_len, num_heads, head_dim, device=device, dtype=dtype)
    V = torch.randn(total_len, num_heads, head_dim, device=device, dtype=dtype)
    
    cu_seqlens = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(full_lens), 0).numpy()),
        device=device, dtype=torch.int32
    )
    max_seqlen = max(full_lens)
    
    return Q, K, V, cu_seqlens, max_seqlen, total_len


def create_common_prefix_data(batch_size, common_len, unique_lens, num_heads, head_dim, device, dtype):
    """Create common prefix data with shared common and packed unique parts."""
    total_len = common_len + sum(unique_lens)
    
    Q = torch.randn(total_len, num_heads, head_dim, device=device, dtype=dtype)
    K = torch.randn(total_len, num_heads, head_dim, device=device, dtype=dtype)
    V = torch.randn(total_len, num_heads, head_dim, device=device, dtype=dtype)
    
    cu_seqlens_list = [0, common_len]
    for ul in unique_lens:
        cu_seqlens_list.append(cu_seqlens_list[-1] + ul)
    cu_seqlens = torch.tensor(cu_seqlens_list, device=device, dtype=torch.int32)
    max_seqlen = common_len + max(unique_lens)
    
    return Q, K, V, cu_seqlens, max_seqlen, total_len


def benchmark_kernel(func, warmup_iters, benchmark_iters):
    """Benchmark a kernel function."""
    for _ in range(warmup_iters):
        func()
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(benchmark_iters):
        func()
    torch.cuda.synchronize()
    
    return (time.perf_counter() - start) / benchmark_iters * 1000  # ms


def run_benchmark(args):
    """Run the benchmark with given arguments."""
    device = "cuda"
    dtype = torch.bfloat16
    
    print(f"{'='*100}")
    print(f"Common Prefix Benchmark")
    print(f"{'='*100}")
    print(f"Config: batch_size={args.batch_size}, avg_unique_len={args.avg_unique_len}, "
          f"num_heads={args.num_heads}, head_dim={args.head_dim}")
    print(f"Warmup: {args.warmup_iters}, Benchmark: {args.benchmark_iters} iterations")
    print(f"{'='*100}")
    print(f"{'common_len':>12} | {'total_len':>10} | {'Baseline':>12} | {'Common':>12} | "
          f"{'Speedup':>8} | {'Mem Save':>10} | {'Compute Save':>12}")
    print(f"{'-'*100}")
    
    for common_len in args.common_lens:
        # Generate unique lengths
        torch.manual_seed(args.seed)
        unique_lens = torch.randint(
            args.avg_unique_len - 100, 
            args.avg_unique_len + 100, 
            (args.batch_size,)
        ).tolist()
        
        # Create baseline data
        Q_base, K_base, V_base, cu_base, max_base, total_base = create_baseline_data(
            args.batch_size, common_len, unique_lens, args.num_heads, args.head_dim, device, dtype
        )
        
        # Create common prefix data
        Q_cp, K_cp, V_cp, cu_cp, max_cp, total_cp = create_common_prefix_data(
            args.batch_size, common_len, unique_lens, args.num_heads, args.head_dim, device, dtype
        )
        
        # Benchmark baseline
        baseline_time = benchmark_kernel(
            lambda: flash_attn_varlen_func(
                Q_base, K_base, V_base, cu_base, cu_base, max_base, max_base, causal=True
            ),
            args.warmup_iters, args.benchmark_iters
        )
        
        # Benchmark common prefix
        common_time = benchmark_kernel(
            lambda: flash_attn_varlen_func(
                Q_cp, K_cp, V_cp, cu_cp, cu_cp, max_cp, max_cp, causal=True, common_len=common_len
            ),
            args.warmup_iters, args.benchmark_iters
        )
        
        # Calculate metrics
        speedup = baseline_time / common_time
        mem_savings = (total_base - total_cp) / total_base * 100
        
        # Compute savings: (batch-1) * common² / 2 vs total compute
        total_compute = sum((common_len + ul) ** 2 / 2 for ul in unique_lens)
        saved_compute = (args.batch_size - 1) * common_len ** 2 / 2
        compute_savings = saved_compute / total_compute * 100
        
        print(f"{common_len:>12} | {common_len + args.avg_unique_len:>10} | "
              f"{baseline_time:>10.3f}ms | {common_time:>10.3f}ms | "
              f"{speedup:>7.2f}x | {mem_savings:>9.1f}% | {compute_savings:>11.1f}%")
    
    print(f"{'='*100}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark common prefix optimization")
    parser.add_argument("--batch_size", type=int, default=50, help="Batch size")
    parser.add_argument("--avg_unique_len", type=int, default=700, help="Average unique sequence length")
    parser.add_argument("--num_heads", type=int, default=32, help="Number of attention heads")
    parser.add_argument("--head_dim", type=int, default=128, help="Head dimension")
    parser.add_argument("--common_lens", type=int, nargs="+", default=[128, 256, 384, 512, 1024, 2048],
                        help="Common prefix lengths to test (must be multiples of 128)")
    parser.add_argument("--warmup_iters", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--benchmark_iters", type=int, default=100, help="Benchmark iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Validate common_lens
    for cl in args.common_lens:
        assert cl % 128 == 0, f"common_len {cl} must be multiple of 128"
    
    run_benchmark(args)


if __name__ == "__main__":
    main()
