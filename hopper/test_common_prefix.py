"""
Test common prefix optimization for FlashAttention.

Data layout:
- Physical: [Common, Unique0, Unique1, ...]
- cu_seqlens: [0, common_len, common_len+u0, common_len+u0+u1, ...]
- bidb=0: processes Common (seqlen_k = common_len)
- bidb>0: processes Unique_i (seqlen_k = unique_len_i + common_len)

Constraints:
- common_len must be multiple of kBlockN (128 for head_dim=128, causal=True)
- causal=True required
"""

import torch
import pytest
from flash_attn_interface import flash_attn_varlen_func


def reference_common_prefix(q_common, k_common, v_common, q_uniques, k_uniques, v_uniques, common_len):
    """Reference implementation: expand common to each batch and run standard attention."""
    batch_size = len(q_uniques)
    unique_lens = [q.shape[0] for q in q_uniques]
    
    q_full_list = [torch.cat([q_common, q_uniques[i]], dim=0) for i in range(batch_size)]
    k_full_list = [torch.cat([k_common, k_uniques[i]], dim=0) for i in range(batch_size)]
    v_full_list = [torch.cat([v_common, v_uniques[i]], dim=0) for i in range(batch_size)]
    
    q_full = torch.cat(q_full_list, dim=0)
    k_full = torch.cat(k_full_list, dim=0)
    v_full = torch.cat(v_full_list, dim=0)
    
    cu_seqlens = [0]
    for i in range(batch_size):
        cu_seqlens.append(cu_seqlens[-1] + common_len + unique_lens[i])
    cu_seqlens = torch.tensor(cu_seqlens, device=q_common.device, dtype=torch.int32)
    max_seqlen = common_len + max(unique_lens)
    
    out_ref = flash_attn_varlen_func(
        q_full, k_full, v_full,
        cu_seqlens, cu_seqlens,
        max_seqlen, max_seqlen,
        causal=True
    )
    return out_ref, cu_seqlens


def run_common_prefix_test(batch_size, common_len, unique_lens, num_heads, head_dim, dtype=torch.bfloat16):
    """Run common prefix test with given parameters."""
    device = "cuda"
    max_seq_len = common_len + max(unique_lens)
    
    # Create data
    q_common = torch.randn(common_len, num_heads, head_dim, device=device, dtype=dtype)
    k_common = torch.randn(common_len, num_heads, head_dim, device=device, dtype=dtype)
    v_common = torch.randn(common_len, num_heads, head_dim, device=device, dtype=dtype)
    
    q_uniques = [torch.randn(l, num_heads, head_dim, device=device, dtype=dtype) for l in unique_lens]
    k_uniques = [torch.randn(l, num_heads, head_dim, device=device, dtype=dtype) for l in unique_lens]
    v_uniques = [torch.randn(l, num_heads, head_dim, device=device, dtype=dtype) for l in unique_lens]
    
    # Pack: [Common, Unique0, Unique1, ...]
    q_packed = torch.cat([q_common] + q_uniques, dim=0)
    k_packed = torch.cat([k_common] + k_uniques, dim=0)
    v_packed = torch.cat([v_common] + v_uniques, dim=0)
    
    # cu_seqlens: [0, common_len, common_len+u0, ...]
    cu_seqlens_list = [0, common_len]
    for ul in unique_lens:
        cu_seqlens_list.append(cu_seqlens_list[-1] + ul)
    cu_seqlens = torch.tensor(cu_seqlens_list, device=device, dtype=torch.int32)
    
    # Run common prefix kernel
    out = flash_attn_varlen_func(
        q_packed, k_packed, v_packed,
        cu_seqlens, cu_seqlens,
        max_seq_len, max_seq_len,
        common_len=common_len,
        causal=True
    )
    
    # Reference
    out_ref, cu_seqlens_ref = reference_common_prefix(
        q_common, k_common, v_common,
        q_uniques, k_uniques, v_uniques,
        common_len
    )
    
    # Compare Common part
    out_common = out[:common_len]
    out_ref_common = out_ref[:common_len]
    diff_common = (out_common - out_ref_common).abs().max().item()
    
    # Compare Unique parts
    diff_unique_max = 0
    offset_out = common_len
    for i in range(batch_size):
        len_i = unique_lens[i]
        out_unique_i = out[offset_out : offset_out + len_i]
        offset_out += len_i
        
        start_ref = cu_seqlens_ref[i] + common_len
        out_ref_unique_i = out_ref[start_ref : start_ref + len_i]
        
        diff_i = (out_unique_i - out_ref_unique_i).abs().max().item()
        diff_unique_max = max(diff_unique_max, diff_i)
    
    return diff_common, diff_unique_max


class TestCommonPrefix:
    """Test suite for common prefix optimization."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
    
    # ==================== Basic functionality ====================
    
    @pytest.mark.parametrize("common_len", [128, 256, 512])
    def test_basic_common_len(self, common_len):
        """Test different common_len values (must be multiple of 128)."""
        batch_size = 10
        unique_lens = [256] * batch_size
        diff_common, diff_unique = run_common_prefix_test(
            batch_size, common_len, unique_lens, num_heads=8, head_dim=128
        )
        assert diff_common < 1e-2, f"Common diff {diff_common} too large"
        assert diff_unique < 1e-2, f"Unique diff {diff_unique} too large"
    
    @pytest.mark.parametrize("batch_size", [1, 2, 10, 50])
    def test_batch_sizes(self, batch_size):
        """Test different batch sizes."""
        common_len = 128
        unique_lens = [256] * batch_size
        diff_common, diff_unique = run_common_prefix_test(
            batch_size, common_len, unique_lens, num_heads=8, head_dim=128
        )
        assert diff_common < 1e-2, f"Common diff {diff_common} too large"
        assert diff_unique < 1e-2, f"Unique diff {diff_unique} too large"
    
    @pytest.mark.parametrize("num_heads", [1, 4, 8, 32])
    def test_num_heads(self, num_heads):
        """Test different number of heads."""
        diff_common, diff_unique = run_common_prefix_test(
            batch_size=5, common_len=128, unique_lens=[256]*5,
            num_heads=num_heads, head_dim=128
        )
        assert diff_common < 1e-2, f"Common diff {diff_common} too large"
        assert diff_unique < 1e-2, f"Unique diff {diff_unique} too large"
    
    # ==================== Variable unique lengths ====================
    
    def test_variable_unique_lengths(self):
        """Test with different unique lengths per batch."""
        batch_size = 10
        common_len = 128
        torch.manual_seed(42)
        unique_lens = torch.randint(128, 512, (batch_size,)).tolist()
        
        diff_common, diff_unique = run_common_prefix_test(
            batch_size, common_len, unique_lens, num_heads=8, head_dim=128
        )
        assert diff_common < 1e-2, f"Common diff {diff_common} too large"
        assert diff_unique < 1e-2, f"Unique diff {diff_unique} too large"
    
    def test_large_variance_unique_lengths(self):
        """Test with large variance in unique lengths."""
        batch_size = 20
        common_len = 256
        # Mix of short and long unique sequences
        unique_lens = [128, 1024, 256, 768, 512, 384, 640, 896, 192, 448] * 2
        
        diff_common, diff_unique = run_common_prefix_test(
            batch_size, common_len, unique_lens, num_heads=8, head_dim=128
        )
        assert diff_common < 1e-2, f"Common diff {diff_common} too large"
        assert diff_unique < 1e-2, f"Unique diff {diff_unique} too large"
    
    # ==================== Edge cases ====================
    
    def test_single_batch(self):
        """Test with single batch (batch_size=1)."""
        diff_common, diff_unique = run_common_prefix_test(
            batch_size=1, common_len=128, unique_lens=[256],
            num_heads=8, head_dim=128
        )
        assert diff_common < 1e-2, f"Common diff {diff_common} too large"
        assert diff_unique < 1e-2, f"Unique diff {diff_unique} too large"
    
    def test_minimal_unique_length(self):
        """Test with minimal unique length (one block)."""
        batch_size = 5
        common_len = 128
        unique_lens = [128] * batch_size  # Minimal: one kBlockN
        
        diff_common, diff_unique = run_common_prefix_test(
            batch_size, common_len, unique_lens, num_heads=8, head_dim=128
        )
        assert diff_common < 1e-2, f"Common diff {diff_common} too large"
        assert diff_unique < 1e-2, f"Unique diff {diff_unique} too large"
    
    def test_large_common_prefix(self):
        """Test with large common prefix (common > unique)."""
        batch_size = 10
        common_len = 1024
        unique_lens = [256] * batch_size
        
        diff_common, diff_unique = run_common_prefix_test(
            batch_size, common_len, unique_lens, num_heads=8, head_dim=128
        )
        assert diff_common < 1e-2, f"Common diff {diff_common} too large"
        assert diff_unique < 1e-2, f"Unique diff {diff_unique} too large"
    
    def test_non_aligned_unique_lengths(self):
        """Test with unique lengths not aligned to kBlockN."""
        batch_size = 5
        common_len = 128
        # Unique lengths not multiples of 128
        unique_lens = [130, 255, 333, 417, 500]
        
        diff_common, diff_unique = run_common_prefix_test(
            batch_size, common_len, unique_lens, num_heads=8, head_dim=128
        )
        assert diff_common < 1e-2, f"Common diff {diff_common} too large"
        assert diff_unique < 1e-2, f"Unique diff {diff_unique} too large"
    
    # ==================== Data types ====================
    
    def test_dtype_bfloat16(self):
        """Test bfloat16 data type."""
        diff_common, diff_unique = run_common_prefix_test(
            batch_size=5, common_len=128, unique_lens=[256]*5,
            num_heads=8, head_dim=128, dtype=torch.bfloat16
        )
        assert diff_common < 1e-2, f"Common diff {diff_common} too large"
        assert diff_unique < 1e-2, f"Unique diff {diff_unique} too large"
    
    # ==================== Stress tests ====================
    
    def test_large_batch(self):
        """Test with large batch size."""
        batch_size = 100
        common_len = 256
        unique_lens = [512] * batch_size
        
        diff_common, diff_unique = run_common_prefix_test(
            batch_size, common_len, unique_lens, num_heads=32, head_dim=128
        )
        assert diff_common < 1e-2, f"Common diff {diff_common} too large"
        assert diff_unique < 1e-2, f"Unique diff {diff_unique} too large"
    
    def test_many_heads(self):
        """Test with many attention heads."""
        diff_common, diff_unique = run_common_prefix_test(
            batch_size=10, common_len=128, unique_lens=[256]*10,
            num_heads=64, head_dim=128
        )
        assert diff_common < 1e-2, f"Common diff {diff_common} too large"
        assert diff_unique < 1e-2, f"Unique diff {diff_unique} too large"
    
    # ==================== Regression tests ====================
    
    def test_regression_random_configs(self):
        """Random configurations for regression testing."""
        torch.manual_seed(12345)
        
        for _ in range(5):
            batch_size = torch.randint(2, 30, (1,)).item()
            common_len = torch.randint(1, 8, (1,)).item() * 128  # Multiple of 128
            num_heads = 2 ** torch.randint(0, 5, (1,)).item()  # 1, 2, 4, 8, 16
            unique_lens = torch.randint(128, 800, (batch_size,)).tolist()
            
            diff_common, diff_unique = run_common_prefix_test(
                batch_size, common_len, unique_lens, num_heads=num_heads, head_dim=128
            )
            assert diff_common < 1e-2, f"Common diff {diff_common} too large"
            assert diff_unique < 1e-2, f"Unique diff {diff_unique} too large"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
