import torch


def dequantize_tensor(
    packed: torch.Tensor,
    scales: torch.Tensor,
    orig_shape: torch.Size,
    block_size: int,
    dtype: torch.dtype,
):
    """
    In-place–friendly dequantization of int4-packed data back to `dtype`,
    mutating `packed` (and reading `scales`) to avoid extra big intermediates.
    """
    # how many bytes encode each block of `block_size` 4-bit values
    num_bytes = (block_size + 1) // 2
    num_blocks = packed.numel() // num_bytes

    # view as [blocks, bytes_per_block]
    pr = packed.view(num_blocks, num_bytes)

    # prepare output in the target dtype
    out = torch.empty((num_blocks, block_size), device=packed.device, dtype=dtype)

    # ---- lower nibble ----
    lower = pr & 0xF  # [blocks, bytes]
    lower = lower.to(torch.int8)  # cast to signed
    lower[lower >= 8] -= 16  # sign-correct

    lo_count = (block_size + 1) // 2
    out[:, 0:block_size:2] = lower[:, :lo_count].to(dtype) * scales.view(-1, 1)

    # ---- upper nibble ----
    pr >>= 4  # in-place shift of the original packed bytes
    upper = pr & 0xF
    upper = upper.to(torch.int8)
    upper[upper >= 8] -= 16

    hi_count = block_size // 2
    out[:, 1:block_size:2] = upper[:, :hi_count].to(dtype) * scales.view(-1, 1)

    # restore original shape
    return out.view(orig_shape)
