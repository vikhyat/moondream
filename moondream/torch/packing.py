import torch


def unpack_int4(packed: torch.Tensor, original_length: int) -> torch.Tensor:
    orig_shape = packed.shape
    last_dim = orig_shape[-1]
    batch_shape = orig_shape[:-1]
    flat_packed = packed.reshape(-1, last_dim)
    batch_size = flat_packed.shape[0]
    flat_bytes = flat_packed.reshape(-1)
    lower = flat_bytes & 0xF
    upper = (flat_bytes >> 4) & 0xF
    unpacked = torch.stack([lower, upper], dim=1).reshape(batch_size, last_dim * 2)
    unpacked = unpacked[:, :original_length]
    unpacked = unpacked.reshape(*batch_shape, original_length)
    return unpacked.to(torch.int8)


def dequantize_tensor(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zero_points: torch.Tensor,
    orig_shape: torch.Size,
    block_size: int,
    dtype: torch.dtype = torch.bfloat16,
):
    out_features, num_blocks, _ = packed.shape
    unpacked = unpack_int4(packed, block_size)
    scales_view = scales.unsqueeze(2)  # Shape: [out_features, num_blocks, 1]
    zero_points_view = zero_points.unsqueeze(2)  # Shape: [out_features, num_blocks, 1]
    dequantized = (unpacked.float() - zero_points_view) * scales_view
    dequantized = dequantized.reshape(out_features, num_blocks * block_size)
    dequantized = dequantized[:, : orig_shape[1]]
    dequantized = dequantized.reshape(orig_shape)
    return dequantized.to(dtype)
