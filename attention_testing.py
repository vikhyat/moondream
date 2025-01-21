import torch

mask = torch.zeros(10, 10)

# 1) For rows 0–4, allow attention only to columns 0–4 (the "prefix" block)
mask[:5, :5] = 1

# 2) For rows 5–9, use causal attention:
#    row i can attend to columns [0..i]
for i in range(5, 10):
    mask[i, : i + 1] = 1

print(mask)