import torch

H = torch.randn(5, 8)

attention_out = torch.randn(5, 8)

print("H shape:")
print(H)

print("\nAttention shape:")
print(attention_out)

H_new = H + attention_out

print("\nOutput shape:")
print(H_new)