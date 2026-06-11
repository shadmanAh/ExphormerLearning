import torch

from models.test.exphormer_block import ExphormerBlock

hidden_dim = 8

block = ExphormerBlock(hidden_dim)

H = torch.randn(5, 8)

attention_out = torch.randn(5,8)

out = block(H, attention_out)
x = block.forward(H , attention_out)

print("out: ")
print(out)

print("\n x: ")
print(x)