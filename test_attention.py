from data.zinc_loader import load_zinc
from models.embedding import AtomEncoder
from models.exphormer_attention import ExphormerAttention

train_loader, _, _ = load_zinc(batch_size=32)

batch = next(iter(train_loader))

encoder = AtomEncoder(128)

x_emb = encoder(batch.x)

attention = ExphormerAttention(128)

out = attention(
    x_emb,
    batch.edge_index
)

print("Input:", x_emb.shape)
print("Output:", out.shape)
print(out[:3])