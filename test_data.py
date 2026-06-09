
from data.zinc_loader import load_zinc
from models.embedding import AtomEncoder

train_loader, val_loader, test_loader = load_zinc(batch_size=4)

for batch in train_loader:
    print(batch)
    print(batch.x.shape)
    print(batch.edge_index.shape)
    print(batch.y.shape)
    break

# print(batch)

# print("x shape:", batch.x.shape)
# print("edge_index shape:", batch.edge_index.shape)

# print("batch shape:", batch.batch.shape)

# print("y shape:", batch.y.shape)

# print("unique node features:")
# print(torch.unique(batch.x))

encoder = AtomEncoder(128)

out = encoder(batch.x)

print(out.shape)