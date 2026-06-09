
from data.zinc_loader import load_zinc

train_loader, val_loader, test_loader = load_zinc(batch_size=4)

for batch in train_loader:
    print(batch)
    print(batch.x.shape)
    print(batch.edge_index.shape)
    print(batch.y.shape)
    break