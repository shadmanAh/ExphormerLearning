from data.zinc_loader import load_zinc

from models.exphormer_model import (
    ExphormerModel
)

train_loader, _, _ = load_zinc(batch_size=32)
batch = next(iter(train_loader))
model = ExphormerModel(hidden_dim=128,num_layers=3)
out = model(batch.x, batch.edge_index, batch.batch)

print("Prediction shape:", out.shape)
print(out[:5])