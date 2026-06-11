import torch
import torch.nn as nn
from torch.optim import AdamW

from data.zinc_loader import load_zinc
from models.exphormer_model import ExphormerModel


# ----------------------
# Device
# ----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------
# Data
# ----------------------
train_loader, val_loader, test_loader = load_zinc(batch_size=32)
sample_batch = next(iter(train_loader))
num_embeddings = int(sample_batch.x.max().item()) + 1
print("max index:", sample_batch.x.max())
print("num_embeddings:", num_embeddings)
print(sample_batch.x.dtype)
print(sample_batch.x[:10])

# ----------------------
# Model
# ----------------------
model = ExphormerModel(hidden_dim=128, num_layers=3, num_embeddings=num_embeddings).to(device)

# ----------------------
# Optimizer
# ----------------------
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

# ----------------------
# Loss (ZINC = MAE)
# ----------------------
criterion = nn.L1Loss()

# ----------------------
# Train function
# ----------------------
def train():
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model( batch.x, batch.edge_index, batch.batch)
        loss = criterion(pred, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)

# ----------------------
# Eval function
# ----------------------
@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss = 0

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.x, batch.edge_index, batch.batch)
        loss = criterion(pred, batch.y.view(-1))
        total_loss += loss.item()

    return total_loss / len(loader)

# ----------------------
# Training Loop
# ----------------------
best_val = float("inf")

for epoch in range(1, 51):
    train_loss = train()
    val_loss = evaluate(val_loader)
    print(
        f"Epoch {epoch:03d} | "
        f"Train MAE: {train_loss:.4f} | "
        f"Val MAE: {val_loss:.4f}"
    )

    if val_loss < best_val:
        best_val = val_loss
        torch.save(model.state_dict(), "best_exphormer.pt")
        print("Model saved!")