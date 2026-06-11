import torch
from torch.optim import AdamW

from data.zinc_loader import load_zinc
from models.test.baseline import BaselineModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, val_loader, test_loader = load_zinc(batch_size=32)

batch = next(iter(train_loader))

sample_batch = next(iter(train_loader))

in_dim = sample_batch.x.shape[1]

model = BaselineModel(in_dim=in_dim,hidden_dim=128).to(device)

optimizer = AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=1e-5
)

criterion = torch.nn.L1Loss()

def train_epoch():
    model.train()

    total_loss = 0

    for batch in train_loader:

        batch = batch.to(device)

        optimizer.zero_grad()

        pred = model(batch.x.float(), batch.batch)

        loss = criterion(pred, batch.y.float())

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


@torch.no_grad()
def evaluate(loader):

    model.eval()

    total_loss = 0

    for batch in loader:

        batch = batch.to(device)

        pred = model(batch.x.float(), batch.batch)

        loss = criterion(pred, batch.y.float())

        total_loss += loss.item()

    return total_loss / len(loader)

for epoch in range(50):

    train_loss = train_epoch()

    val_loss = evaluate(val_loader)

    print(
        f"Epoch {epoch+1:03d} | "
        f"Train {train_loss:.4f} | "
        f"Val {val_loss:.4f}"
    )