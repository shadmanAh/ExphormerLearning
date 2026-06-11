import torch

from models.test.exphormer_model import ExphormerModel

X = torch.tensor([
    [1.0,0.5,2.0],
    [0.3,1.5,0.7],
    [2.0,1.0,0.2],
    [1.2,0.1,1.8],
    [0.0,0.0,0.0]
])

mask = torch.tensor([
    [0,1,0,1,1],
    [1,0,1,0,1],
    [0,1,0,1,1],
    [1,0,1,0,1],
    [1,1,1,1,0]
])

labels = torch.tensor([
    0,
    1,
    0,
    1,
    0
])

model = ExphormerModel(in_dim=3, hidden_dim=8, num_heads=2, num_layers=3)

criterion = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # parameters() چیست؟

for epoch in range(100):

    optimizer.zero_grad() # این چکار میکند؟
    out = model(X, mask)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step() # این دقیقا چکار میکند؟

    if epoch % 10 == 0:
        print(epoch, loss.item())