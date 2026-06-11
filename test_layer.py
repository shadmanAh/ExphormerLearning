from data.zinc_loader import load_zinc

from models.embedding import AtomEncoder
from models.exphormer_layer import ExphormerLayer


train_loader, _, _ = load_zinc(batch_size=32)

batch = next(iter(train_loader))

encoder = AtomEncoder(128)

x = encoder(batch.x)

layer = ExphormerLayer(hidden_dim=128)

out = layer(x,batch.edge_index)

print("Input Shape:",x.shape)

print("Output Shape:",out.shape)

print(out[:3])