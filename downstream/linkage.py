import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import folium
from geopy.distance import geodesic
import warnings
import gc
import torch
import pickle
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.metrics import top_k_accuracy_score

warnings.filterwarnings('ignore')


class StandardScaler:
    def __init__(self, mean=None, std=None, epsilon=1e-7):
        """Standard Scaler.
        The class can be used to normalize PyTorch Tensors using native functions. The module does not expect the
        tensors to be of any specific shape; as long as the features are the last dimension in the tensor, the module
        will work fine.
        :param mean: The mean of the features. The property will be set after a call to fit.
        :param std: The standard deviation of the features. The property will be set after a call to fit.
        :param epsilon: Used to avoid a Division-By-Zero exception.
        """
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def fit(self, values):
        dims = list(range(values.dim() - 1))
        self.mean = torch.mean(values, dim=dims)
        self.std = torch.std(values, dim=dims)

    def transform(self, values):
        return (values - self.mean) / (self.std + self.epsilon)

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)


class Model(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True)

        self.l1 = nn.Linear(64, 128)
        self.l2 = nn.Linear(128, 64)

        self.l3 = nn.Linear(5526, 5526*2)
        self.l4 = nn.Linear(5526*2, 5526)
        self.relu = nn.ReLU()

    def forward(self, x, seq, embed, batch_size):
        embed = self.l2(self.relu(self.l1(embed)))

        packed = pack_padded_sequence(x, seq, batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.gru(packed, None)

        embed = torch.matmul(hidden[-1].squeeze(), embed.T)
        logits = self.l4(self.relu(self.l3(embed)))

        return logits


save_df = open('driver_data.pkl', 'rb')
dataset = pickle.load(save_df)
save_df.close()

save_df = open('driver_embed.pkl', 'rb')
driver_embed = pickle.load(save_df).detach().cpu()

ids = np.loadtxt('driver_id.txt')
ids = ids.astype(np.int32)

seq_len = []
trip_data = []
for trip in tqdm(dataset):
    if 20 <= len(trip) <= 200:
        seq_len.append(len(trip))
        trip_data.append(trip.iloc[:, [0, 1, 2, 3, 6, 7, 8, 9, 10]].values)

dataset = pad_sequence([torch.from_numpy(x) for x in trip_data], batch_first=True).float()
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = Model(8, 64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
batch_size = 128
driver_embed = driver_embed.float().to(device)

celoss = nn.CrossEntropyLoss()

train_index = np.random.choice(len(dataset), int(0.9 * len(dataset)), replace=False)
test_index = np.setdiff1d(np.array([i for i in range(len(dataset))]), train_index)

train_data = dataset[train_index]
train_seq = np.array(seq_len)[train_index]

val_data = dataset[test_index][:20000]
val_seq = np.array(seq_len)[test_index][:20000]
val_label = val_data[:, :, 0][:, 0]
val_data = val_data[:, :, 1:].to(device)
val_y = []
for i in val_label:
    val_y.append(np.argwhere(ids == int(i))[0][0])

test_data = dataset[test_index][20000:]
test_seq = np.array(seq_len)[test_index][20000:]
test_label = test_data[:, :, 0][:, 0]
test_data = test_data[:, :, 1:].to(device)
test_y = []
for i in test_label:
    test_y.append(np.argwhere(ids == int(i))[0][0])


min_val_loss = 1e10
max_100 = 0
for epoch in range(10000):
    rand_idx = np.random.randint(0, int(0.9 * len(dataset)), size=batch_size)
    batch_data = train_data[rand_idx, :, 1:].to(device)
    batch_seq = train_seq[rand_idx]
    batch_label = train_data[rand_idx, :, 0][:, 0].to(device)

    label = []
    for i in batch_label:
        label.append(np.argwhere(ids == int(i))[0][0])

    pred = model(batch_data, batch_seq, driver_embed, batch_size)

    loss = celoss(pred, torch.tensor(label, dtype=torch.int64).to(device))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    val_pred = model(test_data, test_seq, driver_embed, test_data.shape[0])
    val_loss = celoss(val_pred, torch.tensor(test_y, dtype=torch.int64).to(device))

    if val_loss < min_val_loss:
        min_val_loss = val_loss.item()
        best_model = model
    top_k_10 = top_k_accuracy_score(test_y, val_pred.detach().cpu().numpy(), k=100, labels=[i for i in range(5526)])

    if top_k_10 > max_100:
        max_100 = top_k_10
        print(top_k_10)

    if epoch % 2 == 0:
        print('Epoch {}, loss {}, val loss {}, min val loss {}'.format(epoch, loss.item(), val_loss.item(), min_val_loss))






