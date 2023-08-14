import pickle
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from eta_model import ETA
import random
from torch.nn.utils.rnn import pack_padded_sequence
import warnings

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


def main():
    # dataset
    save_df = open('road_dataset.pkl', 'rb')
    road_dataset = pickle.load(save_df)
    save_df.close()

    save_df = open('dataset.pkl', 'rb')
    dataset = np.array(pickle.load(save_df))
    save_df.close()

    save_df = open('data_label.pkl', 'rb')
    label = np.array(pickle.load(save_df))
    save_df.close()

    save_df = open('driver_embed.pkl', 'rb')
    driver_embed = pickle.load(save_df).detach().cpu()
    save_df.close()

    # good driver ids
    ids = np.loadtxt('driver_id.txt')
    ids = ids.astype(np.int32)

    cor_id = []
    for i in range(len(dataset)):
        if int(dataset[i][0]) in ids:
            cor_id.append(i)

    dataset = dataset[cor_id]
    label = label[cor_id]

    # WDR
    # data treatment
    seq_len = []
    for trip in road_dataset:
        seq_len.append(len(trip))

    seq_len = np.array(seq_len)
    seq_len = seq_len[cor_id]
    road_dataset = pad_sequence([torch.from_numpy(np.array(x)) for x in road_dataset], batch_first=True).float()
    road_dataset = torch.tensor(road_dataset, dtype=torch.long)  # / 57846
    road_dataset = road_dataset[cor_id]
    dataset = torch.tensor(dataset)

    dataset_driver_embed = []
    driver_id = dataset[:, 0].long().numpy()
    for driver in driver_id:
        index = np.where(ids == driver)[0][0]
        dataset_driver_embed.append(driver_embed[index])

    dataset_driver_embed = torch.cat(dataset_driver_embed).view(-1, 64)

    # index generation
    train_index = np.random.choice(len(dataset), int(0.9 * len(dataset)), replace=False)
    test_index = np.setdiff1d(np.array([i for i in range(len(dataset))]), train_index)

    train_deep = dataset[train_index]



    train_rnn = road_dataset[train_index] / 57845
    train_label = label[train_index]

    max_label = max(train_label)
    min_label = min(train_label)
    # train_label = (train_label - min_label) / (max_label - min_label)
    train_seq = seq_len[train_index]
    train_driver = dataset_driver_embed[train_index]

    # val
    val_deep = dataset[test_index][:10000]
    val_rnn = road_dataset[test_index][:10000]
    val_label = label[test_index][:10000]
    val_seq = seq_len[test_index][:10000]
    val_driver = dataset_driver_embed[test_index][:10000]

    # test
    test_deep = dataset[test_index][10000:]
    test_rnn = road_dataset[test_index][10000:]
    test_label = label[test_index][10000:]
    test_seq = seq_len[test_index][10000:]
    test_driver = dataset_driver_embed[test_index][10000:]

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = ETA(True).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse_loss = nn.MSELoss()
    batch_size = 512

    min_val_loss = 1

    for epoch in range(50):
        model.train()
        losses = []
        init_idx = 0
        while init_idx <= len(train_deep):
            if init_idx+batch_size <= len(train_deep):
                end_idx = init_idx+batch_size
            else:
                end_idx = len(train_deep)

            batch_deep = train_deep[init_idx:end_idx].float().to(device)
            batch_rnn = train_rnn[init_idx:end_idx].float().to(device)
            batch_seq = train_seq[init_idx:end_idx]
            batch_label = torch.from_numpy(train_label[init_idx:end_idx]).float().to(device)
            batch_driver = train_driver[init_idx:end_idx].float().to(device)

            pred = model(batch_deep, batch_rnn, batch_driver, batch_seq)

            # loss = mse_loss(pred.reshape(-1), batch_label)
            loss = torch.mean(torch.abs(batch_label - pred.reshape(-1)) / batch_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            init_idx += batch_size

        # eval
        model.eval()
        val_pred = model(val_deep.float().to(device), val_rnn.float().to(device), val_driver.float().to(device), val_seq)
        val_loss = torch.mean(torch.abs(torch.from_numpy(val_label).float().to(device) - val_pred.reshape(-1)) /
                              torch.from_numpy(val_label).float().to(device))

        if epoch % 5 == 0:
            print('Epoch {}, loss {}, val_loss {}'.format(epoch, sum(losses) / len(losses), val_loss.item()))

        if val_loss.item() < min_val_loss:
            min_val_loss = val_loss.item()
            best_model = model

    # eval
    best_model.eval()
    test_label = torch.from_numpy(test_label).float().to(device)
    test_pred = best_model(test_deep.float().to(device), test_rnn.float().to(device), test_driver.float().to(device),
                     test_seq)
    rmse = torch.sqrt(mse_loss(test_pred.reshape(-1), test_label))
    mape = torch.mean(torch.abs(test_label - test_pred.reshape(-1)) / test_label)
    mae = torch.mean(torch.abs(test_pred.reshape(-1) - test_label))

    print('RMSE {}, MAPE {}, MAE {}'.format(rmse.item(), mape.item(), mae.item()))

    return rmse, mape, mae


if __name__ == '__main__':
    mape_list = []
    rmse_list = []
    mae_list = []

    rmse, mape, mae = main()

    mape_list.append(mape)
    rmse_list.append(rmse)
    mae_list.append(mae)

    print('MAPE {}, RMSE {}, MAE {}'.format(
        sum(mape_list)/len(mape_list),
        sum(rmse_list)/len(rmse_list),
        sum(mae_list)/len(mae_list)))
