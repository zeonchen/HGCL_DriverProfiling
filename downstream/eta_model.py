import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# layer
class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim):
        super(FeaturesLinear, self).__init__()
        self.fc = torch.nn.Embedding(field_dims, output_dim)  # fc: Embedding:(610 + 193609, 1) 做一维特征的嵌入表示
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))  # Tensor: 1

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        # x:tensor([[554, 2320], [304, 3993]])
        # x: Tensor: 2048, 每个Tensor维度为2, x.new_tensor(self.offsets).unsqueeze(0): tensor([[0, 610]])
        # x: Tensor: [2048, 2]
        return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super(FeaturesEmbedding, self).__init__()
        self.embedding = torch.nn.Embedding(field_dims, embed_dim)  # embeddingL Embedding:(610+193609, 16)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        # embedding weight的初始化通过均匀分布的采用得到

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return self.embedding(x)


class FactorizationMachine(torch.nn.Module):

    def __init__(self):
        super(FactorizationMachine, self).__init__()

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        return 0.5 * ix


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout=0, output_layer=False):
        super(MultiLayerPerceptron, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        return self.mlp(x)


class Regressor(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Regressor, self).__init__()
        self.linear_wide = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.linear_deep = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.linear_recurrent = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.out_layer = MultiLayerPerceptron(output_dim, (output_dim,), output_layer=True)

    def forward(self, wide, deep, recurrent):
        fuse = self.linear_wide(wide) + self.linear_deep(deep) + self.linear_recurrent(recurrent)
        return self.out_layer(fuse)


# model
class ETA(nn.Module):
    def __init__(self, if_driver):
        super(ETA, self).__init__()
        wide_field_dims = np.array([19992, 414, 24, 24])
        wide_embed_dim = 20
        wide_mlp_dims = (128,)

        deep_field_dims = np.array([24, 24])
        deep_embed_dim = 20

        if if_driver:
            deep_real_dim = 2 + 64
        else:
            deep_real_dim = 2
        deep_category_dim = 2
        deep_mlp_input_dim = deep_embed_dim * deep_category_dim + deep_real_dim
        deep_mlp_dims = (128,)

        id_dims = 57846
        id_embed_dim = 20
        slice_dims = 289
        slice_embed_dim = 20
        all_real_dim = 1
        mlp_out_dim = 20
        lstm_hidden_size = 128

        reg_input_dim = 128
        reg_output_dim = 128

        self.wide_embedding = FeaturesEmbedding(sum(wide_field_dims), wide_embed_dim)
        self.fm = torch.nn.Sequential(
            FactorizationMachine(),
            torch.nn.BatchNorm1d(wide_embed_dim),
        )
        self.wide_mlp = MultiLayerPerceptron(wide_embed_dim, wide_mlp_dims)  # 不batchnorm

        self.deep_embedding = FeaturesEmbedding(sum(deep_field_dims), deep_embed_dim)
        self.deep_mlp = MultiLayerPerceptron(deep_mlp_input_dim, deep_mlp_dims)

        self.slice_embedding = nn.Embedding(slice_dims, slice_embed_dim)
        self.id_embedding = nn.Embedding(id_dims, id_embed_dim)
        self.all_mlp = nn.Sequential(
            nn.Linear(id_embed_dim + slice_embed_dim + all_real_dim , mlp_out_dim),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(input_size=mlp_out_dim, hidden_size=lstm_hidden_size, num_layers=2, batch_first=True)
        self.regressor = Regressor(reg_input_dim, reg_output_dim)

    def forward(self, wide_index, wide_value, deep_category, deep_real,
                all_id, seq_len):
        # wide
        wide_embedding = self.wide_embedding(wide_index)  # 对所有item特征做embedding,对连续特征做一维embedding
        cross_term = self.fm(wide_embedding * wide_value.unsqueeze(2))  # wide_value前两列为1，之后为dense feature数值
        wide_output = self.wide_mlp(cross_term)

        # deep part
        batch_size = deep_real.shape[0]
        deep_embedding = self.deep_embedding(deep_category).view(batch_size, -1)
        deep_input = torch.cat([deep_embedding, deep_real], dim=1).float()
        deep_output = self.deep_mlp(deep_input)

        # recurrent part
        all_id_embedding = self.id_embedding(all_id)
        # recurrent_input = self.all_mlp(all_input)
        packed_all_input = pack_padded_sequence(all_id_embedding, seq_len, enforce_sorted=False, batch_first=True)
        out, (hn, cn) = self.lstm(packed_all_input)
        hn = hn.squeeze()
        hn = hn[1, :, :]

        # regressor
        result = self.regressor(wide_output, deep_output, hn)

        return result.squeeze(1)
