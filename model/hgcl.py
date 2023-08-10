import torch.nn.functional as F
import torch
from math import ceil
from torch_geometric.nn import DenseGraphConv, dense_diff_pool, GraphNorm
import torch.nn as nn
import numpy as np
import random
import copy
from torch.nn.utils.rnn import pack_padded_sequence


class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 aggr='mean', num_layer=1):
        super(GNN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        self.convs.append(DenseGraphConv(in_channels, hidden_channels, aggr=aggr))
        self.bns.append(GraphNorm(hidden_channels))

        for _ in range(num_layer):
            self.convs.append(DenseGraphConv(hidden_channels, hidden_channels, aggr=aggr))
            self.bns.append(GraphNorm(hidden_channels))

        self.convs.append(DenseGraphConv(hidden_channels, out_channels, aggr=aggr))
        self.bns.append(GraphNorm(out_channels))

    def forward(self, x, adj, mask=None):
        for step in range(len(self.convs)):
            batch_size, node_num, feat_dim = x.shape
            # x = self.bns[step](F.relu(self.convs[step](x, adj, mask)))
            x = F.relu(self.convs[step](x, adj, mask))
            x = x.view(-1, x.shape[-1])
            x = self.bns[step](x)
            x = x.view(batch_size, node_num, -1)

        return x


class DiffPool(nn.Module):
    def __init__(self, max_nodes, input_dim, hidden_dim, outpu_dim, aggr='mean', ratio=0.5, mode='others'):
        super(DiffPool, self).__init__()
        num_nodes = ceil(ratio * max_nodes)
        self.gnn1_pool = GNN(input_dim, hidden_dim, num_nodes, aggr=aggr, num_layer=3)
        self.gnn1_embed = GNN(input_dim, hidden_dim, hidden_dim, aggr=aggr, num_layer=3)

        num_nodes = ceil(ratio * num_nodes)
        self.gnn2_pool = GNN(hidden_dim, hidden_dim, num_nodes, aggr=aggr, num_layer=3)
        self.gnn2_embed = GNN(hidden_dim, hidden_dim, hidden_dim, aggr=aggr, num_layer=3)

        self.gnn3_embed = GNN(hidden_dim, hidden_dim, hidden_dim, aggr=aggr, num_layer=2)

        self.lin1 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = torch.nn.Linear(hidden_dim, outpu_dim)

    def forward(self, x, adj, mask=None):
        s = self.gnn1_pool(x, adj, mask)
        x = self.gnn1_embed(x, adj, mask)

        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        s = self.gnn2_pool(x, adj)
        x = self.gnn2_embed(x, adj)

        x, adj, l2, e2 = dense_diff_pool(x, adj, s)

        x = self.gnn3_embed(x, adj)

        x = x.mean(dim=1)
        x = F.relu(self.lin1(x))
        x = self.lin2(x)

        return x, l1 + l2, e1 + e2


class HGCL(nn.Module):
    def __init__(self, hidden_dim=16):
        super(HGCL, self).__init__()
        # encoders
        self.gru = nn.GRU(input_size=22, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.relational_traj_linear = nn.Linear(hidden_dim, 1)
        self.traj_gnn = GNN(in_channels=hidden_dim, hidden_channels=64, out_channels=hidden_dim, num_layer=1)
        self.urban_gnn = GNN(in_channels=hidden_dim, hidden_channels=64, out_channels=hidden_dim, num_layer=1)
        self.urban_diffpool = DiffPool(max_nodes=900, input_dim=hidden_dim+19+4, hidden_dim=hidden_dim, outpu_dim=hidden_dim, ratio=0.25)
        self.semantic_diffpool = DiffPool(max_nodes=900, input_dim=hidden_dim+19+4, hidden_dim=hidden_dim, outpu_dim=hidden_dim, ratio=0.25)

        self.driver_encoder = nn.Sequential(nn.Linear(hidden_dim*2, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, hidden_dim))

    def traj_embed_forward(self, x, seq_len):
        packed = pack_padded_sequence(x, seq_len, batch_first=True, enforce_sorted=False)
        packed_outputs, hidden = self.gru(packed, None)
        out = hidden[-1].squeeze()

        return out

    def traj_graph_forward(self, x, adj, mask=None):
        out = self.traj_gnn(x, adj, mask)
        out = torch.mean(out, dim=1)
        # out, l1, l2 = self.traj_diffpool(x, adj, mask)

        return out

    def urban_graph_forward(self, x, adj, mask):
        driver_embed, l1, l2 = self.urban_diffpool(x, adj, mask)

        return driver_embed, l1 + l2

    def semantic_graph_forward(self, x, adj, mask):
        driver_embed, l1, l2 = self.semantic_diffpool(x, adj, mask)

        return driver_embed, l1 + l2

    def driver_forward(self, x):
        driver_encode = self.driver_encoder(x)

        return driver_encode


class BaseMOCO(nn.Module):
    def __init__(self, encoder, hidden_dim, K, device):
        super(BaseMOCO, self).__init__()
        self.q_encoder = encoder
        self.k_encoder = copy.deepcopy(encoder)

        self.device = device
        self.m = 0.999
        self.T = 0.07

        self.K = K

        for param_q, param_k in zip(self.q_encoder.parameters(), self.k_encoder.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # create the queue
        self.register_buffer("queue", torch.randn(hidden_dim, self.K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # projector
        self.l1 = nn.Linear(hidden_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)

        self.relu = nn.ReLU()

    @torch.no_grad()
    def momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.q_encoder.parameters(), self.k_encoder.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # keys = concat_all_gather(keys)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, q, k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # projector
        q = self.l2(self.relu(self.l1(q)))
        k = self.l2(self.relu(self.l1(k)))

        # compute query features
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            # self._momentum_update_key_encoder()  # update the key encoder
            k = nn.functional.normalize(k, dim=1)

        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        loss = F.cross_entropy(logits, labels.to(logits.device))

        return loss


class MOCO(nn.Module):
    def __init__(self, encoder, hidden_dim, device):
        super(MOCO, self).__init__()
        self.eug_moco = BaseMOCO(encoder, hidden_dim, 6000, device)
        self.sug_moco = BaseMOCO(encoder, hidden_dim, 6000, device)
        self.driver_moco = BaseMOCO(encoder, hidden_dim, 6000, device)
        self.traj_moco = BaseMOCO(encoder, hidden_dim, 8000, device)
        self.relation_moco = BaseMOCO(encoder, hidden_dim, 8000, device)
        self.device = device

    def forward(self):
        pass

    def traj_aug(self, x, seq, rm_ratio=0.2, crop_ratio=0.7):
        new_x = []
        new_seq = []
        aug_method = ['crop', 'remove']
        for sub_x, sub_seq in zip(x, seq):
            if int(sub_seq) <= 5:
                new_x.append(sub_x.unsqueeze(0))
                new_seq.append(sub_seq)
                continue

            method = random.choice(aug_method)
            if method == 'remove':
                random_drop_index = np.random.choice(int(sub_seq), int(rm_ratio * sub_seq), replace=False)
                mask = torch.zeros(sub_x.shape[0], dtype=torch.bool).to(self.device)
                mask[random_drop_index] = True
                temp = torch.cat([sub_x[~mask, :], torch.zeros((len(random_drop_index), sub_x.shape[1])).to(self.device)])
                temp = temp.unsqueeze(0)
                new_x.append(temp)
                new_seq.append(sub_seq - len(random_drop_index))

            elif method == 'crop':
                random_index = np.random.choice(int((1 - crop_ratio) * sub_seq), 1)[0]
                temp = torch.cat([sub_x[random_index:random_index + int(crop_ratio * sub_seq), :],
                                  torch.zeros((sub_x.shape[0] - int((crop_ratio) * sub_seq), sub_x.shape[1])).to(
                                      self.device)])
                temp = temp.unsqueeze(0)
                new_x.append(temp)
                new_seq.append(int(crop_ratio * sub_seq))

        new_x = torch.cat(new_x)
        new_seq = torch.tensor(new_seq)

        return new_x, new_seq

    def graph_aug(self, x, adj, method='node_mask', mask_ratio=0.2, sug=False):
        if sug and method == 'node_mask':
            new_adj = adj
            return None, new_adj

        if method == 'node_mask':
            # mask_ratio = np.random.randint(10, 50) / 100
            node_num = x.shape[1]
            mask_num = int(node_num * mask_ratio)
            node_idx = [i for i in range(node_num)]
            mask_idx = [random.sample(node_idx, mask_num) for _ in range(x.shape[0])]
            mask_idx = np.array(mask_idx)
            aug_feature = x
            zeros = torch.zeros_like(aug_feature[0][0])

            for i in range(aug_feature.shape[0]):
                for idx in mask_idx[i]:
                    aug_feature[i, idx, :] = zeros

            return aug_feature, adj

        elif method == 'del_edge':
            idx = torch.nonzero(adj)
            shapes = adj.shape
            rand_idx = np.random.randint(0, idx.shape[0], size=int(mask_ratio*idx.shape[0]))

            mask = torch.from_numpy(np.zeros(shapes)).bool()#.to('cuda:1')
            mask[idx[rand_idx, 0], idx[rand_idx, 1], idx[rand_idx, 2]] = True
            mask = mask.to('cuda:0')

            new_adj = adj*mask

            # for edge in idx[rand_idx, :]:
            #     adj[edge[0], edge[1], edge[2]] = 0

            return x, new_adj

