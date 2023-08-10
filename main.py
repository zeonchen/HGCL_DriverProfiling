import torch
import torch.nn.functional as F
import random
import numpy as np
import warnings
import argparse
from torch_geometric.utils import to_dense_adj, to_dense_batch, dense_to_sparse
from torch.nn.utils.rnn import pad_sequence
from model.hgcl import HGCL, MOCO
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import pickle

warnings.filterwarnings('ignore')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def data_loader(path):
    save_df = open(path, 'rb')
    sample_data = pickle.load(save_df)
    all_traj = sample_data['traj']
    all_len = sample_data['len']
    all_region = sample_data['region']
    all_num = sample_data['num']
    urban_adj = sample_data['urban_adj']
    poi_adj = sample_data['poi_adj']
    poi_nodes = sample_data['poi_nodes']
    node_loc = sample_data['node_loc']

    return all_traj, all_len, all_region, all_num, urban_adj, \
           poi_adj, poi_nodes, node_loc


def construct_traj_rel(moco, batch_traj, batch_len, batch_region, batch_num, batch_size, device):
    contra_rl = []
    traj_embed = moco.traj_moco.q_encoder.traj_embed_forward(batch_traj, batch_len)

    split_traj_embed = torch.split(traj_embed, batch_num.int().numpy().tolist(), dim=0)
    split_region = torch.split(batch_region, batch_num.int().numpy().tolist(), dim=0)

    pad_traj_embed = pad_sequence([x for x in split_traj_embed], batch_first=True)  # .permute(0, 2, 1)
    pad_region = pad_sequence([x for x in split_region], batch_first=True)  # .to(device)

    batch_traj_graph_embed = []

    for idx in range(batch_size):
        # For one driver
        single_driver_embed = pad_traj_embed[idx]
        single_traj_region = pad_region[idx]
        single_non_epty_id = []
        relation_graph = []
        single_traj_embed = torch.zeros((900, pad_traj_embed.shape[-1])).to(device)  # []
        for i in range(900):
            region_idx = np.argwhere(single_traj_region.numpy()[:, i] > 0).reshape(-1)
            if len(region_idx) != 0:
                single_non_epty_id.append(i)
                region_embed = single_driver_embed[region_idx, :]
                sub_diff = torch.abs(region_embed.unsqueeze(1) - region_embed.unsqueeze(0))
                sub_traj_adj = F.sigmoid(moco.relation_moco.q_encoder.relational_traj_linear(sub_diff / 2.0)).permute(2, 0, 1)

                # contrast
                if sub_traj_adj.shape[1] >= 3:
                    contra_rl.append(Data(x=region_embed, edge_index=dense_to_sparse(sub_traj_adj)[0],
                                          edge_attr=dense_to_sparse(sub_traj_adj)[1]))

                relation_graph.append(Data(x=region_embed, edge_index=dense_to_sparse(sub_traj_adj)[0],
                                           edge_attr=dense_to_sparse(sub_traj_adj)[1]))

        sub_driver_dataset = DataLoader(relation_graph, shuffle=False, batch_size=len(relation_graph))
        for data in sub_driver_dataset:
            sub_x = data.x.to(device)
            edge_index = data.edge_index.to(device)
            edge_attr = data.edge_attr.to(device)
            # generate labels
            sub_x, mask = to_dense_batch(sub_x, data.batch.to(device))
            adj = to_dense_adj(edge_index, data.batch.to(device), edge_attr=edge_attr)
            region_embed = moco.relation_moco.q_encoder.traj_graph_forward(sub_x, adj, mask)
            break

        single_traj_embed[np.array(single_non_epty_id)] = region_embed
        batch_traj_graph_embed.append(single_traj_embed.unsqueeze(0))

    return contra_rl, batch_traj_graph_embed


def train(args, moco, batch_traj, batch_len, batch_region, batch_num, urban_adj, \
          poi_adj, poi_nodes, node_loc, device):
    moco.train()
    all_idx = [i for i in range(10)]
    batch_size = args.batch_size
    optimizer = torch.optim.Adam(moco.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epoch):
        random.shuffle(all_idx)
        total_loss_list = []
        for b_id in range(len(all_idx)//batch_size):
            batch_traj = batch_traj.to(device)
            urban_adj = urban_adj.to(device)
            poi_adj = poi_adj.to(device)

            # trajectory contrastive
            rand_traj_idx = np.random.randint(0, batch_traj.shape[0], size=1600)
            contra_traj = batch_traj[rand_traj_idx, :, :]
            contra_seq = batch_len[rand_traj_idx]
            x_1, seq_1 = moco.traj_aug(contra_traj, contra_seq)
            x_2, seq_2 = moco.traj_aug(contra_traj, contra_seq)

            q_traj_embed = moco.traj_moco.q_encoder.traj_embed_forward(x_1, seq_1)
            k_traj_embed = moco.traj_moco.k_encoder.traj_embed_forward(x_2, seq_2)

            traj_cl_loss = moco.traj_moco(q_traj_embed, k_traj_embed)

            if args.stage == 'traj':
                loss = traj_cl_loss
                optimizer.zero_grad()
                traj_cl_loss.backward(retain_graph=True)
                optimizer.step()
                continue

            contra_rl, batch_traj_graph_embed = construct_traj_rel(moco, batch_traj, batch_len, batch_region, batch_num, batch_size, device)

            # contrast relational graph
            aug_method = ['del_edge', 'node_mask']
            region_contra_dataset = DataLoader(contra_rl, batch_size=160)
            for data in region_contra_dataset:
                sub_x = data.x.to(device)
                edge_index = data.edge_index.to(device)
                edge_attr = data.edge_attr.to(device)
                # generate labels
                sub_x, mask = to_dense_batch(sub_x, data.batch.to(device))

                adj = to_dense_adj(edge_index, data.batch.to(device), edge_attr=edge_attr)
                x_1, adj_1 = moco.graph_aug(sub_x, adj, method=random.choice(aug_method))
                x_2, adj_2 = moco.graph_aug(sub_x, adj, method=random.choice(aug_method))

                q_region_embed = moco.relation_moco.q_encoder.traj_graph_forward(x_1, adj_1, mask)
                k_region_embed = moco.relation_moco.k_encoder.traj_graph_forward(x_2, adj_2, mask)
                relation_cl_loss = moco.relation_moco(q_region_embed, k_region_embed)
                break

            if args.stage == 'relation_graph':
                loss = relation_cl_loss + 0.1 * traj_cl_loss
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                continue

            # contrast EUG and SUG
            batch_traj_graph_embed = torch.cat(batch_traj_graph_embed)

            batch_poi = torch.cat([poi_nodes.unsqueeze(0) for _ in range(batch_size)]).to(device)
            batch_loc = torch.cat([node_loc.unsqueeze(0) for _ in range(batch_size)]).to(device)

            combined_embed_eug = torch.cat([batch_traj_graph_embed, batch_poi, batch_loc], dim=2)
            x_1, adj_1 = moco.graph_aug(combined_embed_eug, urban_adj, method=random.choice(aug_method))
            x_2, adj_2 = moco.graph_aug(combined_embed_eug, urban_adj, method=random.choice(aug_method))

            # EUG
            q_embed_eug, l1 = moco.eug_moco.q_encoder.urban_graph_forward(x_1, adj_1, None)
            k_embed_eug, l2 = moco.eug_moco.k_encoder.urban_graph_forward(x_2, adj_2, None)

            eug_cl_loss = moco.eug_moco(q_embed_eug, k_embed_eug)

            # SUG
            combined_embed_sug = torch.cat([batch_traj_graph_embed, batch_poi, batch_loc], dim=2)
            x_1, adj_1 = moco.graph_aug(combined_embed_sug, poi_adj, method=random.choice(aug_method))
            x_2, adj_2 = moco.graph_aug(combined_embed_sug, poi_adj, method=random.choice(aug_method))

            q_embed_sug, l3 = moco.sug_moco.q_encoder.semantic_graph_forward(x_1, adj_1, None)
            k_embed_sug, l4 = moco.sug_moco.k_encoder.semantic_graph_forward(x_2, adj_2, None)

            sug_cl_loss = moco.sug_moco(q_embed_sug, k_embed_sug)

            if args.stage == 'urban':
                loss = 0.1 * relation_cl_loss + 0.1 * traj_cl_loss + eug_cl_loss + sug_cl_loss + \
                       l1 + l2 + l3 + l4
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()
                continue

            # Driver contrast
            q_driver = moco.driver_moco.q_encoder.driver_forward(torch.cat([q_embed_eug, q_embed_sug], dim=1))
            k_driver = moco.driver_moco.k_encoder.driver_forward(torch.cat([k_embed_eug, k_embed_sug], dim=1))

            cl_loss = moco.driver_moco(q_driver, k_driver)

            if args.stage == 'total':
                loss = traj_cl_loss + relation_cl_loss + eug_cl_loss + sug_cl_loss + cl_loss + \
                       l1 + l2 + l3 + l4

                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                optimizer.step()

            total_loss_list.append(loss.item())

        print('epoch {}, loss {:.4}'.format(epoch, np.mean(total_loss_list)))


def eval(args, moco, batch_traj, batch_len, batch_region, batch_num, urban_adj, \
         poi_adj, poi_nodes, node_loc, device):
    moco.eval()
    all_idx = [i for i in range(10)]
    batch_size = args.batch_size

    for b_id in range(len(all_idx)):
        batch_traj = batch_traj.to(device)
        urban_adj = urban_adj.to(device)
        poi_adj = poi_adj.to(device)

        contra_rl, batch_traj_graph_embed = construct_traj_rel(moco, batch_traj, batch_len, batch_region, batch_num, batch_size, device)
        batch_traj_graph_embed = torch.cat(batch_traj_graph_embed)

        batch_poi = torch.cat([poi_nodes.unsqueeze(0) for _ in range(batch_size)]).to(device)
        batch_loc = torch.cat([node_loc.unsqueeze(0) for _ in range(batch_size)]).to(device)

        combined_embed_eug = torch.cat([batch_traj_graph_embed, batch_poi, batch_loc], dim=2)
        # EUG
        embed_eug, _ = moco.eug_moco.q_encoder.urban_graph_forward(combined_embed_eug, urban_adj, None)

        # SUG
        combined_embed_sug = torch.cat([batch_traj_graph_embed, batch_poi, batch_loc], dim=2)
        embed_sug, _ = moco.sug_moco.q_encoder.semantic_graph_forward(combined_embed_sug, poi_adj, None)

        # Driver contrast
        driver_embed = moco.driver_moco.q_encoder.driver_forward(torch.cat([embed_eug, embed_sug], dim=1))

    return driver_embed


def main(args):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    batch_traj, batch_len, batch_region, batch_num, urban_adj, \
    poi_adj, poi_nodes, node_loc = data_loader(args.data_path)

    hgcl_model = HGCL(args.hidden_dim).to(device)
    total_params = sum(p.numel() for p in hgcl_model.parameters())
    print("Number of parameters：{:.3} k".format(total_params / 1024))

    moco = MOCO(hgcl_model, args.hidden_dim, device).to(device)

    if args.mode == 'train':
        train(args, moco, batch_traj, batch_len, batch_region, batch_num, urban_adj, \
              poi_adj, poi_nodes, node_loc, device)
    elif args.mode == 'eval':
        driver_embed = eval(args, moco, batch_traj, batch_len, batch_region, batch_num,
                            urban_adj, poi_adj, poi_nodes, node_loc, device)
        with open('driver_profile', 'wb') as f:
            pickle.dump(driver_embed, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--data_path', type=str, default='sample_data.pkl')
    parser.add_argument('--stage', type=str, default='total', help='Within [traj, relation_graph, urban, total]')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--lr', type=int, default=1e-3)
    parser.add_argument('--weight_decay', type=int, default=1e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--epoch', type=int, default=1000)

    args = parser.parse_args()

    setup_seed(args.seed)
    main(args)
