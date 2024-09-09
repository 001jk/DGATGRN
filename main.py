import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, SAGEConv
from torch_geometric.data import Data
import random
import copy
import config
from model import GAE
from data_split import data_prepare
# from kan import *
from KANLayer import *

file = './Benchmark Dataset/' + config.net_type + ' Dataset/' + config.data_type + '/TFs+' + str(config.num) + '/BL--ExpressionData.csv'

feature = pd.read_csv(file, header=0, index_col=0)
features = torch.from_numpy(feature.values).to(torch.float32)
num_nodes = features.shape[0]

class DirSageConv(torch.nn.Module):
    def __init__(self, out_channels):
        super(DirSageConv, self).__init__()

        self.out_channels = out_channels

        self.conv_src_to_dst_1 = SAGEConv(data.num_node_features, 4 * out_channels, flow="source_to_target", root_weight=False)
        self.conv_src_to_dst_2 = SAGEConv(data.num_node_features, out_channels, flow="source_to_target", root_weight=False)

        self.conv_dst_to_src_1 = SAGEConv(data.num_node_features, out_channels, flow="target_to_source", root_weight=False)
        self.conv_dst_to_src_2 = SAGEConv(data.num_node_features, out_channels, flow="target_to_source", root_weight=False)


        self.lin1 = torch.nn.Linear(data.num_node_features, 4 * out_channels)
        self.lin2 = torch.nn.Linear(4 * out_channels, out_channels)
    def forward(self, x, edge_index):
        F.dropout(x, p=0.6, training=self.training)
        x_in = F.elu(self.conv_src_to_dst(x, edge_index))
        x_out = F.elu(self.conv_dst_to_src(x, edge_index))
        x1 = F.elu(self.lin1(x))
        x_self = F.elu(self.lin2(x1))
        return x_in, x_out, x_self

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn1 = torch.nn.BatchNorm1d(out_channels)
        self.relu = torch.nn.ReLU(inplace=True)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.bn2 = torch.nn.BatchNorm1d(out_channels)
        self.shortcut = torch.nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv1d(in_channels, out_channels, kernel_size=1),
                torch.nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):

        out = self.relu(self.bn1(self.conv1(x)))

        out = self.bn2(self.conv2(out))

        out += self.shortcut(x)

        out = self.relu(out)

        return out

class ResidualNetwork(torch.nn.Module):
    def __init__(self, num_blocks, in_channels, out_channels):
        super(ResidualNetwork, self).__init__()
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        self.residual_blocks = torch.nn.Sequential(*layers)
        self.fc = torch.nn.Linear(out_channels, out_channels)

    def forward(self, x):
        out = self.residual_blocks(x)
        out = F.avg_pool1d(out, kernel_size=out.size(2)).squeeze(2)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

class DGAT(torch.nn.Module):
    def __init__(self, out_channels):
        super(DGAT, self).__init__()
        self.conv1 = GATConv(data.num_node_features, 4 * out_channels, flow='source_to_target',
                             heads=32, concat=False, add_self_loops=True)
        self.conv2 = GATConv( 4 * out_channels, out_channels, flow='source_to_target',
                             heads=32, concat=False, add_self_loops=True)
        self.conv3 = GATConv(data.num_node_features, 4*out_channels , flow='target_to_source',
                             heads=32, concat=False, add_self_loops=True)
        self.conv4 = GATConv( 4 * out_channels, out_channels, flow='target_to_source',
                             heads=32, concat=False, add_self_loops=True)

        self.resblock = ResidualBlock(data.num_node_features, out_channels)
        self.resnet = ResidualNetwork(1, data.num_node_features, out_channels)

    def forward(self, x, edge_index):
        F.dropout(x, p=0.6, training=self.training)
        x_s = F.elu(self.conv1(x, edge_index))
        x_in = F.elu(self.conv2(x_s, edge_index))
        x_t = F.elu(self.conv3(x, edge_index))
        x_out = F.elu(self.conv4(x_t, edge_index))

        x_self = self.resnet(x.unsqueeze(2))
        return x_in, x_out, x_self


def train():
    model.train()
    optimizer.zero_grad()
    z_in, z_out, z_self = model.encode(x, train_pos_edge_index)
    loss = model.recon_loss(z_in, z_out, z_self, train_pos_edge_index)
    # print('loss:', loss)
    loss.backward()
    optimizer.step()
    return float(loss)


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z_in, z_out, z_self = model.encode(x, train_pos_edge_index)
    return model.test(z_in, z_out, z_self, pos_edge_index, neg_edge_index)


def testfinal(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z_in, z_out, z_self = model.encode(x, train_pos_edge_index)
    return model.test(z_in, z_out, z_self, pos_edge_index, neg_edge_index)


def initialize_list():
    lists = [[] for _ in range(7)]
    return [lists[i] for i in range(7)]

# 初始化指标列表
target = ["auc", "ap", "acc", "f1", "pre", "re", "prc"]
auc_list, ap_list, f1_list, acc_list, pre_list, re_list, prc_list = initialize_list()
target_list = [auc_list, ap_list,  f1_list, acc_list, pre_list, re_list, prc_list]


for i in range(config.number):

    auc_l, ap_l, f1_l, acc_l, pre_l, re_l, prc_l = initialize_list()
    target_l = [auc_l, ap_l, f1_l, acc_l, pre_l, re_l, prc_l]
    data = Data(edge_index=None, num_nodes=num_nodes, x=features)
    data = data_prepare(data, config.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_pos_edge_index = data.train_pos_edge_index.to(device)
    x = data.x.to(device)

    model = GAE(DGAT(config.out_channels)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    min_loss_val = config.min_loss_val
    best_model = None
    min_epoch = config.min_epoch
    for epoch in range(1, config.epochs + 1):
        loss = train()
        if epoch % 10 == 0:
            auc, ap, acc, f1, pre, re, prc = test(data.val_pos_edge_index, data.val_neg_edge_index)
            print(
                'Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}, ACC: {:.4f}, F1: {:.4f}, PRE: {:.4f}, RE: {:.4f}, PRC: {:.4f}'
                .format(epoch, auc, ap, acc, f1, pre, re, prc))
        if epoch > min_epoch and loss <= min_loss_val:
            min_loss_val = loss
            best_model = copy.deepcopy(model)
    model = best_model
    auc, ap, acc, f1, pre, re, prc = testfinal(data.test_pos_edge_index, data.test_neg_edge_index)
    print('final. AUC: {:.4f}, AP: {:.4f}, ACC: {:.4f}, F1: {:.4f}, PRE: {:.4f}, RE: {:.4f}, PRC: {:.4f}'
          .format(auc, ap, acc, f1, pre, re, prc))

    for j in range(7):
        target_l[j].append(eval(target[j]))
    for j in range(7):
        target_list[j].append(np.mean(target_l[j]))
for j in range(7):
    print(np.mean(target_list[j]), np.std(target_list[j]))

data_type = config.data_type
net_type = config.net_type
num = config.num
print('data_type:', data_type)
print('net_type:', net_type)
print('num:', num)

