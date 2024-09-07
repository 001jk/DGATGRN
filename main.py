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
# from data_split import direction_specific, direction_blind
from data_split import data_prepare
# edge_list = np.loadtxt('./Benchmark Dataset/STRING Dataset/hESC/TFs+500/Label - 副本.txt', dtype=np.int64)
#
# edge_index = torch.tensor(edge_list).t().contiguous()
# print(edge_index)
# """2*num_edges,每列表示一条边"""
# print('edge_index type:', edge_index.dtype)
#
# num_nodes = len(set(edge_index.flatten().tolist()))
# print('num nodes:', num_nodes)
file = './Benchmark Dataset/' + config.net_type + ' Dataset/' + config.data_type + '/TFs+' + str(config.num) + '/BL--ExpressionData.csv'

feature = pd.read_csv(file, header=0, index_col=0)
features = torch.from_numpy(feature.values).to(torch.float32)
# print('features shape:', features.shape)
num_nodes = features.shape[0]

class DirSageConv(torch.nn.Module):
    def __init__(self, out_channels):
        super(DirSageConv, self).__init__()

        self.out_channels = out_channels

        self.conv_src_to_dst = SAGEConv(data.num_node_features, out_channels+config.beta_on, flow="source_to_target", root_weight=False)
        self.conv_dst_to_src = SAGEConv(data.num_node_features, out_channels+config.beta_on, flow="target_to_source", root_weight=False)

        # 定义线性层，用于特征映射
        self.lin1 = torch.nn.Linear(data.num_node_features, 4 * out_channels)
        self.lin2 = torch.nn.Linear(4 * out_channels, out_channels)
    def forward(self, x, edge_index):
        F.dropout(x, p=0.6, training=self.training)
        x_in = F.elu(self.conv_src_to_dst(x, edge_index))
        x_out = F.elu(self.conv_dst_to_src(x, edge_index))
        x1 = F.elu(self.lin1(x))
        x_self = F.elu(self.lin2(x1))

        # print('x_in shape:', x_in.shape)
        # print('x_out shape:', x_out.shape)
        # print('x_self shape:', x_self.shape)

        return x_in, x_out, x_self
class DGAT(torch.nn.Module):
    def __init__(self, out_channels):
        super(DGAT, self).__init__()
        #两个不同用处的注意力卷积层
        self.conv1 = GATConv(data.num_node_features, out_channels + config.beta_on, flow='source_to_target',
                             heads=32, concat=False, add_self_loops=True)
        self.conv2 = GATConv(data.num_node_features, out_channels + config.beta_on, flow='target_to_source',
                             heads=32, concat=False, add_self_loops=True)
        # 定义线性层，用于特征映射
        self.lin1 = torch.nn.Linear(data.num_node_features, 4 * out_channels)
        self.lin2 = torch.nn.Linear(4 * out_channels, out_channels)

    def forward(self, x, edge_index):
        edge_index_t = torch.stack([edge_index[1], edge_index[0]], dim=0)

        # print('x shape:', x.shape)
        # print('edge_index shape:', edge_index.shape)

        F.dropout(x, p=0.6, training=self.training)
        x_in = F.elu(self.conv1(x, edge_index))
        # x_out = F.elu(self.conv2(x, edge_index))
        x_out = F.elu(self.conv2(x, edge_index_t))

        x1 = F.elu(self.lin1(x))
        x_self = F.elu(self.lin2(x1))

        # print('x_in shape:', x_in.shape)
        # print('x_out shape:', x_out.shape)
        # print('x_self shape:', x_self.shape)

        return x_in, x_out, x_self


def train():
    model.train()
    optimizer.zero_grad()
    z_in, z_out, z_self = model.encode(x, train_pos_edge_index)
    #计算重构损失
    loss = model.recon_loss(z_in, z_out, z_self, train_pos_edge_index)
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
    lists = [[] for _ in range(7)]  # 初始化包含6个空列表的列表，用于存储待打印的值
    return [lists[i] for i in range(7)]

# 初始化指标列表
target = ["auc", "ap", "acc", "f1", "pre", "re", "prc"]
auc_list, ap_list, f1_list, acc_list, pre_list, re_list, prc_list = initialize_list()
target_list = [auc_list, ap_list,  f1_list, acc_list, pre_list, re_list, prc_list]


for i in range(config.number):
    # config.seed = random.randint(0, 10000)

    # if i == 20:
    #     auc_list, ap_list, f1_list, acc_list, pre_list, re_list = initialize_list()
    #     target_list = [auc_list, ap_list, f1_list, acc_list, pre_list, re_list]
    #     config.task1 = False
    # if i > 20:
    #     config.task1 = False

    auc_l, ap_l, f1_l, acc_l, pre_l, re_l, prc_l = initialize_list()
    target_l = [auc_l, ap_l, f1_l, acc_l, pre_l, re_l, prc_l]
    # 创建包含图数据的Data对象
    data = Data(edge_index=None, num_nodes=num_nodes, x=features)

    data = data_prepare(data, config.seed)
    # 根据config.task1选择数据处理方式
    # if config.task1:
    #     data = direction_specific(data,config.seed)
    # else:
    #     data = direction_blind(data, config.seed)
    # print(data)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_pos_edge_index = data.train_pos_edge_index.to(device)
    x = data.x.to(device)

    # 定义模型，将DGAT_DDI作为GAE的编码器
    model = GAE(DGAT(config.out_channels)).to(device)
    # model = GAE(DGAT_DDI(config.out_channels)).to(device)

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
    # print('model:', model)
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

