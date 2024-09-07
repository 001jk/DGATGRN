import argparse
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve,\
    accuracy_score, f1_score, recall_score, precision_score,auc
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)

from torch_geometric.nn.inits import reset
from sklearn import metrics
import pylab as plt
import config


class Role_Align_Predictor(torch.nn.Module):
    """角色对齐预测器类，用于学习节点之间的关系"""
    def __init__(self):
        """初始化方法，定义了模型的结构"""
        super(Role_Align_Predictor, self).__init__()
        self.align_in = torch.nn.Linear(config.out_channels, config.out_channels)
        self.align_out = torch.nn.Linear(config.out_channels, config.out_channels)
        self.dnn = torch.nn.Sequential(
                    # torch.nn.Linear(64, 32),
                    # torch.nn.ReLU(),
                    torch.nn.Linear(64, 16),
                    torch.nn.ReLU(),
                    torch.nn.Linear(16, 4),
                    torch.nn.ReLU(),
                    torch.nn.Linear(4, 1),
        )

    def forward(self, z_in, z_out, z_self, edge_index, sigmoid=True):
        # print('edge index shape:', edge_index.shape)
        # print('edge index shape:', edge_index[0])

        if config.beta_on:
            # value = (z_out[edge_index[0]][:, -1]*config.beta +
            #          (z_out[edge_index[0]][:, :-1] * self.align_out(z_self[edge_index[1]])).sum(dim=1)*config.alpha +
            #          z_in[edge_index[1]][:, -1]*config.beta +
            #          (self.align_in(z_self[edge_index[0]]) * z_in[edge_index[1]][:, :-1]).sum(dim=1)*config.alpha)
            value = ((z_out[edge_index[0]] * self.align_out(z_self[edge_index[1]])).sum(dim=1) * 0.5 +
                     (self.align_in(z_self[edge_index[0]]) * z_in[edge_index[1]]).sum(dim=1) * 0.5)
        else:
            value = ((z_out[edge_index[0]] * self.align_out(z_self[edge_index[1]])).sum(dim=1)*0.5 +
                     (self.align_in(z_self[edge_index[0]]) * z_in[edge_index[1]]).sum(dim=1)*0.5)
            # value = (z_out[edge_index[0]] * self.align_out(z_self[edge_index[1]])).sum(dim=1)
            # value = (self.align_in(z_self[edge_index[0]]) * z_in[edge_index[1]]).sum(dim=1)
            # v1 = torch.cat([z_out[edge_index[0]], z_self[edge_index[1]]], dim=1)
            # v2 = torch.cat((z_self[edge_index[0]], z_in[edge_index[1]]), dim=1)
            # value = self.dnn(torch.cat([v1, v2], dim=1))
        # print('value shape:',value.shape)
        # print('value:', value)
        # # 使用 torch.lt 函数找到小于 1 的元素的位置
        # mask = torch.lt(value, 1)
        # # 使用布尔掩码来获取小于 1 的元素
        # values = value[mask]
        # # 打印结果
        # print("小于1的值：", values)
        return torch.sigmoid(value) if sigmoid else value


class GAE(torch.nn.Module):

    def __init__(self, encoder, decoder=None):
        super(GAE, self).__init__()
        #编码器，用于学习节点表示，使用的是DGAT_DDI
        self.encoder = encoder
        #解码器，用于预测节点关系
        self.decoder = Role_Align_Predictor() if decoder is None else decoder
        GAE.reset_parameters(self)

    def reset_parameters(self):
        """
         重置模型参数的方法。

         这个方法调用了 reset 函数对编码器和解码器的参数进行重置。

         Args:
             None

         Returns:
             None
         """
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        """
        编码器的前向传播方法，用于学习节点表示。

        Args:
            *args: 可变数量的位置参数
            **kwargs: 可变数量的关键字参数

        Returns:
            返回编码器的前向传播结果
        """
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
                解码器的前向传播方法，用于预测节点之间的关系。

                Args:
                    *args: 可变数量的位置参数
                    **kwargs: 可变数量的关键字参数

                Returns:
                    返回解码器的前向传播结果
                """
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z_in, z_out, z_self, pos_edge_index, neg_edge_index=None):
        """
        计算重构损失的方法，包括正样本损失和负样本损失。

        Args:
            z_in: 输入节点的编码表示
            z_out: 输出节点的编码表示
            z_self: 自环节点的编码表示
            pos_edge_index: 正样本边的索引
            neg_edge_index: 负样本边的索引，默认为None

        Returns:
            返回正样本损失和负样本损失的和
        """
        # print('pos_edge_index shape:', pos_edge_index.shape)

        # print('pos_edge_index:', pos_edge_index)
        pos_loss = -torch.log(self.decoder(z_in, z_out, z_self, pos_edge_index, sigmoid=True) + config.EPS).mean()
        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)

        # pos_edge_index, _ = add_self_loops(pos_edge_index)
        # print('pos_edge_index 2:', pos_edge_index)

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z_self.size(0))
        neg_loss = -torch.log(1 - self.decoder(z_in, z_out, z_self, neg_edge_index, sigmoid=True) + config.EPS).mean()
        # print('loss:', pos_loss+neg_loss)
        return pos_loss + neg_loss

    def test(self, z_in, z_out, z_self, pos_edge_index, neg_edge_index):

        pos_y = z_self.new_ones(pos_edge_index.size(1))
        # print('pos_y:', pos_y.shape)
        # print(pos_y)
        neg_y = z_self.new_zeros(neg_edge_index.size(1))
        # print(neg_edge_index)
        # print('neg_y:', neg_y.shape)
        # print(neg_y)
        y = torch.cat([pos_y, neg_y], dim=0)
        pos_pred = self.decoder(z_in, z_out, z_self, pos_edge_index, sigmoid=True)
        # print('pos_pred:', pos_pred.shape)
        # print(pos_pred)
        neg_pred = self.decoder(z_in, z_out, z_self, neg_edge_index, sigmoid=True)
        # print('neg_pred:', neg_pred.shape)
        # print(neg_pred)

        pred = torch.cat([pos_pred, neg_pred], dim=0)
        # print('y:', y)
        # print('pred:', pred)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        precision, recall, thresholds = precision_recall_curve(y, pred)
        prc = auc(recall, precision)
        return roc_auc_score(y, pred), average_precision_score(y, pred), accuracy_score(y, pred.round()), \
               f1_score(y, pred.round()), precision_score(y, pred.round()), recall_score(y, pred.round()), prc
