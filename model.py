import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve,\
    accuracy_score, f1_score, recall_score, precision_score,auc
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)
from torch.nn.modules.distance import PairwiseDistance
from torch_geometric.nn.inits import reset
from sklearn import metrics
import config
# from kan import *
class Decoder(torch.nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.align_in = torch.nn.Linear(config.out_channels, config.out_channels)
        self.align_out = torch.nn.Linear(config.out_channels, config.out_channels)

    def forward(self, z_in, z_out, z_self, edge_index, sigmoid=True):

        value = ((z_out[edge_index[0]] * self.align_out(z_self[edge_index[1]])).sum(dim=1) * config.alpha +
                 (self.align_in(z_self[edge_index[0]]) * z_in[edge_index[1]]).sum(dim=1) * config.beta)

        return torch.sigmoid(value) if sigmoid else value

class GAE(torch.nn.Module):

    def __init__(self, encoder, decoder=None):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.decoder = Decoder() if decoder is None else decoder

        GAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.decoder)

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)

    def recon_loss(self, z_in, z_out, z_self, pos_edge_index, neg_edge_index=None):
        pos_loss = -torch.log(self.decoder(z_in, z_out, z_self, pos_edge_index, sigmoid=True) + config.EPS).mean()
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z_self.size(0))
        neg_loss = -torch.log(1 - self.decoder(z_in, z_out, z_self, neg_edge_index, sigmoid=True) + config.EPS).mean()
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        return pos_loss + neg_loss

    def test(self, z_in, z_out, z_self, pos_edge_index, neg_edge_index):

        pos_y = z_self.new_ones(pos_edge_index.size(1))
        neg_y = z_self.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)
        pos_pred = self.decoder(z_in, z_out, z_self, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z_in, z_out, z_self, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)
        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()
        precision, recall, thresholds = precision_recall_curve(y, pred)
        prc = auc(recall, precision)
        return roc_auc_score(y, pred), average_precision_score(y, pred), accuracy_score(y, pred.round()), \
               f1_score(y, pred.round()), precision_score(y, pred.round()), recall_score(y, pred.round()), prc

