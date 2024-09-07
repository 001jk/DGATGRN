import pandas as pd
import torch
import config
from torch_geometric.data import Data
import numpy as np
torch.manual_seed(config.seed)
edge_list = np.loadtxt('./Benchmark Dataset/STRING Dataset/hESC/TFs+500/Label - 副本.txt', dtype=np.int64)

edge_index = torch.tensor(edge_list).t().contiguous()
print(edge_index)
"""2*num_edges,每列表示一条边"""
print('edge_index type:', edge_index.dtype)
#
# num_nodes = len(set(edge_index.flatten().tolist()))
# print('num nodes:', num_nodes)
feature = pd.read_csv('./Benchmark Dataset/STRING Dataset/hESC/TFs+500/BL--ExpressionData.csv', header=0, index_col=0)
features = torch.from_numpy(feature.values).to(torch.float32)
num_nodes = features.shape[0]

data = Data(edge_index=edge_index, num_nodes=num_nodes, x=features)

train_set_file = './Benchmark Dataset/' + config.net_type + ' Dataset/' + config.data_type + '/TFs+' + str(
    config.num) + '/Train_set.csv'
# 读取CSV文件，跳过第一行和第一列
df = pd.read_csv(train_set_file, header=0, index_col=0)
data.edge_index = None

# 根据第三列的值分割数据
positive_data = df[df.iloc[:, 2] == 1].iloc[:, :2]
negative_data = df[df.iloc[:, 2] == 0].iloc[:, :2]

# 将数据转换为两行的张量
def data_to_tensor(data):
    tensors = []
    for i in range(2):
        tensors.append(torch.tensor(data.iloc[:, i].values, dtype=torch.int64))  # 将 Series 对象转换为数组
    return torch.stack(tensors)

data.positive_tensor = data_to_tensor(positive_data)
# print('positive_tensor:', positive_tensor)
# print('positive_tensor type:', positive_tensor.dtype)
data.negative_tensor = data_to_tensor(negative_data)

print("Positive Tensor:")
print(data.positive_tensor)

print("Negative Tensor:")
print(data.negative_tensor)
