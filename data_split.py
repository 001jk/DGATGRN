import pandas as pd
import torch
import config

def data_prepare(data, seed):
    torch.manual_seed(seed)
    data.edge_index = None
    train_set_file = './Benchmark Dataset/' + config.net_type + ' Dataset/' + config.data_type + '/TFs+' + str(
        config.num) + '/Train_set.csv'
    test_set_file = './Benchmark Dataset/' + config.net_type + ' Dataset/' + config.data_type + '/TFs+' + str(
        config.num) + '/Test_set.csv'
    val_set_file = './Benchmark Dataset/' + config.net_type + ' Dataset/' + config.data_type + '/TFs+' + str(
        config.num) + '/Validation_set.csv'
    # 读取CSV文件
    train = pd.read_csv(train_set_file, header=0, index_col=0)
    test = pd.read_csv(test_set_file, header=0, index_col=0)
    val = pd.read_csv(val_set_file, header=0, index_col=0)

    train_positive_data = train[train.iloc[:, 2] == 1].iloc[:, :2]
    train_negative_data = train[train.iloc[:, 2] == 0].iloc[:, :2]
    test_positive_data = test[test.iloc[:, 2] == 1].iloc[:, :2]
    test_negative_data = test[test.iloc[:, 2] == 0].iloc[:, :2]
    val_positive_data = val[val.iloc[:, 2] == 1].iloc[:, :2]
    val_negative_data = val[val.iloc[:, 2] == 0].iloc[:, :2]

    def data_to_tensor(data):
        tensors = []
        for i in range(2):
            tensors.append(torch.tensor(data.iloc[:, i].values, dtype=torch.int64))
        return torch.stack(tensors)

    data.train_pos_edge_index = data_to_tensor(train_positive_data)
    print(' data.train_pos_edge_index type:', data.train_pos_edge_index.dtype)
    data.train_neg_edge_index = data_to_tensor(train_negative_data)
    data.test_pos_edge_index = data_to_tensor(test_positive_data)
    data.test_neg_edge_index = data_to_tensor(test_negative_data)
    data.val_pos_edge_index = data_to_tensor(val_positive_data)
    data.val_neg_edge_index = data_to_tensor(val_negative_data)

    # print("Positive Tensor:")
    # print(positive_tensor)
    #
    # print("Negative Tensor:")
    # print(negative_tensor)
    return data