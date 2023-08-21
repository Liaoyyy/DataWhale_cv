import torch
def get_k_fold_data(k, i, X_data, y_label):
    assert k > 1
    fold_size = X_data.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j*fold_size, (j+1)*fold_size)
        X_part = X_data[idx, :]
        y_part = y_label[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part#测试集
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)#训练集
            y_train = torch.cat([y_train, y_part], 0)
        return X_train, y_train, X_valid, y_valid