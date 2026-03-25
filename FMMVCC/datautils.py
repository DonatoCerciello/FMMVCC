import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def create_data_loader(train, test, train_labels, test_labels, train_index, test_index, batch_size):
    for data in [train, test]:
        temporal_missing = np.isnan(data).all(axis=-1).any(axis=0)
        if temporal_missing[0] or temporal_missing[-1]:
            data = centerize_vary_length_series(data)

        data = data[~np.isnan(data).all(axis=2).all(axis=1)]

    tensor_train = torch.tensor(train, dtype=torch.float32)
    tensor_test = torch.tensor(test, dtype=torch.float32)
    tensor_train_labels = torch.tensor(train_labels, dtype=torch.long)
    tensor_test_labels = torch.tensor(test_labels, dtype=torch.long)
    tensor_train_index = torch.tensor(train_index, dtype=torch.long)
    tensor_test_index = torch.tensor(test_index, dtype=torch.long)

    train_dataset = TensorDataset(tensor_train, tensor_train_labels, tensor_train_index)
    test_dataset = TensorDataset(tensor_test, tensor_test_labels, tensor_test_index)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, test_loader

def set_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

def centerize_vary_length_series(x):
    prefix_zeros = np.argmax(~np.isnan(x).all(axis=-1), axis=1)
    suffix_zeros = np.argmax(~np.isnan(x[:, ::-1]).all(axis=-1), axis=1)
    offset = (prefix_zeros + suffix_zeros) // 2 - prefix_zeros
    rows, column_indices = np.ogrid[:x.shape[0], :x.shape[1]]
    offset[offset < 0] += x.shape[1]
    column_indices = column_indices - offset[:, np.newaxis]
    return x[rows, column_indices]