import pickle

import numpy as np


import torch
from torch.utils.data import Dataset



class Init_dataset(Dataset):
    def __init__(self, data=None, labels=None):

        self.data = [torch.tensor(item, dtype=torch.float32) for item in data] if data is not None else []
        self.labels = [torch.tensor(label, dtype=torch.int) for label in labels] if labels is not None else []


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]

        return x, y

    def add_sample(self, data, label):

        self.data.append(data)
        self.labels.append(label)


class Track_dataset(Dataset):
    def __init__(self, data, labels):
        """
        初始化数据集
        
        :param data: (N, C, **) 的数组，表示数据
        :param labels: 列表，包含 N 个 (condition, class) 元组
        """
        # 将数据转换为 tensor
        self.data = torch.tensor(data, dtype=torch.float32)
        # 处理 labels，将 condition 转换为浮点数 tensor，class 转换为整数 tensor
        self.labels = [(torch.tensor(condition, dtype=torch.float32), torch.tensor(cls, dtype=torch.int)) for condition, cls in labels]

    def __len__(self):
        """返回数据集的大小"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        获取指定索引的数据和标签
        
        :param idx: 索引
        :return: (x, y)，其中 x 是数据，y 是 (condition, class) 的元组
        """
        x = self.data[idx]  # 获取数据
        y = self.labels[idx]  # 获取标签
        return x, y

