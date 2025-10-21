import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
import yaml
from types import SimpleNamespace
from copy import deepcopy
from collections import OrderedDict

import torch
from torch.utils.data import Dataset
import logging

def load_yaml_config(path):
    with open(path) as f:
        config = yaml.full_load(f)
    return config
    
def seed_everything(seed, cudnn_deterministic=False):
    """
    Function that sets seed for pseudo-random number generators in:
    pytorch, numpy, python.random
    
    Args:
        seed: the integer value seed for global random state
    """
    if seed is not None:
        print(f"Global seed set to {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
def crop_data(driving_cycle, chunk_size):

    chunks = []
    for i in range(0, len(driving_cycle), chunk_size):
        chunk = driving_cycle[i:i+chunk_size]
        if len(chunk) == chunk_size:
            if not if_data_is_DC(chunk):
                chunks.append(chunk)

    return chunks

def crop_data_repeated(driving_cycle, chunk_size, slide_length = 100):
    chunks = []
    for i in range(0, len(driving_cycle) - chunk_size + 1, slide_length):
        chunk = driving_cycle[i:i+chunk_size]
        if not if_data_is_DC(chunk):
            chunks.append(chunk)
    return chunks


def symmetric_padding(data, window_size=21):
    """
    Apply symmetric padding to the data for edge handling.

    Parameters:
        data (array_like): Input data.
        window_size (int): Size of the moving average window.

    Returns:
        array_like: Padded data.
    """
    # Calculate the number of points to pad on each side
    pad_width = window_size // 2
    
    # Pad the data symmetrically
    padded_data = np.pad(data, (pad_width, pad_width), mode='reflect')
    
    return padded_data

def moving_average(data, window_size=21):
    """
    Apply moving average smoothing to the given data.

    Parameters:
        data (array_like): Input data to be smoothed.
        window_size (int): Size of the moving average window.

    Returns:
        array_like: Smoothed data.
    """
    # Apply symmetric padding to the data
    padded_data = symmetric_padding(data, window_size)
    
    # Define the kernel for the moving average
    kernel = np.ones(window_size) / window_size
    
    # Apply the moving average filter
    smoothed_data = np.convolve(padded_data, kernel, mode='valid')
    
    return smoothed_data

def shuffle_list(input_list):
    """
    Return a shuffled version of the input list.

    Parameters:
        input_list (list): Input list to be shuffled.

    Returns:
        list: Shuffled list.
    """
    shuffled_list = input_list[:]  # Make a copy of the input list
    random.shuffle(shuffled_list)  # Shuffle the copy
    return shuffled_list

def shuffle_dataset(dataset):
    # 获取数据集的大小
    dataset_size = len(dataset)
    
    # 生成一个随机排列的索引
    indices = torch.randperm(dataset_size)
    
    # 使用随机排列的索引重新排序数据集
    shuffled_dataset = [dataset[i] for i in indices]
    
    return shuffled_dataset

def draw_figure(data, label='label'):

    plt.plot(data, label=f'{label}')



def show_figure():

    plt.title('vs vs. time for Each Trip')
    plt.xlabel('Time')
    plt.ylabel('vs')
    plt.legend()
    plt.show()

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    logger = logging.getLogger(__name__)

    return logger

def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag

def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)

def get_labels(dataset):
    labels = []
    for x,y in dataset:
        labels.append(y.item())

    return labels

def if_nan(dataset):
    for i, data in enumerate(dataset):
        x, _ = data  # 假设数据集中的每个数据项都是一个元组，其中 x 是要检查的张量
        if torch.isnan(x).any():
            return True
    return False

def if_data_is_DC(data):

    max_val = max(data)
    min_val = min(data)
    if max_val == 0:
        return True
    if (max_val - min_val) / max_val <= 0.1:
        return True
    return False


# 这个类考虑数值，按所有数据的自定义的最值统一拉到-1，1
class MyDataset(Dataset):
    def __init__(self, data=None, labels=None):

        self.data = [self.map_to_range(torch.tensor(item, dtype=torch.float32)).unsqueeze(0) for item in data] if data is not None else []
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


    def map_to_range(self, tensor):
        # 找到最大和最小值
        min_val = 1140
        max_val = 1000

        if min_val == max_val:
            return torch.zeros_like(tensor)
        # 将值映射到 (-1, 1) 范围
        mapped_tensor = 2 * (tensor - min_val) / (max_val - min_val) - 1

        return mapped_tensor

    def modify_labels(self, new_labels):
        # 修改数据集中的标签
        self.labels = new_labels

    def remove_data(self, index):
        for index in index_list:
            del self.data[index]
            del self.targets[index]


# 这个类只保留形状，每一个单独的数据都被拉到-1，1   
class Univariate_Dataset(Dataset):
    def __init__(self, data=None, labels=None):

        self.data = [self.map_to_range(torch.tensor(item, dtype=torch.float32)).unsqueeze(0) for item in data] if data is not None else []
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


    def map_to_range(self, tensor):
        # 找到最大和最小值
        min_val = tensor.min().item()
        max_val = tensor.max().item()

        if min_val == max_val:
            return torch.zeros_like(tensor)
        # 将值映射到 (-1, 1) 范围
        mapped_tensor = 2 * (tensor - min_val) / (max_val - min_val) - 1

        return mapped_tensor

    def modify_labels(self, new_labels):
        # 修改数据集中的标签
        self.labels = new_labels

    def remove_data(self, index):
        for index in index_list:
            del self.data[index]
            del self.targets[index]


class Multivariate_Dataset(Dataset):
    def __init__(self, data=None, labels=None):

        self.data = [self.map_to_range(torch.tensor(item, dtype=torch.float32)).permute(1,0) for item in data] if data is not None else []
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


    def map_to_range(self, tensor):
        # 在 C 维度上找到最小值和最大值
        min_val = tensor.min(dim=0, keepdim=True)[0]  # 每个 C 列的最小值
        max_val = tensor.max(dim=0, keepdim=True)[0]  # 每个 C 列的最大值

        # 防止除以零
        epsilon = 1e-7

        # 将值映射到 (-1, 1) 范围
        mapped_tensor = 2 * (tensor - min_val) / (max_val - min_val + epsilon) - 1
        
        return mapped_tensor

    def modify_labels(self, new_labels):
        # 修改数据集中的标签
        self.labels = new_labels

    def remove_data(self, index):
        for index in index_list:
            del self.data[index]
            del self.targets[index]
            
def find_model(model_name):
    """
    Finds a pre-trained DiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    
    return checkpoint

