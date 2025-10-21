import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from pathlib import Path

from utils.dataset import Init_dataset, Track_dataset
from utils.grid_map import Grid

def normalize(data, min_value, max_value):
    """
    Normalize the input data to the range [0, 1].
    :param data: input array or number
    :param min_value: minimum value of the data
    :param max_value: maximum value of the data
    :return: normalized data
    """
    data = np.asarray(data)
    normalized_data = (data - min_value) / (max_value - min_value) 
    return normalized_data
    
def denormalize(normalized_data, min_value, max_value):
    """
    Denormalize the normalized data (between 0 and 1) to the original range.
    :param normalized_data: normalized array or number (between 0 and 1)
    :param min_value: minimum value of the data
    :param max_value: maximum value of the data
    :return: denormalized data
    """
    normalized_data = np.asarray(normalized_data)
    original_data = normalized_data * (max_value - min_value) + min_value
    return original_data

def restore_original_data(data, my_grid):
    """
    Restore the normalized data to the original data
    :param data: normalized data (N, M, K, 3), where:
    data[..., 0] is the grid score
    data[..., 1] is the normalized y value (935, 955)
    data[..., 2] is the normalized angle (-3.14, 3.14)
    :param my_grid: instance of the Grid class
    :return: restored original data
    """
    restored_data = np.copy(data)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                value = data[i, j, k]
                if value[0] != -1:
                    # 获取 grid_number
                    grid_number = k  # 你的数据的第 0 维是网格编号
                    # 通过 score 和 grid_number 计算 x 值
                    restored_data[i, j, k, 0] = my_grid.get_x_from_grid(grid_number, value[0])
                    # 反归一化 y 值
                    restored_data[i, j, k, 1] = denormalize(value[1], 935, 955)
                    # 反归一化角度值
                    restored_data[i, j, k, 2] = denormalize(value[2], -3.14, 3.14)

    return restored_data

def data_process(data0):
    for i in range(data0.shape[0]):
        for j in range(data0.shape[1]):
            for k in range(data0.shape[2]):
                value = data0[i, j, k]
                if value[0] != -1:        
                    data0[i, j, k, 0] = my_grid.get_score_in_grid(value[0])
                    data0[i, j, k, 1] = normalize(value[1], 935, 955)
                    data0[i, j, k, 2] = normalize(value[2], -3.14, 3.14)
    data0 = np.transpose(data0, (0, 3, 1, 2))
    return data0
###############################################################################################

if __name__ == "__main__":

    np.set_printoptions(precision=3, suppress=True)
    # init_dataset

    base_dir = Path(__file__).parent.resolve()
    load_dir = base_dir / 'processed_data'

    data = np.load(load_dir / 'init_data.npy')
    label = np.load(load_dir / 'init_label.npy')

    data0 = data[label == 0]  
    data1 = data[label == 1]  
    my_grid = Grid(1000, 1140, 5)

    data = data_process(data1)

    label = np.zeros(data.shape[0], dtype=int)

    init_dataset = Init_dataset(data, label)

    save_path = (Path(__file__).parent / "processed_data" / "init_dataset1.pth").resolve()
    torch.save(init_dataset, save_path)
