import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

from utils2.dataset import Init_dataset, Track_dataset
from utils2.grid_map import Grid

def process_tracks(tracks, target_length=140):
    processed_tracks = []
        
    for track in tracks:
        length = track.shape[0]  # 获取当前轨迹的长度
        
        if length < target_length:
            # 长度不足，进行padding
            padding = np.full((target_length - length, 2), -1)
            track = np.vstack((track, padding))
        elif length > target_length:
            # 长度超过，进行裁剪
            track = track[:target_length, :]
        
        processed_tracks.append(track)
    
    # 将所有轨迹拼接到一起
    result = np.array(processed_tracks)
    
    return result

def pad_to_n_max(tracks, n_max=12):
    n, height, width = tracks.shape
    
    if n < n_max:
        # 对第一个维度进行padding，填充-1
        padding = ((0, n_max - n), (0, 0), (0, 0))  # 只在第一个维度上填充
        padded_tracks = np.pad(tracks, padding, mode='constant', constant_values=-1)
    else:
        padded_tracks = tracks  # 如果 n 已经是 48 或更大，不需要填充
    
    return padded_tracks

def normalize_tracks(tracks):
    # 假设 tracks 的形状是 (2, 48, 140)
    # 第一个通道按 1000 到 1140 归一化
    channel_1_min, channel_1_max = 1000, 1140
    # 第二个通道按 935 到 955 归一化
    channel_2_min, channel_2_max = 935, 955
    
    # 创建一个副本，避免修改原数组
    normalized_tracks = np.copy(tracks)
    
    # 处理第一个通道 (1000, 1140)
    mask_channel_1 = tracks[0] != -1  # 创建布尔掩码，选出不等于 -1 的元素
    normalized_tracks[0][mask_channel_1] = (tracks[0][mask_channel_1] - channel_1_min) / (channel_1_max - channel_1_min)
    
    # 处理第二个通道 (935, 955)
    mask_channel_2 = tracks[1] != -1  # 创建布尔掩码，选出不等于 -1 的元素
    normalized_tracks[1][mask_channel_2] = (tracks[1][mask_channel_2] - channel_2_min) / (channel_2_max - channel_2_min)
    
    return normalized_tracks

def normalize_condition(condition):
    """
    对 condition 数组进行归一化，将非 -1 的元素归一化到 [0, 1] 的范围
    
    :param condition: (3, 28) 的数组
    :return: 归一化后的数组
    """
    # 定义归一化的范围
    min_val, max_val = 1000, 1140
    
    # 创建一个副本，避免修改原数组
    normalized_condition = np.copy(condition)
    
    # 创建布尔掩码，选出非 -1 的元素
    mask = normalized_condition != -1
    
    # 对非 -1 的元素进行归一化
    normalized_condition[mask] = (normalized_condition[mask] - min_val) / (max_val - min_val)
    
    return normalized_condition

#################################################反函数########################################


def recon_tracks(tracks, n):
    
    # 步骤 1: 保留前 n 个数据
    processed_tracks = tracks[:n]  # 取前 n 个数据
    
    # 步骤 2: 设置小于 -0.1 的值为 -1，并裁剪大于 -0.1 的值到 [0, 1]
    processed_tracks[processed_tracks < -0.1] = -1  # 设置小于 -0.1 的值为 -1
    processed_tracks[processed_tracks > -0.1] = np.clip(processed_tracks[processed_tracks > -0.1], 0, 1)  # 裁剪到 [0, 1]
    
    # 步骤 3: 反归一化非 -1 值
    channel_1_min, channel_1_max = 1000, 1140
    channel_2_min, channel_2_max = 935, 955
    
    # 反归一化第一个通道
    mask_channel_1 = processed_tracks[:, :, 0] != -1
    processed_tracks[mask_channel_1, 0] = (processed_tracks[mask_channel_1, 0] * (channel_1_max - channel_1_min)) + channel_1_min
    
    # 反归一化第二个通道
    mask_channel_2 = processed_tracks[:, :, 1] != -1
    processed_tracks[mask_channel_2, 1] = (processed_tracks[mask_channel_2, 1] * (channel_2_max - channel_2_min)) + channel_2_min
    
    # 步骤 4: 删除 -1 值并存储在列表中
    cleaned_tracks = []  # 创建一个空列表以存储结果
    for track in processed_tracks:
        # 使用布尔索引过滤掉 -1 值
        cleaned_track = track[track[:, 0] != -1]  # 选择不包含 -1 的行
        cleaned_tracks.append(cleaned_track)  # 将每条清理后的轨迹添加到列表中
    
    return cleaned_tracks

def are_lists_equal(list1, list2):
    # 判断长度是否相同
    if len(list1) != len(list2):
        return False
    
    # 判断每个元素是否相同
    for item1, item2 in zip(list1, list2):
        # 如果元素是列表，递归判断
        if isinstance(item1, list) and isinstance(item2, list):
            if not are_lists_equal(item1, item2):
                return False
        # 如果元素是 NumPy 数组，使用 np.array_equal 比较
        elif isinstance(item1, np.ndarray) and isinstance(item2, np.ndarray):
            if not np.array_equal(item1, item2):
                return False
        # 否则直接比较
        elif item1 != item2:
            return False
            
    return True

def sort_tracks(tracks):
    # tracks 的形状是 (N, L, 2)，其中 2 是 (x, y)
    # 我们需要按 L 上第一个点的 x 值进行排序

    # 获取每条轨迹的第一个 x 值
    first_x_values = tracks[:, 0, 0]  # 形状为 (N,)

    # 按照第一个 x 值从大到小获取排序索引
    sorted_indices = np.argsort(-first_x_values)

    # 根据排序索引对 tracks 进行排序
    sorted_tracks = tracks[sorted_indices]

    return sorted_tracks

def split_into_groups(tracks, group_size=12):
    """
    将 (N, L, 2) 的数组按 (group_size, L, 2) 划分成小组，
    对于最后不足 group_size 的组，将前面的部分与上一组的后部分对齐。
    
    参数：
    tracks: numpy 数组，形状为 (N, L, 2)
    group_size: 每组的大小，默认为 12
    
    返回：
    一个列表，每个元素是形状为 (group_size, L, 2) 的数组
    """
    N, L, _ = tracks.shape
    groups = []
    
    # 按 group_size 划分完整的组
    for i in range(0, N, group_size):
        group = tracks[i:i + group_size]
        
        # 检查是否是最后一个不足 group_size 的组
        if group.shape[0] < group_size:
            # 获取上一组的最后几个数据
            last_group = tracks[i - group_size:i] if i >= group_size else np.zeros((0, L, 2))
            num_needed = group_size - group.shape[0]
            
            # 使用上一组的最后部分填充当前组的前部分
            padding = last_group[-num_needed:] if last_group.shape[0] > 0 else np.zeros((num_needed, L, 2))
            group = np.vstack((padding, group))
        
        groups.append(group)
    
    return groups

###############################################################################################

if __name__ == "__main__":

    # np.set_printoptions(precision=3, suppress=True)
    # init_dataset

    # data = np.load('./processed_data/init_data.npy')
    # label = np.load('./processed_data/init_label.npy')
    # my_grid = Grid(1000, 1140, 5)

    # for i in range(data.shape[0]):
    #     for j in range(data.shape[1]):
    #         for k in range(data.shape[2]):
    #             value = data[i, j, k]
    #             if value != -1:
    #                 # 使用 grid 类的 get_score_in_grid 方法将值转换为 score
    #                 data[i, j, k] = my_grid.get_score_in_grid(value)# 归一化到(0，1）以突出-1. 

    ################### 复原代码 #############################################################
    # for i in range(data.shape[0]):
    #     for j in range(data.shape[1]):
    #         for k in range(data.shape[2]):
    #             value = data[i, j, k]
    #             if value != -1:
    #                 # 使用 grid 类的 get_score_in_grid 方法将值转换为 score
    #                 data[i, j, k] = my_grid.get_x_from_grid(k,value)# 归一化到(0，1）以突出-1. 
    ########################################################################################


    # init_dataset = Init_dataset(data, label)
    # torch.save(init_dataset, "./processed_data/init_dataset.pth")

    with open('./processed_data/track_condition.pkl', 'rb') as file:
        track_condition = pickle.load(file)

    with open('./processed_data/track_data.pkl', 'rb') as file:
        track_data = pickle.load(file)

    with open('./processed_data/track_label.pkl', 'rb') as file:
        track_label = pickle.load(file)

    track_list = []
    # condition_list = []
    label_list = []
    len_list = []
    for _condition, _data, _label in zip(track_condition, track_data, track_label):

        if not _label:
            continue

        condition = _condition[0]
        condition = normalize_condition(condition)

        label = _label[0]

        tracks = process_tracks(_data)
        tracks = sort_tracks(tracks)

        if label == 0:
            tracks = pad_to_n_max(tracks)
            tracks = tracks.transpose(2, 0, 1)
            tracks = normalize_tracks(tracks)
            track_list.append(tracks)
            # condition_list.append(condition)
            # label_list.append(label)
            label_list.append((condition, label))
        if label == 1:

            tracks_groups = split_into_groups(tracks)
            for track in tracks_groups:
                tracks = track.transpose(2, 0, 1)
                tracks = normalize_tracks(tracks)
                track_list.append(tracks)

                # condition_list.append(condition)
                # label_list.append(label)
                label_list.append((condition, label))


        # 复原

        # tracks = tracks.transpose(1, 2, 0)
        # tracks = recon_tracks(tracks, 2)
        # print(are_lists_equal(_data, tracks))

    # reshaped_list = []
    
    # for track in track_list:
    #     # 获取当前数组的形状
    #     B, L, C = track.shape  # B 是第一个维度，L 是第二个维度，C 是通道数（2）
        
    #     # 将 (B, L, C) 变为 (B * L, C)
    #     reshaped_track = track.reshape(B * L, C)
        
    #     # 将结果添加到新的列表中
    #     reshaped_list.append(reshaped_track)
    # track_list = reshaped_list
    Track_dataset = Track_dataset(data=track_list, labels=label_list)

    torch.save(Track_dataset, "./processed_data/track_dataset.pth")
    # np.save('./converted_data', converted_data)
    # np.save('./label', label)
    # all_labels = [dataset[i][1].item() for i in range(len(dataset))]
    # label_distribution = Counter(all_labels)
    # print("num_samples:", len(dataset))
    # print("Label distribution:", label_distribution)




    # loader = DataLoader(
    #     Track_dataset,
    #     batch_size=1,
    #     shuffle=True,
    #     pin_memory=True,
    #     drop_last=True,
    # )
    # for x, y in loader:
    # 	print(x.shape)
    # 	print(y)
    # 	assert 1==2

