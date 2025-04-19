import numpy as np
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from pathlib import Path

from utils.dataset import Init_dataset, Track_dataset
from utils.grid_map import Grid

def process_tracks(tracks, target_length=140):
    processed_tracks = []
        
    for track in tracks:
        length = track.shape[0]  # 获取当前轨迹的长度
        
        if length < target_length:
            # 长度不足，进行padding
            padding = np.full((target_length - length, 3), -1)
            track = np.vstack((track, padding))
        elif length > target_length:
            # 长度超过，进行裁剪
            track = track[:target_length, :]
        
        processed_tracks.append(track)
    
    # Stitch all tracks together
    result = np.array(processed_tracks)
    
    return result

def sort_tracks(tracks):
    # The shape of tracks is (N, L, 3), where 2 is (x, y)
    # We need to sort by the x value of the first point on L

    first_x_values = tracks[:, 0, 0]  
    sorted_indices = np.argsort(first_x_values)
    sorted_tracks = tracks[sorted_indices]

    return sorted_tracks

def pad_to_n_max(tracks, n_max=12):
    n, height, width = tracks.shape
    
    if n < n_max:
        padding = ((0, n_max - n), (0, 0), (0, 0))  
        padded_tracks = np.pad(tracks, padding, mode='constant', constant_values=-1)
    else:
        padded_tracks = tracks  
    
    return padded_tracks

def normalize_tracks(tracks):
    # Assume tracks are of shape (3, 48, 140)
    # Normalize the first channel from 1000 to 1140
    channel_1_min, channel_1_max = 1000, 1140
    # The second channel is normalized from 935 to 955
    channel_2_min, channel_2_max = 935, 955
    # The third channel is normalized from -3.14 to 3.14
    channel_3_min, channel_3_max = -3.14, 3.14
    
    # Create a copy to avoid modifying the original array
    normalized_tracks = np.copy(tracks)
    
    mask_channel_1 = tracks[0] != -1  
    normalized_tracks[0][mask_channel_1] = (tracks[0][mask_channel_1] - channel_1_min) / (channel_1_max - channel_1_min)
    
    mask_channel_2 = tracks[1] != -1  
    normalized_tracks[1][mask_channel_2] = (tracks[1][mask_channel_2] - channel_2_min) / (channel_2_max - channel_2_min)
    
    mask_channel_3 = tracks[2] != -1 
    normalized_tracks[2][mask_channel_3] = (tracks[2][mask_channel_3] - channel_3_min) / (channel_3_max - channel_3_min)
    
    return normalized_tracks

def normalize_condition(condition):
    """
    Normalize the condition array, normalize the non--1 elements to the range of [0, 1] according to the range of different dimensions.

    :param condition: array with shape (3, 28, 3)
    :return: normalized array
    """

    min_vals = np.array([1000, 935, -3.14])
    max_vals = np.array([1140, 955, 3.14])

    normalized_condition = np.copy(condition)
    
    mask = normalized_condition != -1

    for i in range(3):
        dim_mask = mask[:, :, i]
        normalized_condition[:, :, i][dim_mask] = (
            (normalized_condition[:, :, i][dim_mask] - min_vals[i]) / (max_vals[i] - min_vals[i])
        )
    
    return normalized_condition

#################################################反函数########################################


def recon_tracks(tracks, n):
    
    # Step 1: Keep the first n data
    processed_tracks = tracks[:n]  
    
    # Step 2: Set values less than -0.1 to -1, and clip values greater than -0.1 to [0, 1]
    processed_tracks[processed_tracks < -0.1] = -1  # 设置小于 -0.1 的值为 -1
    processed_tracks[processed_tracks > -0.1] = np.clip(processed_tracks[processed_tracks > -0.1], 0, 1)  # 裁剪到 [0, 1]
    
    # Step 3: Denormalize non -1 values
    channel_1_min, channel_1_max = 1000, 1140
    channel_2_min, channel_2_max = 935, 955
    channel_3_min, channel_3_max = -3.14, 3.14
    
    # Denormalize the first channel
    mask_channel_1 = processed_tracks[:, :, 0] != -1
    processed_tracks[mask_channel_1, 0] = (processed_tracks[mask_channel_1, 0] * (channel_1_max - channel_1_min)) + channel_1_min
    
    # Denormalize the second channel
    mask_channel_2 = processed_tracks[:, :, 1] != -1
    processed_tracks[mask_channel_2, 1] = (processed_tracks[mask_channel_2, 1] * (channel_2_max - channel_2_min)) + channel_2_min
    
    # Denormalize the third channel
    mask_channel_3 = processed_tracks[:, :, 2] != -1
    processed_tracks[mask_channel_3, 2] = (processed_tracks[mask_channel_3, 2] * (channel_3_max - channel_3_min)) + channel_3_min
    
    # Step 4: Remove -1 values and store in a list
    cleaned_tracks = []  
    for track in processed_tracks:
        cleaned_track = track[track[:, 0] != -1]  
        cleaned_tracks.append(cleaned_track)  
    
    return cleaned_tracks


def are_lists_equal(list1, list2):
    if len(list1) != len(list2):
        return False

    for item1, item2 in zip(list1, list2):

        if isinstance(item1, list) and isinstance(item2, list):
            if not are_lists_equal(item1, item2):
                return False

        elif isinstance(item1, np.ndarray) and isinstance(item2, np.ndarray):
            rounded_item1 = np.round(item1, 3)
            rounded_item2 = np.round(item2, 3)
            if not np.array_equal(rounded_item1, rounded_item2):
                print(rounded_item1, rounded_item2)  
                return False
        elif np.round(item1, 3) != np.round(item2, 3):
            return False
            
    return True


def split_into_groups(tracks, group_size=12):
    """
    Divide the (N, L, 2) array into groups of (group_size, L, 2).
    For the last group that is less than group_size, align the front part with the back part of the previous group.

    Parameters:
    tracks: numpy array, shape (N, L, 2)
    group_size: the size of each group, default is 12

    Return:
    A list, each element is an array of shape (group_size, L, 2)
    """
    N, L, _ = tracks.shape
    groups = []
    
    # Check if this is the last group that is less than group_size
    for i in range(0, N, group_size):
        group = tracks[i:i + group_size]
        
        # Check if this is the last group that is less than group_size
        if group.shape[0] < group_size:
            # Get the last few data of the previous group
            last_group = tracks[i - group_size:i] if i >= group_size else np.zeros((0, L, 2))
            num_needed = group_size - group.shape[0]
            
            # Split complete groups by group_size
            padding = last_group[-num_needed:] if last_group.shape[0] > 0 else np.zeros((num_needed, L, 2))
            group = np.vstack((padding, group))
        
        groups.append(group)
    
    return groups

###############################################################################################

if __name__ == "__main__":

    np.set_printoptions(precision=3, suppress=True)

    base_dir = Path(__file__).parent.resolve()
    processed_dir = base_dir / 'processed_data'

    with open(processed_dir / 'track_condition.pkl', 'rb') as file:
        track_condition = pickle.load(file)

    with open(processed_dir / 'track_data.pkl', 'rb') as file:
        track_data = pickle.load(file)

    with open(processed_dir / 'track_label.pkl', 'rb') as file:
        track_label = pickle.load(file)

    # print(len(track_condition), len(track_data), len(track_label))

    track_list_smooth = []
    label_list_smooth = []

    track_list_congestion = []
    label_list_congestion = []
    for _condition, _data, _label in zip(track_condition, track_data, track_label):

        if not _label:
            continue

        condition = _condition[0]
        label = _label[0]

        tracks = process_tracks(_data)
        tracks = sort_tracks(tracks)
        
        if label == 0:
            tracks = pad_to_n_max(tracks)
            tracks = tracks.transpose(2, 0, 1)
            tracks = normalize_tracks(tracks)
            track_list_smooth.append(tracks)
            label_list_smooth.append((condition, label))

        if label == 1:
            tracks = pad_to_n_max(tracks, n_max=48)
            tracks = tracks.transpose(2, 0, 1)
            tracks = normalize_tracks(tracks)
            track_list_congestion.append(tracks)
            label_list_congestion.append((condition, label))

        ######################################################### Restore test#############################################################
        # condition = condition[:, :, 0]
        # non_negative_elements = condition[condition != -1]
        # n = len(non_negative_elements)
        # tracks = tracks.transpose(1, 2, 0)
        # tracks = recon_tracks(tracks, n)

        # print(are_lists_equal(_data[0][:,2], tracks[1][:,2]))

    Track_dataset_smooth = Track_dataset(data=track_list_smooth, labels=label_list_smooth)
    Track_dataset_congestion = Track_dataset(data=track_list_congestion, labels=label_list_congestion)
    torch.save(Track_dataset_smooth, save_path / 'Track_dataset_smooth.pth')
    torch.save(Track_dataset_congestion, save_path / 'Track_dataset_congestion.pth')
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

