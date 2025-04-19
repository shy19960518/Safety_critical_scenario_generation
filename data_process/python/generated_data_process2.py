import os

import pickle
import numpy as np

import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline, UnivariateSpline

import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

import pandas as pd

from original_data_process4 import recon_tracks
import numpy as np
from scipy.ndimage import gaussian_filter1d


def calculate_velocity(x, y, t):
    vx = []
    vy = []
    t = t /1000
    for i in range(len(x) - 1):
        delta_x = x[i + 1] - x[i]
        delta_y = y[i + 1] - y[i]
        delta_t = t[i + 1] - t[i]
        
        # 确保 delta_t 不为零以避免除以零的错误
        if delta_t != 0:
            v_x = delta_x / delta_t
            v_y = delta_y / delta_t
        else:
            v_x = 0
            v_y = 0
        
        vx.append(v_x)
        vy.append(v_y)

    vx.append(vx[-1])
    vy.append(vy[-1])
    return np.array(vx), np.array(vy)

def calculate_yaw_angles(x, y):
    yaw_angles = []
    
    for i in range(len(x) - 1):
        delta_x = x[i + 1] - x[i]
        delta_y = y[i + 1] - y[i]
        yaw_angle = np.arctan2(delta_y, delta_x)  # 计算偏航角（弧度）
        yaw_angles.append(yaw_angle)
    yaw_angles.append(yaw_angles[-1])
    return np.array(yaw_angles)

def calculate_acceleration_spline(t, x, y):
    cs_x = CubicSpline(t, x)
    cs_y = CubicSpline(t, y)
    
    # 计算二阶导数
    second_derivative_x = cs_x(t, 2)
    second_derivative_y = cs_y(t, 2)
    
    return second_derivative_x, second_derivative_y

def objective_function(params, t, x, y):
    # 更新样条插值的控制点
    cs_x = CubicSpline(t, x + params[:len(x)])
    cs_y = CubicSpline(t, y + params[len(x):])
    
    # 计算加速度
    second_derivative_x, second_derivative_y = calculate_acceleration_spline(t, x + params[:len(x)], y + params[len(x):])
    
    # 计算违反约束的惩罚
    penalty = np.sum(np.maximum(np.abs(second_derivative_x), np.abs(second_derivative_y)) - 0.5)
    return penalty

def spline_interpolation_with_smoothness_constraint(x, y, t):
    # 初始参数为0
    initial_params = np.zeros(len(x) * 2)
    t = t/1000
    # 最优化
    result = minimize(objective_function, initial_params, args=(t, x, y))
    
    # 更新x和y值
    updated_x = x + result.x[:len(x)]
    updated_y = y + result.x[len(x):]
    
    # 最终样条插值
    cs_x = CubicSpline(t, updated_x)
    cs_y = CubicSpline(t, updated_y)
    
    new_x = cs_x(t)
    new_y = cs_y(t)
    
    return new_x, new_y


np.set_printoptions(precision=3, suppress=True)

current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
target_dir = os.path.join(parent_dir, 'recorded_trackfiles', 'generated_scenarios')

track_path= os.path.join(target_dir, 'track_data.npy')

track_datas = np.load(track_path)


# switch using one of these lines to fit input. 
# track_datas = track_datas.reshape(track_datas.shape[0], 3, 12, track_datas.shape[-1])
track_datas = track_datas.reshape(track_datas.shape[0], 3, 48, track_datas.shape[-1])


conditions_buffer = None
file = None
file_id = 0
n = 5 # only sample first 5 scenarios for visualisation

for tracks in track_datas[:n]:

    file = {'track_id': [],
        'frame_id': [],
        'timestamp_ms': [],
        'agent_type': [],
        'x': [],
        'y': [],
        'vx': [],
        'vy': [],
        'psi_rad': [],
        'length': [],
        'width': [],
    }

    tracks = tracks.transpose(1,2,0)
    # non_negative_elements = condition[condition != -1]
    # sorted_elements = np.sort(non_negative_elements)

    # n = int(len(sorted_elements)/3)

    filtered_data = tracks[tracks[:, 0, 0] >= -0.1]
    n = filtered_data.shape[0]
    track_list = recon_tracks(tracks, n)


    for track_id, track in enumerate(track_list):

        track = track[track[:, 0] < 1140]
        L = track.shape[0]
        x = np.round(track[:, 0].tolist(), 3)
        y = np.round(track[:, 1].tolist(), 3)
        t = np.arange(100, 100 * (L + 1), 100)
        x = gaussian_filter1d(x, sigma=2)
        y = gaussian_filter1d(y, sigma=2)


        if len(x) <= 1:
            continue
        vx, vy = calculate_velocity(x, y, t)
        vx = np.round(vx, 3)
        vy = np.round(vy, 3)
        yaw_angles = np.round(track[:, 2].tolist(), 3)
        yaw_angles = np.round(yaw_angles, 3)
        track_id_list = [track_id ] * len(x)
        frame_id = list(range(1, len(x) + 1))
        timestamp_ms = t.tolist()
        agent_type = ['car'] * len(x)
        length = [3.80] * len(x)
        width = [1.50] * len(x)

        file['track_id'].extend(track_id_list)
        file['frame_id'].extend(frame_id)
        file['timestamp_ms'].extend(timestamp_ms)
        file['agent_type'].extend(agent_type)
        file['x'].extend(x)
        file['y'].extend(y)
        file['vx'].extend(vx)
        file['vy'].extend(vy)
        file['psi_rad'].extend(yaw_angles)
        file['length'].extend(length)
        file['width'].extend(width)

    file_id += 1
    df = pd.DataFrame(file)
    csv_filename = os.path.join(target_dir, f'{file_id}.csv')
    df.to_csv(csv_filename, index=False)
