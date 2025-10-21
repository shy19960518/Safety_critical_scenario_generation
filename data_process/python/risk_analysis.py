from pathlib import Path

import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
from utils.grid_map import Grid, get_y
import bisect

def calculate_yaw_angles(x, y):
    yaw_angles = []
    
    for i in range(len(x) - 1):
        delta_x = x[i + 1] - x[i]
        delta_y = y[i + 1] - y[i]
        yaw_angle = np.arctan2(delta_y, delta_x)  
        yaw_angles.append(yaw_angle)
    yaw_angles.append(yaw_angles[-1])
    return np.array(yaw_angles)

def calculate_smoothness(x, y, dt):

    dx = np.diff(x) 
    dy = np.diff(y)  

    # 计算速度
    vx = dx / dt  
    vy = dy / dt  
    v = np.sqrt(vx**2 + vy**2)  
    
    # 计算速度变化
    dv = np.diff(v)  
    a = dv / dt


    mean = np.mean(a)
    std = np.std(a)
    return (mean, std)

def get_next_index(index, lane_change_indices):
    pos = bisect.bisect_right(lane_change_indices, index)  
    if pos < len(lane_change_indices):
        return lane_change_indices[pos]
    else:
        return None 


def calculate_SDI(row, leader_row):

    if leader_row is None:
        return 0, 0
    if leader_row is None:
        return 0, 0
    L = np.sqrt((row['x'] - leader_row['x']) ** 2 + (row['y'] - leader_row['y']) ** 2)
    SSDL = L + (leader_row['vx']**2 / (2*3.4))
    SSDF = row['vx'] * 1.5 + (row['vx']**2 / (2*3.4))
    SDI = 0 if SSDL > SSDF else 1
    d_SSD = abs(SSDL - SSDF)
    RSL = min(d_SSD / 50, 1) if SDI == 1 else 0
    return SDI, RSL

def get_vehicle_row(frame_df, leader_track_id):
    leader_row = frame_df[frame_df['track_id'] == leader_track_id]

    if not leader_row.empty:

        leader_row = leader_row.iloc[0]  
    else:

        leader_row = None
    return leader_row

def add_REL(follow_df):

    if follow_df.empty:

        follow_df['REL1'] = None
        follow_df['REL2'] = None
        follow_df['REL3'] = None
        return follow_df

    SDI_column1 = follow_df['SDI1']
    SDI_column2 = follow_df['SDI2']
    SDI_column3 = follow_df['SDI3']
    REL1 = (SDI_column1 == 1).sum() / len(SDI_column1)
    REL2 = (SDI_column2 == 1).sum() / len(SDI_column2)
    REL3 = (SDI_column3 == 1).sum() / len(SDI_column3)

    follow_df['REL1'] = REL1
    follow_df['REL2'] = REL2
    follow_df['REL3'] = REL3

    return follow_df



np.set_printoptions(precision=3, suppress=True)


current_file_path = Path(__file__).resolve()
parent_dir = current_file_path.parent.parent
target_dir = parent_dir / 'recorded_trackfiles' / 'risk_cal_generated_scenarios'
map_path = (current_file_path.parent / "../maps/merge.osm").resolve()

risk_list = []
for i in range(1,257):
    file_path = target_dir / f'{i}.csv'
    df = pd.read_csv(file_path)
    # 2. Mark the current lane, target lane, and vehicle status
    unique_track_ids = df['track_id'].unique()
    for track_id in unique_track_ids:
        track_df = df[df['track_id'] == track_id].copy()

        for index, row in track_df.iterrows():
            x = track_df['x'].values
            y = track_df['y'].values

            y_max = get_y(map_path, row['x'], 1000)
            y_2 = get_y(map_path, row['x'], 2000)
            y_3 = get_y(map_path, row['x'], 3000)
            y_min = get_y(map_path, row['x'], 4000)
            assert y_max >= y_2 >= y_3 >= y_min


            if row['y'] >= y_2:
                lane_id = 0
            elif row['y'] >= y_3:
                lane_id = 1
            else:
                lane_id = 2

            track_df.loc[index, 'lane'] = lane_id
        track_df['state'] = 'follow'
        lane_change_indices = track_df.index[(track_df['lane'].diff() != 0) & (track_df['lane'].shift(1).notna())].tolist()

        for change_index in lane_change_indices:

            start_index = max(change_index - 30, track_df.index[0])  
            track_df.loc[start_index:change_index - 1, 'state'] = 'lane_change'
            end_index = min(change_index + 20, track_df.index[-1]) 
            track_df.loc[change_index:end_index, 'state'] = 'lane_change'

        for index, row in track_df.iterrows():
            if row['state'] == 'follow':
                track_df.loc[index, 'target_lane'] = row['lane']
            elif row['state'] == 'lane_change':

                next_change_index = get_next_index(index, lane_change_indices)
                if next_change_index is not None:

                    track_df.loc[index, 'target_lane'] = track_df.loc[next_change_index, 'lane']
                else:

                    track_df.loc[index, 'target_lane'] = row['lane']
                

        df.loc[track_df.index, ['lane', 'state', 'target_lane']] = track_df[['lane', 'state', 'target_lane']]



    # 3.Mark SDI, REL
    delta_t = 0.1
    unique_track_ids = df['track_id'].unique()

    for track_id in unique_track_ids:

        track_df = df[df['track_id'] == track_id].copy()


        track_df['ax'] = (track_df['vx'].shift(-1) - track_df['vx']) / delta_t
        track_df['ay'] = (track_df['vy'].shift(-1) - track_df['vy']) / delta_t

        track_df.loc[track_df.index[-1], 'ax'] = track_df.loc[track_df.index[-2], 'ax']
        track_df.loc[track_df.index[-1], 'ay'] = track_df.loc[track_df.index[-2], 'ay']


        track_df['ax'] = track_df['ax'].apply(lambda x: round(x, 3))
        track_df['ay'] = track_df['ay'].apply(lambda x: round(x, 3))

        df.loc[track_df.index, ['ax', 'ay']] = track_df[['ax', 'ay']]



    unique_track_ids = df['track_id'].unique()

    for track_id in unique_track_ids:
        track_df = df[df['track_id'] == track_id].copy()

        for index, row in track_df.iterrows():
            ego_lane = row['lane']

            frame_df = df[df['frame_id'] == row['frame_id']].copy()
            if row['state'] == 'follow':
                ego_lane_df = frame_df[frame_df['lane'] == ego_lane]
                leader_df = ego_lane_df[ego_lane_df['x'] > row['x']]

                if not leader_df.empty:
                    leader_track_id = leader_df.loc[leader_df['x'].idxmin(), 'track_id']
                else:
                    leader_track_id = None

                leader_row = get_vehicle_row(frame_df, leader_track_id)

                SDI1, RSI1 = calculate_SDI(row, leader_row)
                track_df.loc[index, 'SDI1'] = SDI1
                track_df.loc[index, 'RSI1'] = RSI1

                track_df.loc[index, 'SDI2'] = 0
                track_df.loc[index, 'RSI2'] = 0

                track_df.loc[index, 'SDI3'] = 0
                track_df.loc[index, 'RSI3'] = 0

            if row['state'] == 'lane_change':
                ego_lane_df = frame_df[frame_df['lane'] == ego_lane]
                target_lane_df = frame_df[frame_df['lane'] == row['target_lane']]

                leader_df = ego_lane_df[ego_lane_df['x'] > row['x']]
                target_front_df = ego_lane_df[ego_lane_df['x'] > row['x']]
                target_back_df = ego_lane_df[ego_lane_df['x'] <= row['x']]

                if not leader_df.empty:
                    leader_track_id = leader_df.loc[leader_df['x'].idxmin(), 'track_id']

                else:
                    leader_track_id = None

                if not target_front_df.empty:
                    # 找到 x 最小的那一行对应的 track_id
                    target_leader_track_id = target_front_df.loc[target_front_df['x'].idxmin(), 'track_id']

                else:
                    target_leader_track_id = None

                if not target_back_df.empty:
                    target_follow_track_id = target_back_df.loc[target_back_df['x'].idxmax(), 'track_id']
                else:
                    target_follow_track_id = None

                leader_row = get_vehicle_row(frame_df, leader_track_id)
                target_front_row = get_vehicle_row(frame_df, target_leader_track_id)
                target_follow_row = get_vehicle_row(frame_df, target_follow_track_id)


                SDI1, RSL1 = calculate_SDI(row, leader_row)
                SDI2, RSL2 = calculate_SDI(row, target_front_row)
                SDI3, RSL3 = calculate_SDI(target_follow_row, row)

                track_df.loc[index, 'SDI1'] = SDI1
                track_df.loc[index, 'RSI1'] = RSI1

                track_df.loc[index, 'SDI2'] = SDI2
                track_df.loc[index, 'RSI2'] = RSL2

                track_df.loc[index, 'SDI3'] = SDI3
                track_df.loc[index, 'RSI3'] = RSL3

        follow_df = track_df[track_df['state'] == 'follow']
        follow_df = add_REL(follow_df)

        lane_change_df = track_df[track_df['state'] == 'lane_change']
        lane_change_df = add_REL(lane_change_df)
        
        track_df.loc[track_df['state'] == 'follow', ['REL1', 'REL2', 'REL3']] = follow_df[['REL1', 'REL2', 'REL3']]
        track_df.loc[track_df['state'] == 'lane_change', ['REL1', 'REL2', 'REL3']] = lane_change_df[['REL1', 'REL2', 'REL3']]

        df.loc[df['track_id'] == track_id, ['SDI1', 'RSI1', 'SDI2', 'RSI2', 'SDI3', 'RSI3', 'REL1', 'REL2', 'REL3']] = track_df[['SDI1', 'RSI1', 'SDI2', 'RSI2', 'SDI3', 'RSI3', 'REL1', 'REL2', 'REL3']]




    ###########################################################################


    unique_track_ids = df['track_id'].unique()

    for track_id in unique_track_ids:

        track_df = df[df['track_id'] == track_id].copy()

        REL1 = track_df['REL1'].values[0]
        RSI1 = max(track_df['RSI1'].values)

        REL2 = track_df['REL2'].values[0]
        RSI2 = max(track_df['RSI2'].values)

        REL3 = track_df['REL3'].values[0]
        RSI3 = max(track_df['RSI3'].values)

        risk = 1 - (1 - REL1 * RSI1) * (1 - REL2 * RSI2) * (1 - REL3 * RSI3)
        risk_list.append(risk)


##########################################################################################################################
save_path = Path(parent_dir) / 'recorded_trackfiles' / 'risk_cal_saved_list' / 'generated_risk_list.pkl'

with open(save_path, "wb") as f:
    pickle.dump(risk_list, f)


import matplotlib.pyplot as plt

plt.hist(risk_list, bins=10, edgecolor='k', alpha=0.7)  
plt.title('Distribution of Numbers')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()



