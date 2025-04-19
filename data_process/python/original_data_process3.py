
# extract tracks from Original files
# input csv file
# output - list of scenarios, each element is a scenarios contains tracks.  
#        - condition is the initial states, same type as output of data process 1. 
#        - label, one int. Just in case some mixture model research. In this work it is not used. 

# There are map module in the loop so please wait longer for data process.

from pathlib import Path

import pandas as pd
import numpy as np


from utils.grid_map import Grid, get_y
import pickle


# absolute path
base_dir = Path(__file__).parent.resolve() 
map_path = (base_dir / "../maps/merge.osm").resolve()
folder_path = (base_dir / "../recorded_trackfiles/DR_CHN_Merging_ZS/").resolve()

grid = Grid(1000, 1140, 5)
data_list = []
condition_list = []
label_list = []

for file_path in folder_path.iterdir():
    if file_path.suffix != '.csv':  # check if not CSV file, continue
        continue
    continue_loop = True

    file_number = int(file_path.stem.split('_')[-1])
    
    df = pd.read_csv(file_path)
    max_frame = max(df['frame_id'])

    for frame_id in range(max_frame-500): 
        if frame_id % 300 == 0:  # 每300次循环打印一次
            print(f"Iteration {frame_id} completed.")

        filtered_df = df[df['frame_id'] == frame_id + 1].copy()

        if filtered_df.empty:
            continue_loop = False
            break
        rows_to_delete = []
        for index, row in filtered_df.iterrows():
            y_max = get_y(map_path, row['x'], 1000)
            y_min = get_y(map_path, row['x'], 4000)
            assert y_max > y_min
            if not (y_min <= row['y'] <= y_max):
                rows_to_delete.append(index)

        filtered_df.drop(rows_to_delete, inplace=True)

        condition = np.full((3, 28, 3), -1, dtype=float)

        current_data_list = []
        current_condition_list = []
        current_label_list = []
        if file_number not in [8,9,10]:
            label_ = 0
        else:
            label_ = 1
        
        for index, row in filtered_df.iterrows():
            x = row['x']
            y = row['y']
            psi_rad = row['psi_rad']

            j = grid.get_grid_number_x(x)
            i = grid.get_lane_id(get_y(map_path, x, 1000), get_y(map_path, x, 2000), get_y(map_path, x, 3000), get_y(map_path, x, 4000), y)

            condition[i, j] = np.array([x, y, psi_rad])

            track_id = row['track_id']
            frame_id = row['frame_id']
            track_df = df[(df['track_id'] == track_id) & (df['frame_id'] >= frame_id)].copy()

            x_data = np.array(track_df['x'])
            y_data = np.array(track_df['y'])
            psi_rad_data = np.array(track_df['psi_rad'])
            data = np.column_stack((x_data, y_data, psi_rad_data))

            current_data_list.append(data)
            current_label_list.append(label_)

        for _ in current_data_list:
            current_condition_list.append(condition)

        data_list.append(current_data_list)
        condition_list.append(current_condition_list)
        label_list.append(current_label_list)

    if not continue_loop:
        print(f'{file_number} done.')
        continue


save_dir = (base_dir / "processed_data").resolve()
save_dir.mkdir(parents=True, exist_ok=True)

# 保存数据
with open(save_dir / "track_data.pkl", 'wb') as f:
    pickle.dump(data_list, f)

with open(save_dir / "track_condition.pkl", 'wb') as f:
    pickle.dump(condition_list, f)

with open(save_dir / "track_label.pkl", 'wb') as f:
    pickle.dump(label_list, f)