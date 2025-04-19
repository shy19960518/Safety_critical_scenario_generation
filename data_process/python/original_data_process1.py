# data process from interaction dataset. 
# get frameshot-like initial deployment from Original files

# There are map module in the loop so please wait longer for data process.

from pathlib import Path

import pandas as pd
import numpy as np
from utils.grid_map import Grid, get_y



base_dir = Path(__file__).parent.resolve() 

map_path = (base_dir / "../maps/merge.osm").resolve()

print(base_dir)
print(map_path)
assert 1==2
folder_path = (base_dir / "../recorded_trackfiles/DR_CHN_Merging_ZS/").resolve()

grid = Grid(1000, 1140, 5)
data_list = []
label_list = []


# Extract each file in turn and present the initial state frame by frame
for file_path in folder_path.iterdir():
    if file_path.suffix != '.csv':  # check if not CSV file, continue
        continue
    continue_loop = True
    # get file number 
    file_number = int(file_path.stem.split('_')[-1])
    
    # Read CSV Files
    df = pd.read_csv(file_path)
    
    for frame_id in range(99999): 
        if frame_id % 300 == 0:  # Print once every 300 cycles
            print(f"Iteration {frame_id} completed.")

        filtered_df = df[df['frame_id'] == frame_id + 1].copy()

        if filtered_df.empty:
            continue_loop = False
            break

        # Only keep the data of the target roads (the following three roads):
        rows_to_delete = []
        for index, row in filtered_df.iterrows():
            y_max = get_y(map_path, row['x'], 1000)
            y_min = get_y(map_path, row['x'], 4000)
            assert y_max > y_min
            if not (y_min <= row['y'] <= y_max):
                rows_to_delete.append(index)

        filtered_df.drop(rows_to_delete, inplace=True)

        data = np.full((3, 28, 3), -1, dtype=float)

        for index, row in filtered_df.iterrows():
            x = row['x']
            y = row['y']
            psi_rad = row['psi_rad']
            j = grid.get_grid_number_x(x)
            i = grid.get_lane_id(get_y(map_path, x, 1000), get_y(map_path, x, 2000), get_y(map_path, x, 3000), get_y(map_path, x, 4000), y)
            data[i, j] = np.array([x, y, psi_rad])
        data_list.append(data)
        
        if file_number not in [8,9,10]:
            label_list.append(0)
        else:
            label_list.append(1)

    if not continue_loop:
        print(f'{file_number} done.')
        continue

data = np.array(data_list)
label = np.array(label_list)

# 拼接得到绝对路径
save_dir = (base_dir / "processed_data").resolve()
save_dir.mkdir(parents=True, exist_ok=True)

# 保存数据
np.save(save_dir / "init_data.npy", data)
np.save(save_dir / "init_label.npy", label)

