import os

import numpy as np
import pandas as pd


current_file_path = os.path.abspath(__file__)
parent_dir = os.path.dirname(os.path.dirname(current_file_path))
target_dir = os.path.join(parent_dir, 'recorded_trackfiles', 'init_position')


# 读取 npy 文件
data_path = os.path.join(os.path.dirname(current_file_path), 'generated_data', 'init_data.npy')
data = np.load(data_path)


np.set_printoptions(precision=3, suppress=True)



for index, frame in enumerate(data):

	mask = (frame != -1).all(axis=-1)  # 在最后一个维度上检查是否所有值都不是 -1
	filtered_data = np.vstack([frame[i, mask[i]] for i in range(3)])  # 合并成 (n, 3)
	print(filtered_data)

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

	n =filtered_data.shape[0]
	track_id_list = list(range(1, n + 1))
	frame_id = [index] * n
	timestamp_ms = [100] * n
	agent_type =  ['car'] * n
	x = filtered_data[:, 0]
	y = filtered_data[:, 1]
	yaw_angles = filtered_data[:, 2]
	vx = [0] * n
	vy = [0] * n
	length = [3.80] * n
	width = [1.50] * n

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

	df = pd.DataFrame(file)
	csv_filename = os.path.join(target_dir, f'{index}.csv')
	df.to_csv(csv_filename, index=False)
	