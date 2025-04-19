import os
import pandas as pd
import numpy as np

import xml.etree.ElementTree as xml
import pyproj
import math

class Grid:
    def __init__(self, min_x, max_x, x_step):
        self.min_x = min_x
        self.max_x = max_x
        self.x_step = x_step
    
    def get_grid_number_x(self, x_value):
        
        # 检查输入值是否在范围内
        if x_value <= self.min_x or x_value > self.max_x:
            raise ValueError(f"input {value} out of range ({self.min_x}, {self.max_x})")
        
        # 计算网格编号
        grid_number = (x_value - self.min_x) // 5

        return int(grid_number)

    def get_score_in_grid(self, x_value):
        # 检查输入值是否在范围内
        if x_value < self.min_x or x_value > self.max_x:
            raise ValueError(f"input {x_value} out of range ({self.min_x}, {self.max_x})")
        
        grid_number = self.get_grid_number_x(x_value)

        # 计算当前网格的边界
        grid_start = self.min_x + grid_number * self.x_step
        grid_end = grid_start + self.x_step

        # 计算占比分数
        score = (x_value - grid_start) / (grid_end - grid_start)
        
        return score

    def get_x_from_grid(self, grid_number, score):
        """
        根据 grid_number 和分数计算 x 值。

        参数:
        grid_number (int): 网格编号
        score (float): 在网格中的分数，范围 [0, 1]

        返回:
        float: 对应的 x 值
        """
        # 检查 score 是否在有效范围内
        if not (0 <= score <= 1):
            raise ValueError("Score must be between 0 and 1.")
        
        # 计算该网格的起点和终点
        grid_start = self.min_x + grid_number * self.x_step
        grid_end = grid_start + self.x_step

        # 根据分数计算 x 值
        x_value = grid_start + score * (grid_end - grid_start)
        
        return x_value

    def get_lane_id(self, y1, y2, y3, y4, y_value):
        """
        获取给定 y 值所属的车道 ID。

        参数:
        y1 ~ y4 是车道线的坐标
        y_value (float): 要检查的 y 值。

        返回:
        int: 如果 y_value 在车道范围内，则返回车道 ID，否则引发错误
        """
        # 检查 y_value 是否在车道范围内
        if y1 >= y_value >= y2:
            return 0  # 第一个车道
        elif y2 > y_value >= y3:
            return 1  # 第二个车道
        elif y3 > y_value >= y4:
            return 2  # 第三个车道
        else:
            raise ValueError(f"y_value {y_value} 不在任何车道范围内。")
    def convert_and_normalize(self, array):
        """
        将输入数组中的分数值转换为 1000 到 1140 之间的值，并归一化。

        参数:
        array (numpy.ndarray): 形状为 (B, 3, 28) 的数组，其中每个值是网格中的分数。

        返回:
        numpy.ndarray: 转换为 1000 到 1140 的值并归一化后的数组。
        """

        B, C, W = array.shape
        normalized_array = np.zeros_like(array)
        
        for b in range(B):
            for c in range(C):
                for w in range(W):
                    score = array[b, c, w]
                    if score == -1:  # 保留 -1 值
                        normalized_array[b, c, w] = -1
                    else:
                        # 如果 score 大于 1，将其设置为 1
                        if score > 1:
                            score = 1
                            
                        # 假设我们通过 score 和 grid_number 来恢复 x 值
                        grid_number = w
                        x_value = self.get_x_from_grid(grid_number, score)
                        # 归一化到 0 到 1 范围
                        normalized_value = (x_value - self.min_x) / (self.max_x - self.min_x)
                        # 将归一化后的值赋给数组
                        normalized_array[b, c, w] = normalized_value

        return normalized_array



class LL2XYProjector:
    def __init__(self, lat_origin, lon_origin):
        self.lat_origin = lat_origin
        self.lon_origin = lon_origin
        self.zone = math.floor((lon_origin + 180.) / 6) + 1
        self.p = pyproj.Proj(proj='utm', ellps='WGS84', zone=self.zone, datum='WGS84')
        [self.x_origin, self.y_origin] = self.p(lon_origin, lat_origin)

    def latlon2xy(self, lat, lon):
        [x, y] = self.p(lon, lat)
        return [x - self.x_origin, y - self.y_origin]

def get_way_ids_by_relation(e, relation_id):
    way_ids = []
    for relation in e.findall('relation'):
        if int(relation.get('id')) == relation_id:
            for member in relation.findall('member'):
                if member.get('type') == 'way':
                    way_ids.append(int(member.get('ref')))
    return way_ids

def get_coordinates_by_way(e, way_id, projector):
    coordinates = []
    way = e.find(f'way[@id="{way_id}"]')
    if way is not None:
        for nd in way.findall('nd'):
            ref_id = int(nd.get('ref'))
            node = e.find(f'node[@id="{ref_id}"]')
            if node is not None:
                lat = float(node.get('lat'))
                lon = float(node.get('lon'))
                x, y = projector.latlon2xy(lat, lon)
                coordinates.append((x, y))
    return coordinates

def find_y_for_x(x_input, e, relation_id, projector):
    way_ids = get_way_ids_by_relation(e, relation_id)
    for way_id in way_ids:
        coordinates = get_coordinates_by_way(e, way_id, projector)
        for i in range(len(coordinates) - 1):
            x1, y1 = coordinates[i]
            x2, y2 = coordinates[i + 1]
            if x1 <= x_input <= x2 or x2 <= x_input <= x1:
                if x1 == x2:
                    return y1  # or y2, both are the same
                # Linear interpolation
                y = y1 + (x_input - x1) * (y2 - y1) / (x2 - x1)
                return y
    return None

def get_y(osm_file, x_input, relation_id, lat_origin=0, lon_origin=0):
    projector = LL2XYProjector(lat_origin, lon_origin)
    e = xml.parse(osm_file).getroot()
    y_output = find_y_for_x(x_input, e, relation_id, projector)
    if y_output is not None:
        return y_output
    else:
        print("未找到对应的 y 坐标")

def get_nomorlised_y(osm_file, x_input, relation_id, lat_origin=0, lon_origin=0):

    x_input = x_input * (1140 - 1000) + 1000
    projector = LL2XYProjector(lat_origin, lon_origin)
    e = xml.parse(osm_file).getroot()
    y_output = find_y_for_x(x_input, e, relation_id, projector)
    y_output = (y_output - 935) / (955 - 935)
    if y_output is not None:
        return y_output
    else:
        print("未找到对应的 y 坐标")
        
def denormalize(normalized_data, min_value, max_value):
    """
    将归一化后的数据（0到1之间）反归一化到原始范围。
    :param normalized_data: 归一化后的数组或数字（0到1之间）
    :param min_value: 数据的最小值
    :param max_value: 数据的最大值
    :return: 反归一化后的数据
    """
    # 确保输入的数据是 NumPy 数组
    normalized_data = np.asarray(normalized_data)
    
    # 反归一化公式
    original_data = normalized_data * (max_value - min_value) + min_value
    
    return original_data

def restore_original_data(data, my_grid):
    """
    将归一化后的数据恢复为原始数据
    :param data: 归一化后的数据 (N, M, K, 3)，其中：
                 data[..., 0] 是 grid score
                 data[..., 1] 是归一化的 y 值 (935, 955)
                 data[..., 2] 是归一化的角度 (-3.14, 3.14)
    :param my_grid: Grid 类的实例
    :return: 还原后的原始数据
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

if __name__ == "__main__":
    osm_file = '../map.osm'  # 替换为你的 OSM 文件路径
    x_input = 1000  # 替换为你要查询的 x 坐标
    relation_id = 1000  # 替换为你要查询的关系 ID
    lat_origin = 0  # 替换为投影的纬度原点
    lon_origin = 0 # 替换为投影的经度原点

    y_output = get_y(osm_file, x_input, relation_id, lat_origin, lon_origin)

    print(f"y 坐标: {y_output}")