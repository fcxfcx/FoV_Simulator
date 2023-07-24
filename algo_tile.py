"""
确定性输出对应的算法
确定性指的是预测算法输出为一个终点tile
"""
import random
import math
import json
import sys
import numpy as np
import seaborn as sns
import csv
import pandas as pd
from matplotlib import pyplot as plt

# D代表单个高清tile的数据量，W代表带宽值，LAMBDA代表延迟比重，THETA代表黑边影响比重
D = 3.2e-3
W = []
LAMBDA = 5.5
THETA = 2.34e2

# 计算黑边的公式参数
ALPHA = 3.549
BETA = 3.576
DELTA = 0.879
SIGMA = 1.02

# Map为高9宽16的矩阵
ROW = 9
COL = 16


def make_random_D(tile_count):
    total_D = 0
    for i in range(tile_count):
        # 服从正态分布的随机数，保留六位小数
        random_num = round(random.normalvariate(3e-3, 3.2e-3), 6)
        total_D += random_num
    return total_D


def get_bandwidth(segment_time):
    with open("./network_trace.json", 'r') as file:
        json_data = json.load(file)
    bandwidth = json_data.get(str(segment_time))
    return bandwidth/1024/8


def get_neighbors(tile_set):
    neighbor_set = set()

    for tile in tile_set:
        x, y = tile
        # 遍历tile的九宫格内的所有位置
        for i in range(x - 1, x + 2):
            for j in range(y - 1, y + 2):
                # 将i和j限定在矩阵范围内
                row = i % ROW
                col = j % COL
                neighbor = (row, col)
                # 检查邻居节点是否属于tile_set，并且不等于tile本身
                if neighbor != tile and neighbor not in tile_set:
                    neighbor_set.add(neighbor)

    return neighbor_set


def get_init_set(start, end):
    """获取初始阶段的选中tile

    Args:
        start (point): 起点
        end (point): 终点

    Returns:
        set: 以start和end为中心点的两个3*5区域的并集
    """
    init_set = set()
    for i in range(-1, 2):
        for j in range(-2, 3):
            start_x, start_y = start
            end_x, end_y = end
            init_set.add((start_x+i, start_y+j))
            init_set.add((end_x+i, end_y+j))
    return init_set


def get_fov_set(gaze_x, gaze_y):
    init_set = set()
    for i in range(-1, 2):
        for j in range(-2, 3):
            init_set.add((gaze_x+i, gaze_y+j))
    return init_set


def __line_magnitude(x1, y1, x2, y2):
    # 计算两点之间线段的距离
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
    return lineMagnitude


def distance_to_line(x, y, line_start, line_end):
    x1, y1 = line_start
    x2, y2 = line_end
    line_magnitude = __line_magnitude(x1, y1, x2, y2)
    if line_magnitude < 0.00000001:
        return 9999
    else:
        u1 = (((x - x1) * (x2 - x1)) + ((y - y1) * (y2 - y1)))
        u = u1 / (line_magnitude * line_magnitude)
        if (u < 0.00001) or (u > 1):
            # 点到直线的投影不在线段内
            if is_crossover(line_start, line_end):  # 不在线段内但是边界延伸，则在区域内，用投影算
                ix = x1 + u * (x2 - x1)
                iy = y1 + u * (y2 - y1)
                distance = __line_magnitude(x, y, ix, iy)
            else:  # 不在线段内但是边界未延伸，说明在区域外，用直接距离算
                ix = __line_magnitude(x, y, x1, y1)
                iy = __line_magnitude(x, y, x2, y2)
                return min(ix, iy)
        else:
            # 投影点在线段内部
            if is_crossover(line_start, line_end):  # 投影在线段内，但是边界延伸了，说明在区域外，用直接距离算
                ix = x1 + u * (x2 - x1)
                iy = y1 + u * (y2 - y1)
                distance = __line_magnitude(x, y, ix, iy)
            else:  # 投影在线段内且未发生边界延伸，则在区域内
                ix = x1 + u * (x2 - x1)
                iy = y1 + u * (y2 - y1)
                distance = __line_magnitude(x, y, ix, iy)
        return distance


def is_crossover(line_start, line_end):
    """判断起点到终点的方向

    Args:
        line_start ([x,y]): 起点坐标
        line_end ([x,y]): 终点坐标

    Returns:
        bool: 是否跨越了边界延申
    """
    x1, y1 = line_start[0], line_start[1]
    x2, y2 = line_end[0], line_start[1]

    if x1 == x2:
        if y1 == y2:
            return False
        else:
            return COL-abs(y2-y1) < abs(y2 - y1)
    else:
        if y1 == y2:
            return ROW-abs(x2-x1) < abs(x2-x1)
        else:
            # 计算两个路径的长度
            distance_1 = abs(x2 - x1) + abs(y2 - y1)  # 路径1：从起点到终点
            distance_2 = (ROW - abs(x2 - x1)) + \
                (COL - abs(y2 - y1))  # 路径2：从起点绕一圈到终点
            # 判断点是否位于起点和终点之间
            return distance_1 < distance_2


def get_covered_tiles(line_start, line_end):
    # 获取从起点到终点路径上所有经过的tile，考虑视野区域大小
    x1, y1 = line_start
    x2, y2 = line_end

    cross_over = is_crossover(line_start, line_end)
    # 计算线段的长度
    if not cross_over:
        line_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if line_length == 0:
            line_length = 1
    else:
        if x1 != x2:
            height = ROW - abs(x2-x1)
        else:
            height = 0
        if y1 != y2:
            width = COL - abs(y2-y1)
        else:
            width = 0
        line_length = math.sqrt(height**2 + width**2)

    covered_tiles = set()
    # 遍历从起点到终点路径上的所有点
    for i in range(int(line_length) + 1):
        # 计算当前点的坐标
        if not cross_over:
            x = round(x1 + (x2 - x1) * i / line_length) % ROW
            y = round(y1 + (y2 - y1) * i / line_length) % COL
        else:
            if x1 <= x2 and y1 <= y2:  # 向左上
                x = round(x1 - height * i / line_length) % ROW
                y = round(y1 - width * i / line_length) % COL
            if x1 < x2 and y1 > y2:  # 向左下
                x = round(x1 - height * i / line_length) % ROW
                y = round(y1 + width * i / line_length) % COL
            if x1 >= x2 and y1 <= y2:  # 向右上
                x = round(x1 + height * i / line_length) % ROW
                y = round(y1 - width * i / line_length) % COL
            if x1 > x2 and y1 > y2:  # 向右下
                x = round(x1 + height * i / line_length) % ROW
                y = round(y1 + width * i / line_length) % COL

        # 将当前点视为覆盖的中心点，计算覆盖的范围
        left = (y-2) % COL
        right = (y+2) % COL
        top = (x-1) % ROW
        bottom = (x+1) % ROW

        if left < right and top < bottom:
            # 将范围内的所有tile添加到covered_tiles集合中
            for row in range(top, bottom + 1):
                for col in range(left, right + 1):
                    covered_tiles.add((row, col))
        else:
            if left > right and top < bottom:
                # 发生了左右延伸的情况
                for row in range(top, bottom+1):
                    for col in range(left, 16):
                        covered_tiles.add((row, col))
                for row in range(top, bottom+1):
                    for col in range(0, right+1):
                        covered_tiles.add((row, col))
            if left < right and top > bottom:
                # 发生了上下延伸的情况
                for row in range(top, 9):
                    for col in range(left, right+1):
                        covered_tiles.add((row, col))
                for row in range(0, bottom+1):
                    for col in range(left, right+1):
                        covered_tiles.add((row, col))
            if left > right and top > bottom:
                # 上下左右全部延伸的情况
                for row in range(top, 9):
                    for col in range(left, 16):
                        covered_tiles.add((row, col))
                for row in range(top, 9):
                    for col in range(0, right+1):
                        covered_tiles.add((row, col))
                for row in range(0, bottom+1):
                    for col in range(left, 16):
                        covered_tiles.add((row, col))
                for row in range(0, bottom+1):
                    for col in range(0, right+1):
                        covered_tiles.add((row, col))
    covered_tiles.remove(line_start)
    if line_start != line_end:
        covered_tiles.remove(line_end)
    return covered_tiles


def sort_points_by_distance(start, end, neighbor_tiles):
    # 按照点到线段的距离排序点的列表
    distances = []
    for point in neighbor_tiles:
        x, y = point
        distance = distance_to_line(x, y, start, end)
        distances.append([distance, point])
    # 按照距离从小到大排序
    distances.sort()
    return distances


def get_utility(segment_time, fov_area, cur_set, sorted_neighbors, pre_R, line_width):
    """计算加入新tile后的效用值

    Args:
        fov_area : 预测结果中整个FOV区域的集合元素数(对应公式中的F)
        cur_set : 当前选中的tile集合
        sorted_neighbors : 当前选中的tile的邻居集合,经过了排序并存储了每个距离
        pre_R : 上一次计算的R值
        pre_utility: 上一次计算的效用函数值
        line_width: 起点到终点线段长度

    Returns:
        float, float: 当前的效用值和当前的R值
    """
    cur_distance = sorted_neighbors[0][0]
    sum_distance = 0
    for item in sorted_neighbors:
        sum_distance += item[0]
    c = cur_distance / sum_distance

    z = line_width / 9.2

    cur_R = pre_R + (SIGMA * (1 + c * z) / fov_area)  # 当前的R值

    data_size = make_random_D(len(cur_set))                        # 数据量
    # 时间延迟
    delay = data_size / get_bandwidth(segment_time)
    black_edge = ALPHA * math.exp(-BETA * cur_R) + DELTA  # 黑边影响
    cur_utility = data_size + LAMBDA * delay + THETA*black_edge   # 最终效用

    return cur_utility, cur_R


def plot_matrix_with_set(new, tile_set, index):
    # 创建一个空的矩阵，将每个格子初始化为白色
    matrix = np.zeros((ROW, COL))
    matrix.fill(np.nan)

    # 将之前的 tile 涂成蓝色
    for tile in tile_set:
        row, col = tile
        matrix[row, col] = 1

    # 将新的点涂成绿色
    row, col = new
    matrix[row, col] = 2

    # 绘制矩阵图
    sns.set()
    cmap = sns.color_palette(["blue", "green"])
    plt.figure(figsize=(10, 6))
    sns.heatmap(matrix, cmap=cmap, cbar=False, linewidths=1,
                linecolor="black", annot=True, fmt=".0f")
    plt.xticks([])
    plt.yticks([])
    plt.savefig("./tileSelect/Select_Tile_"+str(index)+".png")
    plt.close()


def run_algo_tile(start, end, segment_time):
    x1, y1 = start
    x2, y2 = end
    line_width = __line_magnitude(x1, y1, x2, y2)  # 线段长度
    cur_set = get_init_set(start, end)            # 获取初始区域
    fov_area = len(get_covered_tiles(start, end))  # 预测FOV区域大小
    pre_R = len(cur_set) / fov_area
    pre_utility = sys.maxsize
    cur_utility = sys.maxsize

    count = 0

    while count < 9*16:
        pre_utility = cur_utility
        neighbor_set = get_neighbors(
            cur_set)                             # 获取邻居集合
        sorted_tiles = sort_points_by_distance(
            start, end, neighbor_set)  # 对邻居集合根据距离排序

        cur_utility, pre_R = get_utility(
            segment_time, fov_area, cur_set, sorted_tiles, pre_R, line_width)
        if cur_utility > pre_utility:
            break
        else:
            new_tile = sorted_tiles[0][1]
            count += 1
            # print("current Utility: " + str(cur_utility) +
            #       " selected Tile: " + str(new_tile))
            # plot_matrix_with_set(new_tile, cur_set, count)
            cur_set.add(new_tile)
    print(count+1)
    return cur_set


def get_evalutate_data(user_id):
    predict, real = [], []
    predict_data_path = "./Predict/user_"+str(user_id)+".csv"
    real_data_path = "./TileFocus/Data/video_1/user_"+str(user_id)+".csv"
    with open(predict_data_path) as predict_f:
        # 读取预测数据,作为终点集合
        count = 3
        csv_reader = csv.reader(predict_f)
        next(csv_reader)
        for row in csv_reader:
            # 从1.5s开始取数据,代表第一个segment的预测数据
            if float(row[0]) < 1.5:
                continue
            # 每隔1s有一个可用数据
            if count == 3:
                count = 0
                predict.append((row[0], row[1], row[2]))
            else:
                count += 1

    with open(real_data_path) as real_f:
        # 读取真实数据,作为起点集合
        csv_reader = csv.reader(real_f)
        next(csv_reader)
        count = 3
        for row in csv_reader:
            # 从1s开始取数据,从1s后开始有预测数据可用
            if float(row[0]) < 1.0:
                continue
            if len(real) == len(predict):
                break
                # 每隔1s有一个可用数据
            if count == 3:
                count = 0
                real.append((row[0], row[1], row[2]))
            else:
                count += 1
    return predict, real


def get_black_edge_single_segment(user_id, segment_start, given_set):
    if segment_start == 201:
        return 0
    real_data_path = "./TileFocus/Data/video_1/user_"+str(user_id)+".csv"
    data_frame = pd.read_csv(real_data_path, index_col="time")
    temp_index = segment_start
    black_edge = 0
    while temp_index < segment_start + 1:
        # 在1s内有四个数据,例如从1s开始,则需要统计1, 1.25, 1.5, 1.75四个时刻的数据
        result_row = data_frame.loc[temp_index]
        x = int(result_row['H'])
        y = int(result_row['W'])
        need_set = get_fov_set(x, y)
        # 统计在当前时刻,在fov区域内但是不在传输区域内的tile数量
        black_edge += len(need_set - given_set)
        temp_index += 0.25
    return black_edge


def evaluate_datasize():
    algo_all = []
    algo_predict = []
    algo_tile = []

    for user_id in range(1, 49):
        predict, real = get_evalutate_data(user_id)
        segment_count = len(real)
        # 使用全传输策略的算法
        algo_all.append(9*16*segment_count)
        # 使用预测传输策略的算法
        temp_sum = 0
        for i in range(segment_count):
            start = (int(real[i][1]), int(real[i][2]))
            end = (int(predict[i][1]), int(predict[i][2]))
            temp_sum += len(get_covered_tiles(start, end))
        algo_predict.append(temp_sum)
        # 使用子模优化算法
        temp_sum = 0
        for i in range(segment_count):
            start = (int(real[i][1]), int(real[i][2]))
            end = (int(predict[i][1]), int(predict[i][2]))
            temp_sum += len(run_algo_tile(start, end, i))
        algo_tile.append(temp_sum)
    save_path = "./AlgoEvaluateData/datasize.csv"
    f = open(save_path, 'w', encoding="utf-8", newline="")
    writer = csv.writer(f)
    writer.writerow(["user id", "algo_all", "algo_predict", "algo_tile"])
    for i in range(len(algo_all)):
        writer.writerow([i+1, algo_all[i], algo_predict[i], algo_tile[i]])
    f.close()
    return algo_all, algo_predict, algo_tile


def evaluate_black_edge():
    for user_id in range(1, 2):
        algo_predict = []
        algo_tile = []
        predict, real = get_evalutate_data(user_id)
        segment_count = len(real)
        # 使用预测传输策略的方法
        for i in range(segment_count):
            time = float(real[i][0])
            start = (int(real[i][1]), int(real[i][2]))
            end = (int(predict[i][1]), int(predict[i][2]))
            given_set = get_covered_tiles(start, end)
            algo_predict.append(
                get_black_edge_single_segment(user_id, time, given_set))
        for i in range(segment_count):
            time = float(real[i][0])
            start = (int(real[i][1]), int(real[i][2]))
            end = (int(predict[i][1]), int(predict[i][2]))
            given_set = run_algo_tile(start, end, i)
            algo_tile.append(get_black_edge_single_segment(
                user_id, time, given_set))
        save_path = "./AlgoEvaluateData/black_edge/user_"+str(user_id)+".csv"
        f = open(save_path, 'w', encoding="utf-8", newline="")
        writer = csv.writer(f)
        writer.writerow(["segment id", "algo_predict", "algo_tile"])
        for i in range(len(algo_predict)):
            writer.writerow([i+1, algo_predict[i], algo_tile[i]])
        f.close()


if __name__ == '__main__':
    evaluate_black_edge()
