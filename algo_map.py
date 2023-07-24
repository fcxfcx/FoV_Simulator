from algo_tile import *
import pandas as pd
import re

# D代表单个高清tile的数据量，W代表带宽值，LAMBDA代表延迟比重，THETA代表黑边影响比重
D = 3.2e-3
W = []
LAMBDA = 1.23
THETA = 1.4e2

# 计算黑边的公式参数
ALPHA = 3.549
BETA = 3.576
DELTA = 0.879
SIGMA = 0.5

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


def __line_magnitude(x1, y1, x2, y2):
    # 计算两点之间线段的距离
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
    return lineMagnitude


def get_probability_map(segment_time, user_id):
    csv_file = "./Predict/Map/user_"+str(user_id)+".csv"
    data_frame = pd.read_csv(csv_file, header=None)
    # 将第一列数据（索引为0的列）转换为字符串列表
    matrix_str_list = data_frame[0].tolist()

    # 使用numpy.fromstring函数将字符串转换回矩阵
    restored_matrix_list = []
    for matrix_str in matrix_str_list:
        if matrix_str == "0":
            continue
        # 按行分割字符串，并使用split方法将每行的数字分开
        matrix_values = [float(num) for num in re.findall(
            r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?', matrix_str)]
        matrix = np.array(matrix_values).reshape(9, 16)  # 这里假设每个矩阵有9行16列
        restored_matrix_list.append(matrix)
    pro_map = restored_matrix_list[segment_time]
    return pro_map


def get_init_set(start):
    init_set = set()
    for i in range(-1, 2):
        for j in range(-2, 3):
            start_x, start_y = start
            init_set.add((start_x+i, start_y+j))
    return init_set


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


def get_utility(segment_time, start, cur_set, new_tile, pro_map):
    # F的大小（FoV区域）
    fov_area = len(get_covered_tiles(start, new_tile))
    # 计算当前new_tile为终点的时候，cur_set内其他点到线段的总距离
    sum_distance = 0
    for item in cur_set:
        x, y = item
        cur_distance = distance_to_line(x, y, start, new_tile)
        sum_distance += cur_distance
    # 计算z的大小
    x1, y1 = start
    x2, y2 = new_tile
    z = __line_magnitude(x1, y1, x2, y2) / 9.2
    # 计算R的大小
    R = 0
    for item in cur_set:
        x, y = item
        cur_distance = distance_to_line(x, y, start, new_tile)
        c = cur_distance / sum_distance
        R += SIGMA * (1 + c*z) / fov_area
    R = R * pro_map[x][y]

    # 计算效用函数结果
    data_size = make_random_D(len(cur_set))                        # 数据量
    # 时间延迟
    delay = data_size / get_bandwidth(segment_time)
    black_edge = ALPHA * math.exp(-BETA * R) + DELTA  # 黑边影响
    cur_utility = data_size + LAMBDA * delay + THETA*black_edge   # 最终效用

    return cur_utility, R


def run_algo_map(start, map, segment_time):
    cur_set = get_init_set(start)
    neighbor_set = get_neighbors(cur_set)
    count = 0
    pre_utility = sys.maxsize
    cur_utility = sys.maxsize

    while count < 9*16:
        selected_tile = (0, 0)
        min_utility = sys.maxsize
        black_edge = 0
        neighbor_set = get_neighbors(cur_set)
        for item in neighbor_set:
            temp_utility, temp_black = get_utility(
                segment_time, start, cur_set, item, map)
            if temp_utility < min_utility:
                min_utility = temp_utility
                selected_tile = item
                black_edge = temp_black
        cur_utility = min_utility
        if cur_utility > pre_utility:
            print(black_edge)
            break
        else:
            pre_utility = cur_utility
            cur_set.add(selected_tile)
            count += 1
            print(black_edge)
    print(count+1)
    return cur_set


if __name__ == '__main__':
    start = (3, 5)
    map = get_probability_map(3, 1)
    run_algo_map(start, map, 3)
