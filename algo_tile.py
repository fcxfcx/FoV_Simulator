"""
确定性输出对应的算法
确定性指的是预测算法输出为一个终点tile
"""
import math
import sys
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

# D代表单个高清tile的数据量，W代表带宽值，LAMBDA代表延迟比重，THETA代表黑边影响比重
D = 1
W = 10
LAMBDA = 1
THETA = 1

# 计算黑边的公式参数
ALPHA = 3.549
BETA = 3.576
DELTA = 0.879
SIGMA = 0.5 

# Map为高9宽16的矩阵
ROW = 9
COL = 16


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
            if is_crossover(line_start, line_end): # 不在线段内但是边界延伸，则在区域内，用投影算
                ix = x1 + u * (x2 - x1)
                iy = y1 + u * (y2 - y1)
                distance = __line_magnitude(x, y, ix, iy)
            else: #不在线段内但是边界未延伸，说明在区域外，用直接距离算
                ix = __line_magnitude(x, y, x1, y1)
                iy = __line_magnitude(x, y, x2, y2)
                return min(ix,iy)
        else:
            # 投影点在线段内部
            if is_crossover(line_start,line_end): # 投影在线段内，但是边界延伸了，说明在区域外，用直接距离算                     
                ix = x1 + u * (x2 - x1)
                iy = y1 + u * (y2 - y1)
                distance = __line_magnitude(x, y, ix, iy)
            else: # 投影在线段内且未发生边界延伸，则在区域内
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
            distance_2 = (ROW - abs(x2 - x1)) + (COL - abs(y2 - y1))  # 路径2：从起点绕一圈到终点
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
            if x1 <= x2 and y1 <= y2: #向左上
                x = round(x1 - height * i / line_length) % ROW
                y = round(y1 - width * i / line_length) % COL
            if x1 < x2 and y1 > y2: #向左下
                x = round(x1 - height * i / line_length) % ROW
                y = round(y1 + width * i / line_length) % COL
            if x1 >= x2 and y1 <= y2: #向右上
                x = round(x1 + height * i / line_length) % ROW
                y = round(y1 - width * i / line_length) % COL
            if x1 > x2 and y1 > y2: #向右下
                x = round(x1 + height * i / line_length) % ROW
                y = round(y1 + width * i / line_length) % COL

        # 将当前点视为覆盖的中心点，计算覆盖的范围
        left = (y-2)%COL
        right = (y+2)%COL
        top = (x-1)%ROW
        bottom = (x+1)%ROW

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
                    for col in range(0,right+1):
                        covered_tiles.add((row, col))
            if left <right and top > bottom:
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
                    for col in range(0,right+1):
                        covered_tiles.add((row, col))
                for row in range(0, bottom+1):
                    for col in range(left, 16):
                        covered_tiles.add((row, col))
                for row in range(0, bottom+1):
                    for col in range(0,right+1):
                        covered_tiles.add((row, col))
    covered_tiles.remove(line_start)
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
    print(distances)
    return distances

def get_utility(fov_area, cur_set, sorted_neighbors, pre_R, line_width):
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
    w1 = cur_distance / sum_distance

    w2 = line_width

    cur_R = pre_R + (SIGMA * w1 * w2 / fov_area) # 当前的R值

    data_size = D * len(cur_set)                         # 数据量
    delay = W/data_size                                  # 时间延迟
    black_edge = ALPHA* math.exp(-BETA * cur_R) + DELTA  # 黑边影响
    cur_utility = data_size + delay + THETA*black_edge   # 最终效用

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
    sns.heatmap(matrix, cmap=cmap, cbar=False, linewidths=1, linecolor="black", annot=True, fmt=".0f")
    plt.xticks([])
    plt.yticks([])
    plt.savefig("./tileSelect/Select_Tile_"+str(index)+".png")
    plt.close()


if __name__ == '__main__':
    start = (3, 5)
    end = (5, 9)
    x1, y1 = start
    x2, y2 = end
    line_width = __line_magnitude(x1, y1, x2, y2) # 线段长度
    cur_set = get_init_set(start, end)            # 获取初始区域
    fov_area = len(get_covered_tiles(start, end)) # 预测FOV区域大小
    pre_R = len(cur_set) / fov_area
    pre_utility = sys.maxsize
    cur_utility = sys.maxsize

    count = 0

    while count < 9*16:
        pre_utility = cur_utility
        neighbor_set = get_neighbors(cur_set)                             # 获取邻居集合
        sorted_tiles = sort_points_by_distance(start, end, neighbor_set)  # 对邻居集合根据距离排序


        cur_utility, pre_R = get_utility(fov_area, cur_set, sorted_tiles, pre_R, line_width)

        if cur_utility > pre_utility:
            print("Utility raise, algorithm ends")
            break
        else: 
            new_tile = sorted_tiles[0][1] 
            count += 1                                      
            plot_matrix_with_set(new_tile,cur_set,count)
            cur_set.add(new_tile)
    
    print(cur_set)