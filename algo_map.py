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
