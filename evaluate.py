import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

predictions_data = []
real_data = []

def count_overlap(x1, y1, x2, y2):
    """计算两个中心点附近九宫格区域的重合部分

    Args:
        x1 (int): 第一个中心点的x坐标
        y1 (int): 第一个中心点的y坐标
        x2 (int): 第二个中心点的x坐标
        y2 (int): 第二个中心点的y坐标

    Returns:
        int: 重合格子的数量
    """
    # 计算第一个视野区域的范围
    left1 = (x1 - 1) % 9
    right1 = (x1 + 1) % 9
    top1 = (y1 - 1) % 16
    bottom1 = (y1 + 1) % 16

    # 计算第二个视野区域的范围
    left2 = (x2 - 1) % 9
    right2 = (x2 + 1) % 9
    top2 = (y2 - 1) % 16
    bottom2 = (y2 + 1) % 16

    # 计算两个视野区域的重叠格子数量
    overlap_count = 0

    if (left1 <= right1 and top1 <= bottom1) and (left2 <= right2 and top2 <= bottom2):
        for i in range(left1, right1 + 1):
            for j in range(top1, bottom1 + 1):
                if i >= left2 and i <= right2 and j >= top2 and j <= bottom2:
                    overlap_count += 1
    else:
        for i in range(left1, 9):
            for j in range(top1, 16):
                if i >= left2 and i <= right2 and j >= top2 and j <= bottom2:
                    overlap_count += 1
        for i in range(0, right1 + 1):
            for j in range(top1, 16):
                if i >= left2 and i <= right2 and j >= top2 and j <= bottom2:
                    overlap_count += 1

    return overlap_count

def generate_heatmap(x1, y1, x2, y2):
    """绘制预测视野和真实视野的热图

    Args:
        x1 (int): 预测视野x坐标
        y1 (int): 预测视野y坐标
        x2 (int): 真实视野x坐标
        y2 (int): 真实视野y坐标

    Returns:
        fig: 图像对象，预测使用用蓝色表示，真实视野用绿色表示
    """
    # 定义矩阵的宽度和高度
    matrix_width = 16
    matrix_height = 9

    # 创建一个白色画布作为热力图
    fig, ax = plt.subplots(figsize=(matrix_width, matrix_height), facecolor='white')

    # 定义边框的宽度
    border_width = 2

    # 绘制九宫格视野图的边框和填充
    for i in range(-1, 2):
        for j in range(-1, 2):
            center_x = (x1 + i) % matrix_height
            center_y = (y1 + j) % matrix_width
            bbox = patches.Rectangle((center_y - 0.5, center_x - 0.5), 1, 1, linewidth=border_width, edgecolor='blue', facecolor='blue', alpha=0.2)
            ax.add_patch(bbox)

    # 绘制x2和y2区域的边框和填充
    for i in range(-1, 2):
        for j in range(-1, 2):
            center_x = (x2 + i) % matrix_height
            center_y = (y2 + j) % matrix_width
            bbox = patches.Rectangle((center_y - 0.5, center_x - 0.5), 1, 1, linewidth=border_width, edgecolor='green', facecolor='green', alpha=0.2)
            ax.add_patch(bbox)

    # 绘制热力图边框
    for i in range(matrix_height + 1):
        ax.axhline(i - 0.5, color='black', linewidth=0.5)
    for j in range(matrix_width + 1):
        ax.axvline(j - 0.5, color='black', linewidth=0.5)

    # 移除刻度
    ax.set_xticks([])
    ax.set_yticks([])

    # 设置图像范围
    ax.set_xlim(-0.5, matrix_width - 0.5)
    ax.set_ylim(matrix_height - 0.5, -0.5)

    # 返回热力图图像
    return fig

def calculate_percentage(numerator, denominator):
    """计算占比并以百分比形式返回

    Args:
        numerator (int): 除数
        denominator (int): 被除数

    Returns:
        float: 保留两位小数的百分比
    """
    result = numerator / denominator
    # 将结果乘以100并保留两位小数
    percentage = round(result * 100, 2)
    return percentage

if __name__ == '__main__':
    preidct_hit = []
    for usr_index in range(1,10):
        real_data = []
        predict_data = []
        real_path = "./Real/user_"+str(usr_index) +".csv"
        predict_path = "./Predict/user_"+str(usr_index)+".csv"
        f1 = open(real_path, 'r')
        f2 = open(predict_path, 'r')
        # 检查图片保存路径
        heatmap_path = "./heatmap/User_"+str(usr_index)
        if not os.path.exists(heatmap_path):
            os.makedirs(heatmap_path)

        # 读取真实数据
        csv_reader = csv.reader(f1)
        next(csv_reader)
        for row in csv_reader:
            # 保证两组数据的启示时间相同
            if float(row[0]) <1.25:
                continue
            temp = []
            temp.append(int(row[4]))
            temp.append(int(row[5]))
            real_data.append(temp)
        
        # 读取预测数据 
        csv_reader = csv.reader(f2)
        next(csv_reader)
        for row in csv_reader:
            if float(row[0]) > 201:
                break
            temp = []
            temp.append(int(row[1]))
            temp.append(int(row[2]))
            predict_data.append(temp)

        overlap = 0
        total = 0
        # 比较预测后tile的命中率
        if len(real_data) == len(predict_data):
            for index in range(len(real_data)):
                # 计算覆盖率
                total += 9
                real_temp = real_data[index]
                predict_temp = predict_data[index]
                overlap += count_overlap(real_temp[0], real_temp[1], predict_temp[0], predict_temp[1])

                # 保存热力图
                fig = generate_heatmap(predict_temp[0], predict_temp[1],real_temp[0], real_temp[1])
                fig_path = "./heatmap/User_"+ str(usr_index)+ "/time_"+str(index) +".png"
                plt.savefig(fig_path, bbox_inches='tight')
                plt.close(fig)
        preidct_hit.append(calculate_percentage(overlap,total))
        print("finish user " + str(usr_index))
    print(preidct_hit)