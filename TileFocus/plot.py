import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def get_focus(video_id, user_id):
    result = []
    f_path = "./TileFocus/Data/video_"+str(video_id)+"/user_"+str(user_id)+".csv"
    with open(f_path) as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        temp_focus = [-1,-1]
        temp_time = 0
        for row in csv_reader:
            if temp_focus[0] == -1:
                temp_focus[0] = row[1]
                temp_focus[1] = row[2]
                continue
            if temp_focus[0] == row[1] or temp_focus[1] == row[2]:
                temp_time+=0.25
                continue
            temp_focus[0] = row[1]
            temp_focus[1] = row[2]
            result.append(temp_time)
            temp_time = 0
        result.append(temp_time)
    return result



def get_overlap(video_id, user_id):
    """获取视频持续时间内用户每个segment的视野覆盖多少个tile

    Args:
        video_id (int): 视频id
        user_id (int): 用户id
    """

    result = []
    f_path = "./TileFocus/Data/video_"+str(video_id)+"/user_"+str(user_id)+".csv"
    with open(f_path) as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        temp = []
        for row in csv_reader:
            x = int(row[1])
            y = int(row[2])
            temp.append([x,y])
            if len(temp) == 8:
                result.append(tile_count(temp))
                temp = []
        result.append(tile_count(temp))
    return result

def tile_count(data):
    """计算一个segment内一共看了多少个tile

    Args:
        data (array): 一个segment的视野中心数据，共八个

    Returns:
        int: 覆盖的tile的数量
    """
    viewed_cells = set()
    for item in data:
        x = item[0]
        y = item[1]
        for i in range(-1, 2):
            for j in range(-1, 2):
                row = (x + i) % 9
                col = (y + j) % 16
                viewed_cells.add((row, col))
    return len(viewed_cells)

def plot_overlay(video_id, user_id):
    # 计算总的tile数量
    total_tiles = 9 * 16

    # 计算每个segment内观看的tile数量
    segment_counts = get_overlap(video_id, user_id)
    segment_tile_percentages = [ ]

    less_than = 0
    for count in segment_counts:
        percentage = count / total_tiles * 100
        if percentage < 20:
            less_than += 1
        segment_tile_percentages.append(percentage)
    percentage_text = round(less_than / len(segment_counts) *100 , 2)


    # 绘制分布图
    kde = sns.kdeplot(segment_tile_percentages, fill=True, color='skyblue')

    # 添加竖线
    plt.axvline(x=20, color='red', linestyle='--')

    # 设置横坐标刻度为百分比形式
    plt.gca().set_xticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_xticks()])

    # 设置图形标题和坐标轴标签
    plt.title("Cumulative Distribution of Tile Percentages in Segments")
    plt.xlabel("Percentage of Tiles Watched in a Segment")
    plt.ylabel("Cumulative Probability")

    # 获取颜色填充的区域
    fill_area = kde.collections[0].get_paths()[0]

    # 确定文本的位置
    text_x = fill_area.vertices[:, 0].mean()
    text_y = fill_area.vertices[:, 1].mean()

    plt.text(text_x, text_y, str(percentage_text)+"%", fontsize=12, color='red')
    # 显示图形
    plt.show()

def plot_overlay_all():
    # 计算总的tile数量
    total_tiles = 9 * 16

    total = []
    segment_tile_percentages = []
    for video_id in range(9):
        for user_id in range(1,49):
            # 计算每个segment内观看的tile数量
            segment_counts = get_overlap(video_id, user_id)
            for count in segment_counts:
                percentage = round(count / total_tiles * 100, 2)
                segment_tile_percentages.append(percentage)
                total.append(percentage)
        sns.kdeplot(segment_tile_percentages, label="Video_"+str(video_id), cut=0)
        segment_tile_percentages = []
    
    sns.kdeplot(total,label="All", color = "black", cut=0)
    # 设置图形标题和坐标轴标签
    plt.title("Cumulative Distribution of Tile Percentages in Segments")
    plt.xlabel("Percentage of Tiles Watched in a Segment")
    plt.ylabel("Cumulative Probability")
    # 设置横坐标刻度为百分比形式
    plt.gca().set_xticklabels(['{:.0f}%'.format(x) for x in plt.gca().get_xticks()])
    plt.legend()
    plt.show()

def plot_focus():
    focus_time = []
    for video_id in range(9):
        for user_id in range(1,49):
            # 计算每个segment内观看的tile数量
            temp_focus = get_focus(video_id, user_id)
            for item in temp_focus:
                focus_time.append(item)
        sns.ecdfplot(focus_time, label="Video_"+str(video_id))
    plt.title("Cumulative Distribution of Focus time in Segments")
    plt.xlabel("Focus time(s) in a Segment")
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    plot_focus()