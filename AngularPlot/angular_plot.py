import math
from pyquaternion import Quaternion
import csv
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def calculate_rotation(quat1, quat2):
    """计算两个四元数之间的角度差值

    Args:
        quat1 (quaternion): 起始四元数
        quat2 (quaternion): 目标四元数

    Returns:
        俯仰角，偏航角和角速度（度）
    """

    # 计算插值因子
    t = 0.5
    
    # 插值得到旋转中间状态的四元数
    quat_interp = Quaternion.slerp(quat1, quat2, t)
    
    # 计算差异四元数
    quat_diff = quat_interp * quat1.conjugate
    
    # 将差异四元数转换为欧拉角
    roll, pitch, yaw = quart_to_rpy(quat_diff.w,quat_diff.x,quat_diff.y,quat_diff.z)
    
    # 提取俯仰角和偏航角，并乘以2
    pitch = 2 * pitch # 俯仰角
    yaw = 2 * yaw  # 偏航角
    
    # 使用球面Haversine距离公式计算角速度，此处返回的d可以认为是单位圆上对应的弧长（半径为1）
    d = Quaternion.distance(quat1,quat2)
    
    # 将角速度（弧度）转换为角速度（度）
    d_deg = math.degrees(d)
    pitch = math.degrees(pitch)
    yaw = math.degrees(yaw)
    
    return pitch, yaw, d, d_deg

def quart_to_rpy(x, y, z, w):
    """由四元数计算欧拉角

    Args:
        x (float): 四元数x
        y (float): 四元数y
        z (float): 四元数z
        w (float): 四元数w

    Returns:
        float: 欧拉角对应的roll,pitch,yaw
    """
    # 下面计算的是分别绕x轴，y轴和z轴的旋转，y轴对应偏航，z轴对应俯仰
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    yaw = math.asin(2 * (w * y - x * z))
    pitch = math.atan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return roll, pitch, yaw

def angular_data():
    """为每个视频的每个用户保存俯仰角、偏航角和角速度数据
    """
    for video_id in range(9):
        for user_id in range(1,49):
            # 构建文件夹
            data_path = "./AngularPlot/Data/video_"+ str(video_id) +"/user_" + str(user_id)
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            pitch_data, yaw_data, degree_data= [],[],[]
            q_data = get_quaternion_from_csv(user_id, video_id)
            previous_q = q_data[0]
            # count等于8的时候代表一个segment结束，计算一次结果并存入数据中
            count = 0
            segment_temp = []
            for index, q in enumerate(q_data):
                if index == 0:
                    continue
                pitch, yaw, _, d_deg = calculate_rotation(previous_q, q)
                previous_q = q
                # 每条数据都存入当前segment的temp集合中
                segment_temp.append([pitch,yaw,d_deg])
                count += 1
                # 一个segment结束的时候，清理temp数据并填入一条绘图数据
                if count == 8 or index==len(q_data)-1:
                    average = segment_average(segment_temp)
                    pitch_data.append(average[0])
                    yaw_data.append(average[1])
                    degree_data.append(average[2])
                    segment_temp = []
                    count = 0
            # 保存数据
            f1 = open(data_path+"/pitch.csv", 'w', encoding="utf-8", newline="")
            f2 = open(data_path+"/yaw.csv", 'w', encoding="utf-8", newline="")
            f3 = open(data_path+"/degree_velocity.csv", 'w', encoding="utf-8", newline="")
            save_data(f1, pitch_data)
            save_data(f2, yaw_data)
            save_data(f3, degree_data)
        print("finish with video "+str(video_id))

def save_data(file, data):
    csv_writer = csv.writer(file)
    csv_writer.writerow(["index","degree"])
    index = 0
    for item in data:
        csv_writer.writerow([index, item])
        index += 1
    file.close()

def cdf_plot(plot_type):
    num_videos = 9
    num_users = 48
    plt.figure(dpi=400)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22']
    sns.set_palette(colors)
    data = []
    # 针对同一用户的不同视频绘图
    for video_index in range(num_videos):
        for user_index in range(1,num_users+1):
            user_file = "./AngularPlot/Data/video_"+str(video_index)+"/user_"+str(user_index)+"/"+plot_type+".csv" 
            user_current = pd.read_csv(user_file)
            data.append(user_current['degree'])
        data= pd.concat(data)   
        sns.ecdfplot(data=data,label='Video_'+str(video_index))
        data = []
    # 添加标签和标题
    plt.xlabel(plot_type)
    plt.ylabel('CDF')
    plt.title('Cumulative Distribution Function')
    # 显示图例
    plt.legend()
    # 显示图形
    plt.savefig("./AngularPlot/Figure/example/"+plot_type+"_video.png")
    plt.close()

    data = []
    plt.figure(dpi=400)
    # 针对同一视频的不同用户绘图
    for user_index in range(1, 9):
        for video_index in range(num_videos):
            user_file = "./AngularPlot/Data/video_"+str(video_index)+"/user_"+str(user_index)+"/"+plot_type+".csv" 
            user_current = pd.read_csv(user_file)
            data.append(user_current['degree'])
        data= pd.concat(data)
        sns.ecdfplot(data=data,label='User_'+str(user_index))
        data = []
    # 添加标签和标题
    plt.xlabel(plot_type)
    plt.ylabel('CDF')
    plt.title('Cumulative Distribution Function')
    # 显示图例
    plt.legend()
    # 显示图形
    plt.savefig("./AngularPlot/Figure/example/"+plot_type+"_user.png")
    plt.close()

def kde_plot(plot_type):
    num_videos = 9
    num_users = 9
    plt.figure(dpi=400)
    # 针对同一用户的不同视频绘图
    for index in range(num_videos):
        user_file = "./AngularPlot/Data/video_"+str(index)+"/user_1/"+plot_type+".csv" 
        user_current = pd.read_csv(user_file)
        data = user_current['degree']
        sns.kdeplot(data=data,label='Video_'+str(index), cut=0, cumulative=True)
    # 添加标签和标题
    plt.xlabel(plot_type)
    plt.ylabel('CDF')
    plt.title('Cumulative Distribution Function')
    # 显示图例
    plt.legend()
    # 显示图形
    plt.savefig("./AngularPlot/Figure/example/kde/"+plot_type+"_video.png")
    plt.close()

    plt.figure(dpi=400)
    # 针对同一视频的不同用户绘图
    for index in range(1, num_users+1):
        user_file = "./AngularPlot/Data/video_0/user_"+str(index)+"/"+plot_type+".csv" 
        user_current = pd.read_csv(user_file)
        sns.kdeplot(data=user_current['degree'],label='User_'+str(index), cut = 0, cumulative=True)
    # 添加标签和标题
    plt.xlabel(plot_type)
    plt.ylabel('CDF')
    plt.title('Cumulative Distribution Function')
    # 显示图例
    plt.legend()
    # 显示图形
    plt.savefig("./AngularPlot/Figure/example/kde/"+plot_type+"_user.png")
    plt.close()

def segment_average(segment_array):
    """计算一个segment(2s)内的俯仰角总和，偏航角总合和球面角速度平均值

    Args:
        segment_array (array): 长度为8的

    Returns:
        _type_: _description_
    """
    pitch, yaw, d_deg = 0,0,0
    length = len(segment_array)
    for item in segment_array:
        pitch += item[0]
        yaw += item[1]
        d_deg += item[2]/0.25
    return [pitch/length, yaw/length, d_deg/length]

def get_quaternion_from_csv(user_id, video_id):
    """从csv文件中获取用户的四元数数据,每250ms提取一个数据

    Args:
        user_id (int): 用户id,0~48
        video_id (int): 视频id,0~8

    Returns:
        array: 四元数数组
    """
    quaternion_Data = []
    UserFile = "./vr-dataset/Experiment_1/" + str(user_id) + "/video_"+ str(video_id) +".csv"
    
    with open(UserFile) as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        t_temp = 0
        for row in csv_reader:
            if t_temp > float(row[1]):
                continue
            q = Quaternion(float(row[5]),float(row[2]),float(row[3]),float(row[4]))
            quaternion_Data.append(q)
            t_temp += 0.25
    return quaternion_Data


if __name__ == '__main__':
    cdf_plot("pitch")
    cdf_plot("yaw")
    cdf_plot("degree_velocity")