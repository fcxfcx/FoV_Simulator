import fov_predict
import csv
from data_process import create_fixation_map, vector_to_ang, ang_to_geoxy
import numpy as np
import seaborn as sns
import cv2
from sklearn import preprocessing
import matplotlib.pyplot as plt
import evaluate

def get_data_test(index):
    data = fov_predict.get_csv(index)
    f_path = "./Real/user_"+str(index) +".csv"
    f = open(f_path, 'w', encoding="utf-8", newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow(["time", "x", "y", "z", "H", "W"])
    H, W = 9, 16
    for item in data:
        # 计算视野落点区域
        v = [item[1], item[2], item[3]]
        theta, phi = vector_to_ang(v)
        hi, wi = ang_to_geoxy(theta, phi, H, W)
        row = [item[0], item[1], item[2], item[3], H-hi-1, W-wi-1]
        csv_writer.writerow(row)  
    f.close()


def get_map_test(index):
    past_pos = []
    file_path = "user_Data_" + str(index) + ".csv"
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        count = 0
        for row in reader:
            cur_pos = []
            cur_pos.append(np.float32(row[1]))
            cur_pos.append(np.float32(row[2]))
            cur_pos.append(np.float32(row[3]))
            past_pos.append(cur_pos)
            count += 1
        print("Total data count:" + str(count))
    H = 9
    W = 16
    mmscaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    fixation_maps = np.array([create_fixation_map(
        past_pos, idx, H, W) for idx, _ in enumerate(past_pos)])
    headmap = np.array(
        [cv2.GaussianBlur(item, (5, 5), 0) for item in fixation_maps])
    fix_maps = mmscaler.fit_transform(
        headmap.ravel().reshape(-1, 1)).reshape(headmap.shape)
    return fixation_maps, fix_maps


if __name__ == '__main__':
    # for i in range(1,10):
    #     get_data_test(i)
    x1 = 4
    y1 = 5
    x2 = 5
    y2 = 6
    fig = evaluate.generate_heatmap(x1,y1,x2,y2)
    plt.savefig("heatmap_image.png",bbox_inches='tight')
    plt.close()

