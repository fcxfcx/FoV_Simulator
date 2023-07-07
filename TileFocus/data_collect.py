import sys
sys.path.append("./")

from fov_predict import get_csv
import csv
import os
from data_process import vector_to_ang, ang_to_geoxy

def get_data(user_id, video_id):
    data = get_csv(userId=user_id,videoId=video_id)
    f_path = "./TileFocus/Data/video_"+str(video_id)
    if not os.path.exists(f_path):
        os.makedirs(f_path)
    f = open(f_path+"/user_"+str(user_id) +".csv", 'w', encoding="utf-8", newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow(["time","H", "W"])
    H, W = 9, 16
    for item in data:
        # 计算视野落点区域
        v = [item[1], item[2], item[3]]
        theta, phi = vector_to_ang(v)
        hi, wi = ang_to_geoxy(theta, phi, H, W)
        row = [item[0], H-hi-1, W-wi-1]
        csv_writer.writerow(row)  
    f.close()

if __name__ == '__main__':
    for video_id in range(9):
        for user_id in range (1,49):
            get_data(user_id,video_id)
        print("finish with video "+str(video_id))