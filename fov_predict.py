import time
from data_process import *
import convlstm
import os
import pandas as pd
import csv
from pyquaternion import Quaternion

sal_maps_list = []
time_array_list = []
FoV_net = convlstm.ConvLSTM_model(input_dim=2,
                                  hidden_dim=6,
                                  kernel_size=(5, 5),
                                  num_layers=1,
                                  batch_first=True)


def FoV(data_batch):
    # for item in req:
    #     print(item)
    inputs, pre_time = get_sal_fix(
        time_array_list[0], sal_maps_list[0], data_batch)

    # return jsonify([pre_time, [-0.7,0,-0.7]])
    # output_list 每个时间点最后一层隐层结果: list
    # output_last 最后一个时间点隐层和细胞层结果: list
    # conv_output 最后一个时间点隐层和细胞层卷积之后结果: list
    # predict_array 每个时间点隐层卷积之后结果 torch.tensor
    output_list, output_last, conv_output, conv_output_list_ret = FoV_net(
        inputs)
    pre_x, pre_y = return_fov(conv_output)
    return pre_time, pre_x, pre_y, conv_output[0, 0].detach().cpu().numpy()


def return_fov(predict_array):
    """
    :param predict_array: predict the array of the image
    :return: (x, y, z)
    """
    pre_image = predict_array[0, 0].detach().cpu().numpy()
    # pre_image[pre_image < args.threshold] = 0
    # pre_image[pre_image > args.threshold] = 1
    H, W = pre_image.shape
    pos = np.argmax(pre_image)
    x, y = divmod(pos, W)   # pre_image: the index of max number
    return x, y


def index_to_xyz(pre_image: np.ndarray):
    H, W = pre_image.shape
    pos = np.argmax(pre_image)
    x, y = divmod(pos, W)   # pre_image: the index of max number
    point = [(H-x-1) % H, (W-y-1) % W]

    phi = np.arcsin(1 - 2/H * point[0])
    temp = 360 / W * point[1]
    theta = 360 - temp

    dx = np.cos(phi/180.0 * np.pi) * np.cos(theta/180.0 * np.pi)
    dy = np.sin(phi/180.0 * np.pi)
    dz = np.cos(phi/180.0 * np.pi) * np.sin(theta/180.0 * np.pi)

    return dx, dy, dz

# 从csv中获取数据


def get_csv(userId, videoId):
    Userdata = []
    UserFile = './vr-dataset/Experiment_1/' + \
        str(userId) + "/video_"+str(videoId)+".csv"
    with open(UserFile) as csvfile:
        csv_reader = csv.reader(csvfile)
        next(csv_reader)
        t_temp = 0
        for row in csv_reader:
            temp = []
            v0 = [0, 0, 1]
            # 每隔250ms采集一次
            if t_temp > float(row[1]):
                continue

            q = Quaternion([float(row[5]), -float(row[4]),
                           float(row[3]), -float(row[2])])
            new_vec = q.rotate(v0)
            temp.append(t_temp)
            temp.append(new_vec[0])
            temp.append(new_vec[1])
            temp.append(new_vec[2])
            Userdata.append(temp)
            t_temp += 0.25
    Userdata = np.array(Userdata)
    return Userdata


if __name__ == '__main__':
    # 载入滑雪视频的显著性数据
    salPath = "./SalData/saliency_ds2_topic1"
    try:
        saliency_array = np.array(pickle.load(
            open(salPath, 'rb'), encoding='bytes'), dtype=object)
    except pickle.UnpicklingError:
        saliency_array = np.load(salPath, allow_pickle=True)

    time_array_list.append(np.array([i[0] for i in saliency_array]))
    sal_array = np.array([i[2] for i in saliency_array])
    sal_maps_list.append(de_interpolate(sal_array, len(sal_array)))

    # 加载模型
    if torch.cuda.is_available():
        FoV_net = FoV_net.cuda()
    else:
        FoV_net = FoV_net.to(device="cpu")
    modelPath = f".\model\convlstm_offline_skiing_1_1s.pth"
    if not os.path.exists(modelPath):
        exit(f"{modelPath} doesn't exit!")

    FoV_net.load_state_dict(torch.load(
        modelPath, map_location=torch.device('cpu')))
    print('Loading model from', modelPath)
    print("Total number of parameters in networks is {}".format(
        sum(x.numel() for x in FoV_net.parameters())))
    FoV_net.eval()

    # 用户id
    for index in range(1, 49):
        # 读取源数据并喂给预测模型
        raw_data = get_csv(index, 1)
        map_list = []
        data_batch = []
        # 写入新的csv中
        f = open("./Predict/user_"+str(index)+".csv",
                 'w', encoding="utf-8", newline="")
        csv_writer = csv.writer(f)
        csv_writer.writerow(["time", "H", "W"])
        H, W = 9, 16
        for item in raw_data:
            data_batch.append(item)
            if len(data_batch) == 4:
                # 预测
                t, x, y, predict_map = FoV(data_batch)
                row = [t, x, y]
                csv_writer.writerow(row)
                map_list.append(predict_map)
                # 去掉最老的数据，加入新数据
                data_batch = data_batch[1:]
        f.close()
        matrix_series = pd.Series(map_list)
        csv_file = "./Predict/Map/user_"+str(index)+".csv"
        matrix_series.to_csv(csv_file, index=False)
        map_list = []
