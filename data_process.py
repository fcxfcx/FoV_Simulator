import numpy as np
from sklearn import preprocessing
import pickle
import bisect
import torch
import cv2



def unit_vector(vector):
    return vector / np.linalg.norm(vector)


# 根据方向计算角度
def degree_distance(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))/np.pi * 180


# 计算一个vector的俯仰角
def vector_to_ang(_v):
    _v = np.array(_v)
    # degree between v and [0, 1, 0]
    alpha = degree_distance(_v, np.array([0, 1, 0]))
    phi = 90.0 - alpha
    # proj1 is the projection of v onto [0, 1, 0] axis
    proj1 = [0, np.cos(alpha/180.0 * np.pi), 0]
    # proj2 is the projection of v onto the plane([1, 0, 0], [0, 0, 1])
    proj2 = _v - proj1
    # theta = degree between project vector to plane and [1, 0, 0]
    theta = degree_distance(proj2, np.array([1, 0, 0]))
    sign = -1.0 if degree_distance(_v, np.array([0, 0, -1])) > 90 else 1.0
    theta = sign * theta
    return theta, phi

# 根据角度得到二维平面的视点坐标
def ang_to_geoxy(_theta, _phi, _h, _w):
    x = _h/2.0 - (_h/2.0) * np.sin(_phi/180.0 * np.pi)
    temp = _theta
    if temp < 0:
        temp = 360 + temp
    temp = 360 - temp
    y = (temp * 1.0/360 * _w)
    return int(x), int(y)


# 根据视点坐标生成fixation map
def create_fixation_map(viewData, idx, H, W):
    v = viewData[idx]
    theta, phi = vector_to_ang(v)
    hi, wi = ang_to_geoxy(theta, phi, H, W)
    result = np.zeros(shape=(H, W))
    result[H-hi-1, W-wi-1] = 1
    return result


def get_sal_fix(time_array, saliency_maps, req, window=4, time_window=0.5):
    """
    :param time_array: the timestamp of saliency maps
    :param saliency_maps: the saliency maps of video
    :param req: past timestamp and viewport direction
    :param window: predict window size
    :param time_window: predict timestamp
    :return: sal_fix_maps to ConvLSTM, predicted time
    """
    # req = [{'time': xx, 'x': xx, 'y': xx, 'z': xx}, ...]
    past_time = [float(i[0]) for i in req]
    past_pos = [[i[1], i[2], i[3]] for i in req]

    if len(past_pos) < window:
        past_pos.extend([past_pos[-1]] * (window - len(past_pos)))

    idx_start = bisect.bisect(time_array, past_time[0])
    if len(time_array) - idx_start < window:
        idx_list = list(range(idx_start, len(time_array)))
        idx_list.extend([idx_list[-1]] * (window - len(idx_list)))
    else:
        idx_list = list(range(idx_start, idx_start + 4))

    sal_maps = saliency_maps[idx_list]
    mmscaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    sal_maps = mmscaler.fit_transform(
        sal_maps.ravel().reshape(-1, 1)).reshape(sal_maps.shape)

    N, H, W = saliency_maps.shape
    mmscaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    fixation_maps = np.array([create_fixation_map(
        past_pos, idx, H, W) for idx, _ in enumerate(past_pos)])
    headmap = np.array(
        [cv2.GaussianBlur(item, (5, 5), 0) for item in fixation_maps])
    fix_maps = mmscaler.fit_transform(
        headmap.ravel().reshape(-1, 1)).reshape(headmap.shape)

    assert sal_maps.shape == fix_maps.shape, "sal_maps.shape != fix_maps.shape"
    sal_fix_list = []
    for i in range(len(fix_maps)):
        sal_fix_list.append(np.stack([sal_maps[i], fix_maps[i]]))
    sal_fix_maps = np.stack(sal_fix_list)
    inputs = torch.from_numpy(
        sal_fix_maps[np.newaxis, :, :, :, :]).to(torch.float32)
    if torch.cuda.is_available():
        inputs = inputs.cuda()

    return inputs, past_time[-1] + time_window

# 根据视点坐标生成fixation map


def de_interpolate(raw_tensor, N):
    """
    F.interpolate(source, scale_factor=scale, mode="nearest")的逆操作！
    :param raw_tensor: [B, C, H, W]
    :param N
    :return: [B, C, H // 2, W // 2]
    """
    out = np.zeros((N, 9, 16))
    for idx in range(10):
        out = out + raw_tensor[:, idx::10, idx::10]
    return out