import os
import mne
import pandas as pd
import numpy as np
import pywt


def get_label_dict(label_path):
    """
    获取实验标签
    :param label_path: 实验标签路径
    :return: 实验标签字典
    """
    # 标签列表
    label_dict = {}

    # 读取csv文件并转为numpy
    df = pd.read_csv(label_path)
    nd = df.to_numpy()

    # 读取标签信息
    name = ''  # 受试者姓名
    nan_flag = False  # nan标记（读到非连续的首个nan时置true）
    answer_sheet = []  # 记录答案
    for seq in nd:
        for i, item in enumerate(seq):
            if i == 0:
                name = item
                label_dict[item] = []
                answer_sheet = []
            elif item != 'F' and item != 'T' and not nan_flag:
                nan_flag = True
                label_dict[name].append(answer_sheet)
                answer_sheet = []
            elif item == 'F' or item == 'T':
                nan_flag = False
                answer_sheet.append(item)
        if len(answer_sheet) > 0:
            label_dict[name].append(answer_sheet)
    return label_dict


def bdf2seq(bdf_file, channel_name, dict_label, mode=0, is_wavelet=True):
    """
    将采集到的bdf数据转换为序列
    :param bdf_file: bdf文件地址
    :param channel_name: 通道
    :param dict_label: 标签字典
    :param mode: 特征提取模式（0：平均，1：最大，2：极差）
    :param is_wavelet: 是否使用小波变换去噪
    :return: 转换后的序列及对应标签
    """
    # 读取bdf文件某一通道
    raw = mne.io.read_raw_bdf(bdf_file, preload=True)
    channel_data = raw.get_data(picks=[channel_name])[0]
    # 获取序列标记信息
    tested_name = bdf_file.split('\\')[1]  # 受试者姓名
    tested_index = int(bdf_file.split('\\')[2][0]) - 1  # 第index次实验
    # 数据采集存在偏差，调整一下
    if tested_name != 'wangyihao' and tested_name != 'lichunqing':
        channel_data = channel_data[15000:]  # 去掉音频前奏
    else:
        channel_data = channel_data[5000:]  # 去掉第一个字
    channel_data = channel_data[:(channel_data.shape[0] - channel_data.shape[0] % 5000)]

    # 小波去噪
    if is_wavelet:
        wavelet = 'db1'  # 选择Daubechies小波基
        level = 3  # 小波分解层数
        coeffs = pywt.wavedec(channel_data, wavelet, level=level)  # 进行小波分解
        sigma = (1 / 0.6745) * np.median(np.abs(coeffs[-level]))  # 对小波系数进行阈值处理以去噪
        threshold = sigma * np.sqrt(2 * np.log(len(channel_data)))  # 对小波系数进行阈值处理以去噪
        coeffs_new = list(coeffs)  # 创建一个新的小波系数列表，其中的噪声系数被设置为0
        for i in range(1, len(coeffs)):
            coeffs_new[i] = pywt.threshold(coeffs[i], value=threshold, mode='hard')
        emg_denoised = pywt.waverec(coeffs_new, wavelet)  # 重构信号(ndarray)
    else:
        emg_denoised = channel_data

    # 预处理数据
    raw_per_50 = []
    for i in range(len(emg_denoised)//50):
        raw_per_50.append(emg_denoised[i*50:(i+1)*50])
    # 1. 使用平均值/最大值/差分值提取序列特征
    process_per_50 = []
    for item in raw_per_50:
        if mode == 0:
            process_per_50.append(np.mean(item))
        elif mode == 1:
            process_per_50.append(np.max(item))
        else:
            process_per_50.append(np.max(item) - np.min(item))

    # 划分序列并打标签
    X = []
    Y = []
    labels = dict_label[tested_name][tested_index]
    label_len = len(labels)
    per_len = len(process_per_50)
    temp = []
    cnt = 0
    for i, aver in enumerate(process_per_50):
        if not cnt == i//(per_len/label_len):
            x = np.array(temp)
            # 维度100维，多裁少补
            if len(x) < 100:
                x = np.pad(x, (100 - len(x)), 'constant', constant_values=0.0)
            elif len(x) > 100:
                x = x[:100]
            X.append(x)
            temp = []
            Y.append(labels[cnt])
            cnt += 1
        temp.append(aver)
    if len(temp) != 0:
        try:
            x = np.array(temp)
            # 维度100维，多裁少补
            if len(x) < 100:
                x = np.pad(x, (100 - len(x)), 'constant', constant_values=0.0)
            elif len(x) > 100:
                x = x[:100]
            y = labels[cnt]
            X.append(x)
            Y.append(y)
        except Exception as e:
            print(e)
            pass
    assert len(Y) == len(labels)
    return X, Y


if __name__ == '__main__':
    bdf_path_list = []
    name_list = os.listdir('./result')
    for name in name_list:
        bdf_path = os.path.join('./result', name)
        bdf_list = os.listdir(bdf_path)
        for bdf in bdf_list:
            bdf_path_list.append(os.path.join(bdf_path, bdf))

    label_dict = get_label_dict('./label.csv')

    modes = {0: 'average', 1: 'max', 2: 'range'}
    is_wavelets = {True: 'wavelet', False: 'not_wavelet'}
    channel_set = ['S01', 'S04', 'S08']
    Y = None
    for mode in modes.keys():
        for is_wavelet in is_wavelets.keys():
            X, Y = None, None
            for bdf in bdf_path_list:
                for channel in channel_set:
                    if X is None and Y is None:
                        X, Y = bdf2seq(bdf, channel, label_dict, mode=mode, is_wavelet=is_wavelet)
                    else:
                        x, y = bdf2seq(bdf, channel, label_dict, mode=mode, is_wavelet=is_wavelet)
                        X += x
                        Y += y
            print("Saved data for mode {} with {} wavelet. The count of data is {}.".format(modes[mode], is_wavelets[is_wavelet], len(X)))
            np.save("./processed/{}_{}.npy".format(modes[mode], is_wavelets[is_wavelet]), np.array(X))
    np.save("./processed/labels.npy", np.array(Y))
