# -*- coding:utf-8 -*-
import random
import time
from pathlib import Path

import pandas as pd
import paramiko
import pyautogui
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio import transforms
import torch.nn.functional as F
from torch.nn import init
from torch import nn

#  变量
local_path = 'G:/mp3/test'
remote_path = '/home/mhn/test'
model_path = r"G:\mp3\checkpoint128.pth.tar"

#  连接Linux服务器
transport = paramiko.Transport(('20.205.39.2', 22))
transport.connect(username='mhn', password='Abc2846488610')
#  循环判断文件是否存在
while True:
    sftp = paramiko.SFTPClient.from_transport(transport)
    try:
        for i in range(10):
            sftp.stat(remote_path + str(i) + '.wav')
            sftp.get(remote_path + str(i) + '.wav', local_path + str(i) + '.wav')  # 下载文件
            print('文件存在')
        break
    except IOError:
        print('文件不存在')
        continue
transport.close()

# 修复音频文件
for i in range(10):
    b = None
    while b is None:
        b = pyautogui.locateOnScreen('02.png', confidence=0.7)
        time.sleep(1)
    # 获取b的中心
    center = pyautogui.center(b)
    # 点击中心
    pyautogui.click(center)
    time.sleep(1)
    c = None
    while c is None:
        c = pyautogui.locateOnScreen('03.png', confidence=0.7)
        time.sleep(1)
    # 键盘输入
    pyautogui.typewrite('test' + str(i) + '.wav')
    time.sleep(1)
    # 获取c的中心
    center = pyautogui.center(c)
    # 点击中心
    pyautogui.click(center)
    time.sleep(1)
    d = None
    while d is None:
        d = pyautogui.locateOnScreen('04.png', confidence=0.7)
        time.sleep(1)
    # 获取d的中心
    center = pyautogui.center(d)
    # 点击中心
    pyautogui.click(center)
    time.sleep(1)
    e = None
    while e is None:
        e = pyautogui.locateOnScreen('05.png', confidence=0.7)
        time.sleep(1)
    # 获取e的中心
    center = pyautogui.center(e)
    # 点击中心
    pyautogui.click(center)
    time.sleep(1)
    f = None
    while f is None:
        f = pyautogui.locateOnScreen('06.png', confidence=0.5)
        time.sleep(1)
    # 获取f的中心
    center = pyautogui.center(f)
    # 点击中心
    pyautogui.click(center)
    time.sleep(1)
g = None
while g is None:
    g = pyautogui.locateOnScreen('07.png', confidence=0.7)
    time.sleep(1)
# 获取g的中心
center = pyautogui.center(g)
# 点击中心
pyautogui.click(center)
time.sleep(1)

#  构造文件路径
file_list = pd.DataFrame({'relative_path': [], 'classID': []})
for f in Path('G:/mp3/').iterdir():
    if f.is_file() & (f.suffix == '.wav'):
        file_list.loc[len(file_list.index)] = [str(f)[7:], 0]
print(file_list)


class AudioUtil():
    # ----------------------------
    # 加载一个音频文件 信号(tensor sample rate)
    @staticmethod
    def open(audio_file):
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)
        # ----------------------------

    # 转换为所需的频道数 channels
    # ----------------------------
    @staticmethod
    def rechannel(aud, new_channel):
        sig, sr = aud

        if (sig.shape[0] == new_channel):
            # 什么都不做
            return aud

        if (new_channel == 1):
            # 转换为单声道（选择第一个channel）
            resig = sig[:1, :]
        else:
            # 转换为多声道（复制第一个channel）
            resig = torch.cat([sig, sig])

        return ((resig, sr))
        # ----------------------------

    # resample一次对一个声道重新采样
    # ----------------------------
    @staticmethod
    def resample(aud, newsr):
        sig, sr = aud

        if (sr == newsr):
            # 什么都不做
            return aud

        num_channels = sig.shape[0]
        # 重新采样第一个声道
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1, :])
        if (num_channels > 1):
            # 重新采样第二个通道，并合并两个通道
            retwo = torchaudio.transforms.Resample(sr, newsr)(sig[1:, :])
            resig = torch.cat([resig, retwo])

        return ((resig, newsr))
        # ----------------------------

    # 把信号截断到固定长度max_ms，以毫秒为单位
    # ----------------------------
    @staticmethod
    def pad_trunc(aud, max_ms):
        sig, sr = aud
        num_rows, sig_len = sig.shape
        max_len = sr // 1000 * max_ms

        if (sig_len > max_len):
            # 把信号截断到指定长度
            sig = sig[:, :max_len]

        elif (sig_len < max_len):
            # 信号结束和开始时要填充的长度
            pad_begin_len = random.randint(0, max_len - sig_len)
            pad_end_len = max_len - sig_len - pad_begin_len

            # 从0s开始
            pad_begin = torch.zeros((num_rows, pad_begin_len))
            pad_end = torch.zeros((num_rows, pad_end_len))

            sig = torch.cat((pad_begin, sig, pad_end), 1)

        return (sig, sr)

    # ----------------------------
    # 信号向左或者向右移动一定的百分比
    # 循环
    # ----------------------------
    @staticmethod
    def time_shift(aud, shift_limit):
        sig, sr = aud
        _, sig_len = sig.shape
        shift_amt = int(random.random() * shift_limit * sig_len)
        return (sig.roll(shift_amt), sr)
        # ----------------------------

    # 生成Spectrogram
    # ----------------------------
    @staticmethod
    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_db = 80

        # spec[channel, n_mels, time],单声道，立体声
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # 转换为分贝
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        return (spec)
        # ----------------------------

    # 掩盖frequency time 两个维度的某些部分---数据增强
    # 防止过拟合
    # 模型更好的泛化
    # 替换为平均值
    # ----------------------------
    @staticmethod
    def spectro_augment(spec, max_mask_pct=0.1, n_freq_masks=1, n_time_masks=1):
        _, n_mels, n_steps = spec.shape
        mask_value = spec.mean()
        aug_spec = spec

        freq_mask_param = max_mask_pct * n_mels
        for _ in range(n_freq_masks):
            aug_spec = transforms.FrequencyMasking(freq_mask_param)(aug_spec, mask_value)

        time_mask_param = max_mask_pct * n_steps
        for _ in range(n_time_masks):
            aug_spec = transforms.TimeMasking(time_mask_param)(aug_spec, mask_value)

        return aug_spec


class SoundDS(Dataset):
    def __init__(self, df, data_path):
        self.df = df
        self.data_path = str(data_path)
        self.duration = 4000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4

    # ----------------------------
    # 项数
    # ----------------------------
    def __len__(self):
        return len(self.df)

    # ----------------------------
    # 从dataset中得到第i项
    # ----------------------------
    def __getitem__(self, idx):
        # 绝对路径文件
        # 连接音频目录与相对路径
        audio_file = self.data_path + self.df.loc[idx, 'relative_path']
        # 得到Class ID 0-9 共10个
        class_id = self.df.loc[idx, 'classID']

        aud = AudioUtil.open(audio_file)
        # sample rate  ,channel 不同  更高 更少
        # 让sample rate 和 channel一样
        # 持续时间相同，采样率不相同 pad_trunc仍然会产生不同长度的数组
        reaud = AudioUtil.resample(aud, self.sr)
        rechan = AudioUtil.rechannel(reaud, self.channel)

        dur_aud = AudioUtil.pad_trunc(rechan, self.duration)
        shift_aud = AudioUtil.time_shift(dur_aud, self.shift_pct)
        sgram = AudioUtil.spectro_gram(shift_aud, n_mels=64, n_fft=1024, hop_len=None)
        aug_sgram = AudioUtil.spectro_augment(sgram, max_mask_pct=0.1, n_freq_masks=2, n_time_masks=2)

        return aug_sgram, class_id


class AudioClassifier(nn.Module):
    # ----------------------------
    # 模型
    # ----------------------------
    def __init__(self):
        super().__init__()
        conv_layers = []

        # 第一个卷积块： Relu ，Batch Norm. Kaiming Initialization
        self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        init.kaiming_normal_(self.conv1.weight, a=0.1)
        self.conv1.bias.data.zero_()
        conv_layers += [self.conv1, self.relu1, self.bn1]

        # 第二个卷积块
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(16)
        init.kaiming_normal_(self.conv2.weight, a=0.1)
        self.conv2.bias.data.zero_()
        conv_layers += [self.conv2, self.relu2, self.bn2]

        # 第三个卷积块
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(32)
        init.kaiming_normal_(self.conv3.weight, a=0.1)
        self.conv3.bias.data.zero_()
        conv_layers += [self.conv3, self.relu3, self.bn3]

        # 第四个卷积块
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm2d(64)
        init.kaiming_normal_(self.conv4.weight, a=0.1)
        self.conv4.bias.data.zero_()
        conv_layers += [self.conv4, self.relu4, self.bn4]

        # 线性分类器
        self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        self.lin = nn.Linear(in_features=64, out_features=10)

        # Wrap卷积块
        self.conv = nn.Sequential(*conv_layers)

    # ----------------------------
    # 向前传播
    # ----------------------------
    def forward(self, x):
        # 运行卷积块
        x = self.conv(x)

        # Adaptive pool 层和 flatten
        x = self.ap(x)
        x = x.view(x.shape[0], -1)

        # 线性分类器
        x = self.lin(x)

        # 最终结果
        return x


# 创建模型
myModel = AudioClassifier()
print(myModel)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel.to(device)

data_path = 'G:/mp3/'
myds = SoundDS(file_list, data_path)
# 加载模型
checkpoint = torch.load(model_path)
myModel.load_state_dict(checkpoint['model_state_dict'])  # 加载模型的参数
# 预测
with torch.no_grad():
    for data in myds:
        # 得到features 和 target labels
        inputs, labels = data[0], data[1]

        # 输出
        inputs_m, inputs_s = inputs.mean(), inputs.std()
        inputs = (inputs - inputs_m) / inputs_s

        # 预测值0-9类
        outputs = myModel(inputs)

        # 预测值 选最高的
        _, prediction = torch.max(outputs, 1)
        print(prediction)
