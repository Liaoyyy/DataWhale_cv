import torch
from torch import nn
import glob
import numpy as np
import torchvision.models as models

# 读取训练集文件路径
train_path = glob.glob('./脑PET图像分析和疾病预测挑战赛公开数据/Train/*/*')
test_path = glob.glob('./脑PET图像分析和疾病预测挑战赛公开数据/Test/*')


# 打乱训练集和测试集的顺序
np.random.shuffle(train_path)
np.random.shuffle(test_path)


class MyNet(nn.Module):
    def __init__(self, input_channels, num_channels, stides = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size= 3, padding= 1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

