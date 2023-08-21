
import glob                # 获取文件路径
import numpy as np
import pandas as pd
import nibabel as nib      # 处理医学图像数据
from nibabel.viewers import OrthoSlicer3D    # 图像可视化
from collections import Counter              # 计数统计
import zipfile
import os


'''
zipFile = zipfile.ZipFile('./脑PET图像分析和疾病预测挑战赛数据集.zip')
zipFile.extractall()
zipFile.close()

# 重命名
name = "─╘PET═╝╧±╖╓╬÷║═╝▓▓í╘ñ▓Γ╠⌠╒╜╚ⁿ╣½┐¬╩²╛▌";
new_name = name.encode('cp437').decode('gbk')
os.rename(name, new_name)
'''

# 读取训练集文件路径
train_path = glob.glob('E:/cvprogram/Data/脑PET图像分析和疾病预测挑战赛公开数据/Train/*/*')
test_path = glob.glob('E:/cvprogram/Data/脑PET图像分析和疾病预测挑战赛公开数据/Test/*')

# 打乱训练集和测试集的顺序
np.random.shuffle(train_path)
np.random.shuffle(test_path)


def extract_feature(path):
    # 加载PET图像数据
    img = nib.load(path)
    # 获取第一个通道的数据
    img = img.dataobj[:, :, :, 0]

    # 随机筛选其中的10个通道提取特征
    random_img = img[:, :, :]

    # 对图片计算统计值
    feat = [
        (random_img != 0).sum(),  # 非零像素的数量
        (random_img == 0).sum(),  # 零像素的数量
        random_img.mean(),  # 平均值
        random_img.std(),  # 标准差
        len(np.where(random_img.mean(0))[0]),  # 在列方向上平均值不为零的数量
        len(np.where(random_img.mean(1))[0]),  # 在行方向上平均值不为零的数量
        random_img.mean(0).max(),  # 列方向上的最大平均值
        random_img.mean(1).max()  # 行方向上的最大平均值
    ]

    # 根据路径判断样本类别（'NC'表示正常，'MCI'表示异常）
    if 'NC' in path:
        return feat + ['NC']
    else:
        return feat + ['MCI']


#对训练集进行30次特征提取，每次提取后的特征以及类别（'NC'表示正常，'MCI'表示异常）被添加到train_feat列表中。
train_data = []
for path in train_path:
    train_data.append(extract_feature(path))
train_feature = np.array(train_data)[:, :-1].astype(np.float32)
train_label = np.array(train_data)[:, -1]

# 对测试集进行30次特征提取
test_data = []
for path in test_path:
    test_data.append(extract_feature(path))
test_feature = np.array(test_data)[:, :-1]
# 使用训练集的特征作为输入，训练集的类别作为输出，对逻辑回归模型进行训练。
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


train_predict = []
m = LogisticRegression(max_iter=500, C = 1)
m.fit(train_feature, train_label)
train_predict.append(accuracy_score(train_label, m.predict(train_feature)))
test_pred = m.predict(test_feature)
print(accuracy_score(train_label, m.predict(train_feature)))

ID = []
for x in os.listdir('./脑PET图像分析和疾病预测挑战赛公开数据/Test/'):
    ID.append(int(x[:-4]))

submit = pd.DataFrame(
    {
        'uuid': ID,  # 提取测试集文件名中的ID
        'label': test_pred  # 预测的类别
    }
)

# 按照ID对结果排序并保存为CSV文件
submit = submit.sort_values(by='uuid')
submit.to_csv('submit_logistic.csv', index=None)

'''plt.figure()
plt.plot(np.linspace(0.01, 2, 10), train_predict)
plt.show()'''