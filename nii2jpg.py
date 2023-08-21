import os
import nibabel as nib
import matplotlib.pyplot as plt


def process_img(input_folder, output_folder):
    # 确保输出文件夹存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的每个文件
    for filename in os.listdir(input_folder):
        # 检查文件扩展名是否为NIfTI
        if filename.endswith('.nii') or filename.endswith('.nii.gz'):
            # 构建输入和输出文件路径
            input_file = os.path.join(input_folder, filename)
            output_file_prefix = os.path.join(output_folder, filename.replace('.nii.gz', '').replace('.nii', ''))

            # 加载NIfTI文件
            nii_image = nib.load(input_file)

            # 提取图像数据
            image_data = nii_image.get_fdata()

            # 获取图像数据的形状和维度
            image_shape = image_data.shape
            num_slices = image_shape[-2]  # 倒数第二个维度表示切片数

            # 迭代处理每个切片和时间点
            t = 0  # 只有一个时间点
            for s in range(num_slices):
                # 提取当前切片和时间点的数据
                slice_data = image_data[..., s, t]  # 使用[..., s, t]选择特定切片和时间点
                # 生成保存的文件名
                file_name = f"_{str(s + 1)}.jpg"
                output_file = output_file_prefix + file_name
                # 保存为PNG文件
                plt.imsave(output_file, slice_data, cmap='gray')



if __name__ == '__main__':
    # 定义输入文件夹路径和输出文件夹路径
    input_folder_1 = 'E:/cvprogram/Data/脑PET图像分析和疾病预测挑战赛公开数据/Train/MCI'
    input_folder_2 = 'E:/cvprogram/Data/脑PET图像分析和疾病预测挑战赛公开数据/Train/NC'
    input_folder_3 = 'E:/cvprogram/Data/脑PET图像分析和疾病预测挑战赛公开数据/Test'

    output_folder_1 = 'E:/cvprogram/Data/ProcessedData/Train/MCI'
    output_folder_2 = 'E:/cvprogram/Data/ProcessedData/Train/NC'
    output_folder_3 = 'E:/cvprogram/Data/ProcessedData/Test'

    process_img(input_folder_1, output_folder_1)
    process_img(input_folder_2, output_folder_2)
    process_img(input_folder_3, output_folder_3)