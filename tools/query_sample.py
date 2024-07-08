import os
import shutil

# 设置源目录和目标目录的路径
source_directory = '/data1/fast-reid/datasets/ViFi/gallery/vision'
destination_directory = '/data1/fast-reid/datasets/ViFi/query/vision'

# 获取原始目录中所有文件夹的列表，并按名称排序
folders = sorted([f for f in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, f))])

# 计算要抽样的文件夹数量（六分之一）
sample_size = len(folders) // 6

# 计算抽样的间隔
sampling_interval = len(folders) // sample_size

# 遍历抽样的文件夹列表
for i in range(0, len(folders), sampling_interval):
    folder = folders[i]

    # 构造完整的文件夹路径
    folder_path = os.path.join(source_directory, folder)

    # 构造目标路径
    destination_path = os.path.join(destination_directory, folder)

    # 移动文件夹
    shutil.move(folder_path, destination_path)