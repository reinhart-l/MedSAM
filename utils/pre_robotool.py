# %% import packages
import numpy as np
import os
join = os.path.join
from skimage import io, transform, measure, morphology
from tqdm import tqdm
import shutil
import torch
from collections import deque
import cv2
from scipy import ndimage
gt_name_suffix = '.png'
img_name_suffix = '.png'
source_path = '../data/robotool/' # path to the images
output_img_path = '../data/robotool_new/images/'  # 输出路径
output_annotation_path = '../data/robotool_new/binary_annotations/'  # 输出路径
# names = sorted(os.listdir(source_path))
# if not os.path.exists(output_img_path):
#     os.makedirs(output_img_path)
# if not os.path.exists(output_annotation_path):
#     os.makedirs(output_annotation_path)
#
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# print(f"Using {device}")
# # 设定最小块大小阈值
# min_size = 100  # 你可以根据需要调整这个阈值
# for name in tqdm(names):
#
#     if name.endswith(gt_name_suffix):
#         if "seg" in name:
#             gt_name = name[:-len(gt_name_suffix)].split("_seg")[0]
#             print(name)
#             gt_data = np.uint8(io.imread(join(source_path, name)))
#             st = [[False] * gt_data.shape[1] for _ in range(gt_data.shape[0])]  # 状态数组
#             q = deque()  # 队列
#             cnt=0
#             # 确保图像是二值图（0和255）
#             _, binary = cv2.threshold(gt_data, 127, 255, cv2.THRESH_BINARY)
#
#             # 使用 SciPy 的 label 函数来标记不同的连通块
#             labeled_array, num_features = ndimage.label(binary == 255)
#             for num in range(1, num_features + 1):
#                 # 创建一个掩码图像，只有当前连通块是 255，其他部分为 0
#                 component = np.where(labeled_array == num, 255, 0).astype(np.uint8)
#                 # 计算连通块的大小（即像素点的数量）
#                 component_size = np.sum(component == 255)
#                 # 保存单独的连通块
#                 if component_size >= min_size:
#                     cv2.imwrite(output_annotation_path + gt_name+f'_class{num}.png', component)
#                     print(f'Saved: component_{num}.png')
#         else:
#             image_data = np.uint8(io.imread(join(source_path, name)))
#             shutil.copy2(source_path +name, output_img_path)



# 定义删除文件的函数
def delete_files_from_list(txt_file_path):
    # 读取txt文件内容
    with open(txt_file_path, 'r') as file:
        file_names = file.readlines()

    # 遍历每一行，检查文件是否存在，并删除
    for file_name in file_names:
        # 去除文件名的前后空格或换行符
        file_name = file_name.strip()
        if file_name.endswith(gt_name_suffix):
            name = file_name.split('.png')[0]
            for i in os.listdir(output_img_path):
                if name in i:
                    try:
                        os.remove(output_img_path+file_name)
                        print(f"Deleted: {output_img_path+file_name}")
                    except Exception as e:
                        print(f"Error deleting {output_img_path+file_name}: {e}")

            for i in os.listdir(output_annotation_path):
                if name in i:
                    try:
                        # os.remove(output_annotation_path+i)
                        print(f"Deleted: {output_annotation_path+i}")
                    except Exception as e:
                        print(f"Error deleting {output_annotation_path+i}: {e}")



# 调用函数，传入txt文件路径
delete_files_from_list("/home3/yuchu/MedSAM/data/robotool_new/bad.txt")









