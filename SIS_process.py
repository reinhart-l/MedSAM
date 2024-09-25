import os
import json
import numpy as np
from PIL import Image
import torch
from sklearn.model_selection import train_test_split
import shutil
#---------------------------标准化数据名称-------------------------
# mask_path="./data/SIS/annotations"
# for root, dirs, files in os.walk(mask_path):
#     if len(dirs) == 0:
#         for file in files:
#             if "class" not in file:
#                 old_file_name = file
#                 # 重命名文件
#                 new_file_name = old_file_name.split('_')[0] + "_class" + old_file_name.split('_')[1]
#                 os.rename(root+'/'+old_file_name, root+'/'+new_file_name)

#---------------------------------数据集划分----------------------------
# 读取 image2label.json 文件
# with open("./data/SIS/image2label.json", 'r') as f:
#     image_to_label = json.load(f)
#
# # 获取所有图像路径
# image_paths = list(image_to_label.keys())
#
# # 切分训练集和测试集，test_size 控制测试集比例
# train_images, test_images = train_test_split(image_paths, test_size=0.1, random_state=42)
#
# # 输出路径
# train_image_path = "./data/SIS/train/images/"
# test_image_path = "./data/SIS/test/images/"
# train_annotation_path = "./data/SIS/train/annotations/"
# test_annotation_path = "./data/SIS/test/annotations/"
#
# # 创建输出文件夹
# for path in [train_image_path, test_image_path, train_annotation_path, test_annotation_path]:
#     if not os.path.exists(path):
#         os.makedirs(path)
#
#
# # 移动图像和对应的标注文件到训练集或测试集
# def move_data(image_list, image_target_path, annotation_target_path):
#     for image in image_list:
#         # print(image)
#         # 移动图像文件
#         image_name = os.path.basename(image)  # 提取图像文件名
#         shutil.copy(image, os.path.join(image_target_path, image_name))
#
#         # 移动对应的标注文件
#         annotation_files = image_to_label[image]
#         for annotation in annotation_files:
#             annotation_name = os.path.basename(annotation)
#             shutil.copy(annotation, os.path.join(annotation_target_path, annotation_name))
#
#
# # 将训练集和测试集的图像和标注文件分别移动到对应文件夹
# move_data(train_images, train_image_path, train_annotation_path)
# move_data(test_images, test_image_path, test_annotation_path)
#
# print(f"训练集图像数量: {len(train_images)}, 测试集图像数量: {len(test_images)}")
# print("数据集切分完成，并移动到训练集和测试集文件夹。")

#------------------------------检查数据集------------------------------------
# 读取 image2label.json 文件
json_file_path = "./data/SIS/train/image2label_train.json"
with open(json_file_path, 'r') as f:
    image_to_label = json.load(f)

image_paths = list(image_to_label.keys())

# 记录需要删除的图像路径
images_to_remove = []

for image_path in image_paths:
    annotation_files = image_to_label[image_path]

    # 用于保存当前图像路径下有效的标注文件
    valid_annotations = []

    for annotation_file in annotation_files:
        # 检查文件是否为图像格式
        if annotation_file.endswith('.png'):
            try:
                # 使用 PIL 打开图像并转为 numpy 数组
                mask = np.array(Image.open(annotation_file))
            except Exception as e:
                print(f"Error loading image file {annotation_file}: {e}")
                continue
        else:
            print(f"Unsupported file format: {annotation_file}")
            continue

        # 统计值为 255 的像素数量
        num_255_pixels = np.sum(mask == 255)

        # 如果 mask 中 255 像素点大于 100，则保留该标注文件
        if num_255_pixels > 100:
            valid_annotations.append(annotation_file)
        else:
            print(f"Mask in {annotation_file} has too few 255 pixels for image: {image_path}")
            print(f"Number of 255 pixels: {num_255_pixels}")

    # 更新当前图像路径的标注文件列表
    if valid_annotations:
        image_to_label[image_path] = valid_annotations
    else:
        # 如果没有有效的标注文件，记录下来以便删除整个图像路径
        images_to_remove.append(image_path)

# 删除没有任何标注文件的图像路径
for image_path in images_to_remove:
    if image_path in image_to_label:
        del image_to_label[image_path]

# 将更新后的字典保存回 JSON 文件
with open(json_file_path, 'w') as f:
    json.dump(image_to_label, f, indent=4)

print(f"共删除 {len(images_to_remove)} 条图像路径，以及相应的无效 mask 文件。")
