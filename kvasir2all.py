import os
import json
import shutil
import torch
from PIL import Image
def get_last_file_number(target_folder):
    files = [f for f in os.listdir(target_folder) if f.endswith('.png')]

    if not files:
        return 0

    files.sort()
    last_file = files[-1]
    last_number = int(last_file.split('_')[0].split('.png')[0])  # 提取文件名中的数字部分
    return last_number + 1


# 处理图像和注释文件，并重命名注释文件
def process_images_and_annotations(source_img_file_path,source_annotation_file_path, target_folder_img,target_folder_annotations):
    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(target_folder_img):
        os.makedirs(target_folder_img)
    if not os.path.exists(target_folder_annotations):
        os.makedirs(target_folder_annotations)

    # 获取目标文件夹中最后一个文件编号
    counter = get_last_file_number(target_folder_img)
    print(counter)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    # 遍历 JSON 文件中的图像和对应的注释文件
    for item in os.listdir(source_img_file_path):
        # 检查图像文件是否存在（你可以根据需要处理图像文件）
        if os.path.exists(source_img_file_path+item):
            print(f"Processing image: {source_img_file_path+item}")
            new_img_name = f"{counter:06d}.png"
            # 复制并重命名注释文件
            target_img_file_path = os.path.join(target_folder_img, new_img_name)
            shutil.copy2(source_img_file_path+item, target_img_file_path)
            print(f"Copied {source_img_file_path+item} to {target_img_file_path}")

        if os.path.exists(source_annotation_file_path):
            # 生成新的文件名，如 000000_class1.png, 000000_class4.png 等
            class_number = "class1"
            new_annotation_name = f"{counter:06d}_{class_number}.png"

            # 构造目标文件路径
            target_annotation_file_path = os.path.join(target_folder_annotations, new_annotation_name)

            # 复制并重命名注释文件
            shutil.copy2(source_annotation_file_path+item, target_annotation_file_path)
            print(f"Copied {source_annotation_file_path+item} to {target_annotation_file_path}")

            # 增加计数器，以确保下一个图像编号递增
            counter += 1


# 示例用法
source_annotation_file_path = "../MedSAM/data/kvasir-instrument/masks/"
source_img_file_path ="../MedSAM/data/kvasir-instrument/images/"
target_folder_annotation = '../MedSAM/data/SIS/annotations/'  # 替换为目标文件夹路径
target_folder_img = '../MedSAM/data/SIS/images/'  # 替换为目标文件夹路径
process_images_and_annotations(source_img_file_path,source_annotation_file_path, target_folder_img,target_folder_annotation)


# # JPG to PNG
# for file_name in os.listdir(source_img_file_path):
#     if file_name.lower().endswith('.jpg'):
#         # 构造完整的源文件路径
#         jpg_path = os.path.join(source_img_file_path, file_name)
#
#         # 生成新的文件名和路径
#         png_file_name = file_name.rsplit('.', 1)[0] + '.png'
#         png_path = os.path.join(source_img_file_path, png_file_name)
#
#         # 打开 JPG 文件并转换为 PNG
#         with Image.open(jpg_path) as img:
#             img.save(png_path, 'PNG')
#             print(f"Converted {jpg_path} to {png_path}")

# # DELETE JPG
# for file_name in os.listdir(source_img_file_path):
#     if file_name.lower().endswith('.jpg'):
#         jpg_path = os.path.join(source_img_file_path, file_name)
#         os.remove(jpg_path)
#         print(f"Deleted {jpg_path}")
