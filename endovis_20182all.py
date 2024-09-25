import os
import json
import shutil
import torch

# 读取 JSON 文件
def read_json(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    return data


# 获取目标文件夹中最后一个文件编号
def get_last_file_number(target_folder):
    files = [f for f in os.listdir(target_folder) if f.endswith('.png')]

    if not files:
        return 0

    files.sort()
    last_file = files[-1]
    last_number = int(last_file.split('_')[0].split('.png')[0])  # 提取文件名中的数字部分
    return last_number + 1


# 处理图像和注释文件，并重命名注释文件
def process_images_and_annotations(json_file_path, target_folder_img,target_folder_annotations):
    # 如果目标文件夹不存在，则创建它
    if not os.path.exists(target_folder_img):
        os.makedirs(target_folder_img)
    if not os.path.exists(target_folder_annotations):
        os.makedirs(target_folder_annotations)
    # 读取 JSON 文件数据
    data = read_json(json_file_path)

    # 获取目标文件夹中最后一个文件编号
    counter = get_last_file_number(target_folder_img)
    print(counter)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")
    # 遍历 JSON 文件中的图像和对应的注释文件
    for image_path, annotation_paths in data.items():
        # 检查图像文件是否存在（你可以根据需要处理图像文件）
        if os.path.exists(image_path):
            print(f"Processing image: {image_path}")
            new_img_name = f"{counter:06d}.png"
            # 复制并重命名注释文件
            target_img_file_path = os.path.join(target_folder_img, new_img_name)
            shutil.copy2(image_path, target_img_file_path)
            print(f"Copied {image_path} to {target_img_file_path}")

            # 遍历该图像对应的所有注释文件
            for annotation_path in annotation_paths:
                if os.path.exists(annotation_path):
                    # 生成新的文件名，如 000000_class1.png, 000000_class4.png 等
                    class_number = annotation_path.split('_')[-1].split('.')[0]
                    new_annotation_name = f"{counter:06d}_{class_number}.png"

                    # 构造目标文件路径
                    target_annotation_file_path = os.path.join(target_folder_annotations, new_annotation_name)

                    # 复制并重命名注释文件
                    shutil.copy2(annotation_path, target_annotation_file_path)
                    print(f"Copied {annotation_path} to {target_annotation_file_path}")

            # 增加计数器，以确保下一个图像编号递增
            counter += 1


# 示例用法
json_file_path = "data/sisvse_new/image2label.json"  # 替换为你的 JSON 文件路径
# json_file_path = "data/robotool_new/image2label.json"  # 替换为你的 JSON 文件路径
# json_file_path = "data/endovis_2018_instrument/val/image2label_test.json"  # 替换为你的 JSON 文件路径
target_folder_annotation = '../MedSAM/data/SIS/annotations/'  # 替换为目标文件夹路径
target_folder_img = '../MedSAM/data/SIS/images/'  # 替换为目标文件夹路径

process_images_and_annotations(json_file_path, target_folder_img,target_folder_annotation)
