import os
import json
import cv2
import numpy as np

category_path = "./data/sisvse_new/category.json"
img_path = "./data/sisvse_new/images/"
mask_path = "./data/sisvse_new/annotations/"
target_path = "./data/sisvse_new/binary_annotations/"

if not os.path.exists(target_path):
    os.makedirs(target_path)

with open(category_path, 'r') as f:
    categories = json.load(f)

# 创建一个 id 到 supercategory 的映射字典
id_to_supercategory = {category['id']: category['supercategory'] for category in categories}

# 为每个 supercategory 分配一个唯一的 ID
supercategory_to_id = {}
current_id = 1

# Supercategory 映射保存路径
supercategory_to_id_file_path = './data/sisvse_new/supercategory_to_id.json'
for category in categories:
    supercategory_name = category['supercategory']
    if supercategory_name not in supercategory_to_id:
        supercategory_to_id[supercategory_name] = current_id
        current_id += 1

# 将 supercategory 到 ID 的映射保存为 JSON 文件
with open(supercategory_to_id_file_path, 'w') as f:
    json.dump(supercategory_to_id, f, indent=4)
    print(f"Saved supercategory to ID mapping to {supercategory_to_id_file_path}")

for root, dirs, files in os.walk(mask_path):
    for file in files:
        if file.endswith('.png'):  # 假设 mask 文件是 PNG 格式
            mask_file_path = os.path.join(root, file)
            mask = cv2.imread(mask_file_path, cv2.IMREAD_GRAYSCALE)

            # 创建一个空的字典来存储每个 supercategory 的二值掩码
            supercategory_masks = {}

            # 遍历 mask 中的每个 object id
            unique_ids = np.unique(mask)
            for object_id in unique_ids:
                if object_id == 0:
                    continue  # 忽略背景（假设 0 是背景）
                if object_id in {24, 25, 26, 27, 28, 29, 31}:
                    continue  # 跳过特定 ID

                # 获取 supercategory 名称
                supercategory_name = id_to_supercategory.get(object_id, f'unknown_{object_id}')

                # 如果该 supercategory 还没有掩码，初始化一个全 0 的二值掩码
                if supercategory_name not in supercategory_masks:
                    supercategory_masks[supercategory_name] = np.zeros_like(mask, dtype=np.uint8)

                # 将属于当前 object id 的区域添加到对应的 supercategory 掩码中
                supercategory_masks[supercategory_name] = np.logical_or(
                    supercategory_masks[supercategory_name], (mask == object_id)
                ).astype(np.uint8) * 255

            # 遍历生成的 supercategory 掩码，并保存到文件
            for supercategory_name, binary_mask in supercategory_masks.items():
                # 查找 supercategory 的 ID
                supercategory_id = supercategory_to_id.get(supercategory_name, 0)

                # 构造输出文件路径
                output_file_name = f"{file.split('.png')[0]}_class_{supercategory_id}.png"
                output_file_path = os.path.join(target_path, output_file_name)

                # 保存二值掩码
                cv2.imwrite(output_file_path, binary_mask)
                print(f"Saved binary mask for supercategory {supercategory_name} (ID: {supercategory_id}) to {output_file_path}")
