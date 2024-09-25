# %% label to image json endovis-2017 train/val
import os
import json
img_path="./data/endovis_2017_instrument/images/"
mask_path="./data/endovis_2017_instrument/binary_annotations/"
mapping = {}
for root, dirs, files in os.walk(mask_path):
    print(root, dirs, files)
    if len(dirs) == 0:
        for file in files:
            img_name = file.split('_class')[0] + '.png'
            seq_num=root.split('/')[-1]
            mapping[root+'/'+file]=img_path+seq_num+'/'+img_name
print(mapping)
# 将字典转换为 JSON 格式并写入文件
with open("./data/endovis_2017_instrument/label2image.json", "w") as json_file:
    json.dump(mapping, json_file, indent=4)

# %%image to label json
import os
import json
img_path="./data/endovis_2017_instrument/images/"
mask_path="./data/endovis_2017_instrument/binary_annotations/"
image_masks = {}
for root, dirs, files in os.walk(img_path):
    if len(dirs) == 0:
        for file in files:
            img_name = file.split('.')[0]
            image_path = os.path.join(root, file)
            # List all mask files corresponding to the image
            seq_num = root.split('/')[-1]
            masks = [f for f in os.listdir(mask_path+seq_num+'/') if img_name in f]
            mask_paths = [os.path.join(mask_path+seq_num+'/', mask) for mask in masks]
            # Add to dictionary
            if mask_paths:
                image_masks[image_path] = mask_paths
print(image_masks)
# 将字典转换为 JSON 格式并写入文件
with open("./data/endovis_2017_instrument/image2label.json", "w") as json_file:
    json.dump(image_masks, json_file, indent=4)