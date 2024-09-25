import os
import json
from PIL import Image
import shutil
img_path="./data/miccai2022_sisvse_dataset/images/"
mask_path="./data/miccai2022_sisvse_dataset/annotations/semantic_masks/"
target_img_path="./data/sisvse_new/images/"
target_mask_path="./data/sisvse_new/annotations/"
if not os.path.exists(target_img_path):
    os.makedirs(target_img_path)
if not os.path.exists(target_mask_path):
    os.makedirs(target_mask_path)

def get_last_file_number(target_folder):
    files = [f for f in os.listdir(target_folder) if f.endswith('.png')]

    if not files:
        return 0

    files.sort()
    last_file = files[-1]
    last_number = int(last_file.split('_')[0].split('.png')[0])  # 提取文件名中的数字部分
    return last_number + 1

cnt=get_last_file_number(target_img_path)
print(cnt)
for root, dirs, files in os.walk(img_path):
    if len(dirs) == 0:
        for file in files:
            if file.endswith(".jpg"):
                png_file_name = f"{cnt:06d}.png"
                png_path = os.path.join(target_img_path, png_file_name)

                # 打开 JPG 文件并转换为 PNG
                with Image.open(root+'/'+file) as img:
                    img.save(png_path, 'PNG')
                    print(f"Converted {root+file} to {png_path}")
                shutil.copy2(mask_path +root.split('images')[-1]+'/' + file.split('.jpg')[0]+'.png', target_mask_path + png_file_name)
                print(f"Copied {mask_path +root.split('images')[-1]+'/' + file.split('.jpg')[0]+'.png'} to {target_mask_path + png_file_name}")
                cnt+=1

