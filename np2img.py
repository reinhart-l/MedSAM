import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="blue", facecolor=(0, 0, 0, 0), lw=2)
    )
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([251 / 255, 252 / 255, 30 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


with np.load('./test_demo/imgs/2DBox_Mammography_demo.npz') as data:
    arrays = data.files
    print("Keys:", arrays)

    # 假设 array1 是图像数据，array2 是边界框数据
    image_data = data[arrays[0]]
    boxes = data[arrays[1]]

    _, axs = plt.subplots(1, 2, figsize=(25, 25))

    # 显示图像
    axs[0].imshow(image_data)  # 确保 image_data 是正确的图像数据
    axs[0].axis("off")

    # 显示边界框
    show_box(boxes[1], axs[0])  # 假设 boxes[0] 包含第一个边界框的坐标

    # 如果 array2 是一个二维口罩图像
    mask = data[arrays[2]] if len(arrays) > 2 else None
    if mask is not None:
        show_mask(mask, axs[1])
        axs[1].axis("off")
    plt.show()
# 加载 .npz 文件
# data = np.load('')
# data2 = np.load('./test_demo/gts/2DBox_Mammography_demo.npz')
# data_keys = data.files
# print(data_keys)
# # 从加载的数据中提取图像数组
# image_array = data['segs']
# image_array2 = data2['gts']
# # 将图像数组保存为 .jpg 文件
# plt.imshow(image_array)  # 如果是灰度图像，使用 cmap='gray' 参数
# plt.imshow(image_array2)
# plt.axis('off')  # 关闭坐标轴
