import torch

# 加载两个 .pth 文件
checkpoint_1 = torch.load('/home3/yuchu/MedSAM/work_dir/models/MedSAM_adapter_25epo/epoch6_sam.pth')
checkpoint_2 = torch.load('/home3/yuchu/MedSAM/work_dir/models_endovis_2018/MedSAM_adapter_25epo/epoch20_sam.pth')

# 获取 state_dict
state_dict_1 = checkpoint_1['model']  # 假设模型保存在 'model' 键下
state_dict_2 = checkpoint_2['model']

# 比较两个 state_dict 的键
keys_1 = set(state_dict_1.keys())
keys_2 = set(state_dict_2.keys())

# 打印键的差异
missing_in_1 = keys_2 - keys_1
missing_in_2 = keys_1 - keys_2

print(keys_1)
print(keys_2)
print(checkpoint_1['optimizer'])
print(checkpoint_2['optimizer'])
