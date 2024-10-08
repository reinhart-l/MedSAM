sa_med16m_folders = [
    'pet_1', 'x_1', 'fundus_1', 'ultrasound_1',
    'ct_1', 'ct_2', 'ct_3', 'ct_4', 'ct_5', 'ct_6', 'ct_7', 'ct_8', 'ct_9', 'ct_10', 'ct_11',
    'dermoscopy_1', 'dermoscopy_2', 'dermoscopy_3', 'mr_1', 'mr_2', 'endoscopy_1',
]
sa_med16m_folders = ['path/to/train_test/' + x for x in sa_med16m_folders]

config_dict = {

    'data_directory': './SAM_Unet/data/endovis_2018_instrument',
    'root_dir': '',
    'random_seed': 000000,
    'checkpoint_path': './work_dir/SAM/sam_vit_b_01ec64.pth',
    'img_size': 1024,
    'pixel_mean': [123.675, 116.28, 103.53],
    'pixel_std': [58.395, 57.12, 57.375],
    'work_dir': "workdir",
}
