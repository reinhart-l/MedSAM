from sam_unet.models.build_sam_unet import sam_unet_registry
import torch.nn as nn
import torch
import argparse
import os
from torch import optim
from torch.utils.data import DataLoader
from DataLoader import TrainingDataset, stack_dict_batched
from utils import FocalDiceloss_IoULoss, get_logger, generate_point, setting_prompt_none
from metrics import SegMetrics
import time
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import numpy as np
import datetime
from torch.nn import functional as F
import random
import argparse
import os
import random
import datetime
import numpy as np
import torch
import torch.distributed as dist
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from utils import FocalDiceloss, get_logger, generate_point, setting_prompt_none
from metrics import SegMetrics
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="work_dir", help="work dir")

    parser.add_argument("--run_name", type=str, default="MedSAM_Unet_25epo", help="run model name")
    # parser.add_argument("--run_name", type=str, default="MedSAM_adapter", help="run model name")
    # parser.add_argument("--run_name", type=str, default="MedSAM_adapter", help="run model name")
    # parser.add_argument("--run_name", type=str, default="MedSAM", help="run model name")

    parser.add_argument("--start_epoch", type=int, default=0, help="start epoch when using resume")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="train batch size")
    parser.add_argument("--image_size", type=int, default=1024, help="image_size")
    parser.add_argument("--mask_num", type=int, default=5, help="get mask number")
    parser.add_argument("--data_path", type=str, default="./data/SIS/train", help="train data path")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--resume", type=str, default=None, help="load resume")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--checkpoint", type=str, default="work_dir/SAM/sam_vit_b_01ec64.pth", help="sam checkpoint")
    parser.add_argument("--iter_point", type=int, default=8, help="point iterations")
    parser.add_argument('--lr_scheduler', type=str, default=None, help='lr scheduler')
    parser.add_argument("--point_list", type=list, default=[1, 3, 5, 9], help="point_list")
    parser.add_argument("--multimask", type=bool, default=False, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=True, help="use adapter")
    # parser.add_argument('--gpu_ids', type=str, default='1,3', help='Comma-separated list of GPU ids to use.')
    # parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()
    if args.resume is not None:

        args.checkpoint = None
    return args

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_main_process():
    return dist.get_rank() == 0

def to_device(batch_input, device):
    device_input = {}
    for key, value in batch_input.items():
        if value is not None:
            if key == 'image' or key == 'label':
                device_input[key] = value.float().to(device)
            elif type(value) is list or type(value) is torch.Size:
                device_input[key] = value
            else:
                device_input[key] = value.to(device)
        else:
            device_input[key] = value
    return device_input


def prompt_and_decoder(args, batched_input, model, image_embeddings, multi_scale_feature,decoder_iter=False):
    if batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None
    # decoder_iter如果为true就冻结prompt_encoder
    if decoder_iter:
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = model.module.prompt_encoder(
                points=points,
                boxes=batched_input.get("boxes", None),
                masks=batched_input.get("mask_inputs", None),
            )

    else:
        sparse_embeddings, dense_embeddings = model.module.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )

    low_res_masks = model.module.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=model.module.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multi_scale_feature=multi_scale_feature,
    )

    # 双线性插值
    masks = F.interpolate(low_res_masks, (args.image_size, args.image_size), mode="bilinear", align_corners=False, )
    return masks, low_res_masks


def train_one_epoch(args, model, optimizer, train_loader, epoch, criterion):
    torch.cuda.empty_cache()
    train_loader = tqdm(train_loader) if is_main_process() else train_loader  # 仅主进程显示进度条
    train_losses = []
    train_iter_metrics = [0] * len(args.metrics)
    for batch, batched_input in enumerate(train_loader):
        batched_input = stack_dict_batched(batched_input)
        batched_input = to_device(batched_input, args.device)

        if random.random() > 0.5:  # 一半是box一半是point prompt
            batched_input["point_coords"] = None
            flag = "boxes"
        else:
            batched_input["boxes"] = None
            flag = "point"

        for n, value in model.module.image_encoder.named_parameters():
            if "Adapter" in n:
                value.requires_grad = True
            else:
                value.requires_grad = False  # 冻结除了adapt以外其他的参数
        # for n, value in model.named_parameters():
        #     print(f"Parameter: {n}, requires_grad: {value.requires_grad}")



        labels = batched_input["label"]
        input_images=batched_input["image"]
        feature_maps_in = model.module.resnet(input_images)  # a list of feature maps for encoder
        feature_maps_out = [None] * 3  # a list of feature maps for decoder
        input_images = model.module.image_encoder.forward_patch_embed(input_images)
        for i in range(len(model.module.global_index)):
            for j in range(2):
                input_images = model.module.image_encoder.forward_block(input_images, i * 3 + j)
            current_feature_map = model.module.adapters_in[i](feature_maps_in[i])
            current_feature_map = current_feature_map.permute(0, 2, 3, 1)
            input_images = input_images + current_feature_map
            input_images = model.module.image_encoder.forward_block(input_images, model.module.global_index[i])
            if i in range(len(model.module.global_index) - 1):  # 0, 1, 2
                permuted_input_images = input_images.permute(0, 3, 1, 2)
                current_out_feature_map = model.module.adapters_bridge[3 - i - 1](
                    permuted_input_images)  # 3 - i - 1 means 2, 1, 0, because the model is a U-Net
                feature_maps_out[3 - i - 1] = current_out_feature_map
        image_embeddings = model.module.image_encoder.forward_neck(input_images)

        model.module.multi_scale_feature = model.module.embedding_encoder(image_embeddings)
        model.module.multi_scale_feature += feature_maps_out[0]
        model.module.multi_scale_feature += feature_maps_out[1]
        model.module.multi_scale_feature += feature_maps_out[2]

        multi_scale_feature = model.module.multi_scale_feature
        masks,low_res_masks = prompt_and_decoder(args, batched_input, model, image_embeddings,multi_scale_feature, decoder_iter=False)

        focal_loss, dice_loss = criterion(masks, labels)
        loss = focal_loss + 20 * dice_loss
        loss.backward(retain_graph=False)

        optimizer.step()
        optimizer.zero_grad()

        if is_main_process() and (batch + 1) % 50 == 0:  # 仅主进程输出日志
            print(
                f'Epoch: {epoch + 1}, Batch: {batch + 1}, first {flag} prompt: {SegMetrics(masks, labels, args.metrics)}')

        point_num = random.choice(args.point_list)
        batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
        batched_input = to_device(batched_input, args.device)

        image_embeddings = image_embeddings.detach().clone()
        multi_scale_feature = multi_scale_feature.detach().clone()
        for n, value in model.named_parameters():
            if "prompt_encoder" or "mask_decoder" in n:
                value.requires_grad = True
            else:
                value.requires_grad = False
        # for n, value in model.named_parameters():
        #     print(f"Parameter: {n}, requires_grad: {value.requires_grad}")
        init_mask_num = np.random.randint(1, args.iter_point - 1)
        for iter in range(args.iter_point):
            if iter == init_mask_num or iter == args.iter_point - 1:
                batched_input = setting_prompt_none(batched_input)

            masks, low_res_masks = prompt_and_decoder(args, batched_input, model, image_embeddings,multi_scale_feature,decoder_iter=True)


            focal_loss, dice_loss = criterion(masks, labels)
            loss = focal_loss + 20 * dice_loss
            loss.backward(retain_graph=True)

            optimizer.step()
            optimizer.zero_grad()

            if iter != args.iter_point - 1:
                point_num = random.choice(args.point_list)

                batched_input = generate_point(masks, labels, low_res_masks, batched_input, point_num)
                batched_input = to_device(batched_input, args.device)

            if is_main_process() and (batch + 1) % 50 == 0:  # 仅主进程输出日志
                if iter == init_mask_num or iter == args.iter_point - 1:
                    print(
                        f'Epoch: {epoch + 1}, Batch: {batch + 1}, mask prompt: {SegMetrics(masks, labels, args.metrics)}')
                else:
                    print(
                        f'Epoch: {epoch + 1}, Batch: {batch + 1}, point {point_num} prompt: {SegMetrics(masks, labels, args.metrics)}')

        if is_main_process() and (batch + 1) % 200 == 0:  # 仅主进程保存模型
            print(f"epoch:{epoch + 1}, iteration:{batch + 1}, loss:{loss.item()}")
            save_path = os.path.join(f"{args.work_dir}/models", args.run_name,
                                     f"epoch{epoch + 1}_batch{batch + 1}_sam.pth")
            state = {'model': model.state_dict(), 'optimizer': optimizer}
            torch.save(state, save_path)

        train_losses.append(loss.item())

        gpu_info = {}
        gpu_info['gpu_name'] = args.device
        if is_main_process():  # 仅主进程显示进度条信息
            train_loader.set_postfix(train_loss=loss.item(), gpu_info=gpu_info)

        train_batch_metrics = SegMetrics(masks, labels, args.metrics)
        train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]

    return train_losses, train_iter_metrics


def main(args):
    set_random_seed(args.seed)
    dist.init_process_group(backend='nccl', init_method='env://')

    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    args.device = torch.device('cuda', local_rank)

    global_rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Build model
    model = sam_unet_registry['res34_sam_unet'](need_ori_checkpoint=True, sam_unet_checkpoint=None)
    model.to(args.device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = FocalDiceloss()

    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10], gamma=0.5)
        print('*******Use MultiStepLR')

    if args.resume is not None:
        with open(args.resume, "rb") as f:
            checkpoint = torch.load(f)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'].state_dict())
            print(f"*******load {args.resume}")

            # 使用resume时，从start_epoch开始
        start_epoch = args.start_epoch
    else:
        start_epoch = 0


    print('*******Do not use mixed precision')

        # Initialize dataset and dataloader with DistributedSampler
    train_dataset = TrainingDataset(args.data_path, image_size=args.image_size, mode='train', point_num=1,
                                    mask_num=args.mask_num, requires_name=False)
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=4,
                              pin_memory=True)
    print('*******Train data:', len(train_dataset))

    if is_main_process():
        loggers = get_logger(
            os.path.join(args.work_dir, "logs",
                         f"{args.run_name}_{datetime.datetime.now().strftime('%Y%m%d-%H%M.log')}"))

    best_loss = 1e10
    l = len(train_loader)

    for epoch in range(start_epoch, start_epoch+args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        if is_main_process():
            start = time.time()
        os.makedirs(os.path.join(f"{args.work_dir}/models", args.run_name), exist_ok=True)
        train_losses, train_iter_metrics = train_one_epoch(args, model, optimizer, train_loader, epoch, criterion)

        if args.lr_scheduler is not None:
            scheduler.step()

        if is_main_process():
            train_iter_metrics = [metric / l for metric in train_iter_metrics]
            train_metrics = {args.metrics[i]: '{:.4f}'.format(train_iter_metrics[i]) for i in
                             range(len(train_iter_metrics))}

            average_loss = np.mean(train_losses)
            lr = scheduler.get_last_lr()[0] if args.lr_scheduler is not None else args.lr
            loggers.info(f"epoch: {epoch + 1}, lr: {lr}, Train loss: {average_loss:.4f}, metrics: {train_metrics}")

            if average_loss < best_loss:
                best_loss = average_loss
                save_path = os.path.join(args.work_dir, "models", args.run_name, f"epoch{epoch + 1}_sam.pth")
                state = {'model': model.float().state_dict(), 'optimizer': optimizer}
                torch.save(state, save_path)

            end = time.time()
            print("Run epoch time: %.2fs" % (end - start))


if __name__ == '__main__':
    args = parse_args()
    main(args)
