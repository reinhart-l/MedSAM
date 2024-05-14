import monai

from segment_anything import sam_model_registry
import torch.nn as nn
from skimage import io, transform, measure, morphology
import torch
import argparse
import os
from utils import FocalDiceloss_IoULoss, generate_point, save_masks,save_imgs_prompts_masks
from torch.utils.data import DataLoader
from DataLoader import TestingDataset
from metrics import SegMetrics
import time
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageColor
from torch.nn import functional as F
import logging
import datetime
import cv2
import random
import csv
import json
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, default="work_dir", help="work dir")
    parser.add_argument("--run_name", type=str, default="SAM_train_endovis_2018", help="run model name")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument("--image_size", type=int, default=1024, help="image_size")
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument("--data_path", type=str, default="data/endovis_2018_instrument/val", help="train data path")
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    parser.add_argument("--model_type", type=str, default="vit_b", help="sam model_type")
    parser.add_argument("--checkpoint", type=str, default="./work_dir/models/MedSAM/epoch15_sam.pth", help="sam checkpoint")
    parser.add_argument("--boxes_prompt", type=bool, default=True, help="use boxes prompt")
    parser.add_argument("--point_num", type=int, default=3, help="point num")
    parser.add_argument("--iter_point", type=int, default=1, help="iter num")
    parser.add_argument("--multimask", type=bool, default=False, help="ouput multimask")
    parser.add_argument("--encoder_adapter", type=bool, default=False, help="use adapter")
    parser.add_argument("--prompt_path", type=str, default=None, help="fix prompt path")
    parser.add_argument("--save_pred", type=bool, default=True, help="save result")
    args = parser.parse_args()
    if args.iter_point > 1:
        args.point_num = 1
    return args


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


def postprocess_masks(low_res_masks, image_size, original_size):
    ori_h, ori_w = original_size
    masks = F.interpolate(
        low_res_masks,
        (image_size, image_size),
        mode="bilinear",
        align_corners=False,
    )

    if ori_h < image_size and ori_w < image_size:
        top = torch.div((image_size - ori_h), 2, rounding_mode='trunc')  # (image_size - ori_h) // 2
        left = torch.div((image_size - ori_w), 2, rounding_mode='trunc')  # (image_size - ori_w) // 2
        masks = masks[..., top: ori_h + top, left: ori_w + left]
        pad = (top, left)
    else:
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        pad = None
    return masks, pad


def prompt_and_decoder(args, batched_input, ddp_model, image_embeddings):
    if batched_input["point_coords"] is not None:
        points = (batched_input["point_coords"], batched_input["point_labels"])
    else:
        points = None

    with torch.no_grad():
        sparse_embeddings, dense_embeddings = ddp_model.prompt_encoder(
            points=points,
            boxes=batched_input.get("boxes", None),
            masks=batched_input.get("mask_inputs", None),
        )

        low_res_masks, iou_predictions = ddp_model.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=ddp_model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=args.multimask,
        )

    if args.multimask:
        max_values, max_indexs = torch.max(iou_predictions, dim=1)
        max_values = max_values.unsqueeze(1)
        iou_predictions = max_values
        low_res = []
        for i, idx in enumerate(max_indexs):
            low_res.append(low_res_masks[i:i + 1, idx])
        low_res_masks = torch.stack(low_res, 0)
    masks = F.interpolate(low_res_masks, (args.image_size, args.image_size), mode="bilinear", align_corners=False, )
    return masks, low_res_masks, iou_predictions


def is_not_saved(save_path, mask_name):
    masks_path = os.path.join(save_path, f"{mask_name}")
    if os.path.exists(masks_path):
        return False
    else:
        return True


def main(args):
    info = []
    print('*' * 100)
    for key, value in vars(args).items():
        info.append(key + ': ' + str(value))
        print(key + ': ' + str(value))
    print('*' * 100)

    # model = sam_model_registry[args.model_type](args).to(args.device)
    model = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(args.device)
    criterion = FocalDiceloss_IoULoss()
    test_dataset = TestingDataset(data_path=args.data_path, image_size=args.image_size, mode='test', requires_name=True,
                                  point_num=args.point_num, return_ori_mask=True, prompt_path=args.prompt_path)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)
    print('Test data:', len(test_loader))

    test_pbar = tqdm(test_loader)
    l = len(test_loader)

    model.eval()
    test_loss = []
    MedSAM_test_loss = []
    test_iter_metrics = [0] * len(args.metrics)
    test_metrics = {}
    prompt_dict = {}
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    for i, batched_input in enumerate(test_pbar):
        batched_input = to_device(batched_input, args.device)
        ori_labels = batched_input["ori_label"]
        original_size = batched_input["original_size"]
        labels = batched_input["label"]
        img_name = batched_input['name'][0]
        if args.prompt_path is None:
            prompt_dict[img_name] = {
                "boxes": batched_input["boxes"].squeeze(1).cpu().numpy().tolist(),
                "point_coords": batched_input["point_coords"].squeeze(1).cpu().numpy().tolist(),
                "point_labels": batched_input["point_labels"].squeeze(1).cpu().numpy().tolist()
            }

        with torch.no_grad():
            image_embeddings = model.image_encoder(batched_input["image"])

        if args.boxes_prompt:
            save_path = os.path.join(args.work_dir, args.run_name, "boxes_prompt")
            save_path_imgs = os.path.join(args.work_dir, args.run_name, "boxes_prompt_with_imgs")
            batched_input["point_coords"], batched_input["point_labels"] = None, None
            masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings)
            points_show = None

        else:
            save_path = os.path.join(f"{args.work_dir}", args.run_name,
                                     f"iter{args.iter_point if args.iter_point > 1 else args.point_num}_prompt")
            save_path_imgs = os.path.join(f"{args.work_dir}", args.run_name,
                                     f"iter{args.iter_point if args.iter_point > 1 else args.point_num}_prompt_with_imgs")
            batched_input["boxes"] = None
            point_coords, point_labels = [batched_input["point_coords"]], [batched_input["point_labels"]]

            for iter in range(args.iter_point):
                masks, low_res_masks, iou_predictions = prompt_and_decoder(args, batched_input, model, image_embeddings)
                if iter != args.iter_point - 1:
                    batched_input = generate_point(masks, labels, low_res_masks, batched_input, args.point_num)
                    batched_input = to_device(batched_input, args.device)
                    point_coords.append(batched_input["point_coords"])
                    point_labels.append(batched_input["point_labels"])
                    batched_input["point_coords"] = torch.concat(point_coords, dim=1)
                    batched_input["point_labels"] = torch.concat(point_labels, dim=1)

            points_show = (torch.concat(point_coords, dim=1), torch.concat(point_labels, dim=1))
            points=[batched_input["point_coords"],batched_input["point_labels"]]

        masks, pad = postprocess_masks(low_res_masks, args.image_size, original_size)
        if args.save_pred:
            if args.boxes_prompt:
                image_np = batched_input["ori_image"].cpu().numpy().squeeze(0)
                save_masks(masks, save_path, img_name, args.image_size, original_size, pad,
                       batched_input.get("boxes", None))
                save_imgs_prompts_masks(masks, save_path_imgs, img_name, args.image_size, original_size, image_np, pad,
                       batched_input.get("boxes", None))
            else:
                image_np = batched_input["ori_image"].cpu().numpy().squeeze(0)
                save_masks(masks, save_path, img_name, args.image_size, original_size, pad,None,
                           points_show)
                # save_imgs_prompts_masks(masks, save_path_imgs, img_name, args.image_size, original_size, image_np, pad,None,
                #                         points_show)
        loss = criterion(masks, ori_labels, iou_predictions)
        MedSAM_loss = seg_loss(masks, ori_labels) + ce_loss(
            masks, ori_labels.float()
        )
        test_loss.append(loss.item())
        MedSAM_test_loss.append(MedSAM_loss.item())
        test_batch_metrics = SegMetrics(masks, ori_labels, args.metrics)
        test_batch_metrics = [float('{:.4f}'.format(metric)) for metric in test_batch_metrics]

        for j in range(len(args.metrics)):
            test_iter_metrics[j] += test_batch_metrics[j]

    test_iter_metrics = [metric / l for metric in test_iter_metrics]
    test_metrics = {args.metrics[i]: '{:.4f}'.format(test_iter_metrics[i]) for i in range(len(test_iter_metrics))}

    average_loss = np.mean(test_loss)
    average_MedSAM_loss = np.mean(MedSAM_test_loss)
    if args.prompt_path is None:
        with open(os.path.join(args.work_dir, args.run_name ,f'{args.image_size}_prompt.json'), 'w') as f:
            json.dump(prompt_dict, f, indent=2)
    if args.boxes_prompt:
        log_Test_type = "boxes_prompt"
    else:
        log_Test_type =  f"iter{args.iter_point if args.iter_point > 1 else args.point_num}_prompt_with_imgs"
    log_Test_loss = f"Test loss: {average_loss:.4f}, metrics: {test_metrics}"
    log_MedSAM_Test_loss =  f"MedSAM Test loss: {average_MedSAM_loss:.4f}, metrics: {test_metrics}"
    with open(os.path.join(args.work_dir, args.run_name, f'{log_Test_type}_log.json'), 'w') as f:
        json.dump(info,f,indent=4)
        json.dump(log_Test_type, f,indent=4)
        json.dump(log_Test_loss,f,indent=4)
        json.dump(log_MedSAM_Test_loss,f,indent=4)
    print(f"Test loss: {average_loss:.4f}, metrics: {test_metrics}")
    print(f"MedSAM Test loss: {average_MedSAM_loss:.4f}, metrics: {test_metrics}")


if __name__ == '__main__':
    args = parse_args()
    main(args)
