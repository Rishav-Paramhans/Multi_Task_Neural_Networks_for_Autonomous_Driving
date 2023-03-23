import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import models
from torchvision import transforms
from rpn_utils import *
import torch.nn.functional as F
import torch
import pandas as pd
import numpy as np
import os
import math
from PIL import Image
import re
from easydict import EasyDict as edict
import cv2
from torch.utils.data import Dataset, DataLoader
from utils import (cells_to_bboxes, iou_width_height as yolo_iou, non_max_suppression as nms, plot_image, plot_image_cv2)


class multi_task_dataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, dataset_dir, mono_3d_label_dir=None, semantic_label_dir= None,
                 det_2d_label_dir= None,depth_estimation_label_dir= None, img_dir=None,
                 tasks=None, yolo_anchors=None, S=None, transform= None):

        self.annotations = pd.read_csv(csv_file)  # csv file in the form of image name, task labels in cosecutive colomns
        self.dataset_dir = dataset_dir
        self.mono_3d_label_dir = mono_3d_label_dir
        self.semantic_label_dir = semantic_label_dir
        self.det_2d_label_dir = det_2d_label_dir
        self.depth_estimation_label_dir= depth_estimation_label_dir
        self.tasks = tasks
        self.transform= transform
        self.yolo_anchors = yolo_anchors
        self.img_dir = img_dir
        self.S = S
        self.yolo_anchors = torch.tensor(yolo_anchors[0] + yolo_anchors[1] + yolo_anchors[2])  # for all 3 scales
        self.yolo_num_anchors = self.yolo_anchors.shape[0]
        self.yolo_num_anchors_per_scale = self.yolo_num_anchors // 3
        self.ignore_iou_thresh = 0.5

        #mono_3D_object_detection
        self.P = np.array([[ 1.00145047e+03, -1.66583918e+03,  3.52945654e+01, -1.74681545e+03],
                 [ 6.76424384e+02, -2.78547356e+01, -1.78625897e+03,  5.27308035e+02],
                    [ 9.99759581e-01,  2.13721864e-02, -4.90013971e-03, -1.70601282e+00],
                     [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]])
        self.P_inv = np.linalg.inv(self.P)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        imobj = edict()
        img_path = os.path.join(self.dataset_dir, self.img_dir, self.annotations.iloc[index, 0])
        #print('img_path', img_path)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        masks= {"semantic_segmentation": None, "object_detection": None, "mono_3D_object_detection": None, "depth_estimation":None}
        dummy_mask = np.ones_like(image)
        dummy_bbox = [[0.5, 0.5, 0.1, 0.1, 0]]
        #image = cv2.resize(image, (960,608), interpolation = cv2.INTER_AREA)
        #image = np.swapaxes(image, 1,2)
        #image = np.swapaxes(image, 0,1)
        #imobj.image = image
        for i, task in enumerate(self.tasks):

            if task[0] == "mono_3D_object_detection":
                label_path = os.path.join(self.dataset_dir, self.mono_3d_label_dir, self.annotations.iloc[index, 2])
                mono_3d_gt = read_A2D2_label(label_path, self.P)
                imobj.P = self.P
                imobj.p2_inv = self.P_inv
                imobj.mono_3D_gts = mono_3d_gt

            elif task[0] == "semantic_segmentation":
                mask_path = os.path.join(self.dataset_dir, self.semantic_label_dir, self.annotations.iloc[index, 1])
                mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
                masks[task[0]] = mask


            elif task[0] == "object_detection":
                label_path = os.path.join(self.dataset_dir,self.det_2d_label_dir, self.annotations.iloc[index, 2])
                bboxes = (np.roll(np.loadtxt(fname=label_path, delimiter=" ", ndmin=2), 4, axis=1)).tolist()
                masks[task[0]]= bboxes
                # Below assumes 3 scale predictions (as paper) and same num of anchors per scale
                targets = [torch.zeros((self.yolo_num_anchors // 3, S, S, 6)) for S in self.S]
            elif task[0] == "depth_estimation":
                label_path= os.path.join(self.dataset_dir, self.depth_estimation_label_dir, self.annotations.iloc[index, 0])
                mask= np.array(cv2.imread(label_path, cv2.IMREAD_GRAYSCALE))    
                masks[task[0]] = mask



        if self.transform:
            augmentations = self.transform(
                image=image,
                mask=masks["semantic_segmentation"] if masks["semantic_segmentation"] is not None else dummy_mask,
                mask2= masks["depth_estimation"] if masks["depth_estimation"] is not None else dummy_mask,
                bboxes=masks["object_detection"] if masks["object_detection"] is not None else dummy_bbox
            )

            imobj.image= augmentations["image"]
            imobj.semantic_label = augmentations["mask"] if masks["semantic_segmentation"] is not None else "empty"
            imobj.depth_estimation_label= augmentations["mask2"] if masks["depth_estimation"] is not None else "empty"
            object_detection_bboxes = augmentations["bboxes"]
            for box in object_detection_bboxes:
                iou_anchors = yolo_iou(torch.tensor(box[2:4]), self.yolo_anchors)
                anchor_indices = iou_anchors.argsort(descending=True, dim=0)
                x, y, width, height, class_label = box
                has_anchor = [False] * 3  # each scale should have one anchor
                for anchor_idx in anchor_indices:

                    scale_idx = anchor_idx // self.yolo_num_anchors_per_scale
                    anchor_on_scale = anchor_idx % self.yolo_num_anchors_per_scale
                    S = self.S[scale_idx]
                    i, j = int(S * y), int(S * x)  # which cell
                    anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
                    if not anchor_taken and not has_anchor[scale_idx]:
                        targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                        x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                        width_cell, height_cell = (
                            width * S,
                            height * S,
                        )  # can be greater than 1 since it's relative to cell
                        box_coordinates = torch.tensor(
                            [x_cell, y_cell, width_cell, height_cell]
                        )
                        targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                        targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                        has_anchor[scale_idx] = True

                    elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                        targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction

            imobj.yolo_target = tuple(targets)
        return imobj

def test():
    from utils import mask_to_colormap
    from torchvision.utils import save_image
    IMAGE_SIZE = 608

    S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]
    print('S',S)
    anchors = [
        [[0.28, 0.22], [0.38, 0.48], [0.9, 0.78]],
        [[0.07, 0.15], [0.15, 0.11], [0.14, 0.29]],
        [[0.02, 0.03], [0.04, 0.07], [0.08, 0.06]],
    ]
    scaled_anchors = torch.tensor(anchors) / (
            1 / torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    )
    print('scaled_anchors', scaled_anchors.size())
    print('scaled_anchors', scaled_anchors)
    train_transform = A.Compose(
        [
            A.Resize(height=609, width=960),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[], ),
        additional_targets={'mask': 'mask'}
    )
    DATASET_DIR = r"A2D2_dataset"
    dataset = multi_task_dataset(csv_file=os.path.join(DATASET_DIR , "train.csv"),
                                 dataset_dir=DATASET_DIR,
                                 mono_3d_label_dir=r"A2D2_3D_Obj_det_label_txt",
                                 semantic_label_dir="seg_label",
                                 det_2d_label_dir='YOLO_Bbox_2D_label',
                                 depth_estimation_label_dir= None,
                                 tasks=[["semantic_segmentation", 22],["object_detection", 10],["mono_3D_object_detection", 10], ["depth_estimation", 1]],
                                 yolo_anchors=anchors,
                                 img_dir="images",
                                 S=S,
                                 transform=train_transform )
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)


    for j, imobj in enumerate(loader):
        if j==0:
            boxes = []
            x= imobj.image
            print('X', x.size())
            y= imobj.yolo_target
            sem_label= imobj.semantic_label
            mono_label= imobj.mono_3D_gts
            depth_estimation_label= imobj.depth_estimation_label
            
  
            for i in range(y[0].shape[1]):
                anchor = scaled_anchors[i]
                print(anchor.shape)
                print(y[i].shape)
                boxes += cells_to_bboxes(
                    y[i], is_preds=False, S=y[i].shape[2], anchors=anchor
                )[0]
            boxes_after_nms = nms(boxes, iou_threshold=1, threshold=0.7, box_format="midpoint")
            print('boxes_after_nms',boxes_after_nms)
            plot_image(x[0].permute(1, 2, 0).to("cpu"), boxes_after_nms)
            print('box',len(boxes_after_nms))
            print('sem_label',sem_label.size())
            print('mono_3D_gts', len(mono_label))
            print('depth_estimation_labels', depth_estimation_label.size())
        else:
            break


if __name__ == "__main__":
    test()

