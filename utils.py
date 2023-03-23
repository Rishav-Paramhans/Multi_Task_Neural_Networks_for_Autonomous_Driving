import matplotlib.pyplot as plt
import matplotlib.patches as patches
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import random
import math
from collections import Counter
import torch
from torch.utils.data import DataLoader
import cv2
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import timeit
from torchvision.utils import save_image
from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader
from rpn_utils import *
# DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"


def save_model(model, name):
    print("----saving model----")
    torch.save(model.state_dict(), name)

    
def save_optimizer(optimizer, name):
    print("----saving model----")
    torch.save(optimizer.state_dict(), name)


def load_model(model, name, DEVICE):
    print("----loading model----")
    model.load_state_dict(torch.load(name, map_location=DEVICE))


def check_accuracies(
        model, loader, tasks=None, check_map=False, anchors=None, iou_threshold=0.5, DEVICE="cuda",
        box_format="midpoint", conf_threshold=0.5, nms_threshold=0.3, class_wise_dice_score=False, conf=None
):
    """
    :param loader: train or test or val loader
    :param tasks: same as tasks mentioned in train.py file
    :param check_map: if True checks the map of the object detection task.
    :param anchors: predefined anchors in train.py
    :param iou_threshold: for calculating mAP
    :param box_format: midpoint or center
    :param conf_threshold: models confidence threshold for calculating mAP
    :param nms_threshold:  non max suppression IOU threshold
    :param class_wise_dice_score:  If True prints the class wise dice score for segmentation tasks
    :return: returns a dictionary with task : mean accuracy over the loader of all tasks.
    """
    print("---checking_accuracies---")
    model.eval()
    segmentation_task = False
    mono_3D_det_task= False
    scores = {}
    for task in tasks:
        if task[0] == "semantic_segmentation":
            scores[task[0]] = torch.zeros(task[1], dtype=torch.float)  # list to store class wise dice score
            segmentation_task = True
        elif task[0] == "object_detection":
            scores[task[0]] = 0
            det_classes = task[1]
        elif task[0] == "mono_3D_object_detection":
            scores[task[0]]= 0
            mono_3D_det_task= True
            mono_det_3D_classes = task[1]

    if segmentation_task:
        Dice_score = []
        for idx, imobj in enumerate(tqdm(loader)):
            x= imobj["image"]
            x = x.to(DEVICE)
            with torch.no_grad():
                out, features = model(x.float())
                for task in tasks:
                    if task[0] == "semantic_segmentation":
                        dice_score= mean_dice_score(preds=out[task[0]], targets=imobj["semantic_label"], num_classes=task[1])
                        Dice_score.append(dice_score)
        mean_dice= sum(Dice_score)/len(Dice_score)
        
        scores["semantic_segmentation"] = mean_dice

    print(f"task_scores : {scores}")
    if check_map:
        print("--checking_mAP_for_2D_Object_Detection--")

        all_pred_boxes, all_true_boxes = get_evaluation_bboxes_with_loader(loader, model, iou_threshold=nms_threshold,
                                                                           threshold=conf_threshold, anchors=anchors,
                                                                           device=DEVICE)
        scores["object_detection"] = mean_average_precision(
            all_pred_boxes, all_true_boxes, iou_threshold=iou_threshold, box_format=box_format, num_classes=det_classes
        ) * 100
    print(f"task_scores : {scores}")    
    if mono_3D_det_task:
        print("---ckecking_mAP_for_monocular_3D_Object_Detection---")
        all_pred_boxes, all_true_boxes= get_rpn_boxes_with_loader(loader, model, DEVICE, conf)
        mAP_bev,mAP_2D = rpn_MAP(all_pred_boxes, all_true_boxes, conf)
        scores["mono_3D_object_detection"] = mAP_2D

    #for task in tasks:
    #    if class_wise_dice_score:
    #        print(f"{task[0]}:{(scores[task[0]] * 100) / len(loader)}")
    #    if task[0] != "object_detection":
    #        scores[task[0]] = ((scores[task[0]].mean() * 100) / len(loader))

    print(f"task_scores : {scores}")

    return scores



# same as check_accuracies but calculates IOU instead of dice coefficient

def check_IOU_accuracies(
        model, loader, tasks=None, check_map=False, anchors=None, iou_threshold=0.5, DEVICE="cuda",
        box_format="midpoint", conf_threshold=0.5, nms_threshold=0.3, class_wise_dice_score=False,
):
    print("---checking_accuracies---")
    model.eval()
    segmentation_task = False
    scores = {}
    for task in tasks:
        if task[0] == "semantic_segmentation":
            scores[task[0]] = torch.zeros(task[1], dtype=torch.float)  # list to store class wise dice score
            segmentation_task = True
        elif task[0] == "lane_marking":
            scores[task[0]] = torch.zeros(task[1], dtype=torch.float)
            segmentation_task = True
        elif task[0] == "drivable_area":
            scores[task[0]] = torch.zeros(task[1], dtype=torch.float)
            segmentation_task = True
        elif task[0] == "object_detection":
            scores[task[0]] = 0
            det_classes = task[1]

    if segmentation_task:

        for idx, (x, y) in enumerate(tqdm(loader)):
            x = x.to(DEVICE)
            with torch.no_grad():
                out, features = model(x)
            for task in tasks:
                if task[0] == "semantic_segmentation":
                    scores[task[0]] += mean_IOU_score(preds=out[task[0]], targets=y[task[0]], num_classes=task[1])
                elif task[0] == "lane_marking":
                    scores[task[0]] += mean_IOU_score(preds=out[task[0]], targets=y[task[0]], num_classes=task[1])
                elif task[0] == "drivable_area":
                    scores[task[0]] += mean_IOU_score(preds=out[task[0]], targets=y[task[0]], num_classes=task[1])
                # elif task[0] == "object_detection":
                #    if check_map:
                #        all_pred_boxes_, all_true_boxes_ = get_evaluation_bboxes(
                #            preds=out[task[0]], targets=y[task[0]], iou_threshold=iou_threshold, train_idx=train_idx,
                #            anchors=anchors, box_format=box_format, threshold=conf_threshold, batch_size=x.shape[0]
                #        )
                #
                #        all_pred_boxes.append(all_pred_boxes_)
                #        all_true_boxes.append(all_true_boxes_)
                #        train_idx += x.shape[0]

    if check_map:
        print("--checking_mAP--")

        all_pred_boxes, all_true_boxes = get_evaluation_bboxes_with_loader(loader, model, iou_threshold=nms_threshold,
                                                                           threshold=conf_threshold, anchors=anchors,
                                                                           device=DEVICE)
        scores["object_detection"] = mean_average_precision(
            all_pred_boxes, all_true_boxes, iou_threshold=iou_threshold, box_format=box_format, num_classes=det_classes
        ) * 100

    for task in tasks:
        if class_wise_dice_score:
            print(f"{task[0]}:{(scores[task[0]] * 100) / len(loader)}")
        if task[0] != "object_detection":
            scores[task[0]] = ((scores[task[0]].mean() * 100) / len(loader))

    print(f"task_scores : {scores}")

    return scores


def save_some_examples(model, loader, save_path=None, tasks=None, anchors=None, iou_threshold=0.45,
                       threshold=0.45, nms_threshold=0.3, DEVICE="cuda", text_file=None):

    """
    :param DEVICE: "cuda" or "cpu"
    :param loader: train or test or val loader
    :param save_path: path to save the predictions
    :param tasks: same as tasks mentioned in train.py file
    :param anchors: predefined anchors in train.py
    :param iou_threshold:  for calculating mAP
    :param threshold: models confidence threshold for calculating mAP
    :param nms_threshold:  non max suppression IOU threshold
    :param text_file: text file to write accuracy against the image ID
    """

    model.eval()
    img_id = 0
    for batch_id, imobj in enumerate(tqdm(loader)):
        x= imobj["image"]
        x = x.to(DEVICE)
        with torch.no_grad():
            out, features = model(x)
        preds = {}
        scores = {}
        for task in tasks:
            if task[0] == "semantic_segmentation":
                preds[task[0]] = preds_to_class_map(out[task[0]])
            elif task[0] == "object_detection":
                bboxes = [[] for _ in range(x.shape[0])]
                target_bb = [[] for _ in range(x.shape[0])]
                yolo_out = out[task[0]]
                for i in range(3):
                    batch_size, A, S, _, _ = yolo_out[i].shape
                    anchor = anchors[i]
                    boxes_scale_i = cells_to_bboxes(
                        yolo_out[i], anchor, S=S, is_preds=True
                    )
                    for idx, (box) in enumerate(boxes_scale_i):
                        bboxes[idx] += box
                target_bb_ = cells_to_bboxes(
                    y[task[0]][2].to(DEVICE), anchor, S=88, is_preds=False
                    # using scale s S=(Image_size/32) to get target bboxes
                )
                # for idx, (box) in enumerate(target_bb):
                #    target_bb[idx] += box

        predictions_final = {}
        for batch_num in range(len(x)):
            text_file.write("\n")
            text_file.write(str(img_id) + ",")
            for task in tasks:
                if task[0] != "object_detection":
                    scores[task[0]] = mean_dice_score(out[task[0]][batch_num:batch_num + 1],
                                                      y[task[0]][batch_num:batch_num + 1],
                                                      num_classes=task[1])
                    concat = pred_to_concat_output(preds[task[0]][batch_num], y[task[0]][batch_num], task=task,
                                                   score=scores[task[0]])
                    predictions_final[task[0]] = concat
                    text_file.write(task[0] + "," + str(round((float(scores[task[0]].mean())), 2)) + ",")
                elif task[0] == "object_detection":
                    nms_boxes = non_max_suppression(
                        bboxes[batch_num], iou_threshold=nms_threshold, threshold=threshold, box_format="midpoint",
                    )
                    for box in (target_bb_[batch_num]):
                        if box[1] > 0:
                            target_bb[batch_num].append(box)
                    input_img_1 = x[batch_num].detach().cpu()
                    det, im_original = plot_image_cv2(input_img_1, nms_boxes)
                    target, im_original = plot_image_cv2_2(input_img_1, target_bb[batch_num])
                    # print([[0] + nms_box for nms_box in nms_boxes])
                    # print([[0] + target_b for target_b in target_bb[batch_num]])

                    map, TP, FP = mean_average_precision_([[0] + nms_box for nms_box in nms_boxes],
                                                          [[0] + target_b for target_b in target_bb[batch_num]],
                                                          num_classes=task[1])
                    scores[task[0]] = round(float((TP / (FP + len(target_bb[batch_num]) + 1e-6)) * 100), 2)
                    all_predictions = "GT:" + str(len(target_bb[batch_num])) + ",TP:" + str(TP) + ",FP:" + str(
                        FP) + ",bb_accuracy:" + str(scores[task[0]])
                    # print(all_predictions)
                    text_file.write(task[0] + "," + all_predictions)
                    # print(scores[task[0]])
                    # scores[task[0]] = (len([[0]+nms_boxes]) / len([[0]+target_bb[batch_num]])) * 100  # precision for this image
                    cv2.rectangle(det, (0, 0), (560, 45), (0, 0, 0), -1)
                    cv2.putText(det, all_predictions, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, thickness=2,
                                color=(0, 0, 255))
                    concat = np.vstack((det, target))
                    predictions_final[task[0]] = concat
            save_file_name = str(round(float(scores[tasks[0][0]].mean()), 2)) + "%_" + str(img_id) if len(
                tasks) == 1 else str(img_id)
            concat_ = np.hstack(
                tuple(predictions_final[key] for key in predictions_final.keys()))  # Horizontally concatenation
            cv2.imwrite(save_path + save_file_name + ".png", concat_)
            img_id += 1
        # if batch_id == 50:
        #    break


def mean_dice_score(preds, targets, num_classes=20, smooth=1):
    preds = torch.softmax(preds, dim=1)
    preds = torch.argmax(preds, dim=1)
    preds = np.asarray(torch.Tensor.cpu(preds))
    targets = np.asarray(torch.Tensor.cpu(targets))

    preds = class_map_to_one_hot_predictions(preds, num_classes=num_classes)
    targets = class_map_to_one_hot_predictions(targets, num_classes=num_classes)

    intersection = (np.logical_and(preds, targets).astype(int))
    union = (np.logical_or(preds, targets).astype(int))

    intersection = intersection.sum(axis=3).sum(axis=2).sum(axis=0)  # sum across batches
    union = union.sum(axis=3).sum(axis=2).sum(axis=0)

    dice_score = (2 * intersection + smooth) / (union + intersection + smooth)

    return torch.Tensor(dice_score)


def mean_IOU_score(preds, targets, num_classes=20, smooth=1):
    preds = torch.softmax(preds, dim=1)
    preds = torch.argmax(preds, dim=1)
    preds = np.asarray(torch.Tensor.cpu(preds))
    targets = np.asarray(torch.Tensor.cpu(targets))

    preds = class_map_to_one_hot_predictions(preds, num_classes=num_classes)
    targets = class_map_to_one_hot_predictions(targets, num_classes=num_classes)

    intersection = (np.logical_and(preds, targets).astype(int))
    union = (np.logical_or(preds, targets).astype(int))

    intersection = intersection.sum(axis=3).sum(axis=2).sum(axis=0)  # sum across batches
    union = union.sum(axis=3).sum(axis=2).sum(axis=0)

    dice_score = (intersection + smooth) / (union + smooth)

    return torch.Tensor(dice_score)


def preds_to_class_map(pred):
    pred = torch.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)
    pred = pred.unsqueeze(1)
    pred = torch.Tensor.cpu(pred)
    pred = torch.permute(pred, (0, 2, 3, 1))

    return pred


def pred_to_concat_output(pred, target, score, task=None):
    pred = np.repeat(np.asarray(pred), 3, 2)  # (H,W,3)
    color_map_preds = mask_to_colormap(pred, task=task[0])
    color_map_preds = color_map_preds.astype("float32")
    cv2.putText(color_map_preds, str(round(float(score.mean() * 100), 2)) + "%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255), thickness=2)
    cv2.imwrite("color.png", color_map_preds)
    target = np.repeat(np.asarray(target.unsqueeze(2)), 3, 2)  # (batch_size,H,W,3)
    color_map_target = mask_to_colormap(target, task=task[0])
    concat = np.vstack((color_map_preds, color_map_target))  # vertical concatination
    # H, W, C = pred.shape
    # cv2.putText(concat, task[0], ((H - 20), 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, 2)

    return concat


def get_evaluation_bboxes(
        preds, targets, iou_threshold, anchors,
        threshold, box_format, batch_size, train_idx, DEVICE="cpu"
):
    all_pred_boxes = []
    all_true_boxes = []
    train_idx = train_idx
    batch_size = batch_size
    bboxes = [[] for _ in range(batch_size)]
    for i in range(3):
        S = preds[i].shape[2]
        anchor = torch.tensor([*anchors[i]]).to(DEVICE) * S
        boxes_scale_i = cells_to_bboxes(
            preds[i], anchor, S=S, is_preds=True
        )
        for idx, (box) in enumerate(boxes_scale_i):
            bboxes[idx] += box

    # we just want one bbox for each label, not one for each scale
    true_bboxes = cells_to_bboxes(
        targets[2], anchor, S=S, is_preds=False
    )

    for idx in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[idx],
            iou_threshold=iou_threshold,
            threshold=threshold,
            box_format=box_format,
        )

        for nms_box in nms_boxes:
            all_pred_boxes.append([train_idx] + nms_box)

        for box in true_bboxes[idx]:
            if box[1] > threshold:
                all_true_boxes.append([train_idx] + box)

        train_idx += 1

    return all_pred_boxes, all_true_boxes


def get_evaluation_bboxes_with_loader(
        loader,
        model,
        iou_threshold,
        anchors,
        threshold,
        box_format="midpoint",
        device="cpu",
):
    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0
    all_pred_boxes = []
    all_true_boxes = []
    for batch_idx, imobj in enumerate(tqdm(loader)):
        x= imobj["image"]
        x = x.to(device)
        with torch.no_grad():
            predictions, features = model(x.float())

        batch_size = x.shape[0]
        predictions = predictions["object_detection"]
        labels = imobj['yolo_target']
        bboxes = [[] for _ in range(batch_size)]
        for i in range(3):
            S = predictions[i].shape[2]
            anchor = torch.tensor([*anchors[i]]).to(device) * S
            boxes_scale_i = cells_to_bboxes(
                predictions[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        # we just want one bbox for each label, not one for each scale
        true_bboxes = cells_to_bboxes(
            labels[2], anchor, S=S, is_preds=False
        )

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

        # print(all_true_boxes)
        # print(all_pred_boxes)

    model.train()
    return all_pred_boxes, all_true_boxes


def mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=10
):
    # list storing all AP for respective classes
    average_precisions = []
    recall_ = []
    precision_ = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1
        # print("class", c, "true positives", sum(TP), "false positives", sum(FP))
        # print(total_true_bboxes)
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        # print(recalls)
        # print(precisions)
        precision_.append(sum(TP) / (sum(TP) + sum(FP) + epsilon))
        recall_.append(sum(TP) / total_true_bboxes + epsilon)

        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))
        # print(average_precisions)
    # print("precisions", precision_)
    # print("recalls", recall_)
    # print("recalls", sum(recall_)/len(recall_))
    # print("precisions", sum(precision_)/len(precision_))
    # print("mAP", average_precisions)

    mAP = sum(average_precisions) / (len(average_precisions) + epsilon)

    return mAP


def mean_average_precision_(
        pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    # list storing all AP for respective classes
    average_precisions = []
    recall_ = []
    precision_ = []
    TP_ = []
    FP_ = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            FP_.append(len(detections))
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1
        # print("class", c, "true positives", sum(TP), "false positives", sum(FP))
        # print(total_true_bboxes)
        TP_.append(sum(TP))
        FP_.append(sum(FP))
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        # print(recalls)
        # print(precisions)
        precision_.append(sum(TP) / (sum(TP) + sum(FP) + epsilon))
        recall_.append(sum(TP) / total_true_bboxes + epsilon)

        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))
        # print(average_precisions)
    # print("precisions", precision_)
    # print("recalls", recall_)
    # print("recalls", sum(recall_)/len(recall_))
    # print("precisions", sum(precision_)/len(precision_))
    # print("mAP", average_precisions)

    mAP = sum(average_precisions) / (len(average_precisions) + epsilon)
    # print(sum(TP_), sum(FP_))

    return mAP, int(sum(TP_)), int(sum(FP_))


def check_class_accuracy(model, loader, threshold, DEVICE="cpu"):
    model.eval()
    tot_class_preds, correct_class = 0, 0
    tot_noobj, correct_noobj = 0, 0
    tot_obj, correct_obj = 0, 0

    for idx, (x, y) in enumerate(tqdm(loader)):
        x = x.to(DEVICE)
        with torch.no_grad():
            out = model(x)
        y = y["object_detection"]
        out = out["object_detection"]
        for i in range(3):
            y[i] = y[i].to(DEVICE)
            obj = y[i][..., 0] == 1  # in paper this is Iobj_i
            noobj = y[i][..., 0] == 0  # in paper this is Iobj_i

            correct_class += torch.sum(
                torch.argmax(out[i][..., 5:][obj], dim=-1) == y[i][..., 5][obj]
            )
            tot_class_preds += torch.sum(obj)

            obj_preds = torch.sigmoid(out[i][..., 0]) > threshold
            correct_obj += torch.sum(obj_preds[obj] == y[i][..., 0][obj])
            tot_obj += torch.sum(obj)
            correct_noobj += torch.sum(obj_preds[noobj] == y[i][..., 0][noobj])
            tot_noobj += torch.sum(noobj)

    print(f"Class accuracy is: {(correct_class / (tot_class_preds + 1e-16)) * 100:2f}%")
    print(f"No obj accuracy is: {(correct_noobj / (tot_noobj + 1e-16)) * 100:2f}%")
    print(f"Obj accuracy is: {(correct_obj / (tot_obj + 1e-16)) * 100:2f}%")
    model.train()


def get_mean_std(loader):
    # var[X] = E[X**2] - E[X]**2
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def plot_couple_examples(model, loader, thresh, iou_thresh, anchors):
    model.eval()
    x, y = next(iter(loader))
    x = x.to("cuda")
    with torch.no_grad():
        start = timeit.default_timer()
        out = model(x)
        bboxes = [[] for _ in range(x.shape[0])]
        for i in range(3):
            batch_size, A, S, _, _ = out[i].shape
            anchor = anchors[i]
            boxes_scale_i = cells_to_bboxes(
                out[i], anchor, S=S, is_preds=True
            )
            for idx, (box) in enumerate(boxes_scale_i):
                bboxes[idx] += box

        model.train()

    for i in range(batch_size):
        nms_boxes = non_max_suppression(
            bboxes[i], iou_threshold=iou_thresh, threshold=thresh, box_format="midpoint",
        )
        stop = timeit.default_timer()
        print('Time: ', stop - start)
        plot_image(x[i].permute(1, 2, 0).detach().cpu(), nms_boxes)


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mask_to_colormap(image, task=None):
    color_map = np.zeros_like(image)
    class_ids = np.unique(image)

    if task == "semantic_segmentation":
        for class_id in class_ids:
            color_map = np.where(image == [[class_id, class_id, class_id]], [[segmentation_color_ids[class_id]]],
                                 color_map)
    elif task == "combined":
        for class_id in class_ids:
            color_map = np.where(image == [[class_id, class_id, class_id]], [[combined_color_ids[class_id]]], color_map)
    else:
        raise ValueError(
            "invalid argument : task should be one of these values 'semantic_segmentation', 'lane_marking', "
            "'drivable_area'")
    return color_map


def check_accuracy(model, loader, DEVICE="cpu"):
    dice_score = 0
    model.eval()
    loop = tqdm(loader, leave=True)

    with torch.no_grad():
        for x, y in loop:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            # labels = torch.argmax(y,dim=1)
            preds = torch.softmax(model(x), dim=1)
            preds = torch.argmax(preds, dim=1)
            preds_correct = (torch.eq(preds, y)).int()
            dice_score += ((preds_correct.sum()) / (
                    torch.sum(y) + 1e-8
            ))

    print(f"Accuracy: {dice_score / len(loader)}")
    model.train()
    return dice_score / len(loader)


def class_map_to_one_hot_predictions(input_img, num_classes=19):
    BS, H, W = input_img.shape
    one_hot = np.zeros((BS, num_classes, H, W))

    class_ids = np.unique(input_img)
    for class_id in class_ids:
        one_hot[:, int(class_id), :, :] = np.where(input_img[:, :, :] == class_id, 1, one_hot[:, int(class_id), :, :])

    return one_hot


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # comment out if your model contains a softmax or equivalent activation layer
        inputs = torch.softmax(inputs, dim=1)

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


def mean_IOU(model, loader, num_classes=19, smooth=1, DEVICE="cpu"):
    loop = tqdm(loader, leave=True)
    IOU_score_avg = torch.zeros(num_classes, dtype=torch.float)  # list to store class wise dice score
    with torch.no_grad():
        for x, y in loop:
            x = x.to(DEVICE)  # input image of size (BS,C,H,W)
            y = y.to(DEVICE)  # labels mask of size (BS,1,H,W)
            preds = model(x)  # preds image of size (BS,num_classes,H,W)
            preds = torch.softmax(preds, dim=1)
            preds = torch.argmax(preds, dim=1)
            preds = np.asarray(torch.Tensor.cpu(preds))
            labels = np.asarray(torch.Tensor.cpu(y))

            preds = class_map_to_one_hot_predictions(preds, num_classes=num_classes)
            labels = class_map_to_one_hot_predictions(labels, num_classes=num_classes)

            intersection = (np.logical_and(preds, labels).astype(int))
            union = (np.logical_or(preds, labels).astype(int))

            intersection = intersection.sum(axis=3).sum(axis=2).sum(axis=0)
            union = union.sum(axis=3).sum(axis=2).sum(axis=0)

            IOU_score = (intersection + smooth) / (union + smooth)
            IOU_score_avg += torch.Tensor(IOU_score)

    print(f"mean_IOU : {((IOU_score_avg.mean() * 100) / len(loader))}%")
    print(f"categorical_IOU_score : {(IOU_score_avg * 100) / len(loader)}")
    return (IOU_score_avg.mean() * 100) / len(loader)


def transforms(IMAGE_WIDTH=None, IMAGE_HEIGHT=None):
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[], ),
        additional_targets={'mask': 'mask', 'mask2':'mask'}
    )

    test_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[], ),
        additional_targets={'mask': 'mask', 'mask2':'mask'}
    )

    return train_transform, test_transform


def iou_width_height(boxes1, boxes2):
    """
    Parameters:
        boxes1 (tensor): width and height of the first bounding boxes
        boxes2 (tensor): width and height of the second bounding boxes
    Returns:
        tensor: Intersection over union of the corresponding boxes
    """
    intersection = torch.min(boxes1[..., 0], boxes2[..., 0]) * torch.min(
        boxes1[..., 1], boxes2[..., 1]
    )
    union = (
            boxes1[..., 0] * boxes1[..., 1] + boxes2[..., 0] * boxes2[..., 1] - intersection
    )
    return intersection / union


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Video explanation of this function:
    https://youtu.be/XXYG5ZWtjj0

    This function calculates intersection over union (iou) given pred boxes
    and target boxes.

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Video explanation of this function:
    https://youtu.be/YDkjWEN8jNA

    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
               or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
               < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def plot_image_cv2(image, bboxes):
    C, W, H, = image.shape
    save_image(image, "img_temp.png")
    img = cv2.imread("img_temp.png")
    img_original = np.array(img)
    # print(type(image))
    for box in bboxes:
        width = W
        height = H
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = int(box[0])
        confidence_score = box[1]
        bbox = box[2:]
        x1 = int((bbox[0] - bbox[2] / 2) * width)
        y1 = int((bbox[1] - bbox[3] / 2) * height)
        x2 = x1 + int(bbox[2] * width)
        y2 = y1 + int(bbox[3] * height)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.putText(img, A2D2_bbox_categories[class_pred], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, 2)
        cv2.putText(img, str(int(confidence_score * 100)) + "%_" + A2D2_bbox_categories[class_pred],
                    (x1, y1), cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=segmentation_color_ids[class_pred]
                    , thickness=2)

    return img, img_original


def plot_image_cv2_2(image, bboxes):
    C, W, H = image.shape
    # save_image(image, "img_temp_2.png")
    img = cv2.imread("img_temp.png")
    img_original = np.array(img)
    # print(type(image))
    for box in bboxes:
        width = W
        height = H
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = int(box[0])
        confidence_score = box[1]
        bbox = box[2:]
        x1 = int((bbox[0] - bbox[2] / 2) * width)
        y1 = int((bbox[1] - bbox[3] / 2) * height)
        x2 = x1 + int(bbox[2] * width)
        y2 = y1 + int(bbox[3] * height)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.putText(img, A2D2_bbox_categories[class_pred], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, 2)
        cv2.putText(img, str(int(confidence_score * 100)) + "%_" + A2D2_bbox_categories[class_pred],
                    (x1, y1), cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=segmentation_color_ids[class_pred]
                    , thickness=2)

    return img, img_original


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    cmap = plt.get_cmap("tab20b")
    class_labels = A2D2_bbox_categories
    colors = [cmap(i) for i in np.linspace(0, 1, len(class_labels))]
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1, figsize=(100, 5))
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle patch
    for box in boxes:
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = box[0]
        confidence_score = box[1]
        box = box[2:]
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=2,
            edgecolor=colors[int(class_pred)],
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)
        plt.text(
            upper_left_x * width,
            upper_left_y * height,
            s=str(int(math.ceil(confidence_score * 100))) + '%' + ', ' + class_labels[int(class_pred)],
            color="white",
            verticalalignment="top",
            bbox={"color": colors[int(class_pred)], "pad": 0},
        )

    # plt.show()
    plt.savefig("im_temp.png")

    image_with_bbox = cv2.imread("img_temp.png")

    return image_with_bbox


def cells_to_bboxes(predictions, anchors, S, is_preds=True, DEVICE="cpu"):
    """
    Scales the predictions coming from the model to
    be relative to the entire image such that they for example later
    can be plotted or.
    INPUT:
    predictions: tensor of size (N, 3, S, S, num_classes+5)
    anchors: the anchors used for the predictions
    S: the number of cells the image is divided in on the width (and height)
    is_preds: whether the input is predictions or the true bounding boxes
    OUTPUT:
    converted_bboxes: the converted boxes of sizes (N, num_anchors, S, S, 1+5) with class index,
                      object score, bounding box coordinates
    """
    BATCH_SIZE = predictions.shape[0]
    num_anchors = len(anchors)
    box_predictions = predictions[..., 1:5]
    #print('box_predictions', box_predictions.size())  #basically perdictions without cls and prob
    if is_preds:
        anchors = anchors.reshape(1, len(anchors), 1, 1, 2)
        box_predictions[..., 0:2] = torch.sigmoid(box_predictions[..., 0:2])
        box_predictions[..., 2:] = torch.exp(box_predictions[..., 2:]) * anchors
        scores = torch.sigmoid(predictions[..., 0:1])
        best_class = torch.argmax(predictions[..., 5:], dim=-1).unsqueeze(-1)
    else:
        scores = predictions[..., 0:1]
        best_class = predictions[..., 5:6]

    cell_indices = (
        torch.arange(S)
            .repeat(predictions.shape[0], 3, S, 1)
            .unsqueeze(-1)
            .to(predictions.device)
    )
    x = 1 / S * (box_predictions[..., 0:1] + cell_indices)
    y = 1 / S * (box_predictions[..., 1:2] + cell_indices.permute(0, 1, 3, 2, 4))
    w_h = 1 / S * box_predictions[..., 2:4]
    converted_bboxes = torch.cat((best_class, scores, x, y, w_h), dim=-1).reshape(BATCH_SIZE, num_anchors * S * S, 6)
    return converted_bboxes.tolist()


def plot_image_cv2_for_overlay(image, bboxes):
    H, W, C = image.shape
    # save_image(image, "img_temp_2.png")
    # img = cv2.imread("img_temp.png")
    # img_original = np.array(img)
    # print(type(image))
    for box in bboxes:
        width = W
        height = H
        assert len(box) == 6, "box should contain class pred, confidence, x, y, width, height"
        class_pred = int(box[0])
        confidence_score = box[1]
        bbox = box[2:]
        x1 = int((bbox[0] - bbox[2] / 2) * width)
        y1 = int((bbox[1] - bbox[3] / 2) * height)
        x2 = x1 + int(bbox[2] * width)
        y2 = y1 + int(bbox[3] * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # cv2.putText(img, A2D2_bbox_categories[class_pred], (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 5, 2)
        cv2.putText(image, str(int(confidence_score * 100)) + "%_" + A2D2_bbox_categories[class_pred],
                    (x1, y1), cv2.FONT_HERSHEY_PLAIN, fontScale=2, color=segmentation_color_ids[class_pred]
                    , thickness=2)

    return image


A2D2_bbox_categories_old = [
    "Car",
    "Pedestrian",
    "Truck",
    "VanSUV",
    "Cyclist",
    "Bus",
    "MotorBiker",
    "Bicycle",
    "UtilityVehicle",
    "Motorcycle",
    "CaravanTransporter",
    "Animal",
    "Trailer",
    "EmergencyVehicle",
]

A2D2_bbox_categories = [
    "Car",
    "Pedestrian",
    "Truck",
    "VanSUV",
    "Cyclist",
    "Bus",
    "MotorBiker",
    "Bicycle",
    "UtilityVehicle",
    "Motorcycle"
]

semantic_segmentation_color_ids = {
    0: "#ff0000",
    1: "#b65906",
    2: "#cc99ff",
    3: "#ff8000",
    4: "#00ff00",
    5: "#0080ff",
    6: "#00ffff",
    7: "#e96400",
    8: "#6e6e00",
    9: "#b97a57",
    10: "#fff68f",
    11: "#ccff99",
    12: "#eea2ad",
    13: "#212cb1",
    14: "#93fdc2",
    15: "#9696c8",
    16: "#b496c8",
    17: "#9f79ee",
    18: "#87ceff",
    19: "#f1e6ff",
    20: "#000000",
}

segmentation_color_ids = {
    0: [0, 0, 255],
    1: [6, 89, 182],
    2: [255, 153, 204],
    3: [0, 128, 255],
    4: [0, 255, 0],
    5: [255, 128, 0],
    6: [255, 255, 0],
    7: [0, 100, 233],
    8: [255, 26, 255],
    9: [87, 122, 185],
    10: [143, 246, 255],
    11: [153, 255, 204],
    12: [173, 162, 238],
    13: [177, 44, 33],
    14: [194, 253, 147],
    15: [200, 150, 150],
    16: [200, 150, 180],
    17: [238, 121, 159],
    18: [255, 206, 135],
    19: [38, 77, 115],
    20: [0, 0, 0]
}

semantic_class_name = {
    0: "Car",
    1: "bicycle",
    2: "pedestrian",
    3: "Utility vehicle",
    4: "small vehicles",
    5: "Traffic signal",
    6: "Traffic sign",
    7: "sidebars",
    8: "road",
    9: "road blocks",
    10: "poles",
    11: "animals",
    12: "grid structure",
    13: "signal corpus",
    14: "nature object",
    15: "parking area",
    16: "sidewalk",
    17: "traffic guide obj.",
    18: "sky",
    19: "buildings",
    20: "background",
}


combined_color_ids = {
    0: [0, 0, 255],
    1: [6, 89, 182],
    2: [255, 153, 204],
    3: [0, 128, 255],
    4: [0, 255, 0],
    5: [255, 128, 0],
    6: [255, 255, 0],
    7: [0, 100, 233],
    8: [255, 26, 255],
    9: [87, 122, 185],
    10: [143, 246, 255],
    11: [153, 255, 204],
    12: [173, 162, 238],
    13: [177, 44, 33],
    14: [194, 253, 147],
    15: [200, 150, 150],
    16: [200, 150, 180],
    17: [238, 121, 159],
    18: [255, 206, 135],
    19: [38, 77, 115],
    20: [255, 255, 255],
    21: [255, 255, 0],
    22: [0, 51, 102],
    23: [153, 0, 0],
    24: [121, 0, 250],
    25: [255, 255, 255],
    26: [77, 77, 255],
    27: [68, 204, 0],
    28: [255, 255, 255]
}
