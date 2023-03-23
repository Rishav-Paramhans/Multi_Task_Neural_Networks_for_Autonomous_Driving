"""
____________________________________________________________________________
This utils file contains all the neccesary functions and class definations
for M3D-RPN network
____________________________________________________________________________
"""


import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import torch
from kitti_3d_multi_warmup import *
from torchvision import models
#from Dataset import *
import pickle
import math
import numpy as np
import os
import re
from easydict import EasyDict as edict
import cv2
import pandas as pd
from nms_new import new_nms


def pickle_write(file_path, obj):
    """
    Serialize an object to a provided file_path
    """

    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)


def pickle_read(file_path):
    """
    De-serialize an object from a provided file_path
    """

    with open(file_path, 'rb') as file:
        return pickle.load(file)


## Anchor Generation

def bbXYWH2Coords(box):
    """
    Convert from [x,y,w,h] to [x1, y1, x2, y2]
    """

    if box.shape[0] == 0: return np.empty([0, 4], dtype=float)

    box[:, 2] += box[:, 0] - 1
    box[:, 3] += box[:, 1] - 1

    return box


def determine_ignores(gts, lbls, ilbls, min_gt_vis=0.99, min_gt_h=0, max_gt_h=10e10, scale_factor=1):
    """
    Given various configuration settings, determine which ground truths
    are ignored and which are relevant.
    """

    igns = np.zeros([len(gts)], dtype=bool)
    rmvs = np.zeros([len(gts)], dtype=bool)

    for gtind, gt in enumerate(gts):
        ign = gt["ign"]
        ign |= gt["visibility"] < min_gt_vis
        ign |= gt["bbox_full"][3] * scale_factor < min_gt_h
        ign |= gt["bbox_full"][3] * scale_factor > max_gt_h
        ign |= gt["cls"] in ilbls

        # rmv = not gt.cls[0] in (lbls + ilbls)
        rmv = not gt["cls"][0] in (lbls)
        igns[gtind] = ign
        rmvs[gtind] = rmv

    return igns, rmvs


def intersect(box_a, box_b, mode='combinations', data_type=None):
    """
    Computes the amount of intersect between two different sets of boxes.
    Args:
        box_a (nparray): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (nparray): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'combinations' or 'list', where combinations will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        data_type (type): either torch.Tensor or np.ndarray, we automatically determine otherwise
    """

    # determine type
    if data_type is None: data_type = type(box_a)

    # this mode computes the intersect in the sense of combinations.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'combinations':

        # np.ndarray
        if data_type == np.ndarray:
            max_xy = np.minimum(box_a[:, 2:4], np.expand_dims(box_b[:, 2:4], axis=1))
            min_xy = np.maximum(box_a[:, 0:2], np.expand_dims(box_b[:, 0:2], axis=1))
            inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

        # unknown type
        else:
            raise ValueError('type {} is not implemented'.format(data_type))

        return inter[:, :, 0] * inter[:, :, 1]

    # this mode computes the intersect in the sense of list_a vs. list_b.
    # i.e., box_a = M x 4, box_b = M x 4 then the output is Mx1
    elif mode == 'list':

        # torch.Tesnor
        if data_type == torch.Tensor:
            max_xy = torch.min(box_a[:, 2:], box_b[:, 2:])
            min_xy = torch.max(box_a[:, :2], box_b[:, :2])
            inter = torch.clamp((max_xy - min_xy), 0)

        # np.ndarray
        elif data_type == np.ndarray:
            max_xy = np.min(box_a[:, 2:], box_b[:, 2:])
            min_xy = np.max(box_a[:, :2], box_b[:, :2])
            inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))

        return inter[:, 0] * inter[:, 1]

    else:
        raise ValueError('unknown mode {}'.format(mode))


def iou(box_a, box_b, mode='combinations', data_type=None):
    """
    Computes the amount of Intersection over Union (IoU) between two different sets of boxes.
    Args:
        box_a (nparray): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (nparray): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'combinations' or 'list', where combinations will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        data_type (type): either torch.Tensor or np.ndarray, we automatically determine otherwise
    """

    # determine type
    if data_type is None: data_type = type(box_a)

    # this mode computes the IoU in the sense of combinations.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'combinations':

        inter = intersect(box_a, box_b, data_type=data_type)
        area_a = ((box_a[:, 2] - box_a[:, 0]) *
                  (box_a[:, 3] - box_a[:, 1]))
        area_b = ((box_b[:, 2] - box_b[:, 0]) *
                  (box_b[:, 3] - box_b[:, 1]))
        union = np.expand_dims(area_a, 0) + np.expand_dims(area_b, 1) - inter

        # torch.Tensor
        if data_type == torch.Tensor:
            return (inter / union).permute(1, 0)

        # np.ndarray
        elif data_type == np.ndarray:
            return (inter / union).T

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))


    # this mode compares every box in box_a with target in box_b
    # i.e., box_a = M x 4 and box_b = M x 4 then output is M x 1
    elif mode == 'list':

        inter = intersect(box_a, box_b, mode=mode)
        area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
        area_b = (box_b[:, 2] - box_b[:, 0]) * (box_b[:, 3] - box_b[:, 1])
        union = area_a + area_b - inter

        return inter / union

    else:
        raise ValueError('unknown mode {}'.format(mode))


def array_mean(x):
    sum_all = 0
    count_all = 0
    for i in range(len(x)):
        ar = x[i]
        sums = 0
        count = 0
        for j in range(len(ar)):
            sums += ar[j]
            count += 1
        sum_all += sums
        count_all += count
    mean_ar = sum_all / count_all

    return (mean_ar)


def generate_anchors(conf, train_data):
    """
    Generates the anchors according to the configuration and
    (optionally) based on the imdb properties.
    """

    anchors = np.zeros([len(conf.anchor_scales) * len(conf.anchor_ratios), 4], dtype=np.float32)

    aind = 0

    # compute simple anchors based on scale/ratios
    for scale in conf.anchor_scales:

        for ratio in conf.anchor_ratios:
            h = scale
            w = scale * ratio

            anchors[aind, 0:4] = anchor_center(w, h, conf.feat_stride)
            aind += 1
    # print("anchor with 2d initialization", anchors)
    # print(anchors)
    # has 3d? then need to compute stats for each new dimension
    # presuming that anchors are initialized in "2d"
    if conf.has_3d:

        # compute the default stats for each anchor
        normalized_gts = []

        # check all images
        for imind, (imobj) in enumerate(train_data):

            imobj.H = 1216
            imobj.W = 1920
            imobj.scale = 1
            # has ground truths?
            if len(imobj.gts) > 0:

                scale = imobj.scale * conf.test_scale / imobj.H

                # determine ignores
                igns, rmvs = determine_ignores(imobj.gts, conf.lbls, conf.ilbls, conf.min_gt_vis,
                                               conf.min_gt_h, np.inf, scale)

                # accumulate boxes
                scaled_gts = np.empty((len(imobj.gts), 4))
                j = 0
                while (j < (len(imobj.gts))):
                    for gt in imobj.gts:
                        for i in range(4):
                            scaled_gts[j, i] = gt.bbox_full[i] * scale
                        j = j + 1
                #    print(np.shape(scaled_gts))
                gts_all = bbXYWH2Coords(scaled_gts)
                # gts_all = bbXYWH2Coords(np.array([(element * scale for element in gt.bbox_full) for gt in imobj.gts]))
                gts_val = gts_all[(rmvs == False) & (igns == False), :]

                # print("gts_val", gts_val)
                gts_3d = np.array([gt.bbox_3d for gt in imobj.gts])
                gts_3d = gts_3d[(rmvs == False) & (igns == False), :]
                # print("gts_3d",gts_3d)
                if gts_val.shape[0] > 0:

                    # center all 2D ground truths
                    for gtind in range(0, gts_val.shape[0]):
                        w = gts_val[gtind, 2] - gts_val[gtind, 0] + 1
                        h = gts_val[gtind, 3] - gts_val[gtind, 1] + 1

                        gts_val[gtind, 0:4] = anchor_center(w, h, conf.feat_stride)

                if gts_val.shape[0] > 0:
                    normalized_gts += np.concatenate((gts_val, gts_3d), axis=1).tolist()

        # convert to np
        normalized_gts = np.array(normalized_gts)
        # print("normalized_gt_shape",np.shape(normalized_gts))
        # expand dimensions
        anchors = np.concatenate((anchors, np.zeros([anchors.shape[0], 5])), axis=1)
        # print("anchors",anchors)
        # bbox_3d order --> [cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY]
        anchors_z3d = [[] for x in range(anchors.shape[0])]
        anchors_w3d = [[] for x in range(anchors.shape[0])]
        anchors_h3d = [[] for x in range(anchors.shape[0])]
        anchors_l3d = [[] for x in range(anchors.shape[0])]
        anchors_rotY = [[] for x in range(anchors.shape[0])]

        # find best matches for each ground truth
        ols = iou(anchors[:, 0:4], normalized_gts[:, 0:4])
        gt_target_ols = np.amax(ols, axis=0)
        gt_target_anchor = np.argmax(ols, axis=0)

        # assign each box to an anchor
        for gtind, gt in enumerate(normalized_gts):

            anum = gt_target_anchor[gtind]

            if gt_target_ols[gtind] > 0.2:
                anchors_z3d[anum].append(gt[6])
                anchors_w3d[anum].append(gt[7])
                anchors_h3d[anum].append(gt[8])
                anchors_l3d[anum].append(gt[9])
                anchors_rotY[anum].append(gt[10])

        # compute global means
        anchors_z3d_gl = np.empty(0)
        anchors_w3d_gl = np.empty(0)
        anchors_h3d_gl = np.empty(0)
        anchors_l3d_gl = np.empty(0)
        anchors_rotY_gl = np.empty(0)

        # update anchors
        for aind in range(0, anchors.shape[0]):

            if len(np.array(anchors_z3d[aind])) > 0:

                if conf.has_3d:
                    anchors_z3d_gl = np.hstack((anchors_z3d_gl, np.array(anchors_z3d[aind])))
                    anchors_w3d_gl = np.hstack((anchors_w3d_gl, np.array(anchors_w3d[aind])))
                    anchors_h3d_gl = np.hstack((anchors_h3d_gl, np.array(anchors_h3d[aind])))
                    anchors_l3d_gl = np.hstack((anchors_l3d_gl, np.array(anchors_l3d[aind])))
                    anchors_rotY_gl = np.hstack((anchors_rotY_gl, np.array(anchors_rotY[aind])))

                    anchors[aind, 4] = array_mean(anchors_z3d_gl)
                    anchors[aind, 5] = array_mean(anchors_w3d_gl)
                    anchors[aind, 6] = array_mean(anchors_h3d_gl)
                    anchors[aind, 7] = array_mean(anchors_l3d_gl)
                    anchors[aind, 8] = array_mean(anchors_rotY_gl)

            else:
                raise ValueError('Non-used anchor #{} found'.format(aind))
    cache_folder = r"E:\Rishav_Thesis\Baseline\Monocular_3D_Object_Det\pickle"
    pickle_write(os.path.join(cache_folder, 'anchors.pkl'), anchors)
    conf.anchors = anchors


def anchor_center(w, h, stride):
    """
    Centers an anchor based on a stride and the anchor shape (w, h).
    center ground truths with steps of half stride
    hence box 0 is centered at (7.5, 7.5) rather than (0, 0)
    for a feature stride of 16 px.
    """

    anchor = np.zeros([4], dtype=np.float32)

    anchor[0] = -w / 2 + (stride - 1) / 2
    anchor[1] = -h / 2 + (stride - 1) / 2
    anchor[2] = w / 2 + (stride - 1) / 2
    anchor[3] = h / 2 + (stride - 1) / 2

    return anchor


## Compute statistics

def clsName2Ind(lbls, cls):
    """
    Converts a cls name to an ind
    """
    if cls in lbls:
        return lbls.index(cls) + 1
    elif not cls:
        pass
    else:
        raise ValueError('unknown class')


def locate_anchors(anchors, feat_size, stride, convert_tensor=False):
    """
    Spreads each anchor shape across a feature map of size feat_size spaced by a known stride.
    Args:
        anchors (ndarray): N x 4 array describing [x1, y1, x2, y2] displacements for N anchors
        feat_size (ndarray): the downsampled resolution W x H to spread anchors across
        stride (int): stride of a network
        convert_tensor (bool, optional): whether to return a torch tensor, otherwise ndarray [default=False]
    Returns:
         ndarray: 2D array = [(W x H) x 5] array consisting of [x1, y1, x2, y2, anchor_index]
    """

    # compute rois
    shift_x = np.array(range(0, feat_size[1], 1)) * float(stride)
    shift_y = np.array(range(0, feat_size[0], 1)) * float(stride)
    [shift_x, shift_y] = np.meshgrid(shift_x, shift_y)

    rois = np.expand_dims(anchors[:, 0:4], axis=1)
    shift_x = np.expand_dims(shift_x, axis=0)
    shift_y = np.expand_dims(shift_y, axis=0)

    shift_x1 = shift_x + np.expand_dims(rois[:, :, 0], axis=2)
    shift_y1 = shift_y + np.expand_dims(rois[:, :, 1], axis=2)
    shift_x2 = shift_x + np.expand_dims(rois[:, :, 2], axis=2)
    shift_y2 = shift_y + np.expand_dims(rois[:, :, 3], axis=2)

    # compute anchor tracker
    anchor_tracker = np.zeros(shift_x1.shape, dtype=float)
    for aind in range(0, rois.shape[0]): anchor_tracker[aind, :, :] = aind

    stack_size = feat_size[0] * anchors.shape[0]

    # torch and numpy MAY have different calls for reshaping, although
    # it is not very important which is used as long as it is CONSISTENT
    if convert_tensor:

        # important to unroll according to pytorch
        shift_x1 = torch.from_numpy(shift_x1).view(1, stack_size, feat_size[1])
        shift_y1 = torch.from_numpy(shift_y1).view(1, stack_size, feat_size[1])
        shift_x2 = torch.from_numpy(shift_x2).view(1, stack_size, feat_size[1])
        shift_y2 = torch.from_numpy(shift_y2).view(1, stack_size, feat_size[1])
        anchor_tracker = torch.from_numpy(anchor_tracker).view(1, stack_size, feat_size[1])

        shift_x1.requires_grad = False
        shift_y1.requires_grad = False
        shift_x2.requires_grad = False
        shift_y2.requires_grad = False
        anchor_tracker.requires_grad = False

        shift_x1 = shift_x1.permute(1, 2, 0).contiguous().view(-1, 1)
        shift_y1 = shift_y1.permute(1, 2, 0).contiguous().view(-1, 1)
        shift_x2 = shift_x2.permute(1, 2, 0).contiguous().view(-1, 1)
        shift_y2 = shift_y2.permute(1, 2, 0).contiguous().view(-1, 1)
        anchor_tracker = anchor_tracker.permute(1, 2, 0).contiguous().view(-1, 1)

        rois = torch.cat((shift_x1, shift_y1, shift_x2, shift_y2, anchor_tracker), 1)

    else:

        shift_x1 = shift_x1.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)
        shift_y1 = shift_y1.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)
        shift_x2 = shift_x2.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)
        shift_y2 = shift_y2.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)
        anchor_tracker = anchor_tracker.reshape(1, stack_size, feat_size[1]).reshape(-1, 1)

        rois = np.concatenate((shift_x1, shift_y1, shift_x2, shift_y2, anchor_tracker), 1)

    return rois


def bbox_transform(ex_rois, gt_rois):
    """
    Compute the bbox target transforms in 2D.
    Translations are done as simple difference, whereas others involving
    scaling are done in log space (hence, log(1) = 0, log(0.8) < 0 and
    log(1.2) > 0 which is a good property).
    """

    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * (ex_widths - 1)
    ex_ctr_y = ex_rois[:, 1] + 0.5 * (ex_heights - 1)

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * (gt_widths - 1.0)
    gt_ctr_y = gt_rois[:, 1] + 0.5 * (gt_heights - 1.0)

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    targets = np.vstack((targets_dx, targets_dy, targets_dw, targets_dh)).transpose()

    return targets


def bbox_transform_3d(ex_rois_2d, ex_rois_3d, gt_rois):
    """
    Compute the bbox target transforms in 3D.
    Translations are done as simple difference, whereas others involving
    scaling are done in log space (hence, log(1) = 0, log(0.8) < 0 and
    log(1.2) > 0 which is a good property).
    """

    ex_widths = ex_rois_2d[:, 2] - ex_rois_2d[:, 0] + 1.0
    ex_heights = ex_rois_2d[:, 3] - ex_rois_2d[:, 1] + 1.0
    ex_ctr_x = ex_rois_2d[:, 0] + 0.5 * (ex_widths - 1)
    ex_ctr_y = ex_rois_2d[:, 1] + 0.5 * (ex_heights - 1)

    gt_ctr_x = gt_rois[:, 0]
    gt_ctr_y = gt_rois[:, 1]

    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights

    delta_z = gt_rois[:, 2] - ex_rois_3d[:, 0]
    scale_w = np.log(gt_rois[:, 3] / ex_rois_3d[:, 1])
    scale_h = np.log(gt_rois[:, 4] / ex_rois_3d[:, 2])
    scale_l = np.log(gt_rois[:, 5] / ex_rois_3d[:, 3])
    deltaRotY = gt_rois[:, 6] - ex_rois_3d[:, 4]

    targets = np.vstack((targets_dx, targets_dy, delta_z, scale_w, scale_h, scale_l, deltaRotY)).transpose()
    targets = np.hstack((targets, gt_rois[:, 7:]))

    return targets


def iou_ign(box_a, box_b, mode='combinations', data_type=None):
    """
    Computes the amount of overap of box_b has within box_a, which is handy for dealing with ignore regions.
    Hence, assume that box_b are ignore regions and box_a are anchor boxes, then we may want to know how
    much overlap the anchors have inside of the ignore regions (hence ignore area_b!)
    Args:
        box_a (nparray): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (nparray): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'combinations' or 'list', where combinations will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        data_type (type): either torch.Tensor or np.ndarray, we automatically determine otherwise
    """

    if data_type is None: data_type = type(box_a)

    # this mode computes the IoU in the sense of combinations.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'combinations':

        inter = intersect(box_a, box_b, data_type=data_type)
        area_a = ((box_a[:, 2] - box_a[:, 0]) *
                  (box_a[:, 3] - box_a[:, 1]))
        area_b = ((box_b[:, 2] - box_b[:, 0]) *
                  (box_b[:, 3] - box_b[:, 1]))
        union = np.expand_dims(area_a, 0) + np.expand_dims(area_b, 1) * 0 - inter * 0

        # torch and numpy have different calls for transpose
        if data_type == torch.Tensor:
            return (inter / union).permute(1, 0)
        elif data_type == np.ndarray:
            return (inter / union).T

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))

    else:
        raise ValueError('unknown mode {}'.format(mode))


def compute_targets(gts_val, gts_ign, box_lbls, rois, fg_thresh, ign_thresh, bg_thresh_lo, bg_thresh_hi, best_thresh,
                    gts_3d=None, anchors=[], tracker=[]):
    """
    Computes the bbox targets of a set of rois and a set
    of ground truth boxes, provided various ignore
    settings in configuration
    """

    ols = None
    has_3d = gts_3d is not None

    # init transforms which respectively hold [dx, dy, dw, dh, label]
    # for labels bg=-1, ign=0, fg>=1
    transforms = np.zeros([len(rois), 5], dtype=np.float32)
    raw_gt = np.zeros([len(rois), 5], dtype=np.float32)

    # if 3d, then init other terms after
    if has_3d:
        transforms = np.pad(transforms, [(0, 0), (0, gts_3d.shape[1])], 'constant')
        raw_gt = np.pad(raw_gt, [(0, 0), (0, gts_3d.shape[1])], 'constant')

    if gts_val.shape[0] > 0 or gts_ign.shape[0] > 0:
        # rint("gts_val_shape loop triggered")

        if gts_ign.shape[0] > 0:

            # compute overlaps ign
            # print("roi",rois.shape)
            # print("gts", gts_ign.shape)
            ols_ign = iou_ign(rois, gts_ign)
            # print("ols_ign",ols_ign.shape)
            ols_ign_max = np.amax(ols_ign, axis=1)

        else:
            ols_ign_max = np.zeros([rois.shape[0]], dtype=np.float32)

        if gts_val.shape[0] > 0:

            # compute overlaps valid
            ols = iou(rois, gts_val)
            ols_max = np.amax(ols, axis=1)
            targets = np.argmax(ols, axis=1)

            # find best matches for each ground truth
            gt_best_rois = np.argmax(ols, axis=0)
            gt_best_ols = np.amax(ols, axis=0)

            gt_best_rois = gt_best_rois[gt_best_ols >= best_thresh]
            gt_best_ols = gt_best_ols[gt_best_ols >= best_thresh]

            fg_inds = np.flatnonzero(ols_max >= fg_thresh)
            fg_inds = np.concatenate((fg_inds, gt_best_rois))
            fg_inds = np.unique(fg_inds)
            # print(fg_inds)

            target_rois = gts_val[targets[fg_inds], :]
            src_rois = rois[fg_inds, :]
            # print("target_rois",target_rois)
            # print("target_rois",src_rois)
            if len(fg_inds) > 0:

                # compute 2d transform
                transforms[fg_inds, 0:4] = bbox_transform(src_rois, target_rois)
                # print("transforms[fg_inds, 0:4]",transforms[fg_inds, 0:4])
                raw_gt[fg_inds, 0:4] = target_rois

                if has_3d:
                    tracker = tracker.astype(np.int64)
                    # print("tracker",tracker)
                    src_3d = anchors[tracker[fg_inds], 4:]
                    target_3d = gts_3d[targets[fg_inds]]

                    raw_gt[fg_inds, 5:] = target_3d

                    # compute 3d transform
                    transforms[fg_inds, 5:] = bbox_transform_3d(src_rois, src_3d, target_3d)
                    # print("transforms[fg_inds, 5:]",transforms[fg_inds, 5:])

                # store labels
                transforms[fg_inds, 4] = [box_lbls[x] for x in targets[fg_inds]]
                assert (all(transforms[fg_inds, 4] >= 1))
                # print("transforms[:,4]",transforms[:,4])

        else:

            ols_max = np.zeros(rois.shape[0], dtype=int)
            fg_inds = np.empty(shape=[0])
            gt_best_rois = np.empty(shape=[0])

        # determine ignores
        ign_inds = np.flatnonzero(ols_ign_max >= ign_thresh)

        # determine background
        bg_inds = np.flatnonzero((ols_max >= bg_thresh_lo) & (ols_max < bg_thresh_hi))

        # subtract fg and igns from background
        bg_inds = np.setdiff1d(bg_inds, ign_inds)
        bg_inds = np.setdiff1d(bg_inds, fg_inds)
        bg_inds = np.setdiff1d(bg_inds, gt_best_rois)

        # mark background
        transforms[bg_inds, 4] = -1
    else:

        # all background
        transforms[:, 4] = -1

    return transforms, ols, raw_gt


def calc_output_size(res, stride):
    """
    Approximate the output size of a network
    Args:
        res (ndarray): input resolution
        stride (int): stride of a network
    Returns:
         ndarray: output resolution
    """

    return np.ceil(np.array(res) / stride).astype(int)


def compute_bbox_stats(conf, imdb):
    """
    Computes the mean and standard deviation for each regression
    parameter (usually pertaining to [dx, dy, sw, sh] but sometimes
    for 3d parameters too).
    Once these stats are known we normalize the regression targets
    to have 0 mean and 1 variance, to hypothetically ease training.
    """

    if conf.has_3d:
        squared_sums = np.zeros([1, 11], dtype=np.float64)
        sums = np.zeros([1, 11], dtype=np.float64)
    else:
        squared_sums = np.zeros([1, 4], dtype=np.float64)
        sums = np.zeros([1, 4], dtype=np.float64)

    class_counts = np.zeros([1], dtype=np.float64) + 1e-10

    # compute the mean first
    # logging.info('Computing bbox regression mean..')

    for imind, imobj in enumerate(train_data):
        imobj.imH = 1216
        imobj.imW = 1920
        imobj.scale = 1
        # has ground truths?
        if len(imobj.gts) > 0:
            scale_factor = imobj.scale * conf.test_scale / imobj.imH
            # print(scale_factor)
            feat_size = calc_output_size(np.array([imobj.imH, imobj.imW]) * scale_factor, conf.feat_stride)
            rois = locate_anchors(conf.anchors, feat_size, conf.feat_stride)
            # print(rois.shape)
            # print("rois",rois)
            # determine ignores
            igns, rmvs = determine_ignores(imobj.gts, conf.lbls, conf.ilbls, conf.min_gt_vis,
                                           conf.min_gt_h, np.inf, scale_factor)

            # accumulate boxes
            scaled_gts = np.empty((len(imobj.gts), 4))
            j = 0
            while (j < (len(imobj.gts))):
                for gt in imobj.gts:
                    for i in range(4):
                        scaled_gts[j, i] = gt.bbox_full[i] * scale_factor
                    j = j + 1
            # print(np.shape(scaled_gts))
            gts_all = bbXYWH2Coords(scaled_gts)
            # print("gts_allshape", gts_all.shape)

            # filter out irrelevant cls, and ignore cls
            gts_val = gts_all[(rmvs == False) & (igns == False), :]
            gts_ign = gts_all[(rmvs == False) & (igns == True), :]

            # print("gts_val_shape", gts_val.shape)
            # print("gts_ign_shape", gts_ign.shape)
            # accumulate labels
            box_lbls = np.array([gt.cls[0] for gt in imobj.gts])
            box_lbls = box_lbls[(rmvs == False) & (igns == False)]

            box_lbls = np.array([clsName2Ind(conf.lbls, cls) for cls in box_lbls])
            # print(box_lbls)
            # print("box_lbls_shape", box_lbls.shape)

            # print("gts_val",gts_val)
            # print("box_lbls",box_lbls)
            if conf.has_3d:

                # accumulate 3d boxes
                gts_3d = np.array([gt.bbox_3d for gt in imobj.gts])
                gts_3d = gts_3d[(rmvs == False) & (igns == False), :]

                # rescale centers (in 2d)
                for gtind, gt in enumerate(gts_3d):
                    gts_3d[gtind, 0:2] *= scale_factor
                # print("gts_3d",gts_3d.shape)
                # compute transforms for all 3d
                transforms, _, _ = compute_targets(gts_val, gts_ign, box_lbls, rois, conf.fg_thresh, conf.ign_thresh,
                                                   conf.bg_thresh_lo, conf.bg_thresh_hi, conf.best_thresh,
                                                   gts_3d=gts_3d,
                                                   anchors=conf.anchors, tracker=rois[:, 4])
            else:

                # compute transforms for 2d
                transforms, _, _ = compute_targets(gts_val, gts_ign, box_lbls, rois, conf.fg_thresh, conf.ign_thresh,
                                                   conf.bg_thresh_lo, conf.bg_thresh_hi, conf.best_thresh)

            # = (np.isnan(transforms))
            # for i in range(transforms.shape[0]):
            #    for j in range(transforms.shape[1]):
            #        if (math.isnan(transforms[i,j])):
            #            print("transforms",(i,j))
            #        else:
            #            pass
            gt_inds = np.flatnonzero(transforms[:, 4] > 0)

            if len(gt_inds) > 0:

                if conf.has_3d:

                    sums[:, 0:4] += np.sum(transforms[gt_inds, 0:4], axis=0)
                    sums[:, 4:] += np.sum(transforms[gt_inds, 5:12], axis=0)
                else:
                    sums += np.sum(transforms[gt_inds, 0:4], axis=0)

                class_counts += len(gt_inds)
                # print("sums",sums)
    means = sums / class_counts

    # logging.info('Computing bbox regression stds..')

    for imind, imobj in enumerate(train_data):
        imobj.imH = 1216
        imobj.imW = 1920
        imobj.scale = 1
        # has ground truths?
        if len(imobj.gts) > 0:
            scale_factor = imobj.scale * conf.test_scale / imobj.imH
            feat_size = calc_output_size(np.array([imobj.imH, imobj.imW]) * scale_factor, conf.feat_stride)
            rois = locate_anchors(conf.anchors, feat_size, conf.feat_stride)

            # determine ignores
            igns, rmvs = determine_ignores(imobj.gts, conf.lbls, conf.ilbls, conf.min_gt_vis,
                                           conf.min_gt_h, np.inf, scale_factor)

            # accumulate boxes
            scaled_gts = np.empty((len(imobj.gts), 4))
            j = 0
            while (j < (len(imobj.gts))):
                for gt in imobj.gts:
                    for i in range(4):
                        scaled_gts[j, i] = gt.bbox_full[i] * scale_factor
                    j = j + 1
            # print(np.shape(scaled_gts))
            gts_all = bbXYWH2Coords(scaled_gts)

            # filter out irrelevant cls, and ignore cls
            gts_val = gts_all[(rmvs == False) & (igns == False), :]
            gts_ign = gts_all[(rmvs == False) & (igns == True), :]

            # accumulate labels
            box_lbls = np.array([gt.cls[0] for gt in imobj.gts])
            box_lbls = box_lbls[(rmvs == False) & (igns == False)]
            box_lbls = np.array([clsName2Ind(conf.lbls, cls) for cls in box_lbls])

            if conf.has_3d:

                # accumulate 3d boxes
                gts_3d = np.array([gt.bbox_3d for gt in imobj.gts])
                gts_3d = gts_3d[(rmvs == False) & (igns == False), :]

                # rescale centers (in 2d)
                for gtind, gt in enumerate(gts_3d):
                    gts_3d[gtind, 0:2] *= scale_factor

                # compute transforms for all 3d
                transforms, _, _ = compute_targets(gts_val, gts_ign, box_lbls, rois, conf.fg_thresh, conf.ign_thresh,
                                                   conf.bg_thresh_lo, conf.bg_thresh_hi, conf.best_thresh,
                                                   gts_3d=gts_3d,
                                                   anchors=conf.anchors, tracker=rois[:, 4])

            else:

                # compute transforms for 2d
                transforms, _, _ = compute_targets(gts_val, gts_ign, box_lbls, rois, conf.fg_thresh, conf.ign_thresh,
                                                   conf.bg_thresh_lo, conf.bg_thresh_hi, conf.best_thresh)

            gt_inds = np.flatnonzero(transforms[:, 4] > 0)

            if len(gt_inds) > 0:

                if conf.has_3d:

                    squared_sums[:, 0:4] += np.sum(np.power(transforms[gt_inds, 0:4] - means[:, 0:4], 2), axis=0)
                    squared_sums[:, 4:] += np.sum(np.power(transforms[gt_inds, 5:12] - means[:, 4:], 2), axis=0)

                else:
                    squared_sums += np.sum(np.power(transforms[gt_inds, 0:4] - means, 2), axis=0)

        stds = np.sqrt((squared_sums / class_counts))

        means = means.astype(float)
        stds = stds.astype(float)
    cache_folder = r"E:\Rishav_Thesis\Baseline\Monocular_3D_Object_Det\pickle"
    pickle_write(os.path.join(cache_folder, 'bbox_means.pkl'), means)
    pickle_write(os.path.join(cache_folder, 'bbox_stds.pkl'), stds)
    conf.bbox_means = means
    conf.bbox_stds = stds


## M3D RPN Model

def dilate_layer(layer, val):
    layer.dilation = val
    layer.padding = val


class LocalConv2d(nn.Module):

    def __init__(self, num_rows, num_feats_in, num_feats_out, kernel=1, padding=0):
        super(LocalConv2d, self).__init__()

        self.num_rows = num_rows
        self.out_channels = num_feats_out
        self.kernel = kernel
        self.pad = padding

        self.group_conv = nn.Conv2d(num_feats_in * num_rows, num_feats_out * num_rows, kernel, stride=1,
                                    groups=num_rows)

    def forward(self, x):
        b, c, h, w = x.size()

        if self.pad: x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='constant', value=0)

        t = int(h / self.num_rows)

        # unfold by rows
        x = x.unfold(2, t + self.pad * 2, t)
        x = x.permute([0, 2, 1, 4, 3]).contiguous()
        x = x.view(b, c * self.num_rows, t + self.pad * 2, (w + self.pad * 2)).contiguous()

        # group convolution for efficient parallel processing
        y = self.group_conv(x)
        y = y.view(b, self.num_rows, self.out_channels, t, w).contiguous()
        y = y.permute([0, 2, 1, 3, 4]).contiguous()
        y = y.view(b, self.out_channels, h, w)

        return y


def flatten_tensor(input):
    """
    Flattens and permutes a tensor from size
    [B x C x W x H] --> [B x (W x H) x C]
    """

    bsize = input.shape[0]
    csize = input.shape[1]

    return input.permute(0, 2, 3, 1).contiguous().view(bsize, -1, csize)


def unflatten_tensor(input, feat_size, anchors):
    """
    Un-flattens and un-permutes a tensor from size
    [B x (W x H) x C] --> [B x C x W x H]
    """

    bsize = input.shape[0]

    if len(input.shape) >= 3:
        csize = input.shape[2]
    else:
        csize = 1

    input = input.view(bsize, feat_size[0] * anchors.shape[0], feat_size[1], csize)
    input = input.permute(0, 3, 1, 2).contiguous()

    return input


class RPN(nn.Module):

    def __init__(self, phase, base, conf, base_features):
        super(RPN, self).__init__()

        self.base = base
        self.base_features= base_features

        # settings
        self.phase = phase
        self.num_classes = len(conf['lbls']) + 1
        self.num_anchors = conf['anchors'].shape[0]

        self.num_rows = int(min(conf.bins, calc_output_size(conf.test_scale, conf.feat_stride)))
        print(self.num_rows)
        self.prop_feats = nn.Sequential(
            nn.Conv2d(self.base_features[-1], 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        #print('xx',self.prop_feats[0].out_channels)
        # outputs
        self.cls = nn.Conv2d(self.prop_feats[0].out_channels, self.num_classes * self.num_anchors, 1, )

        # bbox 2d
        self.bbox_x = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_y = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_w = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_h = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)

        # bbox 3d
        self.bbox_x3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_y3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_z3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_w3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_h3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_l3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_rY3d = nn.Conv2d(self.prop_feats[0].out_channels, self.num_anchors, 1)

        self.prop_feats_loc = nn.Sequential(
            LocalConv2d(self.num_rows, self.base_features[-1], 512, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # outputs
        self.cls_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_classes * self.num_anchors,
                                   1, )

        # bbox 2d
        self.bbox_x_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_y_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_w_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_h_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)

        # bbox 3d
        self.bbox_x3d_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_y3d_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_z3d_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_w3d_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_h3d_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_l3d_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)
        self.bbox_rY3d_loc = LocalConv2d(self.num_rows, self.prop_feats[0].out_channels, self.num_anchors, 1)

        self.cls_ble = nn.Parameter(torch.tensor(10e-5).type(torch.FloatTensor))

        self.bbox_x_ble = nn.Parameter(torch.tensor(10e-5).type(torch.FloatTensor))
        self.bbox_y_ble = nn.Parameter(torch.tensor(10e-5).type(torch.FloatTensor))
        self.bbox_w_ble = nn.Parameter(torch.tensor(10e-5).type(torch.FloatTensor))
        self.bbox_h_ble = nn.Parameter(torch.tensor(10e-5).type(torch.FloatTensor))

        self.bbox_x3d_ble = nn.Parameter(torch.tensor(10e-5).type(torch.FloatTensor))
        self.bbox_y3d_ble = nn.Parameter(torch.tensor(10e-5).type(torch.FloatTensor))
        self.bbox_z3d_ble = nn.Parameter(torch.tensor(10e-5).type(torch.FloatTensor))
        self.bbox_w3d_ble = nn.Parameter(torch.tensor(10e-5).type(torch.FloatTensor))
        self.bbox_h3d_ble = nn.Parameter(torch.tensor(10e-5).type(torch.FloatTensor))
        self.bbox_l3d_ble = nn.Parameter(torch.tensor(10e-5).type(torch.FloatTensor))
        self.bbox_rY3d_ble = nn.Parameter(torch.tensor(10e-5).type(torch.FloatTensor))

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)

        self.feat_stride = conf.feat_stride
        self.feat_size = calc_output_size(np.array(conf.crop_size), self.feat_stride)
        self.rois = locate_anchors(conf.anchors, self.feat_size, conf.feat_stride, convert_tensor=True)
        self.rois = self.rois.type(torch.cuda.FloatTensor)

        self.anchors = conf.anchors
        self.bbox_means = conf.bbox_means
        self.bbox_stds = conf.bbox_stds

    def forward(self, x):

        batch_size = x.size(0)

        # x input is the bottleneck input to m3d-rpn
        #print('rpn_input_size', x.size())
        prop_feats = self.prop_feats(x)
        prop_feats_loc = self.prop_feats_loc(x)

        cls = self.cls(prop_feats)

        # bbox 2d
        bbox_x = self.bbox_x(prop_feats)
        bbox_y = self.bbox_y(prop_feats)
        bbox_w = self.bbox_w(prop_feats)
        bbox_h = self.bbox_h(prop_feats)

        # bbox 3d
        bbox_x3d = self.bbox_x3d(prop_feats)
        bbox_y3d = self.bbox_y3d(prop_feats)
        bbox_z3d = self.bbox_z3d(prop_feats)
        bbox_w3d = self.bbox_w3d(prop_feats)
        bbox_h3d = self.bbox_h3d(prop_feats)
        bbox_l3d = self.bbox_l3d(prop_feats)
        bbox_rY3d = self.bbox_rY3d(prop_feats)

        cls_loc = self.cls_loc(prop_feats_loc)

        # bbox 2d
        bbox_x_loc = self.bbox_x_loc(prop_feats_loc)
        bbox_y_loc = self.bbox_y_loc(prop_feats_loc)
        bbox_w_loc = self.bbox_w_loc(prop_feats_loc)
        bbox_h_loc = self.bbox_h_loc(prop_feats_loc)

        # bbox 3d
        bbox_x3d_loc = self.bbox_x3d_loc(prop_feats_loc)
        bbox_y3d_loc = self.bbox_y3d_loc(prop_feats_loc)
        bbox_z3d_loc = self.bbox_z3d_loc(prop_feats_loc)
        bbox_w3d_loc = self.bbox_w3d_loc(prop_feats_loc)
        bbox_h3d_loc = self.bbox_h3d_loc(prop_feats_loc)
        bbox_l3d_loc = self.bbox_l3d_loc(prop_feats_loc)
        bbox_rY3d_loc = self.bbox_rY3d_loc(prop_feats_loc)

        cls_ble = self.sigmoid(self.cls_ble)

        # bbox 2d
        bbox_x_ble = self.sigmoid(self.bbox_x_ble)
        bbox_y_ble = self.sigmoid(self.bbox_y_ble)
        bbox_w_ble = self.sigmoid(self.bbox_w_ble)
        bbox_h_ble = self.sigmoid(self.bbox_h_ble)

        # bbox 3d
        bbox_x3d_ble = self.sigmoid(self.bbox_x3d_ble)
        bbox_y3d_ble = self.sigmoid(self.bbox_y3d_ble)
        bbox_z3d_ble = self.sigmoid(self.bbox_z3d_ble)
        bbox_w3d_ble = self.sigmoid(self.bbox_w3d_ble)
        bbox_h3d_ble = self.sigmoid(self.bbox_h3d_ble)
        bbox_l3d_ble = self.sigmoid(self.bbox_l3d_ble)
        bbox_rY3d_ble = self.sigmoid(self.bbox_rY3d_ble)

        # blend
        cls = (cls * cls_ble) + (cls_loc * (1 - cls_ble))

        bbox_x = (bbox_x * bbox_x_ble) + (bbox_x_loc * (1 - bbox_x_ble))
        bbox_y = (bbox_y * bbox_y_ble) + (bbox_y_loc * (1 - bbox_y_ble))
        bbox_w = (bbox_w * bbox_w_ble) + (bbox_w_loc * (1 - bbox_w_ble))
        bbox_h = (bbox_h * bbox_h_ble) + (bbox_h_loc * (1 - bbox_h_ble))

        bbox_x3d = (bbox_x3d * bbox_x3d_ble) + (bbox_x3d_loc * (1 - bbox_x3d_ble))
        bbox_y3d = (bbox_y3d * bbox_y3d_ble) + (bbox_y3d_loc * (1 - bbox_y3d_ble))
        bbox_z3d = (bbox_z3d * bbox_z3d_ble) + (bbox_z3d_loc * (1 - bbox_z3d_ble))
        bbox_h3d = (bbox_h3d * bbox_h3d_ble) + (bbox_h3d_loc * (1 - bbox_h3d_ble))
        bbox_w3d = (bbox_w3d * bbox_w3d_ble) + (bbox_w3d_loc * (1 - bbox_w3d_ble))
        bbox_l3d = (bbox_l3d * bbox_l3d_ble) + (bbox_l3d_loc * (1 - bbox_l3d_ble))
        bbox_rY3d = (bbox_rY3d * bbox_rY3d_ble) + (bbox_rY3d_loc * (1 - bbox_rY3d_ble))

        feat_h = cls.size(2)
        feat_w = cls.size(3)

        # reshape for cross entropy
        cls = cls.view(batch_size, self.num_classes, feat_h * self.num_anchors, feat_w)

        # score probabilities
        prob = self.softmax(cls)

        # reshape for consistency
        bbox_x = flatten_tensor(bbox_x.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_y = flatten_tensor(bbox_y.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_w = flatten_tensor(bbox_w.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_h = flatten_tensor(bbox_h.view(batch_size, 1, feat_h * self.num_anchors, feat_w))

        bbox_x3d = flatten_tensor(bbox_x3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_y3d = flatten_tensor(bbox_y3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_z3d = flatten_tensor(bbox_z3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_w3d = flatten_tensor(bbox_w3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_h3d = flatten_tensor(bbox_h3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_l3d = flatten_tensor(bbox_l3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))
        bbox_rY3d = flatten_tensor(bbox_rY3d.view(batch_size, 1, feat_h * self.num_anchors, feat_w))

        # bundle
        bbox_2d = torch.cat((bbox_x, bbox_y, bbox_w, bbox_h), dim=2)
        bbox_3d = torch.cat((bbox_x3d, bbox_y3d, bbox_z3d, bbox_w3d, bbox_h3d, bbox_l3d, bbox_rY3d), dim=2)

        feat_size = [feat_h, feat_w]
        feat_size= torch.tensor(np.array(feat_size))
        cls = flatten_tensor(cls)
        prob = flatten_tensor(prob)

        if self.feat_size[0] != feat_h or self.feat_size[1] != feat_w:
            self.feat_size = [feat_h, feat_w]
            self.rois = locate_anchors(self.anchors, self.feat_size, self.feat_stride, convert_tensor=True)
            self.rois = self.rois.type(torch.cuda.FloatTensor)
            # self.rois = self.rois.type(torch.FloatTensor)
        if self.training:
            return cls, prob, bbox_2d, bbox_3d, feat_size

        else:
            return cls, prob, bbox_2d, bbox_3d, feat_size, self.rois


## Loss and Auxillary

def bbox_transform_inv(boxes, deltas, means=None, stds=None):
    """
    Compute the bbox target transforms in 3D.
    Translations are done as simple difference, whereas others involving
    scaling are done in log space (hence, log(1) = 0, log(0.8) < 0 and
    log(1.2) > 0 which is a good property).
    """

    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    # boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    if stds is not None:
        dx *= stds[0]
        dy *= stds[1]
        dw *= stds[2]
        dh *= stds[3]

    if means is not None:
        dx += means[0]
        dy += means[1]
        dw += means[2]
        dh += means[3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights

    pred_boxes = torch.zeros(deltas.shape)

    # x1, y1, x2, y2
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


class RPN_3D_loss(nn.Module):

    def __init__(self, conf):

        super(RPN_3D_loss, self).__init__()

        self.num_classes = len(conf.lbls) + 1
        self.num_anchors = conf.anchors.shape[0]
        self.anchors = conf.anchors
        self.bbox_means = conf.bbox_means
        self.bbox_stds = conf.bbox_stds
        self.feat_stride = conf.feat_stride
        self.fg_fraction = conf.fg_fraction
        self.box_samples = conf.box_samples
        self.ign_thresh = conf.ign_thresh
        self.nms_thres = conf.nms_thres
        self.fg_thresh = conf.fg_thresh
        self.bg_thresh_lo = conf.bg_thresh_lo
        self.bg_thresh_hi = conf.bg_thresh_hi
        self.best_thresh = conf.best_thresh
        self.hard_negatives = conf.hard_negatives
        self.focal_loss = conf.focal_loss

        self.crop_size = conf.crop_size

        self.cls_2d_lambda = conf.cls_2d_lambda
        self.iou_2d_lambda = conf.iou_2d_lambda
        self.bbox_2d_lambda = conf.bbox_2d_lambda
        self.bbox_3d_lambda = conf.bbox_3d_lambda
        self.bbox_3d_proj_lambda = conf.bbox_3d_proj_lambda

        self.lbls = conf.lbls
        self.ilbls = conf.ilbls

        self.min_gt_vis = conf.min_gt_vis
        self.min_gt_h = conf.min_gt_h
        self.max_gt_h = conf.max_gt_h
        self.device = 'cuda:2'

    def forward(self, cls, prob, bbox_2d, bbox_3d, imobjs, feat_size):

        stats = []

        loss = torch.tensor(0, device=self.device).type(torch.cuda.FloatTensor)

        FG_ENC = 1000
        BG_ENC = 2000

        IGN_FLAG = 3000

        batch_size = cls.shape[0]

        # prob_detach = prob.cpu().detach().numpy()
        prob_detach = prob.cpu().detach().numpy()

        bbox_x = bbox_2d[:, :, 0]
        bbox_y = bbox_2d[:, :, 1]
        bbox_w = bbox_2d[:, :, 2]
        bbox_h = bbox_2d[:, :, 3]

        bbox_x3d = bbox_3d[:, :, 0]
        bbox_y3d = bbox_3d[:, :, 1]
        bbox_z3d = bbox_3d[:, :, 2]
        bbox_w3d = bbox_3d[:, :, 3]
        bbox_h3d = bbox_3d[:, :, 4]
        bbox_l3d = bbox_3d[:, :, 5]
        bbox_ry3d = bbox_3d[:, :, 6]

        bbox_x3d_proj = torch.zeros(bbox_x3d.shape, device=self.device)
        bbox_y3d_proj = torch.zeros(bbox_x3d.shape, device=self.device)
        bbox_z3d_proj = torch.zeros(bbox_x3d.shape, device=self.device)

        labels = np.zeros(cls.shape[0:2])
        labels_weight = np.zeros(cls.shape[0:2])

        labels_scores = np.zeros(cls.shape[0:2])

        bbox_x_tar = np.zeros(cls.shape[0:2])
        bbox_y_tar = np.zeros(cls.shape[0:2])
        bbox_w_tar = np.zeros(cls.shape[0:2])
        bbox_h_tar = np.zeros(cls.shape[0:2])

        bbox_x3d_tar = np.zeros(cls.shape[0:2])
        bbox_y3d_tar = np.zeros(cls.shape[0:2])
        bbox_z3d_tar = np.zeros(cls.shape[0:2])
        bbox_w3d_tar = np.zeros(cls.shape[0:2])
        bbox_h3d_tar = np.zeros(cls.shape[0:2])
        bbox_l3d_tar = np.zeros(cls.shape[0:2])
        bbox_ry3d_tar = np.zeros(cls.shape[0:2])

        bbox_x3d_proj_tar = np.zeros(cls.shape[0:2])
        bbox_y3d_proj_tar = np.zeros(cls.shape[0:2])
        bbox_z3d_proj_tar = np.zeros(cls.shape[0:2])

        bbox_weights = np.zeros(cls.shape[0:2])

        ious_2d = torch.zeros(cls.shape[0:2])
        ious_3d = torch.zeros(cls.shape[0:2])
        coords_abs_z = torch.zeros(cls.shape[0:2], device=self.device)
        coords_abs_ry = torch.zeros(cls.shape[0:2], device=self.device)

        # get all rois
        rois = locate_anchors(self.anchors, feat_size, self.feat_stride, convert_tensor=True)
        rois = torch.tensor(rois, device=self.device).type(torch.cuda.FloatTensor)

        bbox_x3d_dn = bbox_x3d * self.bbox_stds[:, 4][0] + self.bbox_means[:, 4][0]
        bbox_y3d_dn = bbox_y3d * self.bbox_stds[:, 5][0] + self.bbox_means[:, 5][0]
        bbox_z3d_dn = bbox_z3d * self.bbox_stds[:, 6][0] + self.bbox_means[:, 6][0]
        bbox_w3d_dn = bbox_w3d * self.bbox_stds[:, 7][0] + self.bbox_means[:, 7][0]
        bbox_h3d_dn = bbox_h3d * self.bbox_stds[:, 8][0] + self.bbox_means[:, 8][0]
        bbox_l3d_dn = bbox_l3d * self.bbox_stds[:, 9][0] + self.bbox_means[:, 9][0]
        bbox_ry3d_dn = bbox_ry3d * self.bbox_stds[:, 10][0] + self.bbox_means[:, 10][0]

        src_anchors = self.anchors[rois[:, 4].type(torch.cuda.LongTensor).cpu(), :]

        src_anchors = torch.tensor(src_anchors, device=self.device, requires_grad=False).type(torch.cuda.FloatTensor)
        if len(src_anchors.shape) == 1: src_anchors = src_anchors.unsqueeze(0)

        # compute 3d transform
        widths = rois[:, 2] - rois[:, 0] + 1.0
        heights = rois[:, 3] - rois[:, 1] + 1.0
        ctr_x = rois[:, 0] + 0.5 * widths
        ctr_y = rois[:, 1] + 0.5 * heights

        bbox_x3d_dn = bbox_x3d_dn * widths.unsqueeze(0) + ctr_x.unsqueeze(0)
        bbox_y3d_dn = bbox_y3d_dn * heights.unsqueeze(0) + ctr_y.unsqueeze(0)
        bbox_z3d_dn = src_anchors[:, 4].unsqueeze(0) + bbox_z3d_dn
        bbox_w3d_dn = torch.exp(bbox_w3d_dn) * src_anchors[:, 5].unsqueeze(0)
        bbox_h3d_dn = torch.exp(bbox_h3d_dn) * src_anchors[:, 6].unsqueeze(0)
        bbox_l3d_dn = torch.exp(bbox_l3d_dn) * src_anchors[:, 7].unsqueeze(0)
        bbox_ry3d_dn = src_anchors[:, 8].unsqueeze(0) + bbox_ry3d_dn

        for bind in range(0, batch_size):

            # imobj = imobjs[bind]
            imobj = imobjs
            gts = imobj.gts
            imobj.imH = 1216
            imobj.imW = 1920
            imobj.scale = 1

            if len(imobj.gts) > 0:
                scale_factor = imobj.scale * conf.test_scale / imobj.imH

                imobj.scale_factor = scale_factor

            p2_inv = torch.tensor(imobj.p2_inv, device=self.device).type(torch.cuda.FloatTensor)

            # filter gts
            igns, rmvs = determine_ignores(gts, self.lbls, self.ilbls, self.min_gt_vis, self.min_gt_h)

            # accumulate boxes

            scaled_gts = np.empty((len(imobj.gts), 4))
            j = 0
            while (j < (len(imobj.gts))):
                for gt in imobj.gts:
                    for i in range(4):
                        scaled_gts[j, i] = gt.bbox_full[i] * scale_factor
                    j = j + 1
            # print(np.shape(scaled_gts))
            gts_all = bbXYWH2Coords(scaled_gts)
            gts_3d = np.array([gt.bbox_3d for gt in gts])

            for gtind, gt in enumerate(gts_3d):
                gts_3d[gtind, 0:2] *= scale_factor

            if not ((rmvs == False) & (igns == False)).any():
                continue

            # filter out irrelevant cls, and ignore cls
            gts_val = gts_all[(rmvs == False) & (igns == False), :]
            gts_ign = gts_all[(rmvs == False) & (igns == True), :]
            gts_3d = gts_3d[(rmvs == False) & (igns == False), :]

            # accumulate labels
            box_lbls = np.array([gt.cls[0] for gt in gts])
            box_lbls = box_lbls[(rmvs == False) & (igns == False)]
            box_lbls = np.array([clsName2Ind(self.lbls, cls) for cls in box_lbls])

            if gts_val.shape[0] > 0 or gts_ign.shape[0] > 0:

                rois = rois.cpu()

                # bbox regression
                transforms, ols, raw_gt = compute_targets(gts_val, gts_ign, box_lbls, rois.numpy(), self.fg_thresh,
                                                          self.ign_thresh, self.bg_thresh_lo, self.bg_thresh_hi,
                                                          self.best_thresh, anchors=self.anchors, gts_3d=gts_3d,
                                                          tracker=rois[:, 4].numpy())

                # normalize 2d
                transforms[:, 0:4] -= self.bbox_means[:, 0:4]
                transforms[:, 0:4] /= self.bbox_stds[:, 0:4]

                # normalize 3d
                transforms[:, 5:12] -= self.bbox_means[:, 4:]
                transforms[:, 5:12] /= self.bbox_stds[:, 4:]

                labels_fg = transforms[:, 4] > 0
                labels_bg = transforms[:, 4] < 0
                labels_ign = transforms[:, 4] == 0

                fg_inds = np.flatnonzero(labels_fg)
                bg_inds = np.flatnonzero(labels_bg)
                ign_inds = np.flatnonzero(labels_ign)
                # print("fg_inds",fg_inds)
                # print("bg_inds",fg_inds)
                # print("ign_inds",fg_inds)
                # transforms = torch.tensor(np.array(transforms), device= self.device)

                labels[bind, fg_inds] = transforms[fg_inds, 4]
                labels[bind, ign_inds] = IGN_FLAG
                labels[bind, bg_inds] = 0

                bbox_x_tar[bind, :] = transforms[:, 0]
                bbox_y_tar[bind, :] = transforms[:, 1]
                bbox_w_tar[bind, :] = transforms[:, 2]
                bbox_h_tar[bind, :] = transforms[:, 3]

                bbox_x3d_tar[bind, :] = transforms[:, 5]
                bbox_y3d_tar[bind, :] = transforms[:, 6]
                bbox_z3d_tar[bind, :] = transforms[:, 7]
                bbox_w3d_tar[bind, :] = transforms[:, 8]
                bbox_h3d_tar[bind, :] = transforms[:, 9]
                bbox_l3d_tar[bind, :] = transforms[:, 10]
                bbox_ry3d_tar[bind, :] = transforms[:, 11]

                bbox_x3d_proj_tar[bind, :] = raw_gt[:, 12]
                bbox_y3d_proj_tar[bind, :] = raw_gt[:, 13]
                bbox_z3d_proj_tar[bind, :] = raw_gt[:, 14]

                # ----------------------------------------
                # box sampling
                # ----------------------------------------

                if self.box_samples == np.inf:
                    fg_num = len(fg_inds)
                    bg_num = len(bg_inds)

                else:
                    fg_num = min(round(rois.shape[0] * self.box_samples * self.fg_fraction), len(fg_inds))
                    bg_num = min(round(rois.shape[0] * self.box_samples - fg_num), len(bg_inds))

                if self.hard_negatives:

                    if fg_num > 0 and fg_num != fg_inds.shape[0]:
                        scores = prob_detach[bind, fg_inds, labels[bind, fg_inds].astype(int)]
                        fg_score_ascend = (scores).argsort()
                        fg_inds = fg_inds[fg_score_ascend]
                        fg_inds = fg_inds[0:fg_num]

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        scores = prob_detach[bind, bg_inds, labels[bind, bg_inds].astype(int)]
                        bg_score_ascend = (scores).argsort()
                        bg_inds = bg_inds[bg_score_ascend]
                        bg_inds = bg_inds[0:bg_num]

                else:

                    if fg_num > 0 and fg_num != fg_inds.shape[0]:
                        fg_inds = np.random.choice(fg_inds, fg_num, replace=False)

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        bg_inds = np.random.choice(bg_inds, bg_num, replace=False)

                labels_weight[bind, bg_inds] = BG_ENC
                labels_weight[bind, fg_inds] = FG_ENC
                bbox_weights[bind, fg_inds] = 1

                # ----------------------------------------
                # compute IoU stats
                # ----------------------------------------

                if fg_num > 0:

                    # compile deltas pred
                    deltas_2d = torch.cat((bbox_x[bind, :, np.newaxis], bbox_y[bind, :, np.newaxis],
                                           bbox_w[bind, :, np.newaxis], bbox_h[bind, :, np.newaxis]), dim=1)

                    # compile deltas targets
                    deltas_2d_tar = np.concatenate((bbox_x_tar[bind, :, np.newaxis], bbox_y_tar[bind, :, np.newaxis],
                                                    bbox_w_tar[bind, :, np.newaxis], bbox_h_tar[bind, :, np.newaxis]),
                                                   axis=1)

                    # move to gpu
                    deltas_2d_tar = torch.tensor(deltas_2d_tar, device=self.device, requires_grad=False).type(
                        torch.cuda.FloatTensor)

                    means = self.bbox_means[0, :]
                    stds = self.bbox_stds[0, :]

                    rois = rois.to(device)

                    coords_2d = bbox_transform_inv(rois, deltas_2d, means=means, stds=stds)
                    coords_2d_tar = bbox_transform_inv(rois, deltas_2d_tar, means=means, stds=stds)

                    ious_2d[bind, fg_inds] = iou(coords_2d[fg_inds, :], coords_2d_tar[fg_inds, :], mode='list')

                    bbox_x3d_dn_fg = bbox_x3d_dn[bind, fg_inds]
                    bbox_y3d_dn_fg = bbox_y3d_dn[bind, fg_inds]

                    src_anchors = self.anchors[rois[fg_inds, 4].type(torch.LongTensor), :]
                    src_anchors = torch.tensor(src_anchors, device=self.device, requires_grad=False).type(
                        torch.cuda.FloatTensor)
                    if len(src_anchors.shape) == 1: src_anchors = src_anchors.unsqueeze(0)

                    bbox_x3d_dn_fg = bbox_x3d_dn[bind, fg_inds]
                    bbox_y3d_dn_fg = bbox_y3d_dn[bind, fg_inds]
                    bbox_z3d_dn_fg = bbox_z3d_dn[bind, fg_inds]
                    bbox_w3d_dn_fg = bbox_w3d_dn[bind, fg_inds]
                    bbox_h3d_dn_fg = bbox_h3d_dn[bind, fg_inds]
                    bbox_l3d_dn_fg = bbox_l3d_dn[bind, fg_inds]
                    bbox_ry3d_dn_fg = bbox_ry3d_dn[bind, fg_inds]

                    # re-scale all 2D back to original
                    bbox_x3d_dn_fg /= imobj['scale_factor']
                    bbox_y3d_dn_fg /= imobj['scale_factor']

                    coords_2d = torch.cat((bbox_x3d_dn_fg[np.newaxis, :] * bbox_z3d_dn_fg[np.newaxis, :],
                                           bbox_y3d_dn_fg[np.newaxis, :] * bbox_z3d_dn_fg[np.newaxis, :],
                                           bbox_z3d_dn_fg[np.newaxis, :]), dim=0)

                    coords_2d = torch.cat((coords_2d, (torch.ones([1, coords_2d.shape[1]]).to(device))), dim=0)
                    # print("coords_2d",coords_2d.size())
                    coords_3d = torch.squeeze((torch.matmul(p2_inv, coords_2d)), dim=0)
                    # print("coords_3d",coords_3d.size())
                    # print("bbox_x3d_proj",bbox_x3d_proj.size())
                    bbox_x3d_proj[bind, fg_inds] = coords_3d[0, :]
                    bbox_y3d_proj[bind, fg_inds] = coords_3d[1, :]
                    bbox_z3d_proj[bind, fg_inds] = coords_3d[2, :]

                    # absolute targets
                    bbox_z3d_dn_tar = bbox_z3d_tar[bind, fg_inds] * self.bbox_stds[:, 6][0] + self.bbox_means[:, 6][0]
                    bbox_z3d_dn_tar = torch.tensor(bbox_z3d_dn_tar, device=self.device, requires_grad=False).type(
                        torch.cuda.FloatTensor)
                    bbox_z3d_dn_tar = src_anchors[:, 4] + bbox_z3d_dn_tar

                    bbox_ry3d_dn_tar = bbox_ry3d_tar[bind, fg_inds] * self.bbox_stds[:, 10][0] + self.bbox_means[:, 10][
                        0]
                    bbox_ry3d_dn_tar = torch.tensor(bbox_ry3d_dn_tar, requires_grad=False, device=self.device).type(
                        torch.cuda.FloatTensor)
                    bbox_ry3d_dn_tar = src_anchors[:, 8] + bbox_ry3d_dn_tar

                    coords_abs_z[bind, fg_inds] = torch.abs(bbox_z3d_dn_tar - bbox_z3d_dn_fg)
                    coords_abs_ry[bind, fg_inds] = torch.abs(bbox_ry3d_dn_tar - bbox_ry3d_dn_fg)

            else:

                bg_inds = np.arange(0, rois.shape[0])

                if self.box_samples == np.inf:
                    bg_num = len(bg_inds)
                else:
                    bg_num = min(round(self.box_samples * (1 - self.fg_fraction)), len(bg_inds))

                if self.hard_negatives:

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        scores = prob_detach[bind, bg_inds, labels[bind, bg_inds].astype(int)]
                        bg_score_ascend = (scores).argsort()
                        bg_inds = bg_inds[bg_score_ascend]
                        bg_inds = bg_inds[0:bg_num]

                else:

                    if bg_num > 0 and bg_num != bg_inds.shape[0]:
                        bg_inds = np.random.choice(bg_inds, bg_num, replace=False)

                labels[bind, :] = 0
                labels_weight[bind, bg_inds] = BG_ENC

            # grab label predictions (for weighing purposes)
            active = labels[bind, :] != IGN_FLAG
            labels_scores[bind, active] = prob_detach[bind, active, labels[bind, active].astype(int)]

        # ----------------------------------------
        # useful statistics
        # ----------------------------------------

        fg_inds_all = np.flatnonzero((labels > 0) & (labels != IGN_FLAG))
        bg_inds_all = np.flatnonzero((labels == 0) & (labels != IGN_FLAG))

        fg_inds_unravel = np.unravel_index(fg_inds_all, prob_detach.shape[0:2])
        bg_inds_unravel = np.unravel_index(bg_inds_all, prob_detach.shape[0:2])

        cls_pred = cls.argmax(dim=2).cpu().detach().numpy()

        if self.cls_2d_lambda and len(fg_inds_all) > 0:
            acc_fg = np.mean(cls_pred[fg_inds_unravel] == labels[fg_inds_unravel])
            stats.append({'name': 'fg', 'val': acc_fg, 'format': '{:0.2f}', 'group': 'acc'})
            # writer.add_scalar("fg_accuracy_train",acc_fg)
        if self.cls_2d_lambda and len(bg_inds_all) > 0:
            acc_bg = np.mean(cls_pred[bg_inds_unravel] == labels[bg_inds_unravel])
            stats.append({'name': 'bg', 'val': acc_bg, 'format': '{:0.2f}', 'group': 'acc'})

        # ----------------------------------------
        # box weighting
        # ----------------------------------------

        fg_inds = np.flatnonzero(labels_weight == FG_ENC)
        bg_inds = np.flatnonzero(labels_weight == BG_ENC)
        active_inds = np.concatenate((fg_inds, bg_inds), axis=0)

        fg_num = len(fg_inds)
        bg_num = len(bg_inds)

        labels_weight[...] = 0.0
        box_samples = fg_num + bg_num

        fg_inds_unravel = np.unravel_index(fg_inds, labels_weight.shape)
        bg_inds_unravel = np.unravel_index(bg_inds, labels_weight.shape)
        active_inds_unravel = np.unravel_index(active_inds, labels_weight.shape)

        labels_weight[active_inds_unravel] = 1.0

        if self.fg_fraction is not None:

            if fg_num > 0:

                fg_weight = (self.fg_fraction / (1 - self.fg_fraction)) * (bg_num / fg_num)
                labels_weight[fg_inds_unravel] = fg_weight
                labels_weight[bg_inds_unravel] = 1.0

            else:
                labels_weight[bg_inds_unravel] = 1.0

        # different method of doing hard negative mining
        # use the scores to normalize the importance of each sample
        # hence, encourages the network to get all "correct" rather than
        # becoming more correct at a decision it is already good at
        # this method is equivelent to the focal loss with additional mean scaling
        if self.focal_loss:

            weights_sum = 0

            # re-weight bg
            if bg_num > 0:
                bg_scores = labels_scores[bg_inds_unravel]
                bg_weights = (1 - bg_scores) ** self.focal_loss
                weights_sum += np.sum(bg_weights)
                labels_weight[bg_inds_unravel] *= bg_weights

            # re-weight fg
            if fg_num > 0:
                fg_scores = labels_scores[fg_inds_unravel]
                fg_weights = (1 - fg_scores) ** self.focal_loss
                weights_sum += np.sum(fg_weights)
                labels_weight[fg_inds_unravel] *= fg_weights

        # ----------------------------------------
        # classification loss
        # ----------------------------------------
        labels = torch.tensor(labels, requires_grad=False, device=self.device)
        labels = labels.view(-1).type(torch.cuda.LongTensor)

        labels_weight = torch.tensor(labels_weight, requires_grad=False, device=self.device)
        labels_weight = labels_weight.view(-1).type(torch.cuda.FloatTensor)

        cls = cls.view(-1, cls.shape[2])

        if self.cls_2d_lambda:

            # cls loss
            active = labels_weight > 0

            if np.any(active.cpu().numpy()):
                loss_cls = F.cross_entropy(cls[active, :], labels[active], reduction='none', ignore_index=IGN_FLAG)
                loss_cls = (loss_cls * labels_weight[active])

                # simple gradient clipping
                loss_cls = loss_cls.clamp(min=0, max=2000)

                # take mean and scale lambda
                loss_cls = loss_cls.mean()
                loss_cls *= self.cls_2d_lambda

                loss += loss_cls

                stats.append({'name': 'cls', 'val': loss_cls, 'format': '{:0.4f}', 'group': 'loss'})

        # ----------------------------------------
        # bbox regression loss
        # ----------------------------------------

        if np.sum(bbox_weights) > 0:

            bbox_weights = torch.tensor(bbox_weights, requires_grad=False, device=self.device).type(
                torch.cuda.FloatTensor).view(-1)
            active = bbox_weights > 0

            if self.bbox_2d_lambda:
                # bbox loss 2d
                bbox_x_tar = torch.tensor(bbox_x_tar, requires_grad=False, device=self.device).type(
                    torch.cuda.FloatTensor).view(-1)
                bbox_y_tar = torch.tensor(bbox_y_tar, requires_grad=False, device=self.device).type(
                    torch.cuda.FloatTensor).view(-1)
                bbox_w_tar = torch.tensor(bbox_w_tar, requires_grad=False, device=self.device).type(
                    torch.cuda.FloatTensor).view(-1)
                bbox_h_tar = torch.tensor(bbox_h_tar, requires_grad=False, device=self.device).type(
                    torch.cuda.FloatTensor).view(-1)

                bbox_x = bbox_x[:, :].unsqueeze(2).view(-1)
                bbox_y = bbox_y[:, :].unsqueeze(2).view(-1)
                bbox_w = bbox_w[:, :].unsqueeze(2).view(-1)
                bbox_h = bbox_h[:, :].unsqueeze(2).view(-1)

                loss_bbox_x = F.smooth_l1_loss(bbox_x[active], bbox_x_tar[active], reduction='none')
                loss_bbox_y = F.smooth_l1_loss(bbox_y[active], bbox_y_tar[active], reduction='none')
                loss_bbox_w = F.smooth_l1_loss(bbox_w[active], bbox_w_tar[active], reduction='none')
                loss_bbox_h = F.smooth_l1_loss(bbox_h[active], bbox_h_tar[active], reduction='none')

                loss_bbox_x = (loss_bbox_x * bbox_weights[active]).mean()
                loss_bbox_y = (loss_bbox_y * bbox_weights[active]).mean()
                loss_bbox_w = (loss_bbox_w * bbox_weights[active]).mean()
                loss_bbox_h = (loss_bbox_h * bbox_weights[active]).mean()

                bbox_2d_loss = (loss_bbox_x + loss_bbox_y + loss_bbox_w + loss_bbox_h)
                bbox_2d_loss *= self.bbox_2d_lambda

                loss += bbox_2d_loss
                stats.append({'name': 'bbox_2d', 'val': bbox_2d_loss, 'format': '{:0.4f}', 'group': 'loss'})

            if self.bbox_3d_lambda:
                # bbox loss 3d
                bbox_x3d_tar = torch.tensor(bbox_x3d_tar, requires_grad=False, device=self.device).type(
                    torch.cuda.FloatTensor).view(-1)
                bbox_y3d_tar = torch.tensor(bbox_y3d_tar, requires_grad=False, device=self.device).type(
                    torch.cuda.FloatTensor).view(-1)
                bbox_z3d_tar = torch.tensor(bbox_z3d_tar, requires_grad=False, device=self.device).type(
                    torch.cuda.FloatTensor).view(-1)
                bbox_w3d_tar = torch.tensor(bbox_w3d_tar, requires_grad=False, device=self.device).type(
                    torch.cuda.FloatTensor).view(-1)
                bbox_h3d_tar = torch.tensor(bbox_h3d_tar, requires_grad=False, device=self.device).type(
                    torch.cuda.FloatTensor).view(-1)
                bbox_l3d_tar = torch.tensor(bbox_l3d_tar, requires_grad=False, device=self.device).type(
                    torch.cuda.FloatTensor).view(-1)
                bbox_ry3d_tar = torch.tensor(bbox_ry3d_tar, requires_grad=False, device=self.device).type(
                    torch.cuda.FloatTensor).view(-1)

                bbox_x3d = bbox_x3d[:, :].view(-1)
                bbox_y3d = bbox_y3d[:, :].view(-1)
                bbox_z3d = bbox_z3d[:, :].view(-1)
                bbox_w3d = bbox_w3d[:, :].view(-1)
                bbox_h3d = bbox_h3d[:, :].view(-1)
                bbox_l3d = bbox_l3d[:, :].view(-1)
                bbox_ry3d = bbox_ry3d[:, :].view(-1)

                loss_bbox_x3d = F.smooth_l1_loss(bbox_x3d[active], bbox_x3d_tar[active], reduction='none')
                loss_bbox_y3d = F.smooth_l1_loss(bbox_y3d[active], bbox_y3d_tar[active], reduction='none')
                loss_bbox_z3d = F.smooth_l1_loss(bbox_z3d[active], bbox_z3d_tar[active], reduction='none')
                loss_bbox_w3d = F.smooth_l1_loss(bbox_w3d[active], bbox_w3d_tar[active], reduction='none')
                loss_bbox_h3d = F.smooth_l1_loss(bbox_h3d[active], bbox_h3d_tar[active], reduction='none')
                loss_bbox_l3d = F.smooth_l1_loss(bbox_l3d[active], bbox_l3d_tar[active], reduction='none')
                loss_bbox_ry3d = F.smooth_l1_loss(bbox_ry3d[active], bbox_ry3d_tar[active], reduction='none')

                loss_bbox_x3d = (loss_bbox_x3d * bbox_weights[active]).mean()
                loss_bbox_y3d = (loss_bbox_y3d * bbox_weights[active]).mean()
                loss_bbox_z3d = (loss_bbox_z3d * bbox_weights[active]).mean()
                loss_bbox_w3d = (loss_bbox_w3d * bbox_weights[active]).mean()
                loss_bbox_h3d = (loss_bbox_h3d * bbox_weights[active]).mean()
                loss_bbox_l3d = (loss_bbox_l3d * bbox_weights[active]).mean()
                loss_bbox_ry3d = (loss_bbox_ry3d * bbox_weights[active]).mean()

                bbox_3d_loss = (loss_bbox_x3d + loss_bbox_y3d + loss_bbox_z3d)
                bbox_3d_loss += (loss_bbox_w3d + loss_bbox_h3d + loss_bbox_l3d + loss_bbox_ry3d)

                bbox_3d_loss *= self.bbox_3d_lambda

                loss += bbox_3d_loss
                stats.append({'name': 'bbox_3d', 'val': bbox_3d_loss, 'format': '{:0.4f}', 'group': 'loss'})

            if self.bbox_3d_proj_lambda:
                # bbox loss 3d
                bbox_x3d_proj_tar = torch.tensor(bbox_x3d_proj_tar, requires_grad=False, device=self.device).type(
                    torch.cuda.FloatTensor).view(-1)
                bbox_y3d_proj_tar = torch.tensor(bbox_y3d_proj_tar, requires_grad=False, device=self.device).type(
                    torch.cuda.FloatTensor).view(-1)
                bbox_z3d_proj_tar = torch.tensor(bbox_z3d_proj_tar, requires_grad=False, device=self.device).type(
                    torch.cuda.FloatTensor).view(-1)

                bbox_x3d_proj = bbox_x3d_proj[:, :].view(-1)
                bbox_y3d_proj = bbox_y3d_proj[:, :].view(-1)
                bbox_z3d_proj = bbox_z3d_proj[:, :].view(-1)

                loss_bbox_x3d_proj = F.smooth_l1_loss(bbox_x3d_proj[active], bbox_x3d_proj_tar[active],
                                                      reduction='none')
                loss_bbox_y3d_proj = F.smooth_l1_loss(bbox_y3d_proj[active], bbox_y3d_proj_tar[active],
                                                      reduction='none')
                loss_bbox_z3d_proj = F.smooth_l1_loss(bbox_z3d_proj[active], bbox_z3d_proj_tar[active],
                                                      reduction='none')

                loss_bbox_x3d_proj = (loss_bbox_x3d_proj * bbox_weights[active]).mean()
                loss_bbox_y3d_proj = (loss_bbox_y3d_proj * bbox_weights[active]).mean()
                loss_bbox_z3d_proj = (loss_bbox_z3d_proj * bbox_weights[active]).mean()

                bbox_3d_proj_loss = (loss_bbox_x3d_proj + loss_bbox_y3d_proj + loss_bbox_z3d_proj)

                bbox_3d_proj_loss *= self.bbox_3d_proj_lambda

                loss += bbox_3d_proj_loss
                stats.append({'name': 'bbox_3d_proj', 'val': bbox_3d_proj_loss, 'format': '{:0.4f}', 'group': 'loss'})

            coords_abs_z = coords_abs_z.view(-1)
            stats.append({'name': 'z', 'val': coords_abs_z[active].mean(), 'format': '{:0.2f}', 'group': 'misc'})

            coords_abs_ry = coords_abs_ry.view(-1)
            stats.append({'name': 'ry', 'val': coords_abs_ry[active].mean(), 'format': '{:0.2f}', 'group': 'misc'})

            ious_2d = ious_2d.view(-1)
            stats.append({'name': 'iou', 'val': ious_2d[active].mean(), 'format': '{:0.2f}', 'group': 'acc'})

            # use a 2d IoU based log loss
            if self.iou_2d_lambda:
                iou_2d_loss = -torch.log(ious_2d[active])
                iou_2d_loss = torch.tensor(iou_2d_loss, device=self.device)
                iou_2d_loss = (iou_2d_loss * bbox_weights[active])
                iou_2d_loss = iou_2d_loss.mean()

                iou_2d_loss *= self.iou_2d_lambda
                loss += iou_2d_loss

                stats.append({'name': 'iou', 'val': iou_2d_loss, 'format': '{:0.4f}', 'group': 'loss'})

        return loss, stats


def next_iteration(loader, iterator):
    """
    Loads the next iteration of 'iterator' OR makes a new epoch using 'loader'.
    Args:
        loader (object): PyTorch DataLoader object
        iterator (object): python in-built iter(loader) object
    """

    # create if none
    if iterator == None: iterator = iter(loader)

    # next batch
    try:
        imobjs = next(iterator)

    # new epoch / shuffle
    except StopIteration:
        iterator = iter(loader)
        imobjs = next(iterator)

    return iterator, imobjs


## Inference Utils

# NMS
def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def im_detect_3d(im, net, rpn_conf, p2, synced=False, device='cuda'):
    """
    Object detection in 3D
    """
    net.eval()
    imH_orig = 1216
    imW_orig = 1920
    device = device
    # im = preprocess(im)

    imH = im.shape[2]
    imW = im.shape[3]

    scale_factor = imH / imH_orig
    out,features= net(im.float())
    cls,prob,bbox_2d, bbox_3d, feat_size, rois= out['mono_3D_object_detection']
    #cls, prob, bbox_2d, bbox_3d, feat_size, rois = net(im.float())

    # compute feature resolution
    num_anchors = rpn_conf.anchors.shape[0]

    bbox_x = (bbox_2d[:, :, 0])
    bbox_y = (bbox_2d[:, :, 1])
    bbox_w = (bbox_2d[:, :, 2])
    bbox_h = (bbox_2d[:, :, 3])

    bbox_x3d = (bbox_3d[:, :, 0])
    bbox_y3d = (bbox_3d[:, :, 1])
    bbox_z3d = (bbox_3d[:, :, 2])
    bbox_w3d = (bbox_3d[:, :, 3])
    bbox_h3d = (bbox_3d[:, :, 4])
    bbox_l3d = (bbox_3d[:, :, 5])
    bbox_ry3d = (bbox_3d[:, :, 6])

    # detransform 3d
    bbox_x3d = (bbox_x3d * rpn_conf.bbox_stds[:, 4][0] + rpn_conf.bbox_means[:, 4][0])
    bbox_y3d = (bbox_y3d * rpn_conf.bbox_stds[:, 5][0] + rpn_conf.bbox_means[:, 5][0])
    bbox_z3d = (bbox_z3d * rpn_conf.bbox_stds[:, 6][0] + rpn_conf.bbox_means[:, 6][0])
    bbox_w3d = (bbox_w3d * rpn_conf.bbox_stds[:, 7][0] + rpn_conf.bbox_means[:, 7][0])
    bbox_h3d = (bbox_h3d * rpn_conf.bbox_stds[:, 8][0] + rpn_conf.bbox_means[:, 8][0])
    bbox_l3d = (bbox_l3d * rpn_conf.bbox_stds[:, 9][0] + rpn_conf.bbox_means[:, 9][0])
    bbox_ry3d = (bbox_ry3d * rpn_conf.bbox_stds[:, 10][0] + rpn_conf.bbox_means[:, 10][0])

    # find 3d source
    tracker = rois[:, 4].cpu().detach().numpy().astype(np.int64)
    src_3d = torch.from_numpy(rpn_conf.anchors[tracker, 4:]).to(device).type(torch.cuda.FloatTensor)
    # src_3d = torch.from_numpy(rpn_conf.anchors[tracker, 4:]).to(device)

    rois = rois.to(device)
    # compute 3d transform
    widths = (rois[:, 2] - rois[:, 0] + 1.0)
    heights = (rois[:, 3] - rois[:, 1] + 1.0)
    ctr_x = (rois[:, 0] + 0.5 * widths)
    ctr_y = (rois[:, 1] + 0.5 * heights)

    bbox_x3d = bbox_x3d[0, :] * widths + ctr_x
    bbox_y3d = bbox_y3d[0, :] * heights + ctr_y
    bbox_z3d = src_3d[:, 0] + bbox_z3d[0, :]
    bbox_w3d = torch.exp(bbox_w3d[0, :]) * src_3d[:, 1]
    bbox_h3d = torch.exp(bbox_h3d[0, :]) * src_3d[:, 2]
    bbox_l3d = torch.exp(bbox_l3d[0, :]) * src_3d[:, 3]
    bbox_ry3d = src_3d[:, 4] + bbox_ry3d[0, :]

    # bundle
    coords_3d = torch.stack((bbox_x3d, bbox_y3d, bbox_z3d[:bbox_x3d.shape[0]], bbox_w3d[:bbox_x3d.shape[0]],
                             bbox_h3d[:bbox_x3d.shape[0]], bbox_l3d[:bbox_x3d.shape[0]], bbox_ry3d[:bbox_x3d.shape[0]]),
                            dim=1)

    # compile deltas pred
    deltas_2d = torch.cat(
        (bbox_x[0, :, np.newaxis], bbox_y[0, :, np.newaxis], bbox_w[0, :, np.newaxis], bbox_h[0, :, np.newaxis]), dim=1)
    coords_2d = bbox_transform_inv(rois, deltas_2d, means=rpn_conf.bbox_means[0, :], stds=rpn_conf.bbox_stds[0, :])

    # detach onto cpu
    coords_2d = coords_2d.cpu().detach().numpy()
    coords_3d = coords_3d.cpu().detach().numpy()
    prob = prob[0, :, :].cpu().detach().numpy()

    # scale coords
    coords_2d[:, 0:4] /= scale_factor
    coords_3d[:, 0:2] /= scale_factor

    cls_pred = np.argmax(prob[:, 1:], axis=1) + 1
    scores = np.amax(prob[:, 1:], axis=1)

    aboxes = np.hstack((coords_2d, scores[:, np.newaxis]))
    # print('aboxes', aboxes[1])

    sorted_inds = (-aboxes[:, 4]).argsort()
    original_inds = (sorted_inds).argsort()
    aboxes = aboxes[sorted_inds, :]
    # print('sorted_aboxes', aboxes[1])
    coords_3d = coords_3d[sorted_inds, :]
    # print('sorted_coords_3d', coords_3d[1])
    cls_pred = cls_pred[sorted_inds]
    # print('sorted_cls',cls_pred[1])
    tracker = tracker[sorted_inds]
    abox_new = np.hstack((aboxes, cls_pred[:, np.newaxis]))
    abox_new = np.hstack((abox_new, coords_3d))
    # print('abox_new',abox_new.shape)
    # print('abox_first_element', abox_new[1])
    post_nms_det = new_nms(abox_new)
    #print('post_nms_det', len(post_nms_det))


    return post_nms_det


def hill_climb(p2, p2_inv, box_2d, x2d, y2d, z2d, w3d, h3d, l3d, ry3d, step_z_init=0, step_r_init=0, z_lim=0, r_lim=0,
               min_ol_dif=0.0):
    step_z = step_z_init
    step_r = step_r_init

    ol_best, verts_best, _, invalid = test_projection(p2, p2_inv, box_2d, x2d, y2d, z2d, w3d, h3d, l3d, ry3d)

    if invalid: return z2d, ry3d, verts_best

    # attempt to fit z/rot more properly
    while (step_z > z_lim or step_r > r_lim):

        if step_z > z_lim:

            ol_neg, verts_neg, _, invalid_neg = test_projection(p2, p2_inv, box_2d, x2d, y2d, z2d - step_z, w3d, h3d,
                                                                l3d, ry3d)
            ol_pos, verts_pos, _, invalid_pos = test_projection(p2, p2_inv, box_2d, x2d, y2d, z2d + step_z, w3d, h3d,
                                                                l3d, ry3d)

            invalid = ((ol_pos - ol_best) <= min_ol_dif) and ((ol_neg - ol_best) <= min_ol_dif)

            if invalid:
                step_z = step_z * 0.5

            elif (ol_pos - ol_best) > min_ol_dif and ol_pos > ol_neg and not invalid_pos:
                z2d += step_z
                ol_best = ol_pos
                verts_best = verts_pos
            elif (ol_neg - ol_best) > min_ol_dif and not invalid_neg:
                z2d -= step_z
                ol_best = ol_neg
                verts_best = verts_neg
            else:
                step_z = step_z * 0.5

        if step_r > r_lim:

            ol_neg, verts_neg, _, invalid_neg = test_projection(p2, p2_inv, box_2d, x2d, y2d, z2d, w3d, h3d, l3d,
                                                                ry3d - step_r)
            ol_pos, verts_pos, _, invalid_pos = test_projection(p2, p2_inv, box_2d, x2d, y2d, z2d, w3d, h3d, l3d,
                                                                ry3d + step_r)

            invalid = ((ol_pos - ol_best) <= min_ol_dif) and ((ol_neg - ol_best) <= min_ol_dif)

            if invalid:
                step_r = step_r * 0.5

            elif (ol_pos - ol_best) > min_ol_dif and ol_pos > ol_neg and not invalid_pos:
                ry3d += step_r
                ol_best = ol_pos
                verts_best = verts_pos
            elif (ol_neg - ol_best) > min_ol_dif and not invalid_neg:
                ry3d -= step_r
                ol_best = ol_neg
                verts_best = verts_neg
            else:
                step_r = step_r * 0.5

    while ry3d > math.pi: ry3d -= math.pi * 2
    while ry3d < (-math.pi): ry3d += math.pi * 2

    return z2d, ry3d, verts_best


def test_projection(p2, p2_inv, box_2d, cx, cy, z, w3d, h3d, l3d, rotY):
    """
    Tests the consistency of a 3D projection compared to a 2D box
    """

    x = box_2d[0]
    y = box_2d[1]
    x2 = x + box_2d[2] - 1
    y2 = y + box_2d[3] - 1

    p2_inv = torch.squeeze(p2_inv, dim=0)
    # print(p2_inv.shape)
    # coord3d = p2_inv.dot(np.array([cx * z, cy * z, z, 1]))
    # coord3d = p2_inv.dot(torch.tensor(np.array([cx * z, cy * z, z, 1])))
    coord3d = torch.matmul(p2_inv, (torch.tensor(np.array([cx * z, cy * z, z, 1]))))
    #print(coord3d.size())
    cx3d = coord3d[0]
    cy3d = coord3d[1]
    cz3d = coord3d[2]

    # put back on ground first
    # cy3d += h3d/2

    # re-compute the 2D box using 3D (finally, avoids clipped boxes)
    verts3d, corners_3d = project_3d(p2, cx3d, cy3d, cz3d, w3d, h3d, l3d, rotY, return_3d=True)

    invalid = np.any(corners_3d[2, :] <= 0)

    x_new = min(verts3d[:, 0])
    y_new = min(verts3d[:, 1])
    x2_new = max(verts3d[:, 0])
    y2_new = max(verts3d[:, 1])

    b1 = np.array([x, y, x2, y2])[np.newaxis, :]
    b2 = np.array([x_new, y_new, x2_new, y2_new])[np.newaxis, :]

    # ol = iou(b1, b2)[0][0]
    ol = -(np.abs(x - x_new) + np.abs(y - y_new) + np.abs(x2 - x2_new) + np.abs(y2 - y2_new))

    return ol, verts3d, b2, invalid


def convertAlpha2Rot(alpha, z3d, x3d):
    ry3d = alpha + math.atan2(-z3d, x3d) + 0.5 * math.pi
    # ry3d = alpha + math.atan2(x3d, z3d)# + 0.5 * math.pi

    while ry3d > math.pi: ry3d -= math.pi * 2
    while ry3d < (-math.pi): ry3d += math.pi * 2

    return ry3d


def project_3d(p2, x3d, y3d, z3d, w3d, h3d, l3d, ry3d, return_3d=False):
    """
    Projects a 3D box into 2D vertices
    Args:
        p2 (nparray): projection matrix of size 4x3
        x3d: x-coordinate of center of object
        y3d: y-coordinate of center of object
        z3d: z-cordinate of center of object
        w3d: width of object
        h3d: height of object
        l3d: length of object
        ry3d: rotation w.r.t y-axis
    """

    # compute rotational matrix around yaw axis
    R = np.array([[+math.cos(ry3d), -math.sin(ry3d), 0],

                  [+math.sin(ry3d), +math.cos(ry3d), 0], [0, 0, 1]])

    # 3D bounding box corners
    x_corners = np.array([0, l3d, l3d, l3d, l3d, 0, 0, 0])
    y_corners = np.array([0, 0, h3d, h3d, 0, 0, h3d, h3d])
    z_corners = np.array([0, 0, 0, w3d, w3d, w3d, w3d, 0])

    x_corners += -l3d / 2
    y_corners += -h3d / 2
    z_corners += -w3d / 2

    # bounding box in object co-ordinate
    corners_3d = np.array([x_corners, y_corners, z_corners])

    # rotate
    corners_3d = R.dot(corners_3d)

    # translate
    corners_3d = corners_3d + np.array([x3d, y3d, z3d]).reshape((3, 1))

    corners_3D_1 = np.vstack((corners_3d, np.ones((corners_3d.shape[-1]))))

    # corners_2D = p2.dot(corners_3D_1)
    corners_2D = np.matmul(p2, corners_3D_1)
    corners_2D = np.squeeze(corners_2D)
    corners_2D = corners_2D // corners_2D[2, :]
    # corners_2D= np.divide(corners_2D.T, (corners_2D[2,:]).T)
    # print("corners_2D",corners_2D)
    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]

    verts3d = (corners_2D[:, bb3d_lines_verts_idx][:2]).T

    if return_3d:
        return verts3d, corners_3d
    else:
        return verts3d



# function to read A2D2 labels for mono_3d_Object_detection
def read_A2D2_label(file, P, use_3d_for_2d=False):
    """
    Reads the A2D" label file from disc.

    Args:
        file (str): path to single label file for an image
        p2 (ndarray): projection matrix for the given image
    """

    gts = []

    text_file = open(file, 'r')

    '''
     Values    Name      Description
    ----------------------------------------------------------------------------
       1    type         class_id
       1    truncated    Integer (0,1,2,3) indicating truncation state:
                         0 = fully visible, 1 = partly occluded
                         2 = largely occluded, 3 = unknown
       1    occluded     Integer (0,1,2,3) indicating occlusion state:
                         0 = fully visible, 1 = partly occluded
                         2 = largely occluded, 3 = unknown
       1    alpha        Observation angle of object, ranging [-pi..pi]
       4    bbox         2D bounding box of object in the image (0-based index):
                         contains left, top, right, bottom pixel coordinates
       3    dimensions   3D object dimensions: height, width, length (in meters)
       3    location     3D object location x,y,z in camera coordinates (in meters)
       1    rotation_y   Rotation ry around Y-axis in camera coordinates [-pi..pi]


    # step 1: getting P Frame from C frame
    # Facts:
    # - share same origin in 0
    # - y/z plane of C and x/y plane of P coincide
    # - C: right handed system, X pointing along optical axis, y pointing left, z pointing up
    # - P: right handed system, Z pointing into image plane, y pointing down, x pointing right
    A_0P = np.zeros((3,3))
    A_0P[:,2] = A_0C[:,0] # P's z-axis equals C's x-axis 
    A_0P[:,0] = -A_0C[:,1] # P's x-axis equals C's y-axis 
    A_0P[:,1] = -A_0C[:,2] # P's y-axis equals C's z-axis 

    t_P_0 = t_C_0 # translation is the same
    #print(A_0P)
    '''
    for line in text_file:

        pattern = re.compile(('([a-zA-Z\-\?\_]+)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+'
                              + '(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s+(fpat)\s*((fpat)?)\n')
                             .replace('fpat', '[-+]?\d*\.\d+|[-+]?\d+'))

        parsed = pattern.fullmatch(line)
        # bbGt annotation in text format of:
        # cls x y w h occ x y w h ign ang
        if parsed is not None:

            obj = edict()

            ign = False

            cls = parsed.group(1)
            trunc = float(parsed.group(2))
            occ = float(parsed.group(3))
            alpha = float(parsed.group(4))

            x = (float(parsed.group(5)))
            y = (float(parsed.group(6)))
            x2 = (float(parsed.group(7)))
            y2 = (float(parsed.group(8)))

            width = x2 - x + 1
            height = y2 - y + 1

            l3d = (float(parsed.group(9)))
            w3d = (float(parsed.group(11)))
            h3d = (float(parsed.group(10)))

            cx3d = (float(parsed.group(12)))  # center of car in 3d
            cy3d = (float(parsed.group(13)))  # bottom of car in 3d
            cz3d = (float(parsed.group(14)))  # center of car in 3d
            rotY = float(parsed.group(15))

            # actually center the box
            cz3d += (h3d * 0.4)
            cy3d += (w3d * 0.25)
            cx3d += (l3d * 0.5)

            elevation = (1.65 - cy3d)

            coord3d = (P) @ (np.array([cx3d, cy3d, cz3d, 1]))

            # store the projected instead
            cx3d_2d = coord3d[0]
            cy3d_2d = coord3d[1]
            cz3d_2d = coord3d[2]

            cx = cx3d_2d / cz3d_2d
            cy = cy3d_2d / cz3d_2d
            # cx= uv[0]
            # cy= uv[1]

            # encode occlusion with range estimation
            # 0 = fully visible, 1 = partly occluded
            # 2 = largely occluded, 3 = unknown
            if occ == 0:
                vis = 1
            elif occ == 1:
                vis = 0.66
            elif occ == 2:
                vis = 0.33
            else:
                vis = 0.0

            while rotY > math.pi: rotY -= math.pi * 2
            while rotY < (-math.pi): rotY += math.pi * 2

            # recompute alpha
            alpha = convertRot2Alpha(rotY, cz3d, cx3d)

            obj.elevation = elevation
            obj.cls = cls
            obj.occ = occ > 0
            obj.ign = ign
            obj.visibility = vis
            obj.trunc = trunc
            obj.alpha = alpha
            obj.rotY = rotY

            # is there an extra field? (assume to be track)
            if len(parsed.groups()) >= 16 and parsed.group(16).isdigit(): obj.track = int(parsed.group(16))

            obj.bbox_full = [x, y, width, height]
            # print(obj.bbox_full)
            obj.bbox_3d = [cx, cy, cz3d_2d, w3d, h3d, l3d, alpha, cx3d, cy3d, cz3d, rotY]
            #obj.bbox_3d = [cx, cy, cz3d_2d, w3d, h3d, l3d, rotY, cx3d, cy3d, cz3d, alpha]
            # print(obj.bbox_3d)
            obj.center_3d = [cx3d, cy3d, cz3d]
            # print(obj.center_3d)

            gts.append(obj)

    text_file.close()

    return gts


def convertRot2Alpha(ry3d, z3d, x3d):
    alpha = ry3d - math.atan2(-z3d, x3d) - 0.5 * math.pi
    # alpha = ry3d - math.atan2(x3d, z3d)# - 0.5 * math.pi

    while alpha > math.pi: alpha -= math.pi * 2
    while alpha < (-math.pi): alpha += math.pi * 2

    return alpha
"""
----------------------------------------Evaluation part for check accuracies function--------------------------------------------------------------------------
"""
def intersect_rpn_eval(box_a, box_b, mode='list', data_type=None):
    """
    Computes the amount of intersect between two different sets of boxes.
    Args:
        box_a (nparray): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (nparray): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'combinations' or 'list', where combinations will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        data_type (type): either torch.Tensor or np.ndarray, we automatically determine otherwise
    """

    # determine type
    if data_type is None: data_type = type(box_a)

    # this mode computes the intersect in the sense of combinations.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'combinations':

        # np.ndarray
        if data_type == np.ndarray:
            max_xy = np.minimum(box_a[:, 2:4], np.expand_dims(box_b[:, 2:4], axis=1))
            min_xy = np.maximum(box_a[:, 0:2], np.expand_dims(box_b[:, 0:2], axis=1))
            inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

        # unknown type
        else:
            raise ValueError('type {} is not implemented'.format(data_type))

        return inter[:, :, 0] * inter[:, :, 1]

    # this mode computes the intersect in the sense of list_a vs. list_b.
    # i.e., box_a = M x 4, box_b = M x 4 then the output is Mx1
    elif mode == 'list':

        # torch.Tesnor
        if data_type == torch.Tensor:
            max_xy = torch.min(box_a[2:], box_b[2:])
            min_xy = torch.max(box_a[ :2], box_b[:2])
            inter = torch.clamp((max_xy - min_xy), 0)

        # np.ndarray
        elif data_type == np.ndarray:
            max_xy = np.min(box_a[ 2:], box_b[ 2:])
            min_xy = np.max(box_a[ :2], box_b[:2])
            inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))

        return inter[ 0] * inter[1]

    else:
        raise ValueError('unknown mode {}'.format(mode))

def iou_rpn_eval(box_a, box_b, mode='list', data_type=None):
    """
    Computes the amount of Intersection over Union (IoU) between two different sets of boxes.
    Args:
        box_a (nparray): Mx4 boxes, defined by [x1, y1, x2, y2]
        box_a (nparray): Nx4 boxes, defined by [x1, y1, x2, y2]
        mode (str): either 'combinations' or 'list', where combinations will check all combinations of box_a and
                    box_b hence MxN array, and list expects the same size list M == N, hence returns Mx1 array.
        data_type (type): either torch.Tensor or np.ndarray, we automatically determine otherwise
    """

    # determine type
    if data_type is None: data_type = type(box_a)

    # this mode computes the IoU in the sense of combinations.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'combinations':

        inter = intersect_rpn_eval(box_a, box_b, data_type=data_type)
        area_a = ((box_a[ 2] - box_a[0]) *
                  (box_a[ 3] - box_a[ 1]))
        area_b = ((box_b[: 2] - box_b[ 0]) *
                  (box_b[ 3] - box_b[1]))
        union = np.expand_dims(area_a, 0) + np.expand_dims(area_b, 1) - inter

        # torch.Tensor
        if data_type == torch.Tensor:
            return (inter / union).permute(1, 0)

        # np.ndarray
        elif data_type == np.ndarray:
            return (inter / union).T

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))


    # this mode compares every box in box_a with target in box_b
    # i.e., box_a = M x 4 and box_b = M x 4 then output is M x 1
    elif mode == 'list':

        #inter = intersect_rpn_eval(box_a, box_b, mode=mode)
        #area_a = (box_a[2] - box_a[ 0]) * (box_a[ 3] - box_a[ 1])
        #area_b = (box_b[ 2] - box_b[ 0]) * (box_b[ 3] - box_b[1])
        #union = area_a + area_b - inter

        #return inter / union
        box_a= torch.tensor(box_a)
        box_b= torch.tensor(box_b)
        
        
        box1_x1 = box_a[0:1]
        box1_y1 = box_a[1:2]
        box1_x2 = box_a[2:3]
        box1_y2 = box_a[3:4]
        box2_x1 = box_b[0:1]
        box2_y1 = box_b[1:2]
        box2_x2 = box_b[2:3]
        box2_y2 = box_b[3:4]
        

        x1 = torch.max(box1_x1, box2_x1)
        y1 = torch.max(box1_y1, box2_y1)
        x2 = torch.min(box1_x2, box2_x2)
        y2 = torch.min(box1_y2, box2_y2)
        
        intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
        box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
        box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

        return intersection / (box1_area + box2_area - intersection + 1e-6)
    else:
        raise ValueError('unknown mode {}'.format(mode))

from collections import Counter



def mean_average_precision_rpn_eval(pred_boxes, true_boxes, conf,iou_threshold=0.5):
    """
    Calculates mean average precision for 2d bounding boxes 

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, cls, score,x1, y1, x2, y2, w3d, h3d, l3d, x3d[0], y3d[0], z3d[0], ry3d, alpha]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for label in conf.lbls:
        print(label)
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == label:
                detections.append(detection)
        print('detections',len(detections))
        for true_box in true_boxes:
            if true_box[1] == label:
                ground_truths.append(true_box)
        print('groundtruths',len(ground_truths))
        
        
        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        #print(amount_bboxes)
        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        #print('sorted_detections',detections)
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
            #print(num_gts)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou_2d = iou_rpn_eval(
                    detection[3:7],
                    gt[2:6]
                )
                print('iou_2d',iou_2d)
                if iou_2d > best_iou:
                    best_iou = iou_2d
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

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        #torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))
    print(average_precisions)
    return sum(average_precisions) / len(average_precisions)

def mean_average_precision_rpn_eval_bev(pred_boxes, true_boxes, conf,iou_threshold=0.5):
    """
    Calculates mean average precision for 2d bounding boxes 

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, cls, score,x1, y1, x2, y2, w3d, h3d, l3d, x3d[0], y3d[0], z3d[0], ry3d, alpha]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """
    for i in range(len(pred_boxes)):
        bev_corners= generate_pred_bev(pred_boxes[i][8], pred_boxes[i][12], pred_boxes[i][10], pred_boxes[i][7],
                                       pred_boxes[i][13]) 
        #bev_corners shape- 4x2
        #print('bev_corners', np.shape(bev_corners))
        #print('bev_pred', bev_corners)
        pred_boxes[i][3]= bev_corners[2,0]
        pred_boxes[i][4]= bev_corners[2,1]
        pred_boxes[i][5]= bev_corners[0,0]
        pred_boxes[i][6]= bev_corners[0,1]
    for i in range(len(true_boxes)):
        bev_corners= generate_pred_bev(true_boxes[i][7], true_boxes[i][11], true_boxes[i][9], true_boxes[i][6],
                                       true_boxes[i][12]) 
        #bev_corners shape- 4x2
        #print('bev_corners', np.shape(bev_corners))
        #print('bev_true', bev_corners)
        true_boxes[i][2]= bev_corners[2,0]
        true_boxes[i][3]= bev_corners[2,1]
        true_boxes[i][4]= bev_corners[0,1]
        true_boxes[i][5]= bev_corners[0,0]
    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for label in conf.lbls:
        print(label)
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == label:
                detections.append(detection)
        print('detections',len(detections))
        for true_box in true_boxes:
            if true_box[1] == label:
                ground_truths.append(true_box)
        print('groundtruths',len(ground_truths))
        
        
        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])
        #print(amount_bboxes)
        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        #print('sorted_detections',detections)
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
            #print(num_gts)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou_2d = iou_rpn_eval(
                    detection[3:7],
                    gt[2:6]
                )
                print('iou_2d',iou_2d)
                if iou_2d > best_iou:
                    best_iou = iou_2d
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

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum + epsilon)
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        #torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))
    print(average_precisions)
    return sum(average_precisions) / len(average_precisions)

def get_rpn_boxes_with_loader(loader, model, device, conf):
    pred_boxes=[]
    true_boxes= []
    for batch_idx, imobj in enumerate(loader):
        x= imobj["image"]
        x= x.to(device)
        post_nms_boxes= im_detect_3d(im=x, net= model, rpn_conf=conf,p2= imobj["P"])
        
        for boxind in range(post_nms_boxes.shape[0]):
            box = post_nms_boxes[boxind, :]
            score = box[4]
            #print(score)
            final_box=[]
            cls = conf.lbls[int(box[5] - 1)]
            if score >= 0.75:
                x1 = box[0]
                y1 = box[1]
                x2 = box[2]
                y2 = box[3]
                width = (x2 - x1 + 1)
                height = (y2 - y1 + 1)

                # plot 3D box
                x3d = box[6]
                y3d = box[7]
                z3d = box[8]
                w3d = box[9]
                h3d = box[10]
                l3d = box[11]
                ry3d = box[12]

                # convert alpha into ry3d
                coord3d = np.linalg.inv(imobj["P"]).dot(np.array([x3d * z3d, y3d * z3d, 1 * z3d, 1]))
                #print(coord3d.shape)
                ry3d = convertAlpha2Rot(ry3d, coord3d[:,2], coord3d[:,0])

                step_r = 0.3*math.pi
                r_lim = 0.01
                box_2d = np.array([x1, y1, width, height])

                z3d, ry3d, verts_best = hill_climb(imobj["P"], imobj["p2_inv"], box_2d, x3d, y3d, z3d, w3d, h3d, l3d, ry3d,
                                                   step_r_init=step_r, r_lim=r_lim)

                # predict a more accurate projection
                coord3d = np.linalg.inv(imobj["P"]).dot(np.array([x3d * z3d, y3d * z3d, 1 * z3d, 1]))
                alpha = convertRot2Alpha(ry3d, coord3d[:,2], coord3d[:,0])

                x3d = coord3d[:,0]
                y3d = coord3d[:,1]
                z3d = coord3d[:,2]
                
                new_box= [batch_idx,cls ,score, x1, y1, x2, y2, x3d[0], y3d[0], z3d[0], w3d, h3d, l3d,ry3d, alpha]
                pred_boxes.append(new_box)
                #print('new_box', new_box)
        
        
        
        for i in range(len(imobj["mono_3D_gts"])):
            
            x1= float(imobj["mono_3D_gts"][i]["bbox_full"][0][0])
            y1= float(imobj["mono_3D_gts"][i]["bbox_full"][1][0])
            w= float(imobj["mono_3D_gts"][i]["bbox_full"][2][0])
            h= float(imobj["mono_3D_gts"][i]["bbox_full"][3][0])
            x2= (x1 + w) - 1
            y2= (y1 + h) - 1

            ins_box= [batch_idx, imobj["mono_3D_gts"][i]['cls'][0], x1, y1, x2, y2,float(imobj["mono_3D_gts"][i]['bbox_3d'][8][0]),float(imobj["mono_3D_gts"][i]['bbox_3d'][9][0]),
                      float(imobj["mono_3D_gts"][i]['bbox_3d'][10][0]),float(imobj["mono_3D_gts"][i]['bbox_3d'][3][0]),
                      float(imobj["mono_3D_gts"][i]['bbox_3d'][4][0]),float(imobj["mono_3D_gts"][i]['bbox_3d'][5][0])
                      ,float(imobj["mono_3D_gts"][i]['bbox_3d'][6][0])]  
            true_boxes.append(ins_box)
            #print('ins_box',ins_box)
        
    print(len(pred_boxes))
    print(len(true_boxes))

    return pred_boxes, true_boxes
def generate_pred_bev(y3d, l3d, w3d, x3d, ry3d, scale=1):

    l = l3d * scale
    w = w3d * scale
    x = x3d * scale
    y = y3d * scale
    r= ry3d

    corners1 = np.array([
        [-l / 2, -w / 2, 1],
        [+l / 2, -w / 2, 1],
        [+l / 2, +w / 2, 1],
        [-l/ 2, +w / 2, 1]
           ])
    ry = np.array([
        [+math.cos(r), -math.sin(r), 0],
        [+math.sin(r), math.cos(r), 0],
        [0, 0, 1],
    ])

    corners2 = ry.dot(corners1.T).T

    corners2[:, 0] += l/2+ x 
    corners2[:, 1] += w/2 + y 
    
    corners_f= corners2[:,:2]
    return corners_f

def rpn_MAP(pred_boxes, true_boxes, conf):
    mAP_2D= mean_average_precision_rpn_eval(pred_boxes, true_boxes, conf,iou_threshold=0.5)
    mAP_bev= mean_average_precision_rpn_eval_bev(pred_boxes, true_boxes, conf,iou_threshold=0.5)
    
    return mAP_bev, mAP_2D
