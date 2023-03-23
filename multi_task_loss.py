"""
Multi task loss for semantic segmentation,lane_marking,drivable_area and object_detection multi-task network
"""
from easydict import EasyDict as edict
import torch
import numpy as np
import torch.nn as nn
from utils import intersection_over_union
import torch.nn.functional as F
from rpn_utils import *
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
rpn_conf = Config()

class Multi_task_loss_fn(nn.Module):

    def __init__(self, tasks=None,scaled_anchors=None, DEVICE="cuda", WEIGHTED= False, conf=None):
        super(Multi_task_loss_fn, self).__init__()
        self.tasks = tasks  # same as tasks mentioned in train.py file
        self.scaled_anchors = scaled_anchors  # predefined anchors in train.py
        self.device = DEVICE  # "cuda" or "cpu"
        self.weighted = WEIGHTED  # If True calculates weights for each task
        self.conf = conf
        self.mse = nn.MSELoss()
        for task in self.tasks:
            if task[0] == "semantic_segmentation":
                self.seg_loss = WCE_L(num_classes=task[1], DEVICE=self.device)  # Initializing the segmentation loss
            elif task[0] == "object_detection":
                self.yolo_loss = YoloLoss()  # Initializing YOLOv3 loss
            elif task[0] == "mono_3D_object_detection":
                self.m3d_rpn_loss = RPN_3D_loss(self.conf) #Intializing RPN loss
                #self.m3d_rpn_loss.to(self.device)
            elif task[0] == "depth_estimation":
                self.depth_estimation_loss = MaskedMSELoss()

    def forward(self, preds, imobj, features):

        #features_avg = torch.zeros_like(features[self.tasks[0][0]])
        #for task in self.tasks:
        #    features_avg += features[task[0]]
        #features_avg = features_avg / len(features)
        losses = torch.zeros(len(self.tasks)).to(self.device)
        #feature_losses = torch.zeros(len(self.tasks)).to(self.device)
        for i, task in enumerate(self.tasks):
            if task[0] == "semantic_segmentation":
                seg_loss = self.seg_loss(preds[task[0]], imobj["semantic_label"].to(self.device, dtype=torch.long))    
                losses[i] = seg_loss
                
                #feature_losses[i] = self.mse(features[task[0]], features_avg)

            elif task[0] == "mono_3D_object_detection":
                mono_3D_loss = self.m3d_rpn_loss(cls=preds[task[0]][0].to(self.device),
                                                 prob=preds[task[0]][1].to(self.device),
                                                 bbox_2d=preds[task[0]][2].to(self.device),
                                                 bbox_3d=preds[task[0]][3].to(self.device),
                                                 feat_size=preds[task[0]][4].to(self.device),
                                                 imobjs= imobj)
                L= mono_3D_loss.detach().cpu()
                if ((mono_3D_loss == math.inf) or (np.isnan(L)== True)):
                    losses[i] = 1e-6
                else:
                    losses[i] = mono_3D_loss
                
                #feature_losses[i] = self.mse(features[task[0]], features_avg)

            elif task[0] == "object_detection":
                yolo_loss = (
                        self.yolo_loss(preds[task[0]][0], imobj["yolo_target"][0].to(self.device), self.scaled_anchors[0])
                        + self.yolo_loss(preds[task[0]][1], imobj["yolo_target"][1].to(self.device), self.scaled_anchors[1])
                        + self.yolo_loss(preds[task[0]][2], imobj["yolo_target"][2].to(self.device), self.scaled_anchors[2])
                )
                losses[i] = yolo_loss
            
            elif task[0]== "depth_estimation":
                #print('depth_pred',preds[task[0]].shape )
                #print('deppth_gt', imobj["depth_estimation_label"].shape)
                depth_estimation_loss= 0.01* (self.depth_estimation_loss((preds[task[0]].squeeze(axis=1)), imobj["depth_estimation_label"].to(self.device, dtype=torch.long)))

                #feature_losses[i] = self.mse(features[task[0]], features_avg)

        # weights = torch.FloatTensor([(sum(losses)/(len(self.tasks)*losses[i])) for i in range(len(self.tasks))]).to(DEVICE)

        if self.weighted:
            loss_avg = torch.mean(losses)
            weights = torch.FloatTensor([(losses[i]/loss_avg ) for i in range(len(self.tasks))]).to(self.device)
            loss = losses * weights
        else:
            loss = losses

        loss = sum(loss) #+ sum(feature_losses)

        task_losses = losses.detach().cpu()

        #return loss, task_losses, sum(feature_losses)
        return loss, task_losses

class WCE_L(nn.Module):

    def __init__(self, num_classes=22, DEVICE="cuda"):
        super(WCE_L, self).__init__()
        self.num_classes = num_classes
        self.device = DEVICE

    def forward(self, preds, targets, smooth=1):

        weights = np.ones(self.num_classes, dtype=float)
        class_id_counts = np.ones(self.num_classes, dtype=float)
        labels_array = np.asarray(targets.cpu())
        class_ids, count = np.unique(labels_array, return_counts=True)
        length_of_data = labels_array.size
        for idx, class_id in enumerate(class_ids):
            class_id_counts[int(class_id)] += count[idx]
        for idx, class_id_count in enumerate(class_id_counts):
            weights[idx] += length_of_data + smooth / ((self.num_classes * class_id_count) + smooth)
        weights = torch.Tensor(weights).to(self.device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fn(preds, targets)

        return loss


class YoloLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid()

        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1.5
        self.lambda_noobj = 1
        self.lambda_obj = 1.5
        self.lambda_box = 1.5

    def forward(self, predictions, target, anchors):
        # Check where obj and noobj (we ignore if target == -1)
        obj = target[..., 0] == 1  # in paper this is Iobj_i
        noobj = target[..., 0] == 0  # in paper this is Inoobj_i

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        anchors = anchors.reshape(1, 3, 1, 1, 2)
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * anchors], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #

        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )  # width, height coordinates
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #

        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )

        return (
                self.lambda_box * box_loss
                + self.lambda_obj * object_loss
                + self.lambda_noobj * no_object_loss
                + self.lambda_class * class_loss
        )


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
        self.device = 'cuda'

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
            gts = imobj["mono_3D_gts"]
            imobj["imH"] = 1216
            imobj["imW"] = 1920
            imobj["scale"] = 1

            if len(imobj["mono_3D_gts"]) > 0:
                scale_factor = imobj["scale"] * (608 / imobj["imH"])

                imobj["scale_factor"] = scale_factor

            p2_inv = torch.tensor(imobj["p2_inv"], device=self.device).type(torch.cuda.FloatTensor)

            # filter gts
            igns, rmvs = determine_ignores(gts, self.lbls, self.ilbls, self.min_gt_vis, self.min_gt_h)

            # accumulate boxes

            scaled_gts = np.empty((len(imobj["mono_3D_gts"]), 4))
            j = 0
            while (j < (len(imobj["mono_3D_gts"]))):
                for gt in imobj["mono_3D_gts"]:
                    for i in range(4):
                        scaled_gts[j, i] = gt["bbox_full"][i] * scale_factor
                    j = j + 1
            # print(np.shape(scaled_gts))
            gts_all = bbXYWH2Coords(scaled_gts)
            gts_3d = np.array([gt["bbox_3d"] for gt in gts], dtype=object)

            for gtind, gt in enumerate(gts_3d):
                gts_3d[gtind, 0:2] *= scale_factor

            if not ((rmvs == False) & (igns == False)).any():
                continue

            # filter out irrelevant cls, and ignore cls
            gts_val = gts_all[(rmvs == False) & (igns == False), :]
            gts_ign = gts_all[(rmvs == False) & (igns == True), :]
            gts_3d = gts_3d[(rmvs == False) & (igns == False), :]

            # accumulate labels
            box_lbls = np.array([gt["cls"][0] for gt in gts])
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

                    rois = rois.to(DEVICE)

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

                    coords_2d = torch.cat((coords_2d, (torch.ones([1, coords_2d.shape[1]]).to(DEVICE))), dim=0)
                    #print("coords_2d",coords_2d.size())
                    coords_3d = torch.squeeze((torch.matmul(p2_inv, coords_2d)), dim=0)
                    
                    # print("bbox_x3d_proj",bbox_x3d_proj.size())
                    #coords_3d = torch.mm(p2_inv, coords_2d)
                    #print("coords_3d",coords_3d.size())
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

        return loss

class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss
