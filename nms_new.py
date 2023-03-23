import numpy as np
import torch


def intersect(box_a, box_b, mode='list', data_type=None):
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
            max_xy = np.minimum(box_a[2:4], np.expand_dims(box_b[2:4], axis=1))
            min_xy = np.maximum(box_a[ 0:2], np.expand_dims(box_b[ 0:2], axis=1))
            inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)

        # unknown type
        else:
            raise ValueError('type {} is not implemented'.format(data_type))

        return inter[ :, 0] * inter[ :, 1]

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
            max_xy = np.min((box_a[2:], box_b[2:]))
            min_xy = np.max((box_a[:2], box_b[:2]))
            inter = np.clip((max_xy - min_xy), a_min=0, a_max=None)
            print('inter', inter)

        # unknown type
        else:
            raise ValueError('unknown data type {}'.format(data_type))

        return inter[0] * inter[1]

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
    #print('box_a', type(box_a))

    # this mode computes the IoU in the sense of combinations.
    # i.e., box_a = M x 4, box_b = N x 4 then the output is M x N
    if mode == 'combinations':

        inter = intersect(box_a, box_b, data_type=data_type)
        area_a = ((box_a[ 2] - box_a[ 0]) *
                  (box_a[ 3] - box_a[ 1]))
        area_b = ((box_b[ 2] - box_b[0]) *
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
        #print('box_a', np.shape(box_a))
        #print('box_b', np.shape(box_b))
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

        #inter = intersect(box_a, box_b, mode=mode)
        #area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        #area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        #union = area_a + area_b - inter

        #return inter / union

    else:
        raise ValueError('unknown mode {}'.format(mode))


def new_nms(detections, nms_iou_thershold=0.2, prob_filter=0.3, after_nms_prob_filter=0.75):
    filter_detection= []
    final_box=[]
    for det in range(detections.shape[0]):     #First converting selected boxes with prob>0.3 to list
        if detections[det, 4] >= prob_filter:
            detect= detections[det,:].tolist()
            filter_detection.append(detect)
        else:
            pass
    boxes_after_nms= []
    while filter_detection:
        chosen_box= filter_detection.pop(0)
        #print('chosen_box',chosen_box)

        filter_detection= [
            box
            for box in filter_detection
            if box[5] != chosen_box[5]
            or iou(np.array(chosen_box)[0:4], np.array(box)[0:4], mode='list') < nms_iou_thershold
        ]
        boxes_after_nms.append(chosen_box)
    """
    while boxes_after_nms:
        post_nms_chosen_box=boxes_after_nms.pop(0)
        box_after_nms= [
            box
            for box in boxes_after_nms
            if box[5] != post_nms_chosen_box[5]
            or abs(box[7] - post_nms_chosen_box[7]) >0.2
        
        ]
        final_box.append(post_nms_chosen_box)
    return (np.array(final_box))
    """
    return (np.array(boxes_after_nms))
