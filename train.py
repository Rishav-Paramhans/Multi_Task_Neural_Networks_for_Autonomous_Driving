"""
Training file for all single task and multi task networks

"""
import os
import torch
import torch.optim as optim
from multi_task_model import Multi_task_model
from multi_task_loss import Multi_task_loss_fn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from rpn_utils import *
from utils import (
    save_model,
    save_optimizer,
    load_model,
    check_accuracies,
    save_some_examples,
    transforms
)
from kitti_3d_multi_warmup import *
from rpn_utils import *
from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader
from multi_task_dataset import multi_task_dataset
# Hyperparameters

"""
mention the tasks.
TASKS : list [["task name", no. of classes],["task name", no. of classes]....]
task names : "semantic_segmentation", "lane_marking", "drivable_area", "object_detection"

BACKBONE : one of resnet18, resnet34, resnet50, resnet101, resnet150. 

"""
#file_dir= os.path.dirname(__file__) 

rpn_conf = Config()
precomputed = True
cache_folder = 'pickle'

if precomputed == True:
    rpn_conf.anchors = pickle_read(os.path.join(cache_folder, 'anchors.pkl'))
    rpn_conf.bbox_means = pickle_read(os.path.join(cache_folder, 'bbox_means.pkl'))
    rpn_conf.bbox_stds = pickle_read(os.path.join(cache_folder, 'bbox_stds.pkl'))
#print('conf',rpn_conf)

LEARNING_RATE = 0.00001
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print('Using device:',DEVICE)
BATCH_SIZE = 1
TASKS = [["semantic_segmentation", 22], ["object_detection", 10],["mono_3D_object_detection",10], ["depth_estimation",1]]
tasks_name = "four_task"
BACKBONE = "resnet34"  # one of resnet18, resnet34, resnet50, resnet101, resnet150.
IMAGE_WIDTH = 960
IMAGE_HEIGHT = 608
WEIGHT_DECAY = 0.0005
EPOCHS = 200
NUM_WORKERS =8
PIN_MEMORY = True
LOAD_MODEL = False
WRITER = True  # Controlling the tensorboard
CONF_THRESHOLD = 0.5  # confidence of yolo model for the prediction
IOU_THRESH = 0.5
NMS_THRESH = 0.5
LOAD_MODEL_FILE = "models/" + tasks_name + "/" + tasks_name + "_172_.pth"
# LOAD_MODEL_FILE = "models/" + tasks_name + "/" + tasks_name + "5" + "_.pth"
SAVE_MODEL_FILE = "models/" + tasks_name + "/"  # folder to save models
SAVE_PATH = "predictions/" + tasks_name + "/"  # folder to save predictions
if not os.path.exists(SAVE_MODEL_FILE):
    os.makedirs(SAVE_MODEL_FILE)
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)
DATASET_DIR = "A2D2_dataset" # labels main directory
IMG_DIR = "images"  # complete images
TRAIN_CSV= os.path.join(DATASET_DIR , "train.csv")
TEST_CSV=  os.path.join(DATASET_DIR , "test.csv")
MONO_3D_LABEL_DIR= "A2D2_3D_Obj_det_label_txt"
SEMANTIC_LABEL_DIR= "seg_label"
DET_2D_LABEL_DIR= 'YOLO_Bbox_2D_label'
DEPTH_ESTIMATION_LABEL_DIR = "depth_maps_gt"

S = [IMAGE_HEIGHT // 32, IMAGE_HEIGHT // 16, IMAGE_HEIGHT // 8]
ANCHORS = [[[0.28, 0.22], [0.38, 0.48], [0.9, 0.78]],
           [[0.07, 0.15], [0.15, 0.11], [0.14, 0.29]],
           [[0.02, 0.03], [0.04, 0.07], [0.08, 0.06]]]
scaled_anchors = (torch.tensor(ANCHORS) * (torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2))).to(DEVICE)
train_transform, test_transform = transforms(IMAGE_WIDTH=IMAGE_WIDTH, IMAGE_HEIGHT=IMAGE_HEIGHT)


# training loop

def train_fn(train_loader, model, optimizer,loss_fn, GradScaler):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    mean_feature_loss = []
    idv_task_loss = torch.zeros(len(TASKS))

    for batch_idx, imobj in enumerate(loop):

        x= imobj["image"]
        x= x.to(DEVICE)
        optimizer.zero_grad()
        out, features = model(x.float())
        #loss, task_losses, feature_loss = loss_fn(out, imobj,features)
        loss, task_losses = loss_fn(out, imobj,features)
        loss.backward()
        optimizer.step()
        mean_loss.append(loss.item())
        #mean_feature_loss.append(feature_loss)
        idv_task_loss += task_losses
        # update progress bar
        loop.set_postfix(loss=loss.item())
    mean_loss_value = sum(mean_loss) / len(mean_loss)
    #mean_feature_loss_value = sum(mean_feature_loss) / len(mean_feature_loss)
    idv_task_loss = idv_task_loss / len(train_loader)
    print(f"Mean loss was {mean_loss_value}")
    #print(f"Mean feature loss was {mean_feature_loss_value}")
    return mean_loss_value, idv_task_loss


def main():
    model = Multi_task_model(backbone=BACKBONE, in_channels=3, tasks=TASKS, conf= rpn_conf)
    #trained_parameter_path= "./models/three_task/20model_.pth"
    #optimizer_parameter_path= "./models/three_task/20optimizer_.pth"
    #model.load_state_dict(torch.load(trained_parameter_path))
    model= model.to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    #optimizer.load_state_dict(torch.load(optimizer_parameter_path))
    
    train_ds = multi_task_dataset(
        csv_file=TRAIN_CSV, dataset_dir=DATASET_DIR, mono_3d_label_dir=MONO_3D_LABEL_DIR,
        semantic_label_dir=SEMANTIC_LABEL_DIR,
        det_2d_label_dir=DET_2D_LABEL_DIR, depth_estimation_label_dir= DEPTH_ESTIMATION_LABEL_DIR, img_dir=IMG_DIR, tasks=TASKS, transform=train_transform,
        yolo_anchors=ANCHORS, S=S
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )
    test_ds = multi_task_dataset(
        csv_file=TEST_CSV, dataset_dir=DATASET_DIR, mono_3d_label_dir=MONO_3D_LABEL_DIR,
        semantic_label_dir=SEMANTIC_LABEL_DIR,
        det_2d_label_dir=DET_2D_LABEL_DIR, img_dir=IMG_DIR, tasks=TASKS, transform=train_transform,
        yolo_anchors=ANCHORS, S=S
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )

    loss_fn = Multi_task_loss_fn(tasks=TASKS, scaled_anchors=scaled_anchors, DEVICE=DEVICE, WEIGHTED=True, conf=rpn_conf)

    GradScaler = torch.cuda.amp.GradScaler()

    prev_accuracy = 0

    if WRITER:
        writer = SummaryWriter(f'tensorboard/{tasks_name}')

    for epoch in range(EPOCHS):
        model.train()
        print(f"EPOCH : {epoch}")
        mean_loss, task_losses = train_fn(train_loader, model, optimizer, loss_fn, GradScaler)
        if WRITER:
            writer.add_scalars("Losses", {task[0]: float(task_losses[i]) for i, task in enumerate(TASKS)},
                               global_step=epoch)
        save_model(model, SAVE_MODEL_FILE + str(epoch) + "model_.pth") # save model after every epoch
        save_optimizer(optimizer, SAVE_MODEL_FILE + str(epoch) + "optimizer_.pth" )
        """
        save_optimizer
        if epoch > 15 and epoch % 5 ==0:
            task_accuracies = check_accuracies(model, loader=test_loader, tasks=TASKS, anchors=ANCHORS,
                                           nms_threshold=NMS_THRESH, DEVICE=DEVICE,
                                           iou_threshold=IOU_THRESH, conf_threshold=CONF_THRESHOLD,
                                           check_map=True,  # if epoch > 19 and epoch % 5 == 0 else False,
                                           class_wise_dice_score=True, conf= rpn_conf)
            test_accuracy = sum(task_accuracies.values()) / len(task_accuracies)
            if WRITER:
                writer.add_scalars("Accuracies", {key: task_accuracies[key] for key in task_accuracies.keys()},
                               global_step=epoch)
        
        text_file = open(SAVE_PATH + tasks_name + ".txt", "w")   # text file to write test image ID and models accuracy on test image
        save_some_examples(model, test_loader, save_path=SAVE_PATH, epoch=epoch, tasks=TASKS,
                           anchors=scaled_anchors, iou_threshold=IOU_THRESH, threshold=0.65, nms_threshold=0.2,
                           DEVICE=DEVICE, text_file=text_file)  # save some predictions
        text_file.close()
        if test_accuracy > prev_accuracy:
            save_model(model, str(test_accuracy) + tasks_name + "model")  # saving the best model
            prev_accuracy = test_accuracy

        """
if __name__ == "__main__":
    main()
