import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.models.resnet as models
import torch.nn as nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from rpn_utils import *
from kitti_3d_multi_warmup import *
import os
from multi_task_dataset import multi_task_dataset
from torch.utils.data import Dataset, DataLoader
from utils import (cells_to_bboxes, iou_width_height as iou, non_max_suppression as nms, plot_image, plot_image_cv2)
import torchvision
from easydict import EasyDict as edict
"""
config file
tuple : (out_channels, kernel, stride)
"S" : Scale prediction layer
"U" : Up sample layer
"D" : Downsampling , not in original yolo v3 
"""

yolo_config = [
    (512, 2, 2),
    (1024, 3, 1),
    # "D",  # added to match the first output size (img_width//32,img_height//32)
    "S",
    (256, 1, 1),
    "U",
    (256, 1, 1),
    (512, 3, 1),
    "S",
    (128, 1, 1),
    "U",
    (128, 1, 1),
    (256, 3, 1),
    "S",
]
# RPN configuration
rpn_conf = Config()
precomputed = True
file_dir = os.path.dirname(__file__)
cache_folder = os.path.join(file_dir, 'pickle')

if precomputed == True:
    rpn_conf.anchors = pickle_read(os.path.join(cache_folder, 'anchors.pkl'))
    rpn_conf.bbox_means = pickle_read(os.path.join(cache_folder, 'bbox_means.pkl'))
    rpn_conf.bbox_stds = pickle_read(os.path.join(cache_folder, 'bbox_stds.pkl'))


def conv(in_channels, out_channels, kernel_size):
    padding = (kernel_size-1) // 2
    #assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
          nn.Conv2d(in_channels,out_channels,kernel_size,stride=1,padding=padding,bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
        )

def pointwise(in_channels, out_channels):
    return nn.Sequential(
          nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
        )

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, bn_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=not bn_act, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1)
        self.use_bn_act = bn_act

    def forward(self, x):
        if self.use_bn_act:
            return self.leaky(self.bn(self.conv(x)))
        else:
            return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_residual=True, num_repeats=1):
        super().__init__()
        self.layers = nn.ModuleList()
        for repeat in range(num_repeats):
            self.layers += [
                nn.Sequential(
                    CNNBlock(channels, channels // 2, kernel_size=1),
                    CNNBlock(channels // 2, channels, kernel_size=3, padding=1),
                )
            ]

        self.use_residual = use_residual
        self.num_repeats = num_repeats

    def forward(self, x):
        for layer in self.layers:
            if self.use_residual:
                x = x + layer(x)
            else:
                x = layer(x)

        return x


class ScalePrediction(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pred = nn.Sequential(
            CNNBlock(in_channels, 2 * in_channels, kernel_size=3, padding=1),
            CNNBlock(
                2 * in_channels, (num_classes + 5) * 3, bn_act=False, kernel_size=1
            ),
        )
        self.num_classes = num_classes

    def forward(self, x):
        return (
            self.pred(x)
                .reshape(x.shape[0], 3, self.num_classes + 5, x.shape[2], x.shape[3])
                .permute(0, 1, 3, 4, 2)
        )


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


def create_heads(task=None, head=None, features=None, num_classes=22):
    head.append(DoubleConv(features[-2], features[-1]))   #actually taking in the output of the backbone

    for i in range(1, len(features)):
        in_feature = features[-i]
        out_feature = features[-i - 1]
        head.append(
            nn.ConvTranspose2d(
                in_feature, out_feature, kernel_size=(2, 2), stride=(2, 2),
                ))
        head.append(DoubleConv(out_feature * 2, out_feature))
    head.append(nn.Conv2d(features[0], num_classes, kernel_size=(1, 1)))

    return head


def create_yolo_conv_layers(in_channels=None, num_classes=20, head=None):
    layers = head
    in_channels = in_channels
    factor = in_channels // 512
    #print('factor', factor)

    for module in yolo_config:
        if isinstance(module, tuple):
            out_channels, kernel_size, stride = module
            layers.append(
                CNNBlock(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=1 if kernel_size == 3 else 0,
                )
            )
            in_channels = out_channels

        elif isinstance(module, list):
            num_repeats = module[1]
            layers.append(ResidualBlock(in_channels, num_repeats=num_repeats, ))

        elif isinstance(module, str):
            if module == "S":
                layers += [
                    ResidualBlock(in_channels, use_residual=False, num_repeats=1),
                    CNNBlock(in_channels, in_channels // 2, kernel_size=1),
                    ScalePrediction(in_channels // 2, num_classes=num_classes),
                ]
                in_channels = in_channels // 2

            elif module == "U":
                layers.append(nn.Upsample(scale_factor=2))
                in_channels = in_channels + in_channels * 2 * factor  # because resnet layer 4 has 2048 channels

            elif module == "D":
                layers.append(nn.MaxPool2d(2))

    return layers

def create_RPN_layers(phase, base, conf, base_features, head=None):
    head.append(RPN(phase, base, conf, base_features))
    #layer=head
    #layer = RPN_layers
    return head

def create_depth_estimation_head(layers=34, head=None):
    #num_channels=512
    

    conv2 = nn.Conv2d(512, 1024, 1)
    kernel_size = 5
    decode_conv1 = conv(1024, 512, kernel_size)
    decode_conv2 = conv(512, 256, kernel_size)
    decode_conv3 = conv(256, 128, kernel_size)
    decode_conv4 = conv(128, 64, kernel_size)
    decode_conv5 = conv(64, 32, kernel_size)
    decode_conv6 = pointwise(32, 1)

    head.append(conv2)
    head.append(decode_conv1)
    head.append(decode_conv2)
    head.append(decode_conv3)
    head.append(decode_conv4)
    head.append(decode_conv5)
    head.append(decode_conv6)
    return head

def prepare_resnet_model(backbone="resnet34"):
    resnet = None
    features = None

    if backbone == "resnet18":
        resnet = models.resnet18(pretrained=True)
        features = [64, 64, 128, 256, 512, 1024]

    elif backbone == "resnet34":
        resnet = models.resnet34(pretrained=True)
        features = [64, 64, 128, 256, 512, 1024]

    elif backbone == "resnet50":
        resnet = models.resnet50(pretrained=True)
        features = [64, 256, 512, 1024, 2048, 4096]

    elif backbone == "resnet101":
        resnet = models.resnet101(pretrained=True)
        features = [64, 256, 512, 1024, 2048, 4096]

    elif backbone == "resnet152":
        resnet = models.resnet152(pretrained=True)
        features = [64, 256, 512, 1024, 2048, 4096]

    elif backbone is None:
        features = [64, 128, 256, 512, 1024]

    resnet.avgpool = Identity()
    resnet.fc = Identity()
    resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)

    return resnet, features


class Multi_task_model(nn.Module):
    def __init__(self, conf,in_channels=3, backbone=None, tasks=None, rpn_phase='train'):
        super(Multi_task_model, self).__init__()

        """
        in_channels : input channel dimension
        backbone : one of "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
        tasks : list [["task name", no. of classes],["task name", no. of classes]....]
        task names : "semantic_segmentation", "object_detection", "3d_object_det"
        returns : model outputs, features for feature loss
        """

        if tasks is None:
            raise ValueError("tasks are not mentioned")
        self.in_channels = in_channels
        self.resnet, self.features = prepare_resnet_model(backbone=backbone)  # creating tail of multi-task model
        print('resent_features',len(self.features))
        print('features', self.features)
        self.pool = nn.MaxPool2d(2, 2)
        self.tasks = tasks
        self.heads = []
        self.phase = rpn_phase
        self.conf = conf
        self.rpn_conv = nn.Conv2d(in_channels=self.features[-2], out_channels=self.features[-1],
                                          kernel_size=1)
        self.yolo_bottle_neck_transform = torchvision.transforms.Resize((38,38))
        self.yolo_skip2_transform= torchvision.transforms.Resize((76,76))
        for task in self.tasks:
            if task[0] == "semantic_segmentation":  # semantic_segmentation head
                self.segmentation = nn.ModuleList()
                self.heads.append(create_heads(
                    task=task,
                    head=self.segmentation,
                    num_classes=task[1],
                    features=self.features,
                ))

            elif task[0] == "object_detection":  # object_detection head
                self.object_detection = nn.ModuleList()
                self.heads.append(create_yolo_conv_layers(
                    in_channels=self.features[-2],
                    num_classes=task[1],
                    head=self.object_detection,
                ))
            elif task[0] == "mono_3D_object_detection":     #rpn_head
                self.mono_3D_object_detection = nn.ModuleList()
                self.heads.append(create_RPN_layers(
                    phase=self.phase,
                    base=self.resnet,
                    base_features = self.features,
                    conf=self.conf,
                    head=self.mono_3D_object_detection

                ))
            elif task[0]== "depth_estimation":
                self.depth_estimation= nn.ModuleList()
                self.heads.append(create_depth_estimation_head(
                    head= self.depth_estimation
                ))

    def forward(self, x):
        #print('actual_input', x.size())
        #print('no of heads', len(self.heads))
        outputs = {}
        features_track = {task[0]: False for task in self.tasks}
        features = {}
        skip_connections = []

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        
        skip_connections.append(x)  # skip1

        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
      
        skip_connections.append(x)  # skip2

        x = self.resnet.layer2(x)
        skip_connections.append(x)  # skip3

        
        x = self.resnet.layer3(x)
        skip_connections.append(x)  # skip4, also yolo skip2
   
        x = self.resnet.layer4(x)
        skip_connections.append(x)  # skip5, also yolo skip1

        resnet_output = x
        
        #print('resnet_output_hr', resnet_output.shape)
        bottle_neck_input = self.pool(resnet_output)
        #print('bottle_neck', bottle_neck_input.shape)
        rpn_bottle_neck_input= self.rpn_conv(resnet_output)
        yolo_bottle_neck_input= self.yolo_bottle_neck_transform(resnet_output)

        
        skip_connections = skip_connections[::-1]
        #for i,skip in enumerate(skip_connections):
        #    print('skip',skip.size())

        for i, task in enumerate(self.tasks):
            if task[0] == "semantic_segmentation":

                for idx in range(len(self.heads[i])):
                    if idx == 0:
                        x = self.heads[i][idx](bottle_neck_input)
                        if not features_track[task[0]]:
                            features[task[0]] = x
                            features_track[task[0]] = True

                    elif idx % 2 == 0:
                        skip_connection = skip_connections[(idx // 2) - 1]
                        if x.shape != skip_connection.shape:
                            x = TF.resize(x, size=skip_connection.shape[2:])
                        concat_skip = torch.cat((skip_connection, x), dim=1)
                        x = self.heads[i][idx](concat_skip)

                    else:
                        x = self.heads[i][idx](x)

                outputs[task[0]] = x

            elif task[0] == "object_detection":
                idx = 1
                skip_connections_yolo = skip_connections[::-1][3:5]
                #print('skip1', skip_connections_yolo[0].shape)
                #print('skip2', skip_connections_yolo[1].shape)
                skip_connections_yolo_resized=[]
                for j,skip in enumerate(skip_connections_yolo):
                    if j==1:
                        skip= self.yolo_bottle_neck_transform(skip)
                        skip_connections_yolo_resized.append(skip)
                    else:
                        skip= self.yolo_skip2_transform(skip)
                        skip_connections_yolo_resized.append(skip)

                yolo_outputs = []
                #x = resnet_output
                x=yolo_bottle_neck_input
                #print('bottle_nect_input', x.size())
                for layer in self.heads[i]:
                    if isinstance(layer, ScalePrediction) and idx == 1:
                        yolo_outputs.append(layer(x))
                        continue
                    elif isinstance(layer, ScalePrediction):
                        yolo_outputs.append(layer(x))
                        continue

                    x = layer(x)

                    if not features_track[task[0]] and x.shape[1] == 1024:
                        features[task[0]] = x
                        features_track[task[0]] = True

                    if isinstance(layer, nn.Upsample):
                        #print('before_cat', x.size())
                        x = torch.cat([x, skip_connections_yolo_resized[-idx]], dim=1)
                        idx += 1
                        #print('after_cat', x.size())

                outputs[task[0]] = yolo_outputs

            elif task[0] == "mono_3D_object_detection":
                for idx in range(len(self.heads[i])):
                    x = self.heads[i][idx](rpn_bottle_neck_input)
                    rpn_feat = self.pool(rpn_bottle_neck_input)
                    if not features_track[task[0]]:
                        features[task[0]] = rpn_feat
                        features_track[task[0]] = True


                outputs[task[0]] = x
            elif task[0] == "depth_estimation":
                for idx in range(len(self.heads[i])):
                    #print('idx', idx)
                    if idx==0:
                        x= self.heads[i][idx](resnet_output)
                        #if not features_track[task[0]]:
                        #    features[task[0]]= x
                        #    features_track[task[0]]= True
                        #print('depth_output_case_0', x.size())
                    elif (idx== 6 or idx==5):
                        x= self.heads[i][idx](x)
                        #print('depth_output_case_2', x.size())
                    else:
                        x= self.heads[i][idx](x)
                        #print('depth_output_case_1', x.size())
                        #print(skip_connections[-idx].shape)
                        x= F.interpolate(x + skip_connections[idx-1], scale_factor=2, mode='nearest')

                outputs[task[0]] = x
                
        return outputs, features


def test():
    tasks = [["semantic_segmentation", 22], ["object_detection", 10], ["mono_3D_object_detection", 10]]
    #tasks = [["semantic_segmentation", 22]]
    #tasks = [["object_detection", 10]]
    #tasks = [["mono_3D_object_detection", 10]]
    model = Multi_task_model(backbone="resnet34", in_channels=3, tasks=tasks, conf= rpn_conf)
    print('number of heads', len(model.heads))
    x = torch.randn(1, 3, 608, 960)
    #print('model', model)


    y, features = model(x)
    #for key ,val in features.items():
    #    print(key)
    #    print((features[key]).size())
    # print(y["semantic_segmentation"].shape)
    # print(len(features))
    # print(features["semantic_segmentation"].shape)
    # print(features["drivable_area"].shape)

    for task in tasks:
        print(task[0])
        if task[0] == "semantic_segmentation":
            print(y[task[0]].shape)
            print(features[task[0]].shape)
        elif task[0] == "object_detection":
            #print(features[task[0]].shape)
            for j in range(3):
                print(y[task[0]][j].shape)
        elif task[0] == "mono_3D_object_detection":
            for j in range(len(y[task[0]])-1):

                print((y[task[0]][j]).size())
            print(len(y[task[0]][-1]))
        elif task[0] == "depth_estimation":
            print(y[task[0]].shape)
            print(features[task[0]].shape)

def test2():
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
    print('scaled_anchors', scaled_anchors)
    train_transform = A.Compose(
        [
            A.Resize(height=608, width=960),
            ToTensorV2(),
        ],
        bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[], ),
        additional_targets={'mask': 'mask'}
    )
    DATASET_DIR = r'E:\Thesis_Final\A2D2_dataset'
    dataset = multi_task_dataset(csv_file=r'E:\Thesis_Final\A2D2_dataset\train.csv',
                                 dataset_dir=DATASET_DIR,
                                 mono_3d_label_dir=r"A2D2_3D_Obj_det_label_txt",
                                 semantic_label_dir="seg_label",
                                 det_2d_label_dir='YOLO_Bbox_2D_label',
                                 #tasks=[["object_detection", 16]],
                                 tasks=[["semantic_segmentation", 22],["object_detection", 10], ["mono_3D_object_detection", 10]],
                                 yolo_anchors=anchors,
                                 img_dir="images",
                                 S=S,transform=train_transform  )
    a= dataset[0]
    print('a', a.image.size())
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False)
    tasks = [["semantic_segmentation", 22], ["object_detection", 10], ["mono_3D_object_detection", 10]]
    model = Multi_task_model(backbone="resnet50", in_channels=3, tasks=tasks, conf=rpn_conf)

    for j, imobj in enumerate(loader):
        if j==0:
            x = imobj.image
            print('image', x.size())
            y, features = model(x.float())
            for task in tasks:
                print(task[0])
                if task[0] == "semantic_segmentation":
                    print(y[task[0]].shape)
                    print(features[task[0]].shape)
                elif task[0] == "object_detection":
                    print(features[task[0]].shape)
                    for j in range(3):
                        print(y[task[0]][j].shape)
                elif task[0] == "mono_3D_object_detection":
                    for j in range(len(y[task[0]]) - 1):
                        print((y[task[0]][j]).size())
                    print(len(y[task[0]][-1]))
            else:
                break



if __name__ == '__main__':
    test()
