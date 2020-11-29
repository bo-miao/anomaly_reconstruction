import os, sys
import time
import numpy as np
from collections import OrderedDict
import glob
import math
import copy
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import torchvision.transforms as transforms


rng = np.random.RandomState(2020)

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',  # 8
        'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush']

target_names = [0, 1, 2, 3, 5, 7, 9,
                32, 33, 34, 36, 38]


def load_dataset(train_folder, test_folder, label_folder, args):
    if not args.evaluate:
        train_dataset = DataLoaderNew(train_folder, label_folder, resize_height=args.h, resize_width=args.w,
                                      time_step=args.t_length - 1, interval=args.interval, object=args.object_detection)
    else:
        train_dataset = None

    test_dataset = DataLoaderNew(test_folder, label_folder, resize_height=args.h, resize_width=args.w,
                                 time_step=args.t_length - 1, interval=args.interval, object=args.object_detection)
    train_sampler = None
    if not args.evaluate:
        # TODO: Shuffle notice
        train_sampler = data.distributed.DistributedSampler(train_dataset)

    if not args.evaluate:
        train_batch = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                      num_workers=args.workers, collate_fn=my_collate if args.object_detection else None,
                                      pin_memory=True, drop_last=True, sampler=train_sampler)
    else:
        train_batch = None
    test_batch = data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False,
                                 num_workers=args.workers_test, collate_fn=my_collate if args.object_detection else None,
                                 pin_memory=True, drop_last=False)
    return train_batch, test_batch, train_sampler


def my_collate(batch):
    image, label, bbox = zip(*batch)
    return torch.stack(image, 0), torch.stack(label, 0), bbox


def norm_collate(batch):
    image, label = zip(*batch)
    return torch.stack(image, 0), torch.stack(label, 0)


def read_frame(filename, h, w):
    img = cv2.imread(filename)
    sp = img.shape
    real_height, read_weight = sp[0], sp[1]
    # BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(dtype=np.float32)
    # resize
    if int(real_height) != h or int(read_weight) != w:
        img = cv2.resize(img, (w, h)).astype(dtype=np.float32)
    # normalization (-1 to 1)
    img = 2 * (img / 255) - 1.0
    # return H W C
    return img


def read_bounding_box(filename):
    f = filename.split('/')
    f[-3] = 'box_l' if 'shanghai' in filename else 'box_low_l'   # box_low_l
    f[-1] = f[-1].split('.')[0] + '.npy'
    f = '/'.join(f)
    bbox = np.load(f)
    return bbox


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x).astype(np.float32)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y


def xywh2xyxy(x, enlarge_w=1.0, enlarge_h=1.0):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x).astype(np.float32)
    # Enlarge bbox W to ensure whole frames can include object
    x[:, 2] *= enlarge_w
    x[:, 3] *= enlarge_h
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

    return np.clip(y, 0, 1)


def box_iou(box1, box2, video_name, frame_name):
    # XYXY
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    h = min(box1[3], box2[3]) - max(box1[1], box2[1])
    if h <= 0 or w <= 0:
        iou = 0
    else:
        inter = h * w
        iou = inter / (area1 + area2 - inter)
    return iou


def box_multi_merge(boxes, sets):
    idxs = list(sets)
    merged_box = np.array([0.9 * min(boxes[idxs, 0]), 0.9 * min(boxes[idxs, 1]),
                           1.1 * max(boxes[idxs, 2]), 1.1 * max(boxes[idxs, 3])])
    return merged_box


class DataLoaderNew(data.Dataset):
    def __init__(self, video_folder, label_folder,
                 resize_height, resize_width, time_step=4, num_pred=1, interval=1, object=1):
        self.dir = video_folder
        self.labels = label_folder
        self.videos = OrderedDict()
        self._resize_height = resize_height
        self._resize_width = resize_width
        self._time_step = time_step # 4
        self._num_pred = num_pred
        self.setup()
        self.samples = self.get_all_samples()
        self.interval = interval
        self.object_detection = object

    def setup(self):
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            self.videos[video_name] = {}
            self.videos[video_name]['path'] = video  # abs video path
            self.videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
            self.videos[video_name]['frame'].sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))  # frames path
            # self.videos[video_name]['frame'].sort()  # HAVE ISSUE!
            self.videos[video_name]['length'] = len(self.videos[video_name]['frame']) # frames length
            if self.labels:
                self.videos[video_name]['label'] = np.load(os.path.join(self.labels, video_name.split('.')[0] + ".npy"))

                assert self.videos[video_name]['length'] == len(self.videos[video_name]['label']), \
                    "{}.npy doesn't match image length {} vs {}".format(video_name, len(self.videos[video_name]['label']),
                                                                         self.videos[video_name]['length'])

    def get_all_samples(self):
        frames = []
        videos = glob.glob(os.path.join(self.dir, '*'))
        for video in sorted(videos):
            video_name = video.split('/')[-1]
            for i in range(len(self.videos[video_name]['frame']) - self._time_step):
                frames.append(self.videos[video_name]['frame'][i])

        return frames

    def __getitem__(self, index):
        video_name = self.samples[index].split('/')[-2]
        # frame name of the first frame # jpg排序方式有问题
        frame_name = int(self.samples[index].split('/')[-1].split('.')[-2])

        # total frames each sample
        length = self._time_step + self._num_pred
        batch = []

        # get label
        label = self.videos[video_name]['label'][frame_name: frame_name + length: self.interval] \
            if self.labels else np.zeros(length)

        # get N frames 5-10ms
        for i in range(0, length, self.interval):
            image = read_frame(self.videos[video_name]['frame'][frame_name + i], self._resize_height,
                               self._resize_width)
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            batch.append(image)

        # BOXES
        object_type = self.object_detection  # which object extraction method is used
        filter_box = False  # whether filter uncorrelated boxes
        # get objects
        bboxes = []
        if object_type == 1:  # objects from 1st and last frame TODO: 检测到物体框再大一点
            # ped2, 5-10num, 5-25ms
            # shanghai, <5num, 3-12ms
            for i in [0, length - 1]:
                bbox = read_bounding_box(self.videos[video_name]['frame'][frame_name + i])
                num = bbox.shape[0]
                for j in range(num):
                    if filter_box and int(bbox[j][0]) not in target_names:
                        continue
                    box = bbox[j][1:]
                    bboxes.append(box)

            bboxes = np.array(bboxes).astype(np.float32)
            bboxes = np.clip(bboxes, 0.0, 1.0)
            bbox_num = len(bboxes)
            if bbox_num > 0:
                ct = 0
                iterate = 0
                # iteration to merge
                while bbox_num != ct and iterate < 2:
                    iterate += 1
                    ct = bbox_num
                    new_bboxes = []
                    history = {10000}
                    # n*n/2 complexity
                    for i in range(ct):
                        bbox = bboxes[i]
                        sets = {i}
                        for j in range(i+1, ct):
                            iou = box_iou(bbox, bboxes[j], video_name, frame_name)
                            if iou > 0.1:  # TODO：hyper parameters
                                sets.add(j)
                                history.add(j)

                        if len(sets) == 1 and i in history:
                            continue

                        bbox = box_multi_merge(bboxes, sets)
                        new_bboxes.append(bbox)

                    # update boxes
                    bbox_num = len(new_bboxes)
                    bboxes = np.array(new_bboxes).astype(np.float32)
                    bboxes = np.clip(bboxes, 0.0, 1.0)

        elif object_type == 2:  # objects from predicted frame
            # shanghai, <5num, 3-12ms
            # ped2, 10-20num, 2ms
            bbox = read_bounding_box(self.videos[video_name]['frame'][frame_name + (length // 2)])
            num = bbox.shape[0]
            for j in range(num):
                if filter_box and int(bbox[j][0]) not in target_names:
                    continue
                box = bbox[j][1:]
                bboxes.append(box)

            bboxes = np.array(bboxes).astype(np.float32)
            bboxes = np.clip(bboxes, 0.0, 1.0)
            if len(bboxes) > 0:
                bboxes = xyxy2xywh(bboxes)
                bboxes = xywh2xyxy(bboxes, enlarge_w=2, enlarge_h=1.5)

        # organize order 3-5ms
        length = len(batch)

        if length == 1:
            batch = np.concatenate(batch, axis=0)
            label = label[0]
        else:
            batch = batch[:length // 2] + batch[(length // 2) + 1:] + [batch[length // 2]]
            label = label[length // 2]
            batch = np.concatenate(batch, axis=0)

        if object_type:
            return torch.from_numpy(batch), torch.IntTensor([label]), torch.from_numpy(bboxes).float()
        else:
            return torch.from_numpy(batch), torch.IntTensor([label])

    # num frames
    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    p = '/data/miaobo/ucf/'
    video_folder, label_folder, transform, resize_height, resize_width = \
        os.path.join(p, "training_toy/frames"), os.path.join(p, "label"), None, 10, 10

    train_dataset = DataLoaderNew(video_folder, label_folder, transform, resize_height=resize_height, resize_width=resize_width,
                                  time_step=4)