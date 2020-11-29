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

names = ['Arson','Explosion','Fall','Fighting','Normal']

def load_dataset(train_folder, test_folder, label_folder, args):
    train_dataset = DataLoaderNew(train_folder, label_folder, resize_height=args.h, resize_width=args.w,
                                  time_step=args.t_length - 1, interval=args.interval, object=args.object_detection)

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
    # img = img/255
    # return H W C
    return img


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

        # organize order 3-5ms
        length = len(batch)

        batch = batch[:length // 2] + batch[(length // 2) + 1:] + [batch[length // 2]]
        label = label[length // 2]
        labels = ['Normal','Arson','Explosion','Fall','Fighting']
        if label == 1:
            if video_name.startswith(labels[1]):
                label = 1
            elif video_name.startswith(labels[2]):
                label = 2
            elif video_name.startswith(labels[3]):
                label = 3
            elif video_name.startswith(labels[4]):
                label = 4
        batch = np.concatenate(batch, axis=0)

        return torch.from_numpy(batch), torch.LongTensor([label])

    # num frames
    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    p = '/data/miaobo/ucf/'
    video_folder, label_folder, transform, resize_height, resize_width = \
        os.path.join(p, "training_toy/frames"), os.path.join(p, "label"), None, 10, 10

    train_dataset = DataLoaderNew(video_folder, label_folder, transform, resize_height=resize_height, resize_width=resize_width,
                                  time_step=4)