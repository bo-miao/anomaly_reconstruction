from utils import lr_scheduler, metric, prefetch, summary
import os, sys
import time
import numpy as np
from collections import OrderedDict
import glob
import math
import copy
import tqdm
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast

import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
from .datasets import *

rng = np.random.RandomState(2020)


def rel2abs(box, h, w):
    box[:, 0] *= w
    box[:, 2] *= w
    box[:, 1] *= h
    box[:, 3] *= h
    return box


# label resize & loss support
def get_object_images(images, labels, bboxes, args):
    b, c, h, w = images.size()
    bbox_num = []
    bbox_mask = []
    new_bboxes = []
    for i, bbox in enumerate(bboxes):
        bbox_num.append(bbox.size()[0])
        if bbox.size()[0] > 0:
            bbox_mask.append(i)
            bbox = rel2abs(bbox, h, w)  # res to abs coord
            new_bboxes.append(bbox)  # non empty boxes

    if len(new_bboxes) == 0:  # prevent all miss objects
        patches = None
        new_labels = None
        return patches, new_labels, bbox_num

    new_images = images[bbox_mask]  # images with non empty boxes
    patch_h, patch_w = args.path_h, args.path_w  # TODO: Alignsize
    patches = torchvision.ops.roi_align(new_images, new_bboxes, output_size=(patch_h, patch_w))
    patches = patches.view(-1, c, patch_h, patch_w)
    assert patches.size()[0] == sum(bbox_num), "patch number does not match bbox_num"
    new_labels = torch.zeros(sum(bbox_num)).cuda(non_blocking=True)
    start_ = 0
    for i, label in enumerate(labels):
        new_labels[start_:start_ + bbox_num[i]] = label
        start_ += bbox_num[i]

    return patches, new_labels, bbox_num  # [K,C,H,W] [K] [B]


def get_the_number_of_params(model, is_trainable=False):
    """get the number of the model"""
    if is_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def psnr(mse):
    e = 1e-8
    return 10 * math.log10(1 / (mse+e))


# TODO: Max score using 1/3 position large score instead
def anomaly_score(psnr, max_psnr, min_psnr):
    return ((psnr - min_psnr) / (max_psnr - min_psnr))


def anomaly_score_list(psnr_list):
    anomaly_score_list = list()
    for i in range(len(psnr_list)):
        anomaly_score_list.append(anomaly_score(psnr_list[i], np.max(psnr_list), np.min(psnr_list)))
    return anomaly_score_list


def AUC(anomal_scores, labels):
    frame_auc = 0
    try:
        frame_auc = roc_auc_score(y_true=np.squeeze(labels, axis=0), y_score=np.squeeze(anomal_scores))
    except:
        print("AUC Cal ERROR: ", labels, anomal_scores)
    
    return frame_auc


def plot_AUC(anomal_scores, labels):
    try:
        false_positive_rate, true_positive_rate, thresholds = roc_curve(np.squeeze(labels, axis=0), np.squeeze(anomal_scores))
        roc_auc = auc(false_positive_rate, true_positive_rate)
        plt.title('ROC')
        plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.4f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.ylabel('TPR')
        plt.xlabel('FPR')
        plt.savefig('/home/miaobo/project/anomaly_demo/ckpt/'+str(time.strftime("%m%d%H%M",time.localtime()))+'.png')
    except:
        print("PLOTING ROC CURVE ERROR")


def score_sum_single(list1):
    list_result = []
    for i in range(len(list1)):
        list_result.append((list1[i]))
    return list_result


def evaluate_new_D(model, D, test_batch, args):
    print("EVALUATING NORMAL ADVERSARIAL MODEL...")
    return evaluate_object_adversarial(model, D, test_batch, args)


def evaluate_new(model, test_batch, args):
    if 'Classifier' in args.arch:
        if args.object_detection:
            print("EVALUATING NORMAL OBJECT ENCODER CLASSIFIER MODEL...")
            return evaluate_two_stage_object(model, test_batch, args)
        else:
            print("EVALUATING NORMAL ENCODER CLASSIFIER MODEL...")
            return evaluate_two_stage(model, test_batch, args)

    elif args.object_detection:  # object level prediction
        print("EVALUATING NORMAL OBJECT MODEL...")
        return evaluate_object(model, test_batch, args)
    else:  # image-level
        print("EVALUATING NORMAL IMAGE MODEL...")
        return evaluate(model, test_batch, args)


def evaluate_object_adversarial(model, D, test_batch, args):
    avg_loss = metric.AverageMeter('avg_loss', ':.4e')
    single_time = metric.AverageMeter('Time', ':6.3f')
    progress = metric.ProgressMeter(len(test_batch), avg_loss, single_time, prefix="Evaluation: ")

    model.eval()
    D.eval()

    label_list = []
    psnr_list = []
    logit_list = []
    ct = 0
    counter = 0
    for k, (images, labels, bboxes) in enumerate(test_batch):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bboxes = [x.cuda(non_blocking=True) for x in bboxes]
        a = time.time()
        counter += 1

        patches, patch_labels, bbox_num = get_object_images(images, labels, bboxes, args)  # [K,C,H,W] [K] [B]

        if patches is None:
            for i in range(len(labels)):
                label_list.append(labels[i].item())
                psnr_list.append(100.0)
        else:
            del images
            batch_size_now = len(bbox_num)
            ct += patches.size()[0]
            label = patch_labels if args.label else None

            channel = (patches.size()[1] // args.c - 1) * args.c
            input_image = patches[:, 0:channel]
            target_image = patches[:, channel:]


            with autocast():
                output, loss = model.forward(input_image, gt=target_image, label=label, train=False)
                loss = loss['pixel_loss'].view(loss['pixel_loss'].shape[0], -1).mean(1)
            assert len(loss) == len(label), "During inference, loss sample number must match label sample number."

            # dif image as input
            input_image = output - target_image
            logit = D(input_image, train=False)

            start_ = 0
            for i, num_ in enumerate(bbox_num):  # per sample in batch
                logit_per_sample = torch.max(logit[start_: start_ + num_]).item() if num_ > 0 else 0
                loss_per_sample = torch.max(loss[start_: start_ + num_]).item() if num_ > 0 else 0
                psnr_list.append(psnr(loss_per_sample))  # TODO: Max or Mean
                logit_list.append(logit_per_sample)
                label_list.append(labels[i].item())
                avg_loss.update(loss_per_sample, batch_size_now)
                start_ += num_

            assert start_ == logit.size()[0], "patch num and bbox_num doesn't match"
            # statistic
            # for i in range(len(loss)):
            #     l = loss[i]
            #     p = torch.mean(patches[i])
            #     o = torch.mean(output[i])
            #     la = patch_labels[i]
            #     if la == 1 and torch.abs(l/p) <= 0.01:
            #         print("False Negative: ", l, p, o, la, l/p, l/o)
            #     elif la == 0 and torch.abs(l/p) > 0.01:
            #         print("False Positive: ", l, p, o, la, l/p, l/o)

        if args.evaluate_time:
            single_time.update((time.time() - a)*1000)
            progress.print(counter)
            # print("Single batch time cost {}ms, loss {}".format(1000*(time.time()-a), loss.mean().item()))

    anomaly_score_total_list = np.asarray(anomaly_score_list(psnr_list))
    label_list = np.asarray(label_list)
    logit_list = np.asarray(logit_list)
    assert anomaly_score_total_list.size == label_list.size and anomaly_score_total_list.size == logit_list.size, "INFERENCE LENGTH MUST MATCH LABEL LENGTH."

    # final_score = 0.8*logit_list+0.2*(1-anomaly_score_total_list)
    final_score = logit_list
    accuracy = roc_auc_score(y_true=label_list, y_score=final_score)
    accuracy1 = roc_auc_score(y_true=label_list, y_score=1-anomaly_score_total_list)
    # plot_AUC(anomaly_score_total_list, np.expand_dims(1 - label_list, 0))
    print("EVAL FRAME & BOX NUMBER & ACC : ", anomaly_score_total_list.size, ct, accuracy*100, accuracy1*100)

    return accuracy, avg_loss.avg


def evaluate_object(model, test_batch, args):
    avg_loss = metric.AverageMeter('avg_loss', ':.4e')
    single_time = metric.AverageMeter('Time', ':6.3f')
    progress = metric.ProgressMeter(len(test_batch), avg_loss, single_time, prefix="Evaluation: ")
    model.eval()

    label_list = []
    psnr_list = []
    ct = 0
    counter = 0
    for k, (images, labels, bboxes) in enumerate(test_batch):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bboxes = [x.cuda(non_blocking=True) for x in bboxes]
        a = time.time()
        counter += 1

        patches, patch_labels, bbox_num = get_object_images(images, labels, bboxes, args)  # [K,C,H,W] [K] [B]

        if patches is None:
            for i in range(len(labels)):
                label_list.append(labels[i].item())
                psnr_list.append(100.0)
        else:
            del images
            batch_size_now = len(bbox_num)
            ct += patches.size()[0]
            label = patch_labels if args.label else None

            channel = (patches.size()[1] // args.c - 1) * args.c
            input_image = patches[:, 0:channel]
            target_image = patches[:, channel:]

            if args.is_amp:
                with autocast():
                    output, loss = model.forward(input_image, gt=target_image, label=label, train=False)
                    loss = loss['pixel_loss'].view(loss['pixel_loss'].shape[0], -1).mean(1)
            else:
                output, loss = model.forward(input_image, gt=target_image, label=label, train=False)
                loss = loss['pixel_loss'].view(loss['pixel_loss'].shape[0], -1).mean(1)
            assert len(loss) == len(label), "During inference, loss sample number must match label sample number."

            start_ = 0
            for i, num_ in enumerate(bbox_num):  # per sample in batch
                loss_per_sample = torch.max(loss[start_: start_ + num_]).item() if num_ > 0 else 0
                psnr_list.append(psnr(loss_per_sample))  # TODO: Max or Mean
                label_list.append(labels[i].item())
                avg_loss.update(loss_per_sample, batch_size_now)
                start_ += num_

            # statistic
            # for i in range(len(loss)):
            #     l = loss[i]
            #     p = torch.mean(patches[i])
            #     o = torch.mean(output[i])
            #     la = patch_labels[i]
            #     if la == 1 and torch.abs(l/p) <= 0.01:
            #         print("False Negative: ", l, p, o, la, l/p, l/o)
            #     elif la == 0 and torch.abs(l/p) > 0.01:
            #         print("False Positive: ", l, p, o, la, l/p, l/o)

        if args.evaluate_time:
            single_time.update((time.time() - a)*1000)
            progress.print(counter)
            # print("Single batch time cost {}ms, loss {}".format(1000*(time.time()-a), loss.mean().item()))

    anomaly_score_total_list = np.asarray(anomaly_score_list(psnr_list))
    label_list = np.asarray(label_list)
    assert anomaly_score_total_list.size == label_list.size, "INFERENCE LENGTH MUST MATCH LABEL LENGTH."
    accuracy = roc_auc_score(y_true=label_list, y_score=1-anomaly_score_total_list)
    # plot_AUC(anomaly_score_total_list, np.expand_dims(1 - label_list, 0))
    print("EVAL FRAME & BOX NUMBER: ", anomaly_score_total_list.size, ct, len(psnr_list), len(label_list))

    return accuracy, avg_loss.avg


def evaluate(model, test_batch, args):
    avg_loss = metric.AverageMeter('avg_loss', ':.4e')
    single_time = metric.AverageMeter('Time', ':6.3f')
    progress = metric.ProgressMeter(len(test_batch), avg_loss, single_time, prefix="Evaluation: ")
    model.eval()

    label_list = []
    psnr_list = []
    counter = 0
    for k, (images, labels) in enumerate(test_batch):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        a = time.time()
        counter += 1

        label = labels if args.label else None
        channel = (images.size()[1] // args.c - 1) * args.c
        input_image = images[:, 0:channel]
        target_image = images[:, channel:]

        if args.is_amp:
            with autocast():
                output, loss = model.forward(input_image, gt=target_image, label=label, train=False)
                loss = loss['pixel_loss'].view(loss['pixel_loss'].shape[0], -1).mean(1)
        else:
            output, loss = model.forward(input_image, gt=target_image, label=label, train=False)
            loss = loss['pixel_loss'].view(loss['pixel_loss'].shape[0], -1).mean(1)

        assert len(loss) == len(label), "During inference, loss sample number must match label sample number."
        for i in range(len(loss)):
            mse_reconstruction = loss[i].item()
            psnr_list.append(psnr(mse_reconstruction))
            label_list = np.append(label_list, label[i].item())
            avg_loss.update(loss[i].item(), 1)

        if args.evaluate_time:
            single_time.update((time.time() - a)*1000)
            progress.print(counter)
            # print("Single batch time cost {}ms, loss {}".format(1000*(time.time()-a), loss.mean().item()))

    anomaly_score_total_list = np.asarray(anomaly_score_list(psnr_list))
    assert anomaly_score_total_list.size == label_list.size, "INFERENCE LENGTH MUST MATCH LABEL LENGTH."
    accuracy = roc_auc_score(y_true=label_list, y_score=1-anomaly_score_total_list)
    # plot_AUC(anomaly_score_total_list, np.expand_dims(1 - labels_list, 0))
    print("EVALUATE FRAME NUMBER: ", anomaly_score_total_list.size)
    return accuracy, avg_loss.avg


def evaluate_two_stage(model, test_batch, args):
    avg_loss = metric.AverageMeter('avg_loss', ':.4e')
    single_time = metric.AverageMeter('Time', ':6.3f')
    progress = metric.ProgressMeter(len(test_batch), avg_loss, single_time, prefix="Evaluation: ")
    model.eval()

    label_list = []
    psnr_list = []
    logit_list = []
    counter = 0
    for k, (images, labels) in enumerate(test_batch):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        a = time.time()
        counter += 1

        label = labels if args.label else None
        channel = (images.size()[1] // args.c - 1) * args.c
        input_image = images[:, 0:channel]
        target_image = images[:, channel:]

        if args.is_amp:
            with autocast():
                reconstructed_image, loss, logit = model.forward(input_image, gt=target_image, label=label, train=False)
                loss = loss['pixel_loss'].view(loss['pixel_loss'].shape[0], -1).mean(1)
        else:
            reconstructed_image, loss, logit = model.forward(input_image, gt=target_image, label=label, train=False)
            loss = loss['pixel_loss'].view(loss['pixel_loss'].shape[0], -1).mean(1)

        assert len(loss) == len(label), "During inference, loss sample number must match label sample number."
        for i in range(len(loss)):
            mse_reconstruction = loss[i].item()
            psnr_list.append(psnr(mse_reconstruction))
            logit_list.append(logit[i].item())
            label_list.append(label[i].item())
            avg_loss.update(loss[i].item(), 1)

        if args.evaluate_time:
            single_time.update((time.time() - a)*1000)
            progress.print(counter)
            # print("Single batch time cost {}ms, loss {}".format(1000*(time.time()-a), loss.mean().item()))

    anomaly_score_total_list = np.asarray(anomaly_score_list(psnr_list))
    label_list = np.asarray(label_list)
    logit_list = np.asarray(logit_list)
    assert anomaly_score_total_list.size == label_list.size, "INFERENCE LENGTH MUST MATCH LABEL LENGTH."

    final_score = 0.1 * logit_list + 0.9 * (1 - anomaly_score_total_list)
    # final_score = logit_list
    accuracy = roc_auc_score(y_true=label_list, y_score=final_score)
    # plot_AUC(anomaly_score_total_list, np.expand_dims(1 - labels_list, 0))
    print("EVALUATE FRAME NUMBER: ", anomaly_score_total_list.size)
    return accuracy, avg_loss.avg


def evaluate_two_stage_object(model, test_batch, args):
    avg_loss = metric.AverageMeter('avg_loss', ':.4e')
    single_time = metric.AverageMeter('Time', ':6.3f')
    progress = metric.ProgressMeter(len(test_batch), avg_loss, single_time, prefix="Evaluation: ")

    model.eval()
    label_list = []
    psnr_list = []
    logit_list = []
    ct = 0
    counter = 0
    for k, (images, labels, bboxes) in enumerate(test_batch):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bboxes = [x.cuda(non_blocking=True) for x in bboxes]
        a = time.time()
        counter += 1

        patches, patch_labels, bbox_num = get_object_images(images, labels, bboxes, args)  # [K,C,H,W] [K] [B]

        if patches is None:
            for i in range(len(labels)):
                label_list.append(labels[i].item())
                psnr_list.append(100.0)
        else:
            del images
            batch_size_now = len(bbox_num)
            ct += patches.size()[0]

            label = labels if args.label else None
            channel = (patches.size()[1] // args.c - 1) * args.c
            input_image = patches[:, 0:channel]
            target_image = patches[:, channel:]

            with autocast():
                reconstructed_image, loss, logit = model.forward(input_image, gt=target_image, label=label, train=False)
                loss = loss['pixel_loss'].view(loss['pixel_loss'].shape[0], -1).mean(1)
            assert len(loss) == len(label), "During inference, loss sample number must match label sample number."

            start_ = 0
            for i, num_ in enumerate(bbox_num):  # per sample in batch
                logit_per_sample = torch.max(logit[start_: start_ + num_]).item() if num_ > 0 else 0
                loss_per_sample = torch.max(loss[start_: start_ + num_]).item() if num_ > 0 else 0
                psnr_list.append(psnr(loss_per_sample))  # TODO: Max or Mean
                logit_list.append(logit_per_sample)
                label_list.append(labels[i].item())
                avg_loss.update(loss_per_sample, batch_size_now)
                start_ += num_

            assert start_ == logit.size()[0], "patch num and bbox_num doesn't match"

            if args.evaluate_time:
                single_time.update((time.time() - a)*1000)
                progress.print(counter)
                # print("Single batch time cost {}ms, loss {}".format(1000*(time.time()-a), loss.mean().item()))

    anomaly_score_total_list = np.asarray(anomaly_score_list(psnr_list))
    label_list = np.asarray(label_list)
    logit_list = np.asarray(logit_list)
    assert anomaly_score_total_list.size == label_list.size and anomaly_score_total_list.size == logit_list.size, "INFERENCE LENGTH MUST MATCH LABEL LENGTH."

    final_score = 0.1*logit_list+0.9*(1-anomaly_score_total_list)
    # final_score = logit_list
    accuracy = roc_auc_score(y_true=label_list, y_score=final_score)
    # accuracy1 = roc_auc_score(y_true=label_list, y_score=1-anomaly_score_total_list)
    # plot_AUC(anomaly_score_total_list, np.expand_dims(1 - label_list, 0))
    print("EVAL FRAME & BOX NUMBER & ACC : ", anomaly_score_total_list.size, ct, accuracy*100)

    return accuracy, avg_loss.avg

# def evaluate(model, test_batch, args):
#     avg_loss = metric.AverageMeter('avg_loss', ':.4e')
#     model.eval()
#
#     labels = np.load('./data/frame_labels_' + args.dataset_type + '.npy')
#     test_dir = os.path.join(args.dataset_path, args.dataset_type, args.testing_folder)
#
#     videos = OrderedDict()
#     videos_list = sorted(glob.glob(os.path.join(test_dir, '*')))
#     for video in videos_list:
#         video_name = video.split('/')[-1]
#         videos[video_name] = {}
#         videos[video_name]['path'] = video
#         videos[video_name]['frame'] = glob.glob(os.path.join(video, '*.jpg'))
#         videos[video_name]['frame'].sort()
#         # test frame number per video
#         videos[video_name]['length'] = len(videos[video_name]['frame'])
#
#     labels_list = []
#     label_length = 0
#     psnr_list = {}
#     feature_distance_list = {}
#
#     # Setting for video anomaly detection
#     for video in sorted(videos_list):
#         video_name = video.split('/')[-1]
#         # load label  (labels from 4th frame in each video) 1 -> abnormal
#         labels_list = np.append(labels_list,
#                                 labels[0][
#                                 (args.t_length - 1) + label_length:videos[video_name]['length'] + label_length])
#         label_length += videos[video_name]['length']
#         psnr_list[video_name] = []
#         feature_distance_list[video_name] = []
#
#     label_length = 0
#     video_num = 0
#     label_length += videos[videos_list[video_num].split('/')[-1]]['length']
#     loss_record = []
#
#     for k, (imgs, labels) in enumerate(test_batch):
#
#         if k == label_length - 4 * (video_num + 1):
#             video_num += 1
#             label_length += videos[videos_list[video_num].split('/')[-1]]['length']
#
#         imgs = Variable(imgs).cuda()
#         channel = args.c * (args.t_length - 1)
#         output, loss = model.forward(imgs[:, 0:channel], gt=imgs[:, channel:], train=False)
#
#         loss = loss['pixel_loss'].view(loss['pixel_loss'].shape[0], -1).mean(1)
#         for i in range(len(loss)):
#             mse_reconstruction = loss[i].item()
#             psnr_list[videos_list[video_num].split('/')[-1]].append(psnr(mse_reconstruction))
#
#     # Measuring the abnormality score and the AUC
#     anomaly_score_total_list = []
#     for video in sorted(videos_list):
#         video_name = video.split('/')[-1]
#         anomaly_score_total_list += score_sum_single(anomaly_score_list(psnr_list[video_name]))
#
#     avg_loss.update(loss[i].item(), 1)
#
#     anomaly_score_total_list = np.asarray(anomaly_score_total_list)
#     assert labels_list.size == anomaly_score_total_list.size, "INFERENCE LENGTH MUST MATCH LABEL LENGTH."
#     accuracy = AUC(anomaly_score_total_list, np.expand_dims(1 - labels_list, 0))
#     # plot_AUC(anomaly_score_total_list, np.expand_dims(1 - labels_list, 0))
#     return accuracy


def _frame_from_video(video, args):
    while video.isOpened():
        success, frame = video.read()

        if success:
            yield frame  # preprocessed
        else:
            break


def process_predictions(data, model, frame, args):
    start_time = time.time()
    batch = []
    for image in data:
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        batch.append(image)
    batch = torch.from_numpy(np.concatenate(batch, axis=0)).unsqueeze(0)
    images = batch.cuda()
    channel = (images.size()[1] // args.c - 1) * args.c
    if args.is_amp:
        with autocast():
            predict, loss = model.forward(images[:, 0:channel], gt=images[:, channel:], train=False)
            loss = loss['pixel_loss'].mean().item()
    else:
        predict, loss = model.forward(images[:, 0:channel], gt=images[:, channel:], train=False)
        loss = loss['pixel_loss'].mean().item()

    predict = 255 * (predict + 1.) / 2
    predict = predict.squeeze(0).byte().cpu().numpy().transpose((1,2,0))
    predict = cv2.resize(predict, (frame.shape[1], frame.shape[0]))
    predict = cv2.cvtColor(predict, cv2.COLOR_RGB2BGR)

    score = psnr(loss)
    if score < 20:
        res = "Anomaly,"+str(score)
    else:
        res = "Normal,"+str(score)

    print("The category of frame is {}, psnr is {}, time cost: {}".format(res, score, time.time()-start_time))
    return res, predict, frame

# def process_predictions_new(data, model, frame, args):
#     start_time = time.time()
#     batch = []
#     for image in data:
#         image = torch.from_numpy(image).permute(2, 0, 1).float()
#         batch.append(image)
#     batch = torch.from_numpy(np.concatenate(batch, axis=0)).unsqueeze(0)
#     images = Variable(batch).cuda()
#     channel = args.c * (args.t_length - 1)
#     if args.is_amp:
#         with autocast():
#             predict, loss = model.forward(images[:, 0:channel], gt=images[:, channel:], train=False)
#             loss = loss['pixel_loss'].mean().item()
#     else:
#         predict, loss = model.forward(images[:, 0:channel], gt=images[:, channel:], train=False)
#         loss = loss['pixel_loss'].mean().item()
#
#     predict = 255 * (predict + 1.) / 2
#     predict = predict.squeeze(0).byte().cpu().numpy().transpose((1,2,0))
#     predict = cv2.resize(predict, (frame.shape[1], frame.shape[0]))
#     predict = cv2.cvtColor(predict, cv2.COLOR_RGB2BGR)
#
#     score = psnr(loss)
#     if score < 20:
#         res = "Anomaly"
#     else:
#         res = "Normal"
#
#     print("The category of frame is {}, psnr is {}, time cost: {}".format(res, score, time.time()-start_time))
#     return res, predict, frame


def run_on_video(cam, model, args):
    frames = _frame_from_video(cam, args)
    data = []
    args.h, args.w = 64, 64
    frame_length = args.t_length
    frame_interval = args.interval
    for i, frame in enumerate(frames):
        if i % frame_interval != 0:  # interval
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (args.w, args.h)).astype(dtype=np.float32)
        # normalization (-1 to 1)
        image = 2 * (image / 255) - 1.0
        assert len(data) <= frame_length, "fragment length must less than 5."
        if len(data) == frame_length:
            data = data[1:] + [image]
            yield process_predictions(data, model, frame, args)
        elif len(data) == frame_length - 1:
            data.append(image)
            yield process_predictions(data, model, frame, args)
        else:
            data.append(image)


def demo(model, args, video=""):
    model.eval()

    if args.demo == "webcam":
        cam = cv2.VideoCapture(0)
        for res, predict, frame in tqdm.tqdm(run_on_video(cam, model, args)):
            font = cv2.FONT_HERSHEY_SIMPLEX  # 使用默认字体
            frame = np.concatenate((frame, predict), axis=1)
            frame = cv2.putText(frame, res, (100, 100), font, 5, (205, 92, 92), 8)
            # cv2.imwrite(os.path.join("/home/rail/Documents/airs_scene/frames", str(int(time.time()))+".jpg"), vis)
            WINDOW_NAME = "Anomaly Detection"
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) == 27:
                break
        cam.release()
        cv2.destroyAllWindows()
    else:
        print("MUST SPECIFY A DEMO TYPE.")


def visualize(recon, gt):
    b, c, h, w = recon.size()
    for i in range(b):
        img1, img2 = recon[i], gt[i]
        img = torch.cat((img1, img2), dim=2)
        img = 255. * (img + 1.) / 2.
        img = img.squeeze(0).byte().cpu().numpy().transpose((1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (600, 300))
        frame, name = img, str(int(time.time()*1000))
        cv2.imwrite(os.path.join("/data/miaobo/tmp", name+".jpg"), frame)

    return True


def visualize_single(image):
    b, c, h, w = image.size()
    for i in range(b):
        img = image[i]
        img = 255. * (img + 1.) / 2.
        img = img.byte().cpu().numpy().transpose((1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        frame, name = img, str(int(time.time()*1000))
        cv2.imwrite(os.path.join("/data/miaobo/tmp", name+".jpg"), frame)

    return True
