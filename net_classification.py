import os, sys
import numpy as np
import time
import logging
import warnings
import setproctitle
import random
from torch.cuda.amp import autocast
try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch import distributed
from torch.backends import cudnn
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR

# Personal packages
from parser_parameters import *
from utils.utils import *
from utils.demo_classification import *
from utils import lr_scheduler, metric, prefetch, summary
from model.model import *
from utils.datasets_classification import *

logging.basicConfig(level=logging.DEBUG)
logging.info('current time is {}'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

best_acc1 = 0


def main(args):
    # Add global variable here and put it in spawn, then it could be updated by all processes
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    ngpus_per_node = torch.cuda.device_count()
    args.ngpus_per_node = ngpus_per_node
    args.world_size = ngpus_per_node * args.world_size
    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    print("INFO:PyTorch: Initialize process group for distributed training")
    if args.dist_url == "env://" and args.rank == -1:
        args.rank = int(os.environ["RANK"])

    args.rank = args.rank * ngpus_per_node + gpu

    distributed.init_process_group(backend=args.dist_backend,
                                   init_method=args.dist_url,
                                   world_size=args.world_size,
                                   rank=args.rank)

    if args.gpu is not None:
        if not args.evaluate:
            print("INFO:PyTorch: Use GPU: {} for training, the rank of this GPU is {}".format(args.gpu, args.rank))
        else:
            print("INFO:PyTorch: Use GPU: {} for evaluating, the rank of this GPU is {}".format(args.gpu, args.rank))

    criterion = nn.CrossEntropyLoss()
    model = eval(args.arch)()
    print("INFO:PyTorch: The number of parameters is {}". format(get_the_number_of_params(model)))

    if args.is_syncbn:
        print("INFO:PyTorch: convert torch.nn.BatchNormND layer in the model to torch.nn.SyncBatchNorm layer")
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)

        args.batch_size = int(args.batch_size / ngpus_per_node)
        args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)

    else:
        model.cuda()

    param_groups = model.parameters()
    args.param_groups = param_groups

    if args.optim == 'adam':
        optimizer = torch.optim.Adam(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optim == 'adamW':
        optimizer = optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay, eps=args.epsilon)
    else:
        optimizer = torch.optim.SGD(param_groups, args.lr, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=True if args.is_nesterov else False)

    args.scaler = GradScaler()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("INFO:PyTorch: => loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)

            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("INFO:PyTorch: => loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint['epoch']))
        else:
            print("INFO:PyTorch: => no checkpoint found at '{}'".format(args.resume))

    if args.lr_mode == 'step':
        scheduler = MultiStepLR(optimizer,
                                milestones=args.lr_milestones,
                                gamma=args.lr_step_multiplier,
                                last_epoch=args.start_epoch - 1)
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # accelarate the training
    torch.backends.cudnn.benchmark = True

    if args.demo:  # TODO: Update code
        with torch.no_grad():
            demo(model, args)
        return None

    # Data loading code
    traindir = os.path.join(args.dataset_path, args.dataset_type, args.training_folder)
    testdir = os.path.join(args.dataset_path, args.dataset_type, args.testing_folder)
    labeldir = os.path.join(args.dataset_path, args.dataset_type, args.label_folder) if args.label_folder else None
    train_batch, test_batch, train_sampler = load_dataset(traindir, testdir, labeldir, args)

    if args.evaluate:
        with torch.no_grad():
            acc, test_average_loss = evaluate_new(model, test_batch, args)
        print('Test AUC: {}%'.format(acc * 100))
        return None

    # LOGGING
    setproctitle.setproctitle(args.dataset_type + '_' + args.arch + '_rank{}'.format(args.rank))
    log_path = os.path.join('/home/miaobo/project/anomaly_demo2', 'runs', '_'.join([args.suffix, args.dataset_type,
                                              args.arch, args.discriminator if args.discriminator else ""]))
    val_writer = SummaryWriter(log_dir=log_path)
    print("Tensorboard log: {}".format(log_path))

    for epoch in range(args.start_epoch, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        train_average_loss = train(train_batch, model, criterion, optimizer, epoch, args)
        scheduler.step()

        if epoch % args.eval_per_epoch == 0:
            print("Starting EVALUATION ......")
            a = time.time()
            with torch.no_grad():
                acc1 = evaluate_new(model, test_batch, args)

            print("EVALUATION TIME COST: {} min".format(int(time.time()-a)/60))

            is_best = acc1 > best_acc1
            best_acc1 = max(acc1, best_acc1)

            if args.rank % ngpus_per_node == 0:

                print("epoch: {}, EVALUATION AUC: {}, HISTORY BEST AUC: {}".format(epoch, acc1 * 100, best_acc1 * 100))
                # summary per epoch
                val_writer.add_scalar('avg_acc1', acc1, global_step=epoch)
                val_writer.add_scalar('best_acc1', best_acc1, global_step=epoch)
                val_writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step=epoch)

                # save checkpoints
                filename = '_'.join([args.suffix, args.arch, args.discriminator if args.discriminator else "",
                                     args.dataset_type, "checkpoint.pth.tar"])
                ckpt_dict = {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'best_acc1': best_acc1,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                }
                metric.save_checkpoint(ckpt_dict, is_best, args, filename=filename)

                # reload best model every n epoch
                if args.reload_best and epoch % args.reload_interval == 0:
                    print("Reloading model from best_{}, the acc is changed from {} to {}".format(filename, acc1, best_acc1))
                    if args.gpu is None:
                        checkpoint = torch.load(os.path.join(args.model_dir, "best_"+filename))
                    else:
                        # Map model to be loaded to specified single gpu.
                        loc = 'cuda:{}'.format(args.gpu)
                        checkpoint = torch.load(os.path.join(args.model_dir, "best_"+filename), map_location=loc)
                    model.load_state_dict(checkpoint['model'])

    torch.cuda.empty_cache()
    val_writer.close()


def train(train_batch, model, criterion, optimizer, epoch, args):
    batch_time = metric.AverageMeter('Time', ':6.3f')
    data_time = metric.AverageMeter('Data', ':6.3f')
    avg_loss = metric.AverageMeter('avg_loss', ':.4e')
    progress = metric.ProgressMeter(len(train_batch), batch_time, data_time, avg_loss, prefix="Epoch: [{}]".format(epoch))

    model.train()
    end = time.time()

    if args.object_detection:
        prefetcher = prefetch.data_prefetcher_trible(train_batch)
        images, labels, bboxes = prefetcher.next()
    else:
        prefetcher = prefetch.data_prefetcher(train_batch)
        images, labels = prefetcher.next()
        bboxes = None

    optimizer.zero_grad()
    counter = -1
    while images is not None:
        data_time.update(time.time() - end)
        counter += 1

        # whether split into object
        if args.object_detection:
            # 5 - 10 ms
            patches, labels, bbox_num = get_object_images(images, labels, bboxes, args)  # [K,C,H,W] [K] [B]
            del images
            batch_size_now = len(bbox_num)
        else:
            patches = images
            batch_size_now = images.size()[0]

        if patches is None:    # prevent no input
            if args.object_detection:
                images, labels, bboxes = prefetcher.next()
            else:
                images, labels = prefetcher.next()
                bboxes = None

            continue

        label = labels if args.label else None
        input_image = patches

        optimizer.zero_grad()
        with autocast():
            logit = model.forward(input_image)
            loss = criterion(logit, label.view(-1))

        args.scaler.scale(loss).backward()
        args.scaler.step(optimizer)
        args.scaler.update()
        avg_loss.update(loss.mean().item(), batch_size_now)

        batch_time.update(time.time() - end)
        end = time.time()
        if args.rank % args.ngpus_per_node == 0:
            if counter % args.print_freq == 0:
                progress.print(counter)

        if args.object_detection:
            images, labels, bboxes = prefetcher.next()
        else:
            images, labels = prefetcher.next()
            bboxes = None

    print("Training sample number of epoch {} is: {}".format(epoch, counter*int(args.batch_size)))
    return avg_loss.avg


if __name__ == "__main__":
    # Set the random seed manually for reproducibility.
    np.random.seed(2)
    torch.manual_seed(2)
    torch.cuda.manual_seed_all(2)

    args = default_argument_parser().parse_args()
    print(args)
    main(args)
