from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse


def default_argument_parser():
    parser = argparse.ArgumentParser(description="")

    # arch and dataset
    parser.add_argument('--dataset_type', type=str, default='ucf', help='type: ped2, avenue, shanghaitech, ucf')
    parser.add_argument('--dataset_path', type=str, default='/data/miaobo', help='directory of data')
    parser.add_argument('--training_folder', type=str, default='training/frames', help='directory of training data')
    parser.add_argument('--testing_folder', type=str, default='testing/frames', help='directory of testing data')
    parser.add_argument('--label_folder', type=str, default='label', help='directory of label data')
    parser.add_argument('--label', default=1, type=int, help='whether has data label in training')
    parser.add_argument('--model_dir', type=str, default="/home/miaobo/project/anomaly_demo2/ckpt", help='directory of model')

    parser.add_argument('--object_detection', default=0, type=int, help='whether use object level')
    parser.add_argument('--path_h', default=64, type=int, help='object h')
    parser.add_argument('--path_w', default=64, type=int, help='object w')

    parser.add_argument('--arch', metavar='ARCH', default='convAE', help='generator architecture')
    parser.add_argument('--discriminator', metavar='ARCH', default='', help='discriminator architecture')
    parser.add_argument('--memory_arch', metavar='ARCH', default='Memory', help='memory architecture')
    parser.add_argument('--encoder_arch', metavar='ARCH', default='Encoder', help='encoder architecture')
    parser.add_argument('--decoder_arch', metavar='ARCH', default='Decoder', help='decoder architecture')
    parser.add_argument('--suffix', type=str, default='', help='experiment files title suffix')
    parser.add_argument('--proc_name', type=str, default='anomaly_detection', help='The name of the process.')

    parser.add_argument('--visualize_input', default=0, type=int, help='whether visualize input images')
    parser.add_argument('--visualize', default=0, type=int, help='whether visualize reconstruction images')
    parser.add_argument('--demo', type=str, default='', help='webcam, video, robot')

    # test or continue train config
    parser.add_argument("--evaluate", action="store_true", help="evaluate models on validation set")
    parser.add_argument('--evaluate_time', default=0, type=int, help='show evaluate time cost')
    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')

    # train config
    parser.add_argument('--epochs', type=int, default=60, help='number of epochs for training')
    parser.add_argument('-b', '--batch_size', default=4, type=int, metavar='N', help='mini-batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, metavar='N', help='batch size for test')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--eval_per_epoch', default=1, type=int, help='run evaluation per eval_per_epoch')
    parser.add_argument('--reload_best', default=1, type=int, help='reload best model in training')
    parser.add_argument('--reload_interval', default=1, type=int, help='reload best model in training')
    parser.add_argument('--is_syncbn', default=0, type=int, help='use nn.SyncBatchNorm or not')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
    parser.add_argument('--max_ckpt_nums', default=5, type=int, help='maximum number of ckpts.')

    parser.add_argument('--loss_compact', type=float, default=0.1, help='weight of the feature compactness loss')
    parser.add_argument('--loss_separate', type=float, default=0.1, help='weight of the feature separateness loss')
    parser.add_argument('--wgan_gp', default=1, type=int, help='wgan-gp gradient penalty')

    # optimizer
    parser.add_argument('--optim', type=str, choices=['adam', 'sgd', 'adamW'], default='adamW', help='optimizer')
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='optimizer momentum (default: 0.9)')
    parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay')
    parser.add_argument('--epsilon', default=1e-8, type=float, help='epsilon for adamW')
    parser.add_argument('--gradient_clip', type=float, default=None, help='control gradient range')
    parser.add_argument('--is_nesterov', default=1, type=int, help='using Nesterov accelerated gradient or not')
    # lr setting
    parser.add_argument('--lr_mode', type=str, choices=['cos', 'step', 'poly', 'HTD'], default='cos', help='strategy of the learning rate')
    parser.add_argument('--lr_milestones', nargs='+', type=int, default=[20,30,40], help='epochs at which we take a learning-rate step')
    parser.add_argument('--lr_step_multiplier', default=0.1, type=float, metavar='M', help='lr multiplier at lr_milestones (default: 0.1)')
    parser.add_argument('--lr_multiplier', type=float, default=1.0, help='Learning rate multiplier for the unpretrained model.')
    parser.add_argument('--slow_start_epochs', type=int, default=5, help='Training model with small learning rate for few 10 epochs.')
    parser.add_argument('--slow_start_lr', type=float, default=2e-4, help='Learning rate employed during slow start.')
    parser.add_argument('--end_lr', type=float, default=1e-6, help='The ending (minimize) learning rate.')
    # AMP
    parser.add_argument('--is_amp', default=1, type=int, help='using Pytorch1.6 Automatic Mixed Precision (AMP) 32->16')
    parser.add_argument('--is_apex_amp', default=1, type=int, help='using NVIDIA APEX Automatic Mixed Precision (AMP) 32->16')
    parser.add_argument('--amp_opt_level', type=str, default='O1', help='optimization level of apex amp.')

    # GPU config
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--multiprocessing_distributed', action='store_true', help='Use multi-processing distributed training to launch N processes per node')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--workers_test', type=int, default=1, help='number of workers for the test loader')
    parser.add_argument('--dist_backend', type=str, default='nccl', choices=['nccl', 'gloo'], help='Name of the backend to use.')
    parser.add_argument('--world_size', default=1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--dist_url', type=str, default='env://', help='specifying how to initialize the package.')

    # Parameters
    parser.add_argument('--memory', type=int, default=1, help='whether use memory')
    parser.add_argument('--h', type=int, default=256, help='height of input images')
    parser.add_argument('--w', type=int, default=256, help='width of input images')
    parser.add_argument('--multi_size_training', default=0, type=int, help='multi-size training')
    parser.add_argument('--augmentation', default=0, type=int, help='Augmentation.')
    parser.add_argument('--c', type=int, default=3, help='channel of input images')
    parser.add_argument('--t_length', type=int, default=5, help='length of the frame sequences')
    parser.add_argument('--interval', type=int, default=1, help='interval of the frame sequences')
    parser.add_argument('--msize', type=int, default=10, help='number of the memory items')
    parser.add_argument('--mdim', type=int, default=512, help='channel dimension of the memory items')
    parser.add_argument('--gsize', type=int, default=10, help='number of the global memory items')
    parser.add_argument('--gdim', type=int, default=512, help='channel dimension of the global memory items')
    parser.add_argument('--alpha', type=float, default=1, help='weight for the anomality score')

    # Log config
    parser.add_argument('--exp_dir', type=str, default='', help='directory of log')
    parser.add_argument('--is_summary', default=0, type=int, help='only get the Params and FLOPs of the model.')
    parser.add_argument('-p', '--print_freq', default=20, type=int, metavar='N', help='print frequency between batches (default: 30)')

    # Deeplab
    parser.add_argument("--restore_resolution", action="store_true", help="restore resolution or generate 1/4 output")
    parser.add_argument('--bn', type=str, default='nn.BatchNorm2d', help='bn method')

    # optical flow
    parser.add_argument('--loss_opticalflow', type=float, default=0.1, help='weight of the feature compactness loss')
    return parser
