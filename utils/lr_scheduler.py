# coding=utf-8
"""
some training utils.
reference:
	https://github.com/zhanghang1989/PyTorch-Encoding
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import os
import math
# import torch
# from torchvision.utils import make_grid


class lr_scheduler(object):
	"""learning rate scheduler
	step mode: 		``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``
	cosine mode: 	``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``
	poly mode: 		``lr = baselr * (1 - iter/maxiter) ^ 0.9``

	Args:
		init_lr:			initial learnig rate;
		mode:				['cos', 'poly', 'HTD'];
		max_iter:			max iterations of training.
							**Here one iterations means one epoch**.
		slow_start_steps:	slow start steps of training;
		slow_start_lr:		slow start learning rate for slow_start_steps;
		end_lr:				minimum learning rate.
	"""
	def __init__(self, init_lr,
						mode='poly',
						#num_epochs=300,
						max_iter=300,
						lr_milestones=[100, 150],
						slow_start_steps=10,
						slow_start_lr=1e-4,
						end_lr=1e-3,
						lr_step_multiplier=0.1,
						multiplier=1.0,
						lower_bound=-6.0,
						upper_bound=3.0):

		assert mode in ['cos', 'poly', 'HTD']
		self.init_lr = init_lr
		self.now_lr = self.init_lr
		self.mode = mode
		#self.num_epochs = num_epochs
		self.max_iter = max_iter
		self.slow_start_steps = slow_start_steps
		self.slow_start_lr = slow_start_lr
		self.slow_max_iter = self.max_iter - self.slow_start_steps
		self.end_lr = end_lr
		self.multiplier = multiplier
		# step mode
		if self.mode == 'step':
			assert lr_milestones
		self.lr_milestones = lr_milestones
		self.lr_step_multiplier = lr_step_multiplier
		self.lower_bound = lower_bound
		self.upper_bound = upper_bound
		# log info
		print('INFO:PyTorch: Using {} learning rate scheduler!'.format(self.mode))

	def __call__(self, optimizer, global_step):
		"""call method"""
		step_now = 1.0 * global_step

		if global_step <= self.slow_start_steps:
			# slow start strategy -- warm up
			# see 	https://arxiv.org/pdf/1812.01187.pdf
			# 	Bag of Tricks for Image Classification with Convolutional Neural Networks
			# for details.
			lr = (step_now / self.slow_start_steps) * (self.init_lr - self.slow_start_lr)
			lr = lr + self.slow_start_lr
			lr = min(lr, self.init_lr)
		else:
			step_now = step_now - self.slow_start_steps
			# calculate the learning rate
			if self.mode == 'cos':
				lr = 0.5 * self.init_lr * (1.0 + math.cos(step_now / self.slow_max_iter * math.pi))
			elif self.mode == 'poly':
				lr = self.init_lr * pow(1.0 - step_now / self.slow_max_iter, 0.9)
			elif self.mode == 'HTD':
				"""
				Stochastic Gradient Descent with Hyperbolic-Tangent Decay on Classification.
				https://arxiv.org/pdf/1806.01593.pdf
				"""
				ratio = step_now / self.slow_max_iter
				lr = 0.5 * self.init_lr * (1.0 - math.tanh(self.lower_bound + (self.upper_bound - self.lower_bound) * ratio))
			elif self.mode == 'step':

				raise NotImplementedError
			else:
				raise NotImplementedError
			lr = max(lr, self.end_lr)

		self.now_lr = lr
		# adjust learning rate
		self._adjust_learning_rate(optimizer, lr)

	def _adjust_learning_rate(self, optimizer, lr):
		"""adjust the leaning rate"""
		if len(optimizer.param_groups) == 1:
			optimizer.param_groups[0]['lr'] = lr
		else:
			# BE CAREFUL HERE!!!
			# 0 -- the backbone conv weights with weight decay
			# 1 -- the bn params and bias of backbone without weight decay
			# 2 -- the weights of other layers with weight decay
			# 3 -- the bn params and bias of other layers without weigth decay
			optimizer.param_groups[0]['lr'] = lr
			optimizer.param_groups[1]['lr'] = lr
			for i in range(2, len(optimizer.param_groups)):
				optimizer.param_groups[i]['lr'] = lr * self.multiplier


def scale_lr_and_momentum(args):
	"""
	Scale hyperparameters given the adjusted batch_size from input
	hyperparameters and batch size

	Arguements:
		args: holds the script arguments
	"""
	print('=> adjusting learning rate and momentum. '
			'Original lr: {args.lr}, Original momentum: {args.momentum}')
	if 'cifar' in args.dataset:
		std_b_size = 128
	elif 'imagenet' in args.dataset:
		std_b_size = 256
	else:
		raise NotImplementedError

	old_momentum = args.momentum
	args.momentum = old_momentum ** (args.batch_size / std_b_size)
	# args.lr = args.lr * (args.batch_size / std_b_size *
	#                     (1 - args.momentum) / (1 - old_momentum))
	#
	args.lr = args.lr * (args.batch_size / std_b_size)
	print(f'lr adjusted to: {args.lr}, momentum adjusted to: {args.momentum}')

	return args


def get_parameter_groups(model, norm_weight_decay=0):
	"""
	Separate model parameters from scale and bias parameters following norm if
	training imagenet
	"""
	model_params = []
	norm_params = []

	for name, p in model.named_parameters():
		if p.requires_grad:
			# if 'fc' not in name and ('norm' in name or 'bias' in name):
			if 'norm' in name or 'bias' in name:
				norm_params += [p]
			else:
				model_params += [p]

	return [{'params': model_params},
			{'params': norm_params,
				'weight_decay': norm_weight_decay}]
