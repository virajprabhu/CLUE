import os
import random
import numpy as np
import h5py

import PIL
from PIL import Image

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

import utils

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

class DomainNetDataset(torch.utils.data.Dataset):
	def __init__(self, name, domain, split, transforms):
		self.name = 'DomainNet'
		self.domain = domain
		self.split = split
		self.file_path = os.path.join('/srv/testing/prithvi/', 'cond-comp-dg', \
									  'DomainNet-Large-tv_1.0', '{}_{}.h5'.format(self.domain, self.split))

		self.data, self.labels = None, None
		with h5py.File(self.file_path, 'r') as file:
			self.dataset_len = len(file["images"])
			self.num_classes = len(set(list(np.array(file['labels']))))
		self.transforms = transforms

	def __len__(self):
		return self.dataset_len

	def __getitem__(self, idx):
		if self.data is None:
			self.data = h5py.File(self.file_path, 'r')["images"]
			self.labels = h5py.File(self.file_path, 'r')["labels"]
		datum, label = Image.fromarray(np.uint8(np.array(self.data[idx]))), np.array(self.labels[idx])
		return (self.transforms(datum), int(label))

	def get_num_classes(self):
		# return self.num_classes
		#! Hardcoded
		return 345

class ASDADataset:
	# Active Semi-supervised DA Dataset class
	def __init__(self, name, data_dir='data', valid_ratio=0.2, batch_size=128, augment=False):
		self.name = name
		self.data_dir = data_dir
		self.valid_ratio = valid_ratio
		self.batch_size = batch_size
		self.train_size = None
		self.train_dataset = None
		self.num_classes = None

	def get_num_classes(self):
		return self.num_classes

	def get_dsets(self, normalize=True, apply_transforms=True):
		if self.name == "mnist":
			mean, std = 0.5, 0.5
			normalize_transform = transforms.Normalize((mean,), (std,)) \
								  if normalize else transforms.Normalize((0,), (1,))
			train_transforms = transforms.Compose([
									   transforms.ToTensor(),
									   normalize_transform
								   ])
			test_transforms = transforms.Compose([
									   transforms.ToTensor(),
									   normalize_transform
									])

			train_dataset = datasets.MNIST(self.data_dir, train=True, download=True, transform=train_transforms)
			val_dataset = datasets.MNIST(self.data_dir, train=True, download=True, transform=test_transforms)
			test_dataset = datasets.MNIST(self.data_dir, train=False, download=True, transform=test_transforms)
			train_dataset.name, val_dataset.name, test_dataset.name = 'DIGITS','DIGITS', 'DIGITS'
			self.num_classes = 10
		
		elif self.name == "svhn":
			mean, std = 0.5, 0.5
			normalize_transform = transforms.Normalize((mean,), (std,)) \
								  if normalize else transforms.Normalize((0,), (1,))
			RGB2Gray = transforms.Lambda(lambda x: x.convert('L'))
			train_transforms = transforms.Compose([
								   RGB2Gray,
								   transforms.Resize((28, 28)),
								   transforms.ToTensor(),
								   normalize_transform
							   ])
			test_transforms = transforms.Compose([
								   RGB2Gray,
								   transforms.Resize((28, 28)),
								   transforms.ToTensor(),
								   normalize_transform
							   ])

			train_dataset = datasets.SVHN(self.data_dir, split='train', download=True, transform=train_transforms)
			val_dataset = datasets.SVHN(self.data_dir, split='train', download=True, transform=test_transforms)
			test_dataset = datasets.SVHN(self.data_dir, split='test', download=True, transform=test_transforms)
			self.num_classes = 10

		elif self.name in ["real", "quickdraw", "sketch", "infograph", "clipart", "painting"]:

			normalize_transform = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) \
								  if normalize else transforms.Normalize([0, 0, 0], [1, 1, 1])
			
			if apply_transforms:
				data_transforms = {
					'train': transforms.Compose([
						transforms.Resize(256),
						transforms.RandomCrop(224),
						transforms.RandomHorizontalFlip(),
						transforms.ToTensor(),
						normalize_transform
					]),
				}
			else:
				data_transforms = {
					'train': transforms.Compose([
						transforms.Resize(224),
						transforms.ToTensor(),
						normalize_transform
					]),
				}

			data_transforms['test'] = transforms.Compose([
					transforms.Resize(224),
					transforms.ToTensor(),
					normalize_transform
				])

			train_dataset = DomainNetDataset('DomainNet', self.name, 'train', data_transforms['train'])
			val_dataset = DomainNetDataset('DomainNet', self.name, 'val', data_transforms['test'])
			test_dataset = DomainNetDataset('DomainNet', self.name, 'test', data_transforms['test'])

			self.num_classes = train_dataset.get_num_classes()

		self.train_dataset = train_dataset
		self.val_dataset = val_dataset
		self.test_dataset = test_dataset

		return train_dataset, val_dataset, test_dataset

	def get_loaders(self, shuffle=True, num_workers=4, normalize=True):
		if not self.train_dataset: self.get_dsets(normalize=normalize)
		
		num_train = len(self.train_dataset)
		self.train_size = num_train

		if self.name in ["mnist", "svhn"]:
			
			indices = list(range(num_train))
			split = int(np.floor(self.valid_ratio * num_train))
			if shuffle == True: np.random.shuffle(indices)
			train_idx, valid_idx = indices[split:], indices[:split]
			
			train_sampler = SubsetRandomSampler(train_idx)
			valid_sampler = SubsetRandomSampler(valid_idx)

		elif self.name in ["real", "quickdraw", "sketch", "infograph", "painting", "clipart"]:

			train_idx = np.arange(len(self.train_dataset))
			train_sampler = SubsetRandomSampler(train_idx)
			valid_sampler = SubsetRandomSampler(np.arange(len(self.val_dataset)))

		train_loader = torch.utils.data.DataLoader(self.train_dataset, sampler=train_sampler, \
												   batch_size=self.batch_size, num_workers=num_workers)
		val_loader = torch.utils.data.DataLoader(self.val_dataset, sampler=valid_sampler, batch_size=self.batch_size)
		test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size)

		return train_loader, val_loader, test_loader, train_idx