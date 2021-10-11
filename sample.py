# -*- coding: utf-8 -*-
"""
Implements active learning sampling strategies
Adapted from https://github.com/ej0cl6/deep-active-learning
"""

import os
import copy
import random
import numpy as np

import scipy
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.utils.data.sampler import Sampler, SubsetRandomSampler

import utils
from utils import ActualSequentialSampler
from adapt.solvers.solver import get_solver

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

al_dict = {}
def register_strategy(name):
    def decorator(cls):
        al_dict[name] = cls
        return cls
    return decorator

def get_strategy(sample, *args):
	if sample not in al_dict: raise NotImplementedError
	return al_dict[sample](*args)

class SamplingStrategy:
	""" 
	Sampling Strategy wrapper class
	"""
	def __init__(self, dset, train_idx, model, discriminator, device, args, balanced=False):
		self.dset = dset
		if dset.name == 'DomainNet':
			self.num_classes = self.dset.get_num_classes()
		else:
			self.num_classes = len(set(dset.targets.numpy()))
		self.train_idx = np.array(train_idx)
		self.model = model
		self.discriminator = discriminator
		self.device = device
		self.args = args
		self.idxs_lb = np.zeros(len(self.train_idx), dtype=bool)

	def query(self, n):
		pass

	def update(self, idxs_lb):
		self.idxs_lb = idxs_lb

	def train(self, target_train_dset, da_round=1, src_loader=None, src_model=None):
		"""
		Driver train method
		"""
		best_val_acc, best_model = 0.0, None
		
		train_sampler = SubsetRandomSampler(self.train_idx[self.idxs_lb])
		tgt_sup_loader = torch.utils.data.DataLoader(target_train_dset, sampler=train_sampler, num_workers=4, \
												 	 batch_size=self.args.batch_size, drop_last=False)
		tgt_unsup_loader = torch.utils.data.DataLoader(target_train_dset, shuffle=True, num_workers=4, \
													   batch_size=self.args.batch_size, drop_last=False)
		opt_net_tgt = optim.Adam(self.model.parameters(), lr=self.args.adapt_lr, weight_decay=self.args.wd)

		# Update discriminator adversarially with classifier
		lr_scheduler = optim.lr_scheduler.StepLR(opt_net_tgt, 20, 0.5)
		solver = get_solver(self.args.da_strat, self.model, src_loader, tgt_sup_loader, tgt_unsup_loader, \
							self.train_idx, opt_net_tgt, da_round, self.device, self.args)
		
		for epoch in range(self.args.adapt_num_epochs):
			if self.args.da_strat == 'dann':
				opt_dis_adapt = optim.Adam(self.discriminator.parameters(), lr=self.args.adapt_lr, \
										   betas=(0.9, 0.999), weight_decay=0)
				solver.solve(epoch, self.discriminator, opt_dis_adapt)
			elif self.args.da_strat in ['ft', 'mme']:
				solver.solve(epoch)
			else:
				raise NotImplementedError
		
			lr_scheduler.step()

		return self.model

@register_strategy('uniform')
class RandomSampling(SamplingStrategy):
	"""
	Uniform sampling 
	"""
	def __init__(self, dset, train_idx, model, discriminator, device, args, balanced=False):
		super(RandomSampling, self).__init__(dset, train_idx, model, discriminator, device, args)
		self.labels = dset.labels if dset.name == 'DomainNet' else dset.targets
		self.classes = np.unique(self.labels)
		self.dset = dset
		self.balanced = balanced

	def query(self, n):
		return np.random.choice(np.where(self.idxs_lb==0)[0], n, replace=False)

@register_strategy('AADA')
class AADASampling(SamplingStrategy):
	"""
	Implements Active Adversarial Domain Adaptation (https://arxiv.org/abs/1904.07848)
	"""
	def __init__(self, dset, train_idx, model, discriminator, device, args, balanced=False):
		super(AADASampling, self).__init__(dset, train_idx, model, discriminator, device, args)
		self.D = None
		self.E = None

	def query(self, n):
		"""
		s(x) = frac{1-G*_d}{G_f(x))}{G*_d(G_f(x))} [Diversity] * H(G_y(G_f(x))) [Uncertainty]
		"""
		self.model.eval()
		idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
		train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
		data_loader = torch.utils.data.DataLoader(self.dset, sampler=train_sampler, num_workers=4, batch_size=64, drop_last=False)

		# Get diversity and entropy
		all_log_probs, all_scores = [], []
		with torch.no_grad():
			for batch_idx, (data, target) in enumerate(data_loader):
				data, target = data.to(self.device), target.to(self.device)
				scores = self.model(data)
				log_probs = nn.LogSoftmax(dim=1)(scores)
				all_scores.append(scores)
				all_log_probs.append(log_probs)

		all_scores = torch.cat(all_scores)
		all_log_probs = torch.cat(all_log_probs)

		all_probs = torch.exp(all_log_probs)
		disc_scores = nn.Softmax(dim=1)(self.discriminator(all_scores))
		# Compute diversity
		self.D = torch.div(disc_scores[:, 0], disc_scores[:, 1])
		# Compute entropy
		self.E = -(all_probs*all_log_probs).sum(1)
		scores = (self.D*self.E).sort(descending=True)[1]
		# Sample from top-2 % instances, as recommended by authors
		top_N = int(len(scores) * 0.02)
		q_idxs = np.random.choice(scores[:top_N].cpu().numpy(), n, replace=False)

		return idxs_unlabeled[q_idxs]

@register_strategy('BADGE')
class BADGESampling(SamplingStrategy):
	"""
	Implements BADGE: Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds (https://arxiv.org/abs/1906.03671)
	"""
	def __init__(self, dset, train_idx, model, discriminator, device, args, balanced=False):
		super(BADGESampling, self).__init__(dset, train_idx, model, discriminator, device, args)

	def query(self, n):
		idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
		train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
		data_loader = torch.utils.data.DataLoader(self.dset, sampler=train_sampler, num_workers=4, batch_size=self.args.batch_size, drop_last=False)
		self.model.eval()

		if self.args.cnn == 'LeNet':
			emb_dim = 500
		elif self.args.cnn == 'ResNet34':
			emb_dim = 512

		tgt_emb = torch.zeros([len(data_loader.sampler), self.num_classes])
		tgt_pen_emb = torch.zeros([len(data_loader.sampler), emb_dim])
		tgt_lab = torch.zeros(len(data_loader.sampler))
		tgt_preds = torch.zeros(len(data_loader.sampler))
		batch_sz = self.args.batch_size
		
		with torch.no_grad():
			for batch_idx, (data, target) in enumerate(data_loader):
				data, target = data.to(self.device), target.to(self.device)
				e1, e2 = self.model(data, with_emb=True)
				tgt_pen_emb[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e2.shape[0]), :] = e2.cpu()
				tgt_emb[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e1.shape[0]), :] = e1.cpu()
				tgt_lab[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e1.shape[0])] = target
				tgt_preds[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e1.shape[0])] = e1.argmax(dim=1, keepdim=True).squeeze()

		# Compute uncertainty gradient
		tgt_scores = nn.Softmax(dim=1)(tgt_emb)
		tgt_scores_delta = torch.zeros_like(tgt_scores)
		tgt_scores_delta[torch.arange(len(tgt_scores_delta)), tgt_preds.long()] = 1
		
		# Uncertainty embedding
		badge_uncertainty = (tgt_scores-tgt_scores_delta)

		# Seed with maximum uncertainty example
		max_norm = utils.row_norms(badge_uncertainty.cpu().numpy()).argmax()

		_, q_idxs = utils.kmeans_plus_plus_opt(badge_uncertainty.cpu().numpy(), tgt_pen_emb.cpu().numpy(), n, init=[max_norm])

		return idxs_unlabeled[q_idxs]

@register_strategy('CLUE')
class CLUESampling(SamplingStrategy):
	"""
	Implements CLUE: CLustering via Uncertainty-weighted Embeddings
	"""
	def __init__(self, dset, train_idx, model, discriminator, device, args, balanced=False):
		super(CLUESampling, self).__init__(dset, train_idx, model, discriminator, device, args)
		self.random_state = np.random.RandomState(1234)
		self.T = self.args.clue_softmax_t

	def query(self, n):
		idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
		train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
		data_loader = torch.utils.data.DataLoader(self.dset, sampler=train_sampler, num_workers=4, \
												  batch_size=self.args.batch_size, drop_last=False)
		self.model.eval()
		
		if self.args.cnn == 'LeNet':
			emb_dim = 500
		elif self.args.cnn == 'ResNet34':
			emb_dim = 512

		# Get embedding of target instances
		tgt_emb, tgt_lab, tgt_preds, tgt_pen_emb = utils.get_embedding(self.model, data_loader, self.device, self.num_classes, \
																	   self.args, with_emb=True, emb_dim=emb_dim)		
		tgt_pen_emb = tgt_pen_emb.cpu().numpy()
		tgt_scores = nn.Softmax(dim=1)(tgt_emb / self.T)
		tgt_scores += 1e-8
		sample_weights = -(tgt_scores*torch.log(tgt_scores)).sum(1).cpu().numpy()
		
		# Run weighted K-means over embeddings
		km = KMeans(n)
		km.fit(tgt_pen_emb, sample_weight=sample_weights)
		
		# Find nearest neighbors to inferred centroids
		dists = euclidean_distances(km.cluster_centers_, tgt_pen_emb)
		sort_idxs = dists.argsort(axis=1)
		q_idxs = []
		ax, rem = 0, n
		while rem > 0:
			q_idxs.extend(list(sort_idxs[:, ax][:rem]))
			q_idxs = list(set(q_idxs))
			rem = n-len(q_idxs)
			ax += 1

		return idxs_unlabeled[q_idxs]