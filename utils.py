# -*- coding: utf-8 -*-
import os
import json
import random
import math
from tqdm import tqdm, trange
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
import seaborn as sns

import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.autograd import Function, Variable
import torch.nn.functional as F
import torchvision.transforms
from torchvision import datasets, transforms
from torch.utils.data.sampler import Sampler, SubsetRandomSampler

from adapt.models.models import get_model
from adapt.solvers.solver import get_solver

random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed(1234)

######################################################################
##### Miscellaneous utilities and helper classes
######################################################################
class objectview(object):
	def __init__(self, d):
		self.__dict__ = d

class ActualSequentialSampler(Sampler):
	r"""Samples elements sequentially, always in the same order.

	Arguments:
		data_source (Dataset): dataset to sample from
	"""

	def __init__(self, data_source):
		self.data_source = data_source

	def __iter__(self):
		return iter(self.data_source)

	def __len__(self):
		return len(self.data_source)

######################################################################
##### Training utilities
######################################################################

class ReverseLayerF(Function):
	"""
	Gradient negation utility class
	"""				 
	@staticmethod
	def forward(ctx, x):
		return x.view_as(x)

	@staticmethod
	def backward(ctx, grad_output):
		output = grad_output.neg()
		return output, None

class ConditionalEntropyLoss(torch.nn.Module):
	"""
	Conditional entropy loss utility class
	"""				 
	def __init__(self):
		super(ConditionalEntropyLoss, self).__init__()

	def forward(self, x):
		b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
		b = b.sum(dim=1)
		return -1.0 * b.mean(dim=0)

######################################################################
##### Sampling utilities
######################################################################

def row_norms(X, squared=False):
	"""Row-wise (squared) Euclidean norm of X.
	Equivalent to np.sqrt((X * X).sum(axis=1)), but also supports sparse
	matrices and does not create an X.shape-sized temporary.
	Performs no input validation.
	Parameters
	----------
	X : array_like
		The input array
	squared : bool, optional (default = False)
		If True, return squared norms.
	Returns
	-------
	array_like
		The row-wise (squared) Euclidean norm of X.
	"""
	norms = np.einsum('ij,ij->i', X, X)

	if not squared:
		np.sqrt(norms, norms)
	return norms

def outer_product_opt(c1, d1, c2, d2):
	"""Computes euclidean distance between a1xb1 and a2xb2 without evaluating / storing cross products
	"""
	B1, B2 = c1.shape[0], c2.shape[0]
	t1 = np.matmul(np.matmul(c1[:, None, :], c1[:, None, :].swapaxes(2, 1)), np.matmul(d1[:, None, :], d1[:, None, :].swapaxes(2, 1)))
	t2 = np.matmul(np.matmul(c2[:, None, :], c2[:, None, :].swapaxes(2, 1)), np.matmul(d2[:, None, :], d2[:, None, :].swapaxes(2, 1)))
	t3 = np.matmul(c1, c2.T) * np.matmul(d1, d2.T)
	t1 = t1.reshape(B1, 1).repeat(B2, axis=1)
	t2 = t2.reshape(1, B2).repeat(B1, axis=0)
	return t1 + t2 - 2*t3

def kmeans_plus_plus_opt(X1, X2, n_clusters, init=[0], random_state=np.random.RandomState(1234), n_local_trials=None):
	"""Init n_clusters seeds according to k-means++ (adapted from scikit-learn source code)
	Parameters
	----------
	X1, X2 : array or sparse matrix
		The data to pick seeds for. To avoid memory copy, the input data
		should be double precision (dtype=np.float64).
	n_clusters : integer
		The number of seeds to choose
	init : list
		List of points already picked
	random_state : int, RandomState instance
		The generator used to initialize the centers. Use an int to make the
		randomness deterministic.
		See :term:`Glossary <random_state>`.
	n_local_trials : integer, optional
		The number of seeding trials for each center (except the first),
		of which the one reducing inertia the most is greedily chosen.
		Set to None to make the number of trials depend logarithmically
		on the number of seeds (2+log(k)); this is the default.
	Notes
	-----
	Selects initial cluster centers for k-mean clustering in a smart way
	to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
	"k-means++: the advantages of careful seeding". ACM-SIAM symposium
	on Discrete algorithms. 2007
	Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
	which is the implementation used in the aforementioned paper.
	"""

	n_samples, n_feat1 = X1.shape
	_, n_feat2 = X2.shape
	# x_squared_norms = row_norms(X, squared=True)
	centers1 = np.empty((n_clusters+len(init)-1, n_feat1), dtype=X1.dtype)
	centers2 = np.empty((n_clusters+len(init)-1, n_feat2), dtype=X1.dtype)

	idxs = np.empty((n_clusters+len(init)-1,), dtype=np.long)

	# Set the number of local seeding trials if none is given
	if n_local_trials is None:
		# This is what Arthur/Vassilvitskii tried, but did not report
		# specific results for other than mentioning in the conclusion
		# that it helped.
		n_local_trials = 2 + int(np.log(n_clusters))

	# Pick first center randomly
	center_id = init

	centers1[:len(init)] = X1[center_id]
	centers2[:len(init)] = X2[center_id]
	idxs[:len(init)] = center_id

	# Initialize list of closest distances and calculate current potential
	distance_to_candidates = outer_product_opt(centers1[:len(init)], centers2[:len(init)], X1, X2).reshape(len(init), -1)

	candidates_pot = distance_to_candidates.sum(axis=1)
	best_candidate = np.argmin(candidates_pot)
	current_pot = candidates_pot[best_candidate]
	closest_dist_sq = distance_to_candidates[best_candidate]

	# Pick the remaining n_clusters-1 points
	for c in range(len(init), len(init)+n_clusters-1):
		# Choose center candidates by sampling with probability proportional
		# to the squared distance to the closest existing center
		rand_vals = random_state.random_sample(n_local_trials) * current_pot
		candidate_ids = np.searchsorted(closest_dist_sq.cumsum(),
										rand_vals)
		# XXX: numerical imprecision can result in a candidate_id out of range
		np.clip(candidate_ids, None, closest_dist_sq.size - 1,
				out=candidate_ids)

		# Compute distances to center candidates
		distance_to_candidates = outer_product_opt(X1[candidate_ids], X2[candidate_ids], X1, X2).reshape(len(candidate_ids), -1)

		# update closest distances squared and potential for each candidate
		np.minimum(closest_dist_sq, distance_to_candidates,
				   out=distance_to_candidates)
		candidates_pot = distance_to_candidates.sum(axis=1)

		# Decide which candidate is the best
		best_candidate = np.argmin(candidates_pot)
		current_pot = candidates_pot[best_candidate]
		closest_dist_sq = distance_to_candidates[best_candidate]
		best_candidate = candidate_ids[best_candidate]

		idxs[c] = best_candidate

	return None, idxs[len(init)-1:]

def get_embedding(model, loader, device, num_classes, args, with_emb=False, emb_dim=512):
	model.eval()
	embedding = torch.zeros([len(loader.sampler), num_classes])
	embedding_pen = torch.zeros([len(loader.sampler), emb_dim])
	labels = torch.zeros(len(loader.sampler))
	preds = torch.zeros(len(loader.sampler))
	batch_sz = args.batch_size
	with torch.no_grad():
		for batch_idx, (data, target) in enumerate(loader):
			data, target = data.to(device), target.to(device)
			if with_emb:
				e1, e2 = model(data, with_emb=True)
				embedding_pen[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e2.shape[0]), :] = e2.cpu()
			else:
				e1 = model(data, with_emb=False)

			embedding[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e1.shape[0]), :] = e1.cpu()
			labels[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e1.shape[0])] = target
			preds[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e1.shape[0])] = e1.argmax(dim=1, keepdim=True).squeeze()

	return embedding, labels, preds, embedding_pen

def train(model, device, train_loader, optimizer, epoch):
	"""
	Test model on provided data for single epoch
	"""
	model.train()
	total_loss, correct = 0.0, 0
	for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = nn.CrossEntropyLoss()(output, target)		
		total_loss += loss.item()
		pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
		corr =  pred.eq(target.view_as(pred)).sum().item()
		correct += corr
		loss.backward()
		optimizer.step()

	train_acc = 100. * correct / len(train_loader.sampler)
	avg_loss = total_loss / len(train_loader.sampler)
	print('\nTrain Epoch: {} | Avg. Loss: {:.3f} | Train Acc: {:.3f}'.format(epoch, avg_loss, train_acc))
	return avg_loss

def test(model, device, test_loader, split="test"):
	"""
	Test model on provided data
	"""
	print('\nEvaluating model on {}...'.format(split))
	model.eval()
	test_loss = 0
	correct = 0
	test_acc = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			loss = nn.CrossEntropyLoss()(output, target) 
			test_loss += loss.item() # sum up batch loss
			pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
			corr =  pred.eq(target.view_as(pred)).sum().item()
			correct += corr
			del loss, output

	test_loss /= len(test_loader.sampler)
	test_acc = 100. * correct / len(test_loader.sampler)

	return test_acc, test_loss

def run_unsupervised_da(model, src_train_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, num_classes, device, args):
	"""
	Unsupervised adaptation of source model to target at round 0
	Returns:
		Model post adaptation
	"""
	adapt_net_file = os.path.join('checkpoints', 'adapt', '{}_{}_{:s}_net_{:s}_{:s}.pth'.format(args.da_strat, \
								  args.uda_lr, args.cnn, args.source, args.target))
	if os.path.exists(adapt_net_file):
		print('Found pretrained checkpoint, loading...')
		adapt_model = get_model('AdaptNet', num_cls=num_classes, weights_init=adapt_net_file, model=args.cnn)
	else:
		print('No pretrained checkpoint found, training...')
		source_file = '{}_{}_source.pth'.format(args.source, args.cnn)
		source_path = os.path.join('checkpoints', 'source', source_file)	
		adapt_model = get_model('AdaptNet', num_cls=num_classes, src_weights_init=source_path, model=args.cnn)
		opt_net_tgt = optim.Adam(adapt_model.tgt_net.parameters(), lr=args.lr, weight_decay=args.wd)
		uda_solver = get_solver(args.da_strat, adapt_model.tgt_net, src_train_loader, tgt_sup_loader, tgt_unsup_loader, \
								train_idx, opt_net_tgt, 0, device, args)
		for epoch in range(args.uda_num_epochs):
			if args.da_strat == 'dann':
				opt_dis_adapt = optim.Adam(discriminator.parameters(), lr=args.uda_lr, betas=(0.9, 0.999), weight_decay=0)
				uda_solver.solve(epoch, discriminator, opt_dis_adapt)
			elif args.da_strat in ['mme', 'ft']:
				uda_solver.solve(epoch)
		adapt_model.save(adapt_net_file)
	
	model, src_model, discriminator = adapt_model.tgt_net, adapt_model.src_net, adapt_model.discriminator
	return model, src_model, discriminator
			
######################################################################
##### Interactive visualization utilities
######################################################################

def log(target_accs, fname):
	"""
	Log results as JSON
	"""
	with open(os.path.join('results', 'perf_{}.json'.format(fname)), 'w') as f:
		json.dump(target_accs, f, indent=4)

def interactive_test(model, device, test_loader, split="test", num_classes=10):
	"""
	Utility to test model and generate confusion matrix
	"""
	model.eval()
	correct = 0
	confusion_matrix = torch.zeros(num_classes, num_classes).long()
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
			scores = nn.Softmax(dim=1)(output)
			correct += pred.eq(target.view_as(pred)).sum().item()
			for t, p in zip(target.view(-1), pred.view(-1)):
				confusion_matrix[t.long(), p.long()] += 1
	test_acc = 100. * correct / len(test_loader.sampler)
	return test_acc, confusion_matrix

def plot_cm(ax, conf_matrix, annot=False, label='Confusion matrix', normalize=False, num_classes=10):
	"""
	Utility to plot confusion matrix
	"""
	if normalize:
		conf_matrix = conf_matrix.float()
		conf_matrix /= conf_matrix.sum(dim=1)
		
	df_cm = pd.DataFrame(conf_matrix.cpu().numpy(), \
						 index = [i for i in range(num_classes)], \
						 columns = [i for i in range(num_classes)])

	sns.heatmap(df_cm, annot=True, fmt="d", ax=ax)
	ax.set_title(label, fontsize=18)

def plot_examples(dset, x_lbl, num_classes=10, method='uniform'):
	"""
	Utility to plot label histograms of selected instances as well as qualitative examples
	"""
	fig = plt.figure(figsize=(7, 3.5))
	NUM_IMGS = min(math.floor(math.sqrt(len(x_lbl))), 7)
	nrows, ncols = NUM_IMGS, NUM_IMGS
	gs = gridspec.GridSpec(nrows, ncols*2)
	ax = fig.add_subplot(gs[:, :NUM_IMGS])
	counts = Counter([dset[el][1] for el in x_lbl])
	fig.suptitle(r'SVHN$\rightarrow$MNIST: Instances picked via {}'.format(method), fontsize=16, y=1.05)

	X = sorted(list(counts.keys()))
	ax.bar(X, height=[counts[el] for el in X])
	ax.set_title('Label Histogram')
	ax.set_xticks(np.arange(num_classes))
	ax.set_xlabel('Ground Truth Class')
	ax.set_ylabel('Count')
	data = [dset[el][0].unsqueeze(0) for el in x_lbl]
	join = list(zip(x_lbl, data))
	random.shuffle(join)
	x_lbl, data = zip(*join)
	data_flat = torch.cat(data, dim=0)
	matplotlib.rcParams.update({'font.size': 12})
	
	for row in range(nrows):
		for col in range(ncols):
			ix = (row * ncols) + col
			img = torchvision.transforms.ToPILImage()(data_flat[ix])
			ax = fig.add_subplot(gs[row, NUM_IMGS+col])
			ax.imshow(img, cmap='gray')
			ax.set_xticks([])
			ax.set_yticks([])

def representative_examples(dset, num_classes=10, dset_name='MNIST'):
	"""
	Utility to plot representative examples from a given dataset (only works with MNIST/SVHN)
	"""
	fig = plt.figure(figsize=(3.5, 3.5))
	x_lbl = np.random.choice(np.arange(50000), 25)
	NUM_IMGS = min(math.floor(math.sqrt(len(x_lbl))), 5)
	nrows, ncols = NUM_IMGS, NUM_IMGS
	gs = gridspec.GridSpec(nrows, ncols)
	ax = fig.add_subplot(gs[:, :NUM_IMGS])
	fig.suptitle(r'{}'.format(dset_name), fontsize=16)
	data = [dset[el][0].unsqueeze(0) for el in x_lbl]
	join = list(zip(x_lbl, data))
	random.shuffle(join)
	x_lbl, data = zip(*join)
	data_flat = torch.cat(data, dim=0)
	matplotlib.rcParams.update({'font.size': 12})    
	plt.xticks([])
	plt.yticks([])
	cmap = 'gray' if dset_name == 'MNIST' else None
	for row in range(nrows):
		for col in range(ncols):
			ix = (row * ncols) + col
			img = torchvision.transforms.ToPILImage()(data_flat[ix])
			ax = fig.add_subplot(gs[row, col])
			ax.imshow(img, cmap=cmap)
			ax.set_xticks([])
			ax.set_yticks([])
	fig.subplots_adjust(hspace=0.1, wspace=0.01)

######################################################################
##### Plotting utilities
######################################################################

COLORS = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#e78ac3", "#a65628"]
COLORS_DICT = {'uniform': COLORS[6], 'BADGE': COLORS[2], 'AADA': COLORS[4], 'CLUE': COLORS[0] }

def interactive_plot(ax, ablations, IDS, runs, cnn, source, target, target_maxp, num_rounds, max_points=10, custom_keys=None, \
					 custom_title=None, size=0, custom_filename=None, upper_bound=None, lower_bound=None):
	"""
	Utility to plot result curves
	"""				 
	ablations = ablations.split(",")
	IDS = IDS.split(",")    
	target_accs = {}
	suffix = 'maxb' if target_maxp > 1 else 'maxp'
	for (ID, ablation, n_runs) in zip(IDS, ablations, runs):
		method, sample = ablation.split('_')        
		exp_name = '{}_{}_{}_{}runs_{}split_{}{}_iterall'.format(ID, method, sample, n_runs, num_rounds, target_maxp, suffix)
		fname = os.path.join('results', 'perf_{}.json'.format(exp_name))
		key = ablation.split('_')[-1]
		if os.path.exists(fname):
			target_accs['{}_{}'.format(ID, key)] = json.load(open(fname, 'rb'))[ablation]
		else:
			print('{} not found'.format(fname))
			raise FileNotFoundError
	
	return plot_perf_curve(ax, target_accs, cnn, source, target, target_maxp, num_rounds, max_points, custom_keys, \
						   custom_title, size, custom_filename, upper_bound, lower_bound)

def plot_perf_curve(ax, target_accs, cnn, source, target, target_maxp, num_rounds, max_points, custom_keys=None, custom_title=None, \
					size=0, custom_filename=None, upper_bound=None, lower_bound=None, set_xticks=False):
	"""
	Main result curve plotting code
	"""				 
	keys = sorted(list(target_accs.keys()))
	font_sz = 12
	matplotlib.rcParams.update({'font.size': font_sz})
	ax.grid(linestyle='--') 
	
	title = custom_title
		
	ax.set_title(title, fontsize=font_sz)
	ax.set_xlabel('# Labels from {} Train'.format(target), fontsize=font_sz)
	ax.set_ylabel('{} Test Accuracy'.format(target), fontsize=font_sz)
	
	ax.set_xlim(0, target_maxp)
	
	x = np.array([(1.0/num_rounds) * target_maxp * n for n in range(num_rounds+1)])
	keys = target_accs.keys()

	MAX = max_points+1
	min_y, max_y = 100, 0
	linestyle_ix = 0
	keys_counter = defaultdict(int)
	lines = []
	for ix, k in enumerate(keys):
		v = target_accs[k]
		if not v: continue    
		if 'args' in v.keys(): del v['args']

		sorted_keys = sorted(list([float(el) for el in v.keys()]))
		ym = np.array([np.mean(v[str(k)]) for k in sorted_keys])
		yv = np.array([np.std(v[str(k)]) for k in sorted_keys])
		
		if ym[:MAX][-1] > max_y: max_y = ym[:MAX][-1] + 2
		if ym[0] < min_y: min_y = ym[0] - 2

		method = k.split('_')[1]
		linestyle_ix = keys_counter[method]
		keys_counter[method] += 1

		if linestyle_ix == 0:
			line, = ax.plot(np.array(x[:MAX]), ym[:MAX], alpha=0.75, marker='.', linestyle='-', linewidth=1.5, color=COLORS_DICT[method])
			ax.fill_between(np.array(x[:MAX]), ym[:MAX]-yv[:MAX], ym[:MAX]+yv[:MAX], alpha=0.25, linewidth=1, color=COLORS_DICT[method])
			REM_COLORS = list(set(COLORS) - set([COLORS_DICT[method]]))
		else:
			color = COLORS[5]
			line, = ax.plot(np.array(x[:MAX]), ym[:MAX], alpha=0.75, marker='.', linestyle='-', linewidth=1.5, color=color)
			ax.fill_between(np.array(x[:MAX]), ym[:MAX]-yv[:MAX], ym[:MAX]+yv[:MAX], alpha=0.25, linewidth=1, color=color)
			REM_COLORS = list(set(REM_COLORS) - set([color]))
		lines.append(line)
	
	if custom_keys is None:
		custom_keys = [key.split('_')[-1] for key in keys]
		# custom_keys = [dict_map[key] for key in custom_keys]
	
	if upper_bound is not None: max_y = upper_bound
	if lower_bound is not None: min_y = lower_bound

	if set_xticks: ax.set_xticks(x[:MAX])
	ax.set_ylim(min_y, max_y)	
	tick_frequency = (max_y-min_y) // 5
	ax.yaxis.set_major_locator(ticker.MultipleLocator(base=tick_frequency))

	return lines