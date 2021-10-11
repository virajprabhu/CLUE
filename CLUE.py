import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import torch
import torch.nn as nn

class CLUESampling(SamplingStrategy):
	"""
	Implements CLUE: CLustering via Uncertainty-weighted Embeddings
	"""
	def __init__(self, dset, train_idx, model, discriminator, device, args, balanced=False):
		super(CLUESampling, self).__init__(dset, train_idx, model, discriminator, device, args)
		self.random_state = np.random.RandomState(1234)
		self.T = args.clue_softmax_t # Typically smaller values (0.1-0.5) work better

    def get_embedding(self, model, loader, device, num_classes, args, with_emb=False, emb_dim=512):
        model.eval()
        embedding = torch.zeros([len(loader.sampler), num_classes])
        embedding_pen = torch.zeros([len(loader.sampler), emb_dim])
        labels = torch.zeros(len(loader.sampler))
        preds = torch.zeros(len(loader.sampler))
        batch_sz = args.batch_size
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(tqdm(loader)):
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

	def query(self, n):
		idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
		train_sampler = utils.ActualSequentialSampler(self.train_idx[idxs_unlabeled])
		data_loader = torch.utils.data.DataLoader(self.dset, sampler=train_sampler, num_workers=4, \
												  batch_size=self.args.batch_size, drop_last=False)
		self.model.eval()
		
		if self.args.cnn == 'LeNet':
			emb_dim = 500
		elif self.args.cnn in ['ResNet18', 'ResNet34', 'ResNet34FS']:
			emb_dim = 512

		# Get embedding of target instances
		tgt_emb, tgt_lab, tgt_preds, tgt_pen_emb = self.get_embedding(self.model, data_loader, self.device, self.num_classes, \
																	   
                                                                       self.args, with_emb=True, emb_dim=emb_dim)
		# Using penultimate layer embeddings works a little better but it is okay to use logits as well
        # if speed is a concern
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