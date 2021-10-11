import os
import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import utils

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--id', type=str, default='svhn2mnist', help="Experiment identifier")
	parser.add_argument('--al_strats', type=list, default=['uniform', 'BADGE', 'uniform', 'AADA', 'CLUE'], \
						 help="List of AL strats. Supported: {uniform, BADGE, AADA, CLUE}")
	parser.add_argument('--da_strats', type=list, default=['ft', 'ft', 'mme', 'dann', 'mme'], \
						help="List of DA strats. Supported: {ft, DANN, MME}")
	parser.add_argument('--model_inits', type=list, default=['source', 'source', 'source', 'source', 'source'], \
						 help="List of model initializations.")
	parser.add_argument('--runs', type=int, default=3, help="Number of experimental runs")
	parser.add_argument('--source', default="svhn", help="Source dataset")
	parser.add_argument('--target', default="mnist", help="Target dataset")
	parser.add_argument('--total_budget', type=int, default=300, help="Total target budget")
	parser.add_argument('--num_rounds', type=int, default=30, help="Target dataset number of splits")

	args = parser.parse_args()
	target_accs, custom_keys = {}, []
	for (al_strat, da_strat, model_init) in zip(args.al_strats, args.da_strats, args.model_inits):
		exp_name = '{}_{}_{}_{}_{}runs_{}rounds_{}budget'.format(args.id, model_init, al_strat, da_strat, \
																 args.runs, args.num_rounds, args.total_budget)
		results_fname = os.path.join('results', 'perf_{}.json'.format(exp_name))
		
		key = '{}_{}_{}'.format(model_init, al_strat, da_strat)
		if os.path.exists(results_fname):
			custom_keys.append(key)
			target_accs[key] = json.load(open(results_fname, 'rb'))
		else:
			print('{} not found'.format(results_fname))
			continue
			
		outstr = key 
		for ix in range(0, args.num_rounds+1):
			rat = str((1.0/args.num_rounds) * args.total_budget * ix)
			outstr += '\nRound {:2d}: {:.2f}+/-{:.2f}'.format(ix, np.mean(target_accs[key][rat]), np.std(target_accs[key][rat]))
		print(outstr)
	
	fig, axs = plt.subplots(1, 1, figsize=(4.5, 4.5))
	custom_title=r'{}$\rightarrow${}'.format(args.source, args.target)
	lines = utils.plot_perf_curve(axs, target_accs, '', args.source, args.target, args.total_budget, \
								  args.num_rounds, args.num_rounds, custom_title=custom_title, custom_keys=custom_keys)
	plt.legend(lines, custom_keys, labelspacing=0.2)
	plt.tight_layout(pad=3)
	os.makedirs('plots', exist_ok=True)
	plt.savefig(os.path.join('plots', '{}.png'.format(args.id)), bbox_inches='tight')

if __name__ == "__main__":
	main()