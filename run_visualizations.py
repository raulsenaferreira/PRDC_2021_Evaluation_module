import os
import argparse
from config.novelty_detection import plot_pos_neg_comparison


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	parser.add_argument("config_id", type=int, help="ID for a set of pre-defined visualizations")

	parser.add_argument("path_for_saving_plots", help="Root path for saving visualizations")

	args = parser.parse_args()

	if args.config_id == 1:
		root_path = os.path.join(args.path_for_saving_plots, 'novelty_detection')
		arr_exp = ['PHD-111', 'PHD-112']
		names = ['oob isomap', 'oob PCA']
		title = 'ID=GTSRB; OOD=BTSC;'
		plot_pos_neg_comparison.plot(arr_exp, names, title, root_path)