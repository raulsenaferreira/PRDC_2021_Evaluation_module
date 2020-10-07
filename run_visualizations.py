import os
import argparse
from config.novelty_detection import plot_pos_neg_comparison


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	parser.add_argument("config_id", type=int, help="ID for a set of pre-defined visualizations")

	parser.add_argument("path_for_saving_plots", help="Root path for saving visualizations")

	args = parser.parse_args()

	if args.config_id == 1:
		arr_exp = ['PHD-107', 'PHD-108', 'PHD-109']
		plot_pos_neg_comparison.plot(arr_exp)