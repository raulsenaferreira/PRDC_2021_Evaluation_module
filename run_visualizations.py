import os
import argparse
from config.novelty_detection import plot_pos_neg_comparison
from config.novelty_detection import eval_sm_performance
from config.novelty_detection import eval_sm_impact_on_the_system
from tensorflow.keras.datasets import cifar10, mnist
from src import plot_functions as pf


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	parser.add_argument("config_id", type=int, help="ID for a set of pre-defined visualizations")

	parser.add_argument("path_for_saving_plots", help="Root path for saving visualizations")

	args = parser.parse_args()

	if args.config_id == 1:
		exp_type = 'novelty_detection'
		names = ['oob', 'oob isomap', 'oob PCA']
		arr_exp = ['NOV-1', 'NOV-2', 'NOV-3']

		root_path = os.path.join(args.path_for_saving_plots, exp_type)
		path_for_load_neptune = exp_type.replace('_', '-') # correcting for loading neptune experiments
		
		#############################
		# SM results for a specific dataset
		label = 'table_1'
		caption = 'Table 1: comparing outside-of-the-box-based monitors for\
		 GTSRB as ID dataset, and BTSC as OOD dataset.'
		
		eval_sm_performance.plot1(arr_exp, names, label, caption, root_path, path_for_load_neptune)
		#############################

		#############################
		# SM results across datasets 
		#arr_exp = {}
		#names = ['oob', 'oob isomap', 'oob PCA']
		#arr_exp.update({names[0]: ['PHD-111', 'PHD-112', 'PHD-111', 'PHD-112', 'PHD-111']})
		#arr_exp.update({names[1]: ['PHD-111', 'PHD-112', 'PHD-111', 'PHD-112', 'PHD-111']})
		
		#eval_sm_performance.plot2(arr_exp, root_path)
		#############################

		#############################
		# SM's impact on the system for a specific dataset
		label = 'table_2'
		caption = 'Table 2: comparing the impact of outside-of-the-box-based monitors for\
		 GTSRB as ID dataset, and BTSC as OOD dataset.'
		
		eval_sm_impact_on_the_system.plot1(arr_exp, names, label, caption, root_path, path_for_load_neptune)
		#############################

		#############################
		# SM's impact on the system across datasets
		#eval_sm_impact_on_the_system.plot2(arr_exp, names, root_path)
		#############################
		

		#plot_pos_neg_comparison.plot(arr_exp, names, caption, root_path)
		'''
		(x_train, y_train), (x_test, y_test) = cifar10.load_data() # mnist.load_data()
		dataset_name = 'CIFAR-10' # 'MNIST'
		model_name = 'resNet_'+dataset_name+'.h5' # 'leNet_'+dataset_name+'.h5'
		
		#path to load the model
		models_folder = os.path.join("aux_data", "temp")
		model_file = os.path.join(models_folder, model_name)
		pf.visualize_distributions(x_train, y_train, dataset_name, model_file)
		'''

		#pf.visualize_pair_distributions(x_train, y_train, dataset_name, model_file\
		#	x_train_2, y_train_2, dataset_name_2, model_file_2)