import os
import argparse
from config import plot_pos_neg_comparison
from config import eval_sm_performance
from config import eval_sm_impact_on_the_system
from tensorflow.keras.datasets import cifar10, mnist
from src import plot_functions as pf
from src import util
import neptune_config as npte 
import numpy as np


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	parser.add_argument("config_id", type=int, help="ID for a set of pre-defined visualizations")

	parser.add_argument("path_for_saving_plots", help="Root path for saving visualizations")

	args = parser.parse_args()

	exp_type = 'distributional_shift'


	if args.config_id == 1:
		technique_name = "OOB"
		
		instances = [22665, 26001]

		arr_exp_cifar10 = ['DSHIFT-1', 'DSHIFT-3', 'DSHIFT-6', 'DSHIFT-9', 'DSHIFT-10', 'DSHIFT-15', 
		'DSHIFT-18', 'DSHIFT-19', 'DSHIFT-20', 'DSHIFT-21', 'DSHIFT-22']
		arr_exp_gtsrb = ['DSHIFT-2', 'DSHIFT-4', 'DSHIFT-5', 'DSHIFT-7', 'DSHIFT-8', 'DSHIFT-11', 
		'DSHIFT-12', 'DSHIFT-13', 'DSHIFT-14', 'DSHIFT-16', 'DSHIFT-17']

		matrix_experiments = [arr_exp_cifar10, arr_exp_gtsrb]
		datasets = ['cifar10', 'gtsrb']

		#same order of arr_exp
		variations = ['snow_severity_2', 'fog_severity_2', 'brightness_severity_2', 'contrast_severity_2', 'saturate_severity_2', 'rotated',
		'snow_severity_5', 'fog_severity_5', 'brightness_severity_5', 'contrast_severity_5', 'saturate_severity_5']
		
		path_for_saving_plots = os.path.join(args.path_for_saving_plots, exp_type)
		path_for_load_neptune = exp_type.replace('_', '-') # correcting for loading neptune experiments


		#############################
		######## one base dataset each time
		#############################
		i=1
		for arr_exp, dataset in zip(matrix_experiments, datasets):
			project = npte.neptune_init(path_for_load_neptune)
			experiments = project.get_experiments(arr_exp)

			# readouts
			arr_ml_pred = util.load_artifact('arr_classification_pred.npy', experiments)
			arr_ml_true = util.load_artifact('arr_classification_true.npy', experiments)

			arr_detection_SM = util.load_artifact('arr_detection_SM.npy', experiments)
			arr_detection_true = util.load_artifact('arr_detection_true.npy', experiments)

			arr_reaction_SM = util.load_artifact('arr_reaction_SM.npy', experiments)
			arr_reaction_true = util.load_artifact('arr_reaction_true.npy', experiments)
			
			readouts_ML = [arr_ml_pred, arr_ml_true]
			readouts_SM_detection = [arr_detection_SM, arr_detection_true]
			readouts_SM_reaction = [arr_reaction_SM, arr_reaction_true]
			
			indices_tables = len(arr_exp)

			# SM results  
			label = 'table_{}_{}_{}'.format(i, technique_name, dataset)
			caption = 'Table {}: comparing {}-based monitors for the {} variations for OOD detection in {} dataset.'.format(i,technique_name, exp_type, dataset)
			i+=1
			eval_sm_performance.plot(indices_tables, readouts_SM_detection, variations, label, caption, path_for_saving_plots)
		#############################
		#############################


	elif args.config_id == 2:
		technique_name = "ODIN"
		
		instances = [22665, 26001]

		arr_exp_cifar10 = ['DSHIFT-45', 'DSHIFT-46', 'DSHIFT-47', 'DSHIFT-48', 'DSHIFT-49', 'DSHIFT-50', 
		'DSHIFT-51', 'DSHIFT-52', 'DSHIFT-53', 'DSHIFT-54', 'DSHIFT-55']
		arr_exp_gtsrb = ['DSHIFT-56', 'DSHIFT-57', 'DSHIFT-58', 'DSHIFT-59', 'DSHIFT-60', 'DSHIFT-61', 
		'DSHIFT-62', 'DSHIFT-63', 'DSHIFT-64', 'DSHIFT-65', 'DSHIFT-66']

		matrix_experiments = [arr_exp_cifar10, arr_exp_gtsrb]
		datasets = ['cifar10', 'gtsrb']

		#same order of arr_exp
		variations = ['snow_severity_2', 'fog_severity_2', 'brightness_severity_2', 'contrast_severity_2', 'saturate_severity_2', 'rotated',
		'snow_severity_5', 'fog_severity_5', 'brightness_severity_5', 'contrast_severity_5', 'saturate_severity_5']
		
		path_for_saving_plots = os.path.join(args.path_for_saving_plots, exp_type)
		path_for_load_neptune = exp_type.replace('_', '-') # correcting for loading neptune experiments


		#############################
		######## one base dataset each time
		#############################
		i=1
		for arr_exp, dataset in zip(matrix_experiments, datasets):
			project = npte.neptune_init(path_for_load_neptune)
			experiments = project.get_experiments(arr_exp)

			# readouts
			arr_ml_pred = util.load_artifact('arr_classification_pred.npy', experiments)
			arr_ml_true = util.load_artifact('arr_classification_true.npy', experiments)

			arr_detection_SM = util.load_artifact('arr_detection_SM.npy', experiments)
			arr_detection_true = util.load_artifact('arr_detection_true.npy', experiments)

			arr_reaction_SM = util.load_artifact('arr_reaction_SM.npy', experiments)
			arr_reaction_true = util.load_artifact('arr_reaction_true.npy', experiments)
			
			readouts_ML = [arr_ml_pred, arr_ml_true]
			readouts_SM_detection = [arr_detection_SM, arr_detection_true]
			readouts_SM_reaction = [arr_reaction_SM, arr_reaction_true]
			
			indices_tables = len(arr_exp)

			# SM results  
			label = 'table_{}_{}_{}'.format(i, technique_name, dataset)
			caption = 'Table {}: comparing {}-based monitors for the {} variations for OOD detection in {} dataset.'.format(i,technique_name, exp_type, dataset)
			i+=1
			eval_sm_performance.plot(indices_tables, readouts_SM_detection, variations, label, caption, path_for_saving_plots)

		# Time
		#readouts_time = util.load_time_info(experiments, instances, indices_experiments, path_for_saving_plots, dataset, names)
		#############################
		#############################


	elif args.config_id == 3:
		technique_name = "ALOOC"
		
		instances = [22665, 26001]

		arr_exp_gtsrb = ['DSHIFT-23', 'DSHIFT-25', 'DSHIFT-27', 'DSHIFT-29', 'DSHIFT-31', 'DSHIFT-33', 
		'DSHIFT-35', 'DSHIFT-37', 'DSHIFT-39', 'DSHIFT-41', 'DSHIFT-43']
		arr_exp_cifar10 = ['DSHIFT-24', 'DSHIFT-26', 'DSHIFT-28', 'DSHIFT-30', 'DSHIFT-32', 'DSHIFT-34', 
		'DSHIFT-36', 'DSHIFT-38', 'DSHIFT-40', 'DSHIFT-42', 'DSHIFT-44']

		matrix_experiments = [arr_exp_cifar10, arr_exp_gtsrb]
		datasets = ['cifar10', 'gtsrb']

		#same order of arr_exp
		variations = ['rotated', 'snow_severity_5', 'fog_severity_5', 'brightness_severity_5', 'contrast_severity_5', 'saturate_severity_5',
		'snow_severity_2', 'fog_severity_2', 'brightness_severity_2', 'contrast_severity_2', 'saturate_severity_2']
		
		path_for_saving_plots = os.path.join(args.path_for_saving_plots, exp_type)
		path_for_load_neptune = exp_type.replace('_', '-') # correcting for loading neptune experiments


		#############################
		######## one base dataset each time
		#############################
		i=1
		for arr_exp, dataset in zip(matrix_experiments, datasets):
			project = npte.neptune_init(path_for_load_neptune)
			experiments = project.get_experiments(arr_exp)

			# readouts
			arr_ml_pred = util.load_artifact('arr_classification_pred.npy', experiments)
			arr_ml_true = util.load_artifact('arr_classification_true.npy', experiments)

			arr_detection_SM = util.load_artifact('arr_detection_SM.npy', experiments)
			arr_detection_true = util.load_artifact('arr_detection_true.npy', experiments)

			arr_reaction_SM = util.load_artifact('arr_reaction_SM.npy', experiments)
			arr_reaction_true = util.load_artifact('arr_reaction_true.npy', experiments)
			
			readouts_ML = [arr_ml_pred, arr_ml_true]
			readouts_SM_detection = [arr_detection_SM, arr_detection_true]
			readouts_SM_reaction = [arr_reaction_SM, arr_reaction_true]
			
			indices_tables = len(arr_exp)

			# SM results  
			label = 'table_{}_{}_{}'.format(i, technique_name, dataset)
			caption = 'Table {}: comparing {}-based monitors for the {} variations for OOD detection in {} dataset.'.format(i,technique_name, exp_type, dataset)
			i+=1
			eval_sm_performance.plot(indices_tables, readouts_SM_detection, variations, label, caption, path_for_saving_plots)

		# Time
		#readouts_time = util.load_time_info(experiments, instances, indices_experiments, path_for_saving_plots, dataset, names)
		#############################
		#############################