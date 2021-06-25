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

	exp_type = 'anomaly_detection'


	if args.config_id == 1:
		technique_name = "OOB"
		
		instances = [22665, 26001]

		arr_exp_cifar10 = ['AN-1', 'AN-3', 'AN-5', 'AN-8', 'AN-11', 'AN-12']
		arr_exp_gtsrb = ['AN-2', 'AN-4', 'AN-6', 'AN-7', 'AN-9', 'AN-10']

		matrix_experiments = [arr_exp_cifar10, arr_exp_gtsrb]
		datasets = ['cifar10', 'gtsrb']

		#same order of arr_exp
		variations = ['pixel trap severity 1', 'row add logic severity 1', 'shifted pixel severity 1', 
		'pixel trap severity 3', 'row add logic severity 3', 'shifted pixel severity 3']
		
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

		arr_exp_cifar10 = ['AN-25', 'AN-26', 'AN-27', 'AN-28', 'AN-29', 'AN-30']
		arr_exp_gtsrb = ['AN-31', 'AN-32', 'AN-33', 'AN-34', 'AN-35', 'AN-36']

		matrix_experiments = [arr_exp_cifar10, arr_exp_gtsrb]
		datasets = ['cifar10', 'gtsrb']

		#same order of arr_exp
		variations = ['pixel trap severity 1', 'row add logic severity 1', 'shifted pixel severity 1', 
		'pixel trap severity 3', 'row add logic severity 3', 'shifted pixel severity 3']
		
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

		arr_exp_cifar10 = ['AN-14', 'AN-16', 'AN-18', 'AN-20', 'AN-22', 'AN-24']
		arr_exp_gtsrb = ['AN-13', 'AN-15', 'AN-17', 'AN-19', 'AN-21', 'AN-23']

		matrix_experiments = [arr_exp_cifar10, arr_exp_gtsrb]
		datasets = ['cifar10', 'gtsrb']

		#same order of arr_exp
		variations = ['pixel trap severity 3', 'row add logic severity 3', 'shifted pixel severity 3',
		'pixel trap severity 1', 'row add logic severity 1', 'shifted pixel severity 1']
		
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