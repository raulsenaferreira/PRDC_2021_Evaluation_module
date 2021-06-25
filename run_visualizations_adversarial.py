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

	exp_type = 'adversarial_attack'


	if args.config_id == 1:
		names = ["ODIN"] 
		
		instances = [22665, 26001]

		arr_exp = [
		 'AD-1', # cifar10
		 'AD-2', # gtsrb
		]
		#same order of arr_exp
		exp_datasets = ['cifar10', 'gtsrb']
		classes_ID_OOD = [10, 43]  # cifar10 # gtsrb
		
		path_for_saving_plots = os.path.join(args.path_for_saving_plots, exp_type)
		path_for_load_neptune = exp_type.replace('_', '-') # correcting for loading neptune experiments

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

		#############################
		######## one dataset each time
		#############################
		#'''
		
		indices_tables = len(arr_exp)

		for i in range(indices_tables):
			# SM's impact on the system
			label = 'table_{}_{}_{}'.format(i+1, names[0], exp_datasets[i])
			caption = 'Table {}: comparing the impact of {}-based monitors for the datasets divided into ID and OOD.'.format(i+1, names[0])
			classes_ID, classes_OOD = classes_ID_OOD[0], classes_ID_OOD[1]
			
			eval_sm_impact_on_the_system.plot1([i], classes_ID, classes_OOD, readouts_ML, readouts_SM_reaction,
			 names, label, caption, path_for_saving_plots, is_pytorch=True)
			#############################

			# SM results  
			label = 'table_{}_{}_{}'.format(indices_tables+i+1, names[0], exp_datasets[i])
			caption = 'Table {}: comparing {}-based monitors for the datasets divided into ID and OOD.'.format(indices_tables+i+1,names[0])
			
			eval_sm_performance.plot1([i], readouts_SM_detection,
			 names, label, caption, path_for_saving_plots)
		#############################
		#############################


	elif args.config_id == 2:
		names = ["ALOOC"] 
		
		instances = [22665, 26001]

		arr_exp = [
		 'AD-3', # cifar10
		 'AD-4', # gtsrb
		]
		#same order of arr_exp
		exp_datasets = ['cifar10', 'gtsrb']
		classes_ID_OOD = [10, 43]  # cifar10 # gtsrb
		
		path_for_saving_plots = os.path.join(args.path_for_saving_plots, exp_type)
		path_for_load_neptune = exp_type.replace('_', '-') # correcting for loading neptune experiments

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

		#############################
		######## one dataset each time
		#############################
		#indices_experiments = list(range(0, 2))
		
		indices_tables = len(arr_exp)

		for i in range(indices_tables):
			# SM's impact on the system
			label = 'table_{}_{}_{}'.format(i+1, names[0], exp_datasets[i])
			caption = 'Table {}: comparing the impact of {}-based monitors for the datasets divided into ID and OOD.'.format(i+1, names[0])
			classes_ID, classes_OOD = classes_ID_OOD[0], classes_ID_OOD[1]
			
			eval_sm_impact_on_the_system.plot1([i], classes_ID, classes_OOD, readouts_ML, readouts_SM_reaction,
			 names, label, caption, path_for_saving_plots, is_pytorch=False)
			#############################

			# SM results  
			label = 'table_{}_{}_{}'.format(indices_tables+i+1, names[0], exp_datasets[i])
			caption = 'Table {}: comparing {}-based monitors for the datasets divided into ID and OOD.'.format(indices_tables+i+1,names[0])
			
			eval_sm_performance.plot1([i], readouts_SM_detection,
			 names, label, caption, path_for_saving_plots)

		# Time
		#readouts_time = util.load_time_info(experiments, instances, indices_experiments, path_for_saving_plots, dataset, names)
		#############################
		#############################


	elif args.config_id == 3:
		names = ["OOB"] 
		
		instances = [22665, 26001]

		arr_exp = [
		 'AD-5', # gtsrb
		 'AD-6', # cifar10
		]
		#same order of arr_exp
		exp_datasets = ['gtsrb', 'cifar10']
		classes_ID_OOD = [43, 10]  # cifar10 # gtsrb
		
		path_for_saving_plots = os.path.join(args.path_for_saving_plots, exp_type)
		path_for_load_neptune = exp_type.replace('_', '-') # correcting for loading neptune experiments

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

		#############################
		######## one dataset each time
		#############################
		#indices_experiments = list(range(0, 2))
		
		indices_tables = len(arr_exp)

		for i in range(indices_tables):
			# SM's impact on the system
			label = 'table_{}_{}_{}'.format(i+1, names[0], exp_datasets[i])
			caption = 'Table {}: comparing the impact of {}-based monitors for the datasets divided into ID and OOD.'.format(i+1, names[0])
			classes_ID, classes_OOD = classes_ID_OOD[0], classes_ID_OOD[1]
			
			eval_sm_impact_on_the_system.plot1([i], classes_ID, classes_OOD, readouts_ML, readouts_SM_reaction,
			 names, label, caption, path_for_saving_plots, is_pytorch=False)
			#############################

			# SM results  
			label = 'table_{}_{}_{}'.format(indices_tables+i+1, names[0], exp_datasets[i])
			caption = 'Table {}: comparing {}-based monitors for the datasets divided into ID and OOD.'.format(indices_tables+i+1,names[0])
			
			eval_sm_performance.plot1([i], readouts_SM_detection,
			 names, label, caption, path_for_saving_plots)

		# Time
		#readouts_time = util.load_time_info(experiments, instances, indices_experiments, path_for_saving_plots, dataset, names)
		#############################
		#############################