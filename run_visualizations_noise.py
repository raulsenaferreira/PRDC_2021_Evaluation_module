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

	exp_type = 'noise'

	if args.config_id == 0:

		technique_name = "ODIN"
		
		instances = [22665, 26001]

		arr_exp_cifar10 = ['NS-96', 'NS-46', 'NS-48', 'NS-50', 'NS-52']
		arr_exp_gtsrb = ['NS-45', 'NS-47', 'NS-49', 'NS-51', 'NS-54']

		matrix_experiments = [arr_exp_cifar10, arr_exp_gtsrb]
		datasets = ['cifar10', 'gtsrb']

		#same order of arr_exp
		variations = ['gaussian_noise_severity_2', 'gaussian_noise_severity_5', 'impulse_noise_severity_2', 'impulse_noise_severity_5', 'shot_noise_severity_2']
		
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


	if args.config_id == 1:
		technique_name = "ODIN"
		
		instances = [22665, 26001]

		arr_exp_cifar10 = ['NS-44', 'NS-46', 'NS-48', 'NS-50', 'NS-52', 'NS-55', 'NS-57', 'NS-59', 'NS-61', 'NS-63', 'NS-65', #'NS-68', 
		'NS-82', 'NS-83', 'NS-84', 'NS-86', 'NS-87', 'NS-88', 'NS-89', 'NS-90']
		arr_exp_gtsrb = ['NS-45', 'NS-47', 'NS-49', 'NS-51', 'NS-54', 'NS-56', 'NS-58', 'NS-60', 'NS-62', 'NS-64', 'NS-66', #'NS-69',
		'NS-70', 'NS-71', 'NS-72', 'NS-73', 'NS-74', 'NS-75', 'NS-77', 'NS-78']

		matrix_experiments = [arr_exp_cifar10, arr_exp_gtsrb]
		datasets = ['cifar10', 'gtsrb']

		#same order of arr_exp
		variations = ['gaussian_noise_severity_2', 'gaussian_noise_severity_5', 'impulse_noise_severity_2', 'impulse_noise_severity_5', 'shot_noise_severity_2',
		'shot_noise_severity_5', 'spatter_severity_2', 'spatter_severity_5', 'speckle_noise_severity_2', 'speckle_noise_severity_5', 'defocus_blur_severity_2', #'defocus_blur_severity_5', 
		'elastic_transform_severity_2', 'elastic_transform_severity_5', 'glass_blur_severity_2', 'glass_blur_severity_5',
		'zoom_blur_severity_2', 'zoom_blur_severity_5', 'gaussian_blur_severity_2', 'gaussian_blur_severity_5']
		
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
			#arr_ml_pred = util.load_artifact('arr_classification_pred.npy', experiments)
			#arr_ml_true = util.load_artifact('arr_classification_true.npy', experiments)

			arr_detection_SM = util.load_artifact('arr_detection_SM.npy', experiments)
			arr_detection_true = util.load_artifact('arr_detection_true.npy', experiments)

			#arr_reaction_SM = util.load_artifact('arr_reaction_SM.npy', experiments)
			#arr_reaction_true = util.load_artifact('arr_reaction_true.npy', experiments)
			
			#readouts_ML = [arr_ml_pred, arr_ml_true]
			readouts_SM_detection = [arr_detection_SM, arr_detection_true]
			#readouts_SM_reaction = [arr_reaction_SM, arr_reaction_true]
			
			indices_tables = len(arr_exp)

			# SM results  
			label = 'table_{}_{}_{}'.format(i, technique_name, dataset)
			caption = 'Table {}: comparing {}-based monitors for the {} variations for OOD detection in {} dataset.'.format(i,technique_name, exp_type, dataset)
			i+=1
			eval_sm_performance.plot(indices_tables, readouts_SM_detection, variations, label, caption, path_for_saving_plots)
		#############################
		#############################


	elif args.config_id == 2:
		technique_name = "ALOOC"

		datasets = ['cifar10', 'gtsrb']
		
		instances = [22665, 26001]

		
		arr_exp_cifar10 = ['NS-36', 'NS-38', 'NS-40', 'NS-42', 'NS-53', 'NS-76', 'NS-80', 'NS-85', 'NS-91', 'NS-92',
		'NS-101', 'NS-103', 'NS-105', 'NS-107', 'NS-109', 'NS-111', 'NS-113', 'NS-115', 'NS-117', 'NS-119']

		#same order of arr_exp
		variations1 = ['defocus_blur_severity_2', 'defocus_blur_severity_5', 'elastic_transform_severity_2', 'elastic_transform_severity_5', 'glass_blur_severity_2', 'glass_blur_severity_5',
		'zoom_blur_severity_2', 'zoom_blur_severity_5', 'gaussian_blur_severity_2', 'gaussian_blur_severity_5',
		'gaussian_noise_severity_2', 'gaussian_noise_severity_5', 'impulse_noise_severity_2', 'impulse_noise_severity_5', 
		'shot_noise_severity_2', 'shot_noise_severity_5', 'spatter_severity_2', 'spatter_severity_5', 'speckle_noise_severity_2', 'speckle_noise_severity_5'
		]

		
		arr_exp_gtsrb = ['NS-100', 'NS-102', 'NS-104', 'NS-106', 'NS-108', 'NS-110', 'NS-112', 'NS-114', 'NS-116', 'NS-118', 
		'NS-120', 'NS-121', 'NS-122', 'NS-123', 'NS-124', 'NS-125', 'NS-126', 'NS-127', 'NS-128', 'NS-129']

		#same order of arr_exp
		variations2 = ['gaussian_noise_severity_2', 'gaussian_noise_severity_5', 'impulse_noise_severity_2', 'impulse_noise_severity_5', 
		'shot_noise_severity_2', 'shot_noise_severity_5', 'spatter_severity_2', 'spatter_severity_5', 'speckle_noise_severity_2', 'speckle_noise_severity_5',
		'defocus_blur_severity_2', 'defocus_blur_severity_5', 'elastic_transform_severity_2', 'elastic_transform_severity_5', 'glass_blur_severity_2', 'glass_blur_severity_5',
		'zoom_blur_severity_2', 'zoom_blur_severity_5', 'gaussian_blur_severity_2', 'gaussian_blur_severity_5'
		]

		matrix_experiments = [arr_exp_cifar10, arr_exp_gtsrb]

		arr_variations = [variations1, variations2]		
		
		path_for_saving_plots = os.path.join(args.path_for_saving_plots, exp_type)
		path_for_load_neptune = exp_type.replace('_', '-') # correcting for loading neptune experiments


		#############################
		######## one base dataset each time
		#############################
		i=1
		for arr_exp, dataset, variations in zip(matrix_experiments, datasets, arr_variations):
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
		technique_name = "OOB"
		
		instances = [22665, 26001]

		arr_exp_cifar10 = ['NS-3', 'NS-4', 'NS-7', 'NS-10', 'NS-13', 'NS-14', 'NS-16', 'NS-18', 'NS-19', 'NS-21', 'NS-22', 
		'NS-23', 'NS-24', 'NS-25', 'NS-26', 'NS-27', 'NS-28', 'NS-29', 'NS-32', 'NS-34']

		arr_exp_gtsrb = ['NS-5', 'NS-6', 'NS-8', 'NS-9', 'NS-11', 'NS-12', 'NS-15', 'NS-17', 'NS-19', 'NS-20', 
		'NS-30', 'NS-31', 'NS-33', 'NS-35', 'NS-37', 'NS-39', 'NS-41', 'NS-43', 'NS-67', 'NS-79']

		matrix_experiments = [arr_exp_cifar10, arr_exp_gtsrb]
		datasets = ['cifar10', 'gtsrb']

		#same order of arr_exp
		variations = ['gaussian_noise_severity_2', 'gaussian_noise_severity_5', 'impulse_noise_severity_2', 'impulse_noise_severity_5', 
		'shot_noise_severity_2', 'shot_noise_severity_5', 'spatter_severity_2', 'spatter_severity_5', 'speckle_noise_severity_2', 'speckle_noise_severity_5',
		'defocus_blur_severity_2', 'defocus_blur_severity_5', 'elastic_transform_severity_2', 'elastic_transform_severity_5', 'glass_blur_severity_2', 'glass_blur_severity_5',
		'zoom_blur_severity_2', 'zoom_blur_severity_5', 'gaussian_blur_severity_2', 'gaussian_blur_severity_5'
		]
		
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