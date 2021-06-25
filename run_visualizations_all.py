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


	#############################
	######## Statistical analysis
	#############################
	if args.config_id == 0:
		names = ['ALOOC', 'OOB', 'ODIN']
		# avg ranks: 
		# ALOOC = indice 0
		# OOB = indice 1
		# ODIN = indice 2
		matrix_ranks = [[3, 2, 1], [3,2,1], [3,1,2], [3,2,1], [1,3,2], 
				[1,2,3], [3,1,1], [1,2,2], [1,2,3], [2,2,1], [1,2,3], [1,2,2], [3,1,2], [1,2,1], [1,2,2], [2,2,1], [3,2,1],
				[2,1,3], [3,2,1], [1,2,3], [2,2,1], [1,3,2], [1,2,2], [1,3,2], [2,3,1], [1,2,2], [2,3,1], [2,1,3],
				[2,1,3], [1,3,2], [3,1,1], [1,3,2], [3,2,1], [1,2,2], [1,3,2], [3,2,1], [1,2,3], [3,2,1], [3,1,2],
				[3,1,1], [3,2,1], [3,2,1], [3,2,1], [3,2,1], [3,1,1], [3,1,1], [3,1,1], [3,2,1], [1,2,2], [1,3,2], 
				[1,3,2], [1,3,2], [1,3,2], [1,3,2], [1,3,2], [1,3,2], [1,3,2], [1,3,2], [1,3,2],
				[1,2,3], [1,2,3], [1,2,3], [1,2,3], [1,1,3], [1,1,3], [1,1,3], [1,1,3], [1,2,3], [1,2,3], [3,1,2],
				[3,2,1], [3,2,1], [3,2,1], [3,2,1], [3,2,1], [3,2,1], [3,2,1], [3,2,1], [3,2,1]]
		matrix_ranks = np.asarray(matrix_ranks)

		num_datasets = len(matrix_ranks) # benchmark datasets

		avranks = matrix_ranks.sum(axis=0)/num_datasets
		#print(avranks)
		
		pf.plot_critical_difference(names, avranks, num_datasets)

	#############################
	######## DISTRIBUTIONAL SHIFT
	#############################
	if args.config_id == 1:
		exp_type = 'distributional_shift'

		matrix_experiments = {}
		variations = {}
		
		datasets = ['cifar10', 'gtsrb']
		instances = [22665, 26001]

		path_for_saving_plots = os.path.join(args.path_for_saving_plots, exp_type)
		path_for_load_neptune = exp_type.replace('_', '-') # correcting for loading neptune experiments

		method_names = 'ALOOC \\\ OOB \\\ ODIN'

		indices_tables = 11 #len(arr_exp)


		#############################
		######## ALOOC
		#############################

		arr_exp_gtsrb = ['DSHIFT-23', 'DSHIFT-25', 'DSHIFT-27', 'DSHIFT-29', 'DSHIFT-31', 'DSHIFT-33', 
		'DSHIFT-35', 'DSHIFT-37', 'DSHIFT-39', 'DSHIFT-41', 'DSHIFT-43']
		arr_exp_cifar10 = ['DSHIFT-24', 'DSHIFT-26', 'DSHIFT-28', 'DSHIFT-30', 'DSHIFT-32', 'DSHIFT-34', 
		'DSHIFT-36', 'DSHIFT-38', 'DSHIFT-40', 'DSHIFT-42', 'DSHIFT-44']

		matrix_experiments.update({'cifar10': [arr_exp_cifar10]})
		matrix_experiments.update({'gtsrb': [arr_exp_gtsrb]})

		#same order of arr_exp
		variations_alooc = ['rotated', 'snow_severity_5', 'fog_severity_5', 'brightness_severity_5', 'contrast_severity_5', 'saturate_severity_5',
		'snow_severity_2', 'fog_severity_2', 'brightness_severity_2', 'contrast_severity_2', 'saturate_severity_2']

		variations.update({'cifar10': [variations_alooc]})
		variations.update({'gtsrb': [variations_alooc]})


		#############################
		######## OOB
		#############################
		technique_name_2 = "OOB"

		arr_exp_cifar10 = ['DSHIFT-1', 'DSHIFT-3', 'DSHIFT-6', 'DSHIFT-9', 'DSHIFT-10', 'DSHIFT-15', 
		'DSHIFT-18', 'DSHIFT-19', 'DSHIFT-20', 'DSHIFT-21', 'DSHIFT-22']
		arr_exp_gtsrb = ['DSHIFT-2', 'DSHIFT-4', 'DSHIFT-5', 'DSHIFT-7', 'DSHIFT-8', 'DSHIFT-11', 
		'DSHIFT-12', 'DSHIFT-13', 'DSHIFT-14', 'DSHIFT-16', 'DSHIFT-17']

		matrix_experiments['cifar10'].append(arr_exp_cifar10)
		matrix_experiments['gtsrb'].append(arr_exp_gtsrb)

		#same order of arr_exp
		variations_oob = ['snow_severity_2', 'fog_severity_2', 'brightness_severity_2', 'contrast_severity_2', 'saturate_severity_2', 'rotated',
		'snow_severity_5', 'fog_severity_5', 'brightness_severity_5', 'contrast_severity_5', 'saturate_severity_5']

		variations['cifar10'].append(variations_oob)
		variations['gtsrb'].append(variations_oob)
		
		#############################
		######## ODIN
		#############################

		technique_name_3 = "ODIN"

		arr_exp_cifar10 = ['DSHIFT-45', 'DSHIFT-46', 'DSHIFT-47', 'DSHIFT-48', 'DSHIFT-49', 'DSHIFT-50', 
		'DSHIFT-51', 'DSHIFT-52', 'DSHIFT-53', 'DSHIFT-54', 'DSHIFT-55']
		arr_exp_gtsrb = ['DSHIFT-56', 'DSHIFT-57', 'DSHIFT-58', 'DSHIFT-59', 'DSHIFT-60', 'DSHIFT-61', 
		'DSHIFT-62', 'DSHIFT-63', 'DSHIFT-64', 'DSHIFT-65', 'DSHIFT-66']

		matrix_experiments['cifar10'].append(arr_exp_cifar10)
		matrix_experiments['gtsrb'].append(arr_exp_gtsrb)

		#same order of arr_exp
		variations_odin = ['snow_severity_2', 'fog_severity_2', 'brightness_severity_2', 'contrast_severity_2', 'saturate_severity_2', 'rotated',
		'snow_severity_5', 'fog_severity_5', 'brightness_severity_5', 'contrast_severity_5', 'saturate_severity_5']
		
		variations['cifar10'].append(variations_odin)
		variations['gtsrb'].append(variations_odin)


		#############################
		######## one base dataset each time
		#############################

		for dataset in datasets:
			project = npte.neptune_init(path_for_load_neptune)
			experiments_alooc = project.get_experiments(matrix_experiments[dataset][0])
			experiments_oob = project.get_experiments(matrix_experiments[dataset][1])
			experiments_odin = project.get_experiments(matrix_experiments[dataset][2])

			# readouts
			#arr_ml_pred = util.load_artifact('arr_classification_pred.npy', experiments)
			#arr_ml_true = util.load_artifact('arr_classification_true.npy', experiments)

			arr_detection_SM_alooc = util.load_artifact('arr_detection_SM.npy', experiments_alooc)
			arr_detection_true_alooc = util.load_artifact('arr_detection_true.npy', experiments_alooc)

			arr_detection_SM_oob = util.load_artifact('arr_detection_SM.npy', experiments_oob)
			arr_detection_true_oob = util.load_artifact('arr_detection_true.npy', experiments_oob)

			arr_detection_SM_odin = util.load_artifact('arr_detection_SM.npy', experiments_odin)
			arr_detection_true_odin = util.load_artifact('arr_detection_true.npy', experiments_odin)

			#arr_reaction_SM = util.load_artifact('arr_reaction_SM.npy', experiments)
			#arr_reaction_true = util.load_artifact('arr_reaction_true.npy', experiments)
			
			#readouts_ML = [arr_ml_pred, arr_ml_true]
			readouts_SM_detection_alooc = [arr_detection_SM_alooc, arr_detection_true_alooc]
			readouts_SM_detection_oob = [arr_detection_SM_oob, arr_detection_true_oob]
			readouts_SM_detection_odin = [arr_detection_SM_odin, arr_detection_true_odin]

			readouts_SM_detection = [readouts_SM_detection_alooc, readouts_SM_detection_oob, readouts_SM_detection_odin]
			#readouts_SM_reaction = [arr_reaction_SM, arr_reaction_true]

			# SM results  
			label = 'table_noise_{}'.format(dataset)
			caption = 'Comparing data-based monitors for {} dataset with different types of noise.'.format(dataset)
			eval_sm_performance.plot_new(indices_tables, readouts_SM_detection, variations[dataset], method_names, label, caption, path_for_saving_plots)
		#############################
		#############################

	#############################
	######## NOISE DETECTION
	#############################

	elif args.config_id == 2:
		exp_type = 'noise'

		matrix_experiments = {}
		variations = {}
		
		datasets = ['cifar10', 'gtsrb']
		instances = [22665, 26001]

		path_for_saving_plots = os.path.join(args.path_for_saving_plots, exp_type)
		path_for_load_neptune = exp_type.replace('_', '-') # correcting for loading neptune experiments

		method_names = 'ALOOC \\\ OOB \\\ ODIN'

		indices_tables = 20 #len(arr_exp)


		#############################
		######## ALOOC
		#############################

		arr_exp_cifar10 = ['NS-36', 'NS-38', 'NS-40', 'NS-42', 'NS-53', 'NS-76', 'NS-80', 'NS-85', 'NS-91', 'NS-92',
		'NS-101', 'NS-103', 'NS-105', 'NS-107', 'NS-109', 'NS-111', 'NS-113', 'NS-115', 'NS-117', 'NS-119']
		arr_exp_gtsrb = ['NS-100', 'NS-102', 'NS-104', 'NS-106', 'NS-108', 'NS-110', 'NS-112', 'NS-114', 'NS-116', 'NS-118', 
		'NS-120', 'NS-121', 'NS-122', 'NS-123', 'NS-124', 'NS-125', 'NS-126', 'NS-127', 'NS-128', 'NS-129']

		matrix_experiments.update({'cifar10': [arr_exp_cifar10]})
		matrix_experiments.update({'gtsrb': [arr_exp_gtsrb]})

		#same order of arr_exp
		variations_alooc = ['gaussian_noise_severity_2', 'gaussian_noise_severity_5', 'impulse_noise_severity_2', 'impulse_noise_severity_5', 
		'shot_noise_severity_2', 'shot_noise_severity_5', 'spatter_severity_2', 'spatter_severity_5', 'speckle_noise_severity_2', 'speckle_noise_severity_5',
		'defocus_blur_severity_2', 'defocus_blur_severity_5', 'elastic_transform_severity_2', 'elastic_transform_severity_5', 'glass_blur_severity_2', 'glass_blur_severity_5',
		'zoom_blur_severity_2', 'zoom_blur_severity_5', 'gaussian_blur_severity_2', 'gaussian_blur_severity_5'
		]

		variations.update({'cifar10': [variations_alooc]})
		variations.update({'gtsrb': [variations_alooc]})


		#############################
		######## OOB
		#############################
		technique_name_2 = "OOB"

		arr_exp_cifar10 = ['NS-3', 'NS-4', 'NS-7', 'NS-10', 'NS-13', 'NS-14', 'NS-16', 'NS-18', 'NS-19', 'NS-21', 'NS-22', 
		'NS-23', 'NS-24', 'NS-25', 'NS-26', 'NS-27', 'NS-28', 'NS-29', 'NS-32', 'NS-34']
		arr_exp_gtsrb = ['NS-5', 'NS-6', 'NS-8', 'NS-9', 'NS-11', 'NS-12', 'NS-15', 'NS-17', 'NS-19', 'NS-20', 
		'NS-30', 'NS-31', 'NS-33', 'NS-35', 'NS-37', 'NS-39', 'NS-41', 'NS-43', 'NS-67', 'NS-79']

		matrix_experiments['cifar10'].append(arr_exp_cifar10)
		matrix_experiments['gtsrb'].append(arr_exp_gtsrb)

		#same order of arr_exp
		variations_oob = ['gaussian_noise_severity_2', 'gaussian_noise_severity_5', 'impulse_noise_severity_2', 'impulse_noise_severity_5', 
		'shot_noise_severity_2', 'shot_noise_severity_5', 'spatter_severity_2', 'spatter_severity_5', 'speckle_noise_severity_2', 'speckle_noise_severity_5',
		'defocus_blur_severity_2', 'defocus_blur_severity_5', 'elastic_transform_severity_2', 'elastic_transform_severity_5', 'glass_blur_severity_2', 'glass_blur_severity_5',
		'zoom_blur_severity_2', 'zoom_blur_severity_5', 'gaussian_blur_severity_2', 'gaussian_blur_severity_5'
		]

		variations['cifar10'].append(variations_oob)
		variations['gtsrb'].append(variations_oob)
		
		#############################
		######## ODIN
		#############################

		technique_name_3 = "ODIN"

		arr_exp_cifar10 = ['NS-44', 'NS-46', 'NS-48', 'NS-50', 'NS-52', 'NS-55', 'NS-57', 'NS-59', 'NS-61', 'NS-63', 'NS-65', 'NS-23', 
		'NS-82', 'NS-83', 'NS-84', 'NS-86', 'NS-87', 'NS-88', 'NS-89', 'NS-90']
		arr_exp_gtsrb = ['NS-45', 'NS-47', 'NS-49', 'NS-51', 'NS-54', 'NS-56', 'NS-58', 'NS-60', 'NS-62', 'NS-64', 'NS-66', 'NS-31',
		'NS-70', 'NS-71', 'NS-72', 'NS-73', 'NS-74', 'NS-75', 'NS-77', 'NS-78']

		matrix_experiments['cifar10'].append(arr_exp_cifar10)
		matrix_experiments['gtsrb'].append(arr_exp_gtsrb)

		#same order of arr_exp
		variations_odin = ['gaussian_noise_severity_2', 'gaussian_noise_severity_5', 'impulse_noise_severity_2', 'impulse_noise_severity_5', 
		'shot_noise_severity_2', 'shot_noise_severity_5', 'spatter_severity_2', 'spatter_severity_5', 'speckle_noise_severity_2', 'speckle_noise_severity_5',
		'defocus_blur_severity_2', 'defocus_blur_severity_5', 'elastic_transform_severity_2', 'elastic_transform_severity_5', 'glass_blur_severity_2', 'glass_blur_severity_5',
		'zoom_blur_severity_2', 'zoom_blur_severity_5', 'gaussian_blur_severity_2', 'gaussian_blur_severity_5'
		]
		
		variations['cifar10'].append(variations_odin)
		variations['gtsrb'].append(variations_odin)


		#############################
		######## one base dataset each time
		#############################

		for dataset in datasets:
			project = npte.neptune_init(path_for_load_neptune)
			experiments_alooc = project.get_experiments(matrix_experiments[dataset][0])
			experiments_oob = project.get_experiments(matrix_experiments[dataset][1])
			experiments_odin = project.get_experiments(matrix_experiments[dataset][2])

			# readouts
			#arr_ml_pred = util.load_artifact('arr_classification_pred.npy', experiments)
			#arr_ml_true = util.load_artifact('arr_classification_true.npy', experiments)

			arr_detection_SM_alooc = util.load_artifact('arr_detection_SM.npy', experiments_alooc)
			arr_detection_true_alooc = util.load_artifact('arr_detection_true.npy', experiments_alooc)

			arr_detection_SM_oob = util.load_artifact('arr_detection_SM.npy', experiments_oob)
			arr_detection_true_oob = util.load_artifact('arr_detection_true.npy', experiments_oob)

			arr_detection_SM_odin = util.load_artifact('arr_detection_SM.npy', experiments_odin)
			arr_detection_true_odin = util.load_artifact('arr_detection_true.npy', experiments_odin)

			#arr_reaction_SM = util.load_artifact('arr_reaction_SM.npy', experiments)
			#arr_reaction_true = util.load_artifact('arr_reaction_true.npy', experiments)
			
			#readouts_ML = [arr_ml_pred, arr_ml_true]
			readouts_SM_detection_alooc = [arr_detection_SM_alooc, arr_detection_true_alooc]
			readouts_SM_detection_oob = [arr_detection_SM_oob, arr_detection_true_oob]
			readouts_SM_detection_odin = [arr_detection_SM_odin, arr_detection_true_odin]

			readouts_SM_detection = [readouts_SM_detection_alooc, readouts_SM_detection_oob, readouts_SM_detection_odin]
			#readouts_SM_reaction = [arr_reaction_SM, arr_reaction_true]

			# SM results  
			label = 'table_noise_{}'.format(dataset)
			caption = 'Comparing data-based monitors for {} dataset with different types of noise.'.format(dataset)
			eval_sm_performance.plot_new(indices_tables, readouts_SM_detection, variations[dataset], method_names, label, caption, path_for_saving_plots)

		# Time
		#readouts_time = util.load_time_info(experiments, instances, indices_experiments, path_for_saving_plots, dataset, names)
		#############################
		#############################

	#############################
	######## ANOMALY DETECTION
	#############################

	elif args.config_id == 3:
		exp_type = 'anomaly_detection'
		matrix_experiments = {}
		variations = {}
		
		datasets = ['cifar10', 'gtsrb']
		instances = [22665, 26001]

		path_for_saving_plots = os.path.join(args.path_for_saving_plots, exp_type)
		path_for_load_neptune = exp_type.replace('_', '-') # correcting for loading neptune experiments

		method_names = 'ALOOC \\\ OOB \\\ ODIN'

		indices_tables = 6 #len(arr_exp)


		#############################
		######## ALOOC
		#############################

		arr_exp_cifar10 = ['AN-14', 'AN-16', 'AN-18', 'AN-20', 'AN-22', 'AN-24']
		arr_exp_gtsrb = ['AN-13', 'AN-15', 'AN-17', 'AN-19', 'AN-21', 'AN-23']

		matrix_experiments.update({'cifar10': [arr_exp_cifar10]})
		matrix_experiments.update({'gtsrb': [arr_exp_gtsrb]})

		#same order of arr_exp
		variations_alooc = ['pixel trap severity 3', 'row add logic severity 3', 'shifted pixel severity 3',
		'pixel trap severity 1', 'row add logic severity 1', 'shifted pixel severity 1']

		variations.update({'cifar10': [variations_alooc]})
		variations.update({'gtsrb': [variations_alooc]})


		#############################
		######## OOB
		#############################
		technique_name_2 = "OOB"

		arr_exp_cifar10 = ['AN-1', 'AN-3', 'AN-5', 'AN-8', 'AN-11', 'AN-12']
		arr_exp_gtsrb = ['AN-2', 'AN-4', 'AN-6', 'AN-7', 'AN-9', 'AN-10']

		matrix_experiments['cifar10'].append(arr_exp_cifar10)
		matrix_experiments['gtsrb'].append(arr_exp_gtsrb)

		#same order of arr_exp
		variations_oob = ['pixel trap severity 1', 'row add logic severity 1', 'shifted pixel severity 1', 
		'pixel trap severity 3', 'row add logic severity 3', 'shifted pixel severity 3']

		variations['cifar10'].append(variations_oob)
		variations['gtsrb'].append(variations_oob)
		
		#############################
		######## ODIN
		#############################

		technique_name_3 = "ODIN"

		arr_exp_cifar10 = ['AN-25', 'AN-26', 'AN-27', 'AN-28', 'AN-29', 'AN-30']
		arr_exp_gtsrb = ['AN-31', 'AN-32', 'AN-33', 'AN-34', 'AN-35', 'AN-36']

		matrix_experiments['cifar10'].append(arr_exp_cifar10)
		matrix_experiments['gtsrb'].append(arr_exp_gtsrb)

		#same order of arr_exp
		variations_odin = ['pixel trap severity 1', 'row add logic severity 1', 'shifted pixel severity 1', 
		'pixel trap severity 3', 'row add logic severity 3', 'shifted pixel severity 3']
		
		variations['cifar10'].append(variations_odin)
		variations['gtsrb'].append(variations_odin)


		#############################
		######## one base dataset each time
		#############################

		for dataset in datasets:
			project = npte.neptune_init(path_for_load_neptune)
			experiments_alooc = project.get_experiments(matrix_experiments[dataset][0])
			experiments_oob = project.get_experiments(matrix_experiments[dataset][1])
			experiments_odin = project.get_experiments(matrix_experiments[dataset][2])

			# readouts
			#arr_ml_pred = util.load_artifact('arr_classification_pred.npy', experiments)
			#arr_ml_true = util.load_artifact('arr_classification_true.npy', experiments)

			arr_detection_SM_alooc = util.load_artifact('arr_detection_SM.npy', experiments_alooc)
			arr_detection_true_alooc = util.load_artifact('arr_detection_true.npy', experiments_alooc)

			arr_detection_SM_oob = util.load_artifact('arr_detection_SM.npy', experiments_oob)
			arr_detection_true_oob = util.load_artifact('arr_detection_true.npy', experiments_oob)

			arr_detection_SM_odin = util.load_artifact('arr_detection_SM.npy', experiments_odin)
			arr_detection_true_odin = util.load_artifact('arr_detection_true.npy', experiments_odin)

			#arr_reaction_SM = util.load_artifact('arr_reaction_SM.npy', experiments)
			#arr_reaction_true = util.load_artifact('arr_reaction_true.npy', experiments)
			
			#readouts_ML = [arr_ml_pred, arr_ml_true]
			readouts_SM_detection_alooc = [arr_detection_SM_alooc, arr_detection_true_alooc]
			readouts_SM_detection_oob = [arr_detection_SM_oob, arr_detection_true_oob]
			readouts_SM_detection_odin = [arr_detection_SM_odin, arr_detection_true_odin]

			readouts_SM_detection = [readouts_SM_detection_alooc, readouts_SM_detection_oob, readouts_SM_detection_odin]
			#readouts_SM_reaction = [arr_reaction_SM, arr_reaction_true]

			# SM results  
			label = 'table_anomaly_{}'.format(dataset)
			caption = 'Comparing data-based monitors for {} dataset with different types of anomalies.'.format(dataset)
			eval_sm_performance.plot_new(indices_tables, readouts_SM_detection, variations[dataset], method_names, label, caption, path_for_saving_plots)

		# Time
		#readouts_time = util.load_time_info(experiments, instances, indices_experiments, path_for_saving_plots, dataset, names)
		#############################
		#############################

	#############################
	######## ADVERSARIAL ATTACK
	#############################
	elif args.config_id == 4:
		exp_type = 'adversarial_attack'

		matrix_experiments = {}
		variations = {}
		
		datasets = ['cifar10', 'gtsrb']
		instances = [22665, 26001]

		path_for_saving_plots = os.path.join(args.path_for_saving_plots, exp_type)
		path_for_load_neptune = exp_type.replace('_', '-') # correcting for loading neptune experiments

		method_names = 'ALOOC \\\ OOB \\\ ODIN'

		indices_tables = 1 #len(arr_exp)


		#############################
		######## ALOOC
		#############################

		arr_exp_gtsrb = ['AD-3']
		arr_exp_cifar10 = ['AD-4']

		matrix_experiments.update({'cifar10': [arr_exp_cifar10]})
		matrix_experiments.update({'gtsrb': [arr_exp_gtsrb]})

		#same order of arr_exp
		variations_alooc = ['FGSM']

		variations.update({'cifar10': [variations_alooc]})
		variations.update({'gtsrb': [variations_alooc]})


		#############################
		######## OOB
		#############################
		technique_name_2 = "OOB"

		arr_exp_cifar10 = ['AD-5']
		arr_exp_gtsrb = ['AD-6']

		matrix_experiments['cifar10'].append(arr_exp_cifar10)
		matrix_experiments['gtsrb'].append(arr_exp_gtsrb)

		#same order of arr_exp
		variations_oob = ['FGSM']

		variations['cifar10'].append(variations_oob)
		variations['gtsrb'].append(variations_oob)
		
		#############################
		######## ODIN
		#############################

		technique_name_3 = "ODIN"

		arr_exp_cifar10 = ['AD-1']
		arr_exp_gtsrb = ['AD-2']

		matrix_experiments['cifar10'].append(arr_exp_cifar10)
		matrix_experiments['gtsrb'].append(arr_exp_gtsrb)

		#same order of arr_exp
		variations_odin = ['FGSM']
		
		variations['cifar10'].append(variations_odin)
		variations['gtsrb'].append(variations_odin)


		#############################
		######## one base dataset each time
		#############################

		for dataset in datasets:
			project = npte.neptune_init(path_for_load_neptune)
			experiments_alooc = project.get_experiments(matrix_experiments[dataset][0])
			experiments_oob = project.get_experiments(matrix_experiments[dataset][1])
			experiments_odin = project.get_experiments(matrix_experiments[dataset][2])

			# readouts
			#arr_ml_pred = util.load_artifact('arr_classification_pred.npy', experiments)
			#arr_ml_true = util.load_artifact('arr_classification_true.npy', experiments)

			arr_detection_SM_alooc = util.load_artifact('arr_detection_SM.npy', experiments_alooc)
			arr_detection_true_alooc = util.load_artifact('arr_detection_true.npy', experiments_alooc)

			arr_detection_SM_oob = util.load_artifact('arr_detection_SM.npy', experiments_oob)
			arr_detection_true_oob = util.load_artifact('arr_detection_true.npy', experiments_oob)

			arr_detection_SM_odin = util.load_artifact('arr_detection_SM.npy', experiments_odin)
			arr_detection_true_odin = util.load_artifact('arr_detection_true.npy', experiments_odin)

			#arr_reaction_SM = util.load_artifact('arr_reaction_SM.npy', experiments)
			#arr_reaction_true = util.load_artifact('arr_reaction_true.npy', experiments)
			
			#readouts_ML = [arr_ml_pred, arr_ml_true]
			readouts_SM_detection_alooc = [arr_detection_SM_alooc, arr_detection_true_alooc]
			readouts_SM_detection_oob = [arr_detection_SM_oob, arr_detection_true_oob]
			readouts_SM_detection_odin = [arr_detection_SM_odin, arr_detection_true_odin]

			readouts_SM_detection = [readouts_SM_detection_alooc, readouts_SM_detection_oob, readouts_SM_detection_odin]
			#readouts_SM_reaction = [arr_reaction_SM, arr_reaction_true]

			# SM results  
			label = 'table_noise_{}'.format(dataset)
			caption = 'Comparing data-based monitors for {} dataset with different types of noise.'.format(dataset)
			eval_sm_performance.plot_new(indices_tables, readouts_SM_detection, variations[dataset], method_names, label, caption, path_for_saving_plots)