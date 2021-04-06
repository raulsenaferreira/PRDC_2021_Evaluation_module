import os
import argparse
from config.novelty_detection import plot_pos_neg_comparison
from config.novelty_detection import eval_sm_performance
from config.novelty_detection import eval_sm_impact_on_the_system
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

	exp_type = 'novelty_detection'

	classes_ID_OOD = [
	 [43, 62], # gtsrb + btsc
	 [43, 10], # gtsrb + cifar10
	 [10, 43] # cifar10 + gtsrb
	]

	# gtsrb+btsc = 26600 images for test (4005 ood instances)
	# gtsrb+cifar10 = 72600 images for test (60000 ood instances)
	# cifar10+gtsrb = 61800 images for test (51800 ood instances)

	#ploting statistics
	if args.config_id == 0:
		names = ["ALOOC", "OOB", "ODIN"] 
		num_datasets = 6 # benchmark datasets
		# avg MCC results: 
		# ALOOC = 0.01 + 0.02 + 0.05
		# OOB = 0.23 + 0.11 + 0.15
		# ODIN = 0.03 + 0.23 + 0.07
		avMCCs =  [0.02, 0.66, 0.11]

		# avg ranks: 
		# ALOOC = 3 + 3 + 3
		# OOB = 1 + 2 + 2
		# ODIN = 1 + 1 + 2
		avranks = [3, 1.66, 1.33]
		
		pf.plot_critical_difference(names, avranks, num_datasets)

	# experiments ODIN
	if args.config_id == 1:
		
		instances = [26600, 72600, 61800]

		arr_exp = [
		 'NOV-12', # gtsrb + btsc
		 'NOV-10', # cifar10 + gtsrb
		 'NOV-13' # gtsrb + cifar10
		]
		
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
		exp_id = 0
		indices_experiments = [0]
		for i in range(3):
			# SM's impact on the system
			label = 'table_{}'.format(7+i)
			caption = 'Table {}: comparing the impact of ODIN-based monitors for the datasets divided into ID and OOD.'.format(7+i)
			classes_ID, classes_OOD = classes_ID_OOD[i][0], classes_ID_OOD[i][1]
			
			eval_sm_impact_on_the_system.plot1([i], classes_ID, classes_OOD, readouts_ML, readouts_SM_reaction,
			 names, label, caption, path_for_saving_plots)
			#############################

			# SM results  
			label = 'table_{}'.format(10+i)
			caption = 'Table {}: comparing ODIN-based monitors for the datasets divided into ID and OOD.'.format(10+i)
			
			eval_sm_performance.plot1([i], readouts_SM_detection,
			 names, label, caption, path_for_saving_plots)
		#############################
		#############################
		#'''

		'''
		#############################
		# results across datasets: gtsrb + btsc, cifar10 + gtsrb, gtsrb + cifar10
		#############################
		arr_exp_dict = {}
		arr_exp_dict.update({names[0]: [experiments[0]]}) #gtsrb + btsc
		arr_exp_dict.update({names[1]: [experiments[1]]}) #cifar10 + gtsrb
		arr_exp_dict.update({names[2]: [experiments[2]]}) #gtsrb + cifar10
		
		# SM's impact 
		eval_sm_impact_on_the_system.plot2(arr_exp_dict, names, path_for_saving_plots)
		#############################
		
		# SM results
		eval_sm_performance.plot2(arr_exp_dict, path_for_saving_plots)
		#############################
		#############################
		'''
		#############################
		# Specific metrics for ID x OOD detection from the SM
		#eval_sm_performance.plot3(arr_exp, names, label, caption, path_for_saving_plots, path_for_load_neptune)
		#############################
		#############################

		'''
		# varables for plot_B
		datasets = ['GTSRB', 'CIFAR-10']
		#indices_experiments = {datasets[0]: [0, 1, 2, 3], datasets[1]: [4, 5, 6, 7]}
		indices_experiments = {datasets[0]: [4, 5, 6, 7], datasets[1]: [8, 9, 10, 11]}
		classes_ID = {datasets[0]: 43, datasets[1]: 10}
		label, caption = 'table_99', 'Table 1: MCC for measuring the overall impact of data-based SM in the system.'
		
		eval_sm_impact_on_the_system.plot1_B(datasets, indices_experiments, classes_ID, readouts_ML, readouts_SM_reaction,
		 names, label, caption, path_for_saving_plots)
		'''
		#plot_pos_neg_comparison.plot(arr_exp, names, caption, path_for_saving_plots)
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


	# experiments OOB
	elif args.config_id == 2:
		a = [26600]*12
		b = [72600]*12
		c = [61800]*12
		instances = np.concatenate((a, b,c), axis=None)
		#print(len(instances))

		arr_exp = [
		## gtsrb + btsc
		 'NOV-53', # simple + 3 + 1
		 'NOV-56', # simple + 3 + 1.1
		 'NOV-59', # simple + 3 + 1.35
		 'NOV-54', # isomap + 3 + 1
		 'NOV-57', # isomap + 3 + 1.1
		 'NOV-60', # isomap + 3 + 1.35
		 'NOV-55', # pca + 3 + 1
		 'NOV-58', # pca + 3 + 1.1
		 'NOV-61', # pca + 3 + 1.35

		 'NOV-50', # simple + 17 + 1
		 'NOV-51', # simple + 17 + 1.1
		 'NOV-52', # simple + 17 + 1.35

		 ## gtsrb + cifar10
		 'NOV-26', # simple + 0 + 1
		 'NOV-23', # simple + 0 + 1.1
		 'NOV-32', # simple + 0 + 1.35
		 'NOV-27', # isomap + 0 + 1
		 'NOV-24', # isomap + 0 + 1.1
		 'NOV-33', # isomap + 0 + 1.35
		 'NOV-28', # pca + 0 + 1
		 'NOV-25', # pca + 0 + 1.1
		 'NOV-34', # pca + 0 + 1.35

		 'NOV-7', # simple + 3 + 1
		 'NOV-8', # isomap + 3 + 1
		 'NOV-9', # pca + 3 + 1

		 ## cifar10 + gtsrb
		 'NOV-41', # simple + 5 + 1
		 'NOV-44', # simple + 5 + 1.1
		 'NOV-47', # simple + 5 + 1.35
		 'NOV-42', # isomap + 5 + 1
		 'NOV-45', # isomap + 5 + 1.1
		 'NOV-48', # isomap + 5 + 1.35
		 'NOV-43', # pca + 5 + 1
		 'NOV-46', # pca + 5 + 1.1
		 'NOV-49', # pca + 5 + 1.35

		 'NOV-4', # simple + 3 + 1
		 'NOV-5', # isomap + 3 + 1
		 'NOV-6' # pca + 3 + 1
		 ]
		
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

		'''
		arr_ml_time = util.load_values(experiments)
		arr_sm_time = util.load_values(experiments)
		arr_total_time = util.load_values(experiments)
		arr_total_memory = util.load_values(experiments)
		'''

		readouts_ML = [arr_ml_pred, arr_ml_true]
		readouts_SM_detection = [arr_detection_SM, arr_detection_true]
		readouts_SM_reaction = [arr_reaction_SM, arr_reaction_true]

		#############################
		######## gtsrb + btsc
		#############################
		dataset = 'GTSRB + BTSC'
		exp_id = 0
		
		names = ['simple + 3 + 1', 'simple + 3 + 1.1', 'simple + 3 + 1.35', 'isomap + 3 + 1', 'isomap + 3 + 1.1', 'isomap + 3 + 1.35',
		 'pca + 3 + 1','pca + 3 + 1.1', 'pca + 3 + 1.35', 'simple + 17 + 1', 'simple + 17 + 1.1', 'simple + 17 + 1.35']
		
		indices_experiments = list(range(0, 12))

		# SM's impact on the system
		label = 'table_1'
		caption = 'Table 1: comparing the impact of data-based monitors for GTSRB as ID dataset, and BTSC as OOD dataset.'
		classes_ID, classes_OOD = classes_ID_OOD[exp_id][0], classes_ID_OOD[exp_id][1]
		
		eval_sm_impact_on_the_system.plot1(indices_experiments, classes_ID, classes_OOD, readouts_ML, readouts_SM_reaction,
		 names, label, caption, path_for_saving_plots)
		#############################

		# SM results  
		label = 'table_4'
		caption = 'Table 4: comparing data-based monitors for GTSRB as ID dataset, and BTSC as OOD dataset.'
		
		eval_sm_performance.plot1(indices_experiments, readouts_SM_detection,
		 names, label, caption, path_for_saving_plots)

		# Time
		readouts_time = util.load_time_info(experiments, instances, indices_experiments, path_for_saving_plots, dataset, names)
		#############################
		#############################
		
		#############################
		######## gtsrb + cifar10
		#############################
		dataset = 'GTSRB + CIFAR-10'
		exp_id = 1
		names = ['simple + 0 + 1', 'simple + 0 + 1.1', 'simple + 0 + 1.35', 'isomap + 0 + 1', 'isomap + 0 + 1.1', 'isomap + 0 + 1.35',
		 'pca + 0 + 1', 'pca + 0 + 1.1', 'pca + 0 + 1.35', 'simple + 3 + 1', 'isomap + 3 + 1', 'pca + 3 + 1']
		
		indices_experiments = list(range(12, 24))

		# SM's impact on the system
		label = 'table_2'
		caption = 'Table 2: comparing the impact of data-based monitors for GTSRB as ID dataset, and CIFAR-10 as OOD dataset.'
		classes_ID, classes_OOD = classes_ID_OOD[exp_id][0], classes_ID_OOD[exp_id][1]
		
		eval_sm_impact_on_the_system.plot1(indices_experiments, classes_ID, classes_OOD, readouts_ML, readouts_SM_reaction,
		 names, label, caption, path_for_saving_plots)
		#############################
		
		# SM results  
		label = 'table_5'
		caption = 'Table 5: comparing data-based monitors for GTSRB as ID dataset, and CIFAR-10 as OOD dataset.'
		
		eval_sm_performance.plot1(indices_experiments, readouts_SM_detection,
		 names, label, caption, path_for_saving_plots)

		# Time
		readouts_time = util.load_time_info(experiments, instances, indices_experiments, path_for_saving_plots, dataset, names)
		#############################
		#############################

		#############################
		######## cifar10 + gtsrb
		#############################
		dataset = 'CIFAR-10 + GTSRB' 
		exp_id = 2
		names = ['simple + 5 + 1', 'simple + 5 + 1.1', 'simple + 5 + 1.35', 'isomap + 5 + 1', 'isomap + 5 + 1.1', 'isomap + 5 + 1.35',
		 'pca + 5 + 1', 'pca + 5 + 1.1', 'pca + 5 + 1.35', 'simple + 3 + 1', 'isomap + 3 + 1', 'pca + 3 + 1']
		
		indices_experiments = list(range(24, 36))

		# SM's impact on the system
		label = 'table_3'
		caption = 'Table 3: comparing the impact of data-based monitors for CIFAR-10 as ID dataset, and GTSRB as OOD dataset.'
		classes_ID, classes_OOD = classes_ID_OOD[exp_id][0], classes_ID_OOD[exp_id][1]
		
		eval_sm_impact_on_the_system.plot1(indices_experiments, classes_ID, classes_OOD, readouts_ML, readouts_SM_reaction,
		 names, label, caption, path_for_saving_plots)
		#############################

		# SM results 
		label = 'table_6'
		caption = 'Table 6: comparing data-based monitors for CIFAR-10 as ID dataset, and GTSRB as OOD dataset.'
		
		eval_sm_performance.plot1(indices_experiments, readouts_SM_detection,
		 names, label, caption, path_for_saving_plots)

		# Time
		readouts_time = util.load_time_info(experiments, instances, indices_experiments, path_for_saving_plots, dataset, names)
		#############################
		#############################
		'''
		#############################
		# results across datasets: gtsrb + btsc, cifar10 + gtsrb, gtsrb + cifar10
		#############################
		arr_exp_dict = {}
		arr_exp_dict.update({names[0]: [experiments[0], experiments[4], experiments[8]]}) #oob
		arr_exp_dict.update({names[1]: [experiments[1], experiments[5], experiments[9]]}) #oob isomap
		arr_exp_dict.update({names[2]: [experiments[2], experiments[6], experiments[10]]}) #oob pca
		arr_exp_dict.update({names[3]: [experiments[3], experiments[7], experiments[11]]}) #odin

		# SM's impact 
		#eval_sm_impact_on_the_system.plot2(arr_exp_dict, names, path_for_saving_plots)
		#############################
		
		# SM results
		#eval_sm_performance.plot2(arr_exp_dict, path_for_saving_plots)
		#############################
		#############################

		#############################
		# Specific metrics for ID x OOD detection from the SM
		#eval_sm_performance.plot3(arr_exp, names, label, caption, path_for_saving_plots, path_for_load_neptune)
		#############################
		#############################

		# variables for plot_B
		datasets = ['GTSRB', 'CIFAR-10']
		#indices_experiments = {datasets[0]: [0, 1, 2, 3], datasets[1]: [4, 5, 6, 7]}
		indices_experiments = {datasets[0]: [4, 5, 6, 7], datasets[1]: [8, 9, 10, 11]}
		classes_ID = {datasets[0]: 43, datasets[1]: 10}
		label, caption = 'table_99', 'Table 1: MCC for measuring the overall impact of data-based SM in the system.'
		
		eval_sm_impact_on_the_system.plot1_B(datasets, indices_experiments, classes_ID, readouts_ML, readouts_SM_reaction,
		 names, label, caption, path_for_saving_plots)
		'''
		#plot_pos_neg_comparison.plot(arr_exp, names, caption, path_for_saving_plots)
		
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

	# alooc
	elif args.config_id == 3:
		arr_exp = [
		 ## gtsrb + btsc
		 'NOV-67', # adam
		 'NOV-66', # rmsprop

		 ## gtsrb + cifar10
		 'NOV-65', # adam
		 'NOV-64', # rmsprop

		 ## cifar10 + gtsrb
		 'NOV-63', # adam
		 'NOV-62' # rmsprop
		 
		 ]

		instances = [26600, 26600, 72600, 72600, 61800, 61800]
		
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
		######## gtsrb + btsc
		#############################
		dataset = 'GTSRB + BTSC'
		exp_id = 0
		
		names = ['adam', 'rmsprop']
		
		indices_experiments = list(range(0, 2))

		# SM's impact on the system
		label = 'table_7'
		caption = 'Table 7: comparing the impact of alooc-based monitors for GTSRB as ID dataset, and BTSC as OOD dataset.'
		classes_ID, classes_OOD = classes_ID_OOD[exp_id][0], classes_ID_OOD[exp_id][1]
		
		eval_sm_impact_on_the_system.plot1(indices_experiments, classes_ID, classes_OOD, readouts_ML, readouts_SM_reaction,
		 names, label, caption, path_for_saving_plots)

		#############################
		# SM results  
		label = 'table_10'
		caption = 'Table 10: comparing alooc-based monitors for GTSRB as ID dataset, and BTSC as OOD dataset.'
		
		eval_sm_performance.plot1(indices_experiments, readouts_SM_detection,
		 names, label, caption, path_for_saving_plots)

		# Time
		readouts_time = util.load_time_info(experiments, instances, indices_experiments, path_for_saving_plots, dataset, names)
		#############################
		#############################
		
		#############################
		######## gtsrb + cifar10
		#############################
		dataset = 'GTSRB + CIFAR-10'
		exp_id = 1
		names = ['adam', 'rmsprop']
		
		indices_experiments = list(range(2, 4))

		# SM's impact on the system
		label = 'table_8'
		caption = 'Table 8: comparing the impact of alooc-based monitors for GTSRB as ID dataset, and CIFAR-10 as OOD dataset.'
		classes_ID, classes_OOD = classes_ID_OOD[exp_id][0], classes_ID_OOD[exp_id][1]
		
		eval_sm_impact_on_the_system.plot1(indices_experiments, classes_ID, classes_OOD, readouts_ML, readouts_SM_reaction,
		 names, label, caption, path_for_saving_plots)
		#############################
		
		# SM results  
		label = 'table_11'
		caption = 'Table 11: comparing alooc-based monitors for GTSRB as ID dataset, and CIFAR-10 as OOD dataset.'
		
		eval_sm_performance.plot1(indices_experiments, readouts_SM_detection,
		 names, label, caption, path_for_saving_plots)

		# Time
		readouts_time = util.load_time_info(experiments, instances, indices_experiments, path_for_saving_plots, dataset, names)
		#############################
		#############################

		#############################
		######## cifar10 + gtsrb
		#############################
		dataset = 'CIFAR-10 + GTSRB'
		exp_id = 2
		names = ['adam', 'rmsprop']
		
		indices_experiments = list(range(4, 6))

		# SM's impact on the system
		label = 'table_9'
		caption = 'Table 9: comparing the impact of alooc-based monitors for CIFAR-10 as ID dataset, and GTSRB as OOD dataset.'
		classes_ID, classes_OOD = classes_ID_OOD[exp_id][0], classes_ID_OOD[exp_id][1]
		
		eval_sm_impact_on_the_system.plot1(indices_experiments, classes_ID, classes_OOD, readouts_ML, readouts_SM_reaction,
		 names, label, caption, path_for_saving_plots)
		#############################

		# SM results 
		label = 'table_12'
		caption = 'Table 12: comparing alooc-based monitors for CIFAR-10 as ID dataset, and GTSRB as OOD dataset.'
		
		eval_sm_performance.plot1(indices_experiments, readouts_SM_detection,
		 names, label, caption, path_for_saving_plots)

		# Time
		readouts_time = util.load_time_info(experiments, instances, indices_experiments, path_for_saving_plots, dataset, names)
		#############################
		#############################