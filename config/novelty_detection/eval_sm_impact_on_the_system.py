import os
import neptune_config as npte 
from src.plot_functions import pos_neg_stacked_bars
from src import util
from src.Classes.readout import Readout
from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import precision_recall_curve
import numpy as np
import disarray
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def print_dataframe_to_latex(df_results, caption, label, path_for_saving_plots, file_name):
	# it should be a for instead
	ind = df_results['Balanced accuracy'].idxmax()
	baseline = df_results['Balanced accuracy'][0]
	
	for i in range(1, len(df_results['Balanced accuracy'])):
		proposal = df_results['Balanced accuracy'][i]
		delta = round((proposal - baseline) / baseline * 100, 2)
		
		delta = '(+'+str(delta)+'\%)' if delta > 0 else '(-'+str(delta)+'\%)'
		df_results['Balanced accuracy'][i] = '{} {}'.format(df_results['Balanced accuracy'][i], delta)

	df_results['Balanced accuracy'].at[ind] = '\textbf{'+str(df_results['Balanced accuracy'][ind])+'}'

	tex_path = os.path.join(path_for_saving_plots, 'tex', file_name)
	
	df_results.to_latex(tex_path, caption=caption, label=label, escape=False, index=False)


def plot1(arr_id, names, label, caption, path_for_saving_plots, path_for_load_neptune):

	#table_ML = {'Architecture': [], 'Accuracy': [], 'F1': [], 'F1-Micro': []}
	#table_SM = {'Method': [], 'Detection': [], 'Confidence': [], 'Memory': [], 'Time': []}
	table_system = {'Experiment': [], 'Balanced accuracy': []}
	#table_novelty = {'TPR_ID_FPR_OOD': [], 'AUPR_ID': [], 'AUPR_OOD': []}

	project = npte.neptune_init(path_for_load_neptune)
	experiments = project.get_experiments(arr_id)
	
	arr_ml_pred = util.load_artifact('arr_classification_pred.npy', experiments)[0]
	arr_ml_true = util.load_artifact('arr_classification_true.npy', experiments)[0]
	print('arr_ml_pred', arr_ml_pred)
	print('arr_ml_true', arr_ml_true)
	
	# baseline (ML alone)
	table_system['Experiment'].append('Baseline')
	table_system['Balanced accuracy'].append(round(balanced_accuracy_score(arr_ml_true, arr_ml_pred), 2))

	# SM methods
	arr_reaction_SM = util.load_artifact('arr_reaction_SM.npy', experiments)
	arr_reaction_true = util.load_artifact('arr_reaction_true.npy', experiments)

	for name, y_true, y_pred in zip(names, arr_reaction_true, arr_reaction_SM):

		table_system['Experiment'].append(name)
		table_system['Balanced accuracy'].append(round(balanced_accuracy_score(y_true, y_pred), 2))

	df_results = pd.DataFrame.from_dict(table_system)
	#print(df_results)

	print_dataframe_to_latex(df_results, caption, label, path_for_saving_plots, '{}.tex'.format(label))


def plot2(arr_id, listOfMethods, path_for_saving_plots):
	
	def plot_precision_recall_curves(listOfResults):
		fig, ax = plt.subplots(1, 1)
		
		for (labels, predicted) in listOfResults:
			precision, recall, _ = precision_recall_curve(labels, predicted)
			decreasing_max_precision = np.maximum.accumulate(precision[::-1])[::-1]
			
			ax.step(perc_threats, decreasing_max_precision, '-')

		plt.title("Precision-Recall curve")
		plt.legend(listOfMethods)
		#plt.yticks(bins)
		#plt.xticks(bins)
		plt.ylabel("Precision")
		plt.xlabel("Recall")
		#plt.grid()
		plt.show()


	listOfResults = []
	project = npte.neptune_init('novelty-detection')
	experiments = project.get_experiments(arr_id)
	
	arr_pnc_id = util.load_artifact('Pos_Neg_Classified_ID.npy', experiments)
	arr_pnl_id = util.load_artifact('Pos_Neg_Labels_ID.npy', experiments)
	arr_pnc_ood = util.load_artifact('Pos_Neg_Classified_OOD.npy', experiments)
	arr_pnl_ood = util.load_artifact('Pos_Neg_Labels_OOD.npy', experiments)
	
	# total = ID + ODD
	for pnl_id, pnl_ood, pnc_id, pnc_ood in zip(arr_pnl_id, arr_pnl_ood, arr_pnc_id, arr_pnc_ood):
		
		y_true = np.hstack([pnl_id, pnl_ood])
		y_pred = np.hstack([pnc_id, pnc_ood])

		listOfResults.append((y_true, y_pred))

	plot_precision_recall_curves(listOfResults)