import os
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


def print_dataframe_to_latex(df_results, caption, label, cols, path_for_saving_plots, file_name):
	 
	def format_df_to_latex(df_results, col):
		baseline = df_results[col][0]
		
		print('df_results[{}] = {}'.format(col, df_results[col]))

		ind = df_results[col].idxmax()

		for i in range(1, len(df_results[col])):
			proposal = df_results[col][i]
			# formula 1
			delta = round((proposal - baseline) / baseline * 100, 2)
			# formula 2
			#delta = round((proposal / baseline) * 100, 2) - 100

			delta = '(+'+str(delta)+'\%)' if delta > 0 else '(-'+str(delta)+'\%)'
			df_results[col][i] = '{} {}'.format(df_results[col][i], delta)

		try:
			df_results[col].at[ind] = '\textbf{'+str(df_results[col][ind])+'}'
		except:
			print('Some error here...', 'col', col, 'ind', ind)

		return df_results

	df_results = format_df_to_latex(df_results, cols[1])
	df_results = format_df_to_latex(df_results, cols[2])
	df_results = format_df_to_latex(df_results, cols[3])

	tex_path = os.path.join(path_for_saving_plots, 'tex', file_name)
	
	df_results.to_latex(tex_path, caption=caption, label=label, escape=False, index=False)


def plot1(indices_experiments, classes_ID, classes_OOD, readouts_ML, readouts_SM_reaction,
		 names, label, caption, path_for_saving_plots, is_pytorch=False):
	
	def acc(arr_true, arr_pred):
		return round(balanced_accuracy_score(arr_true, arr_pred), 2)

	col_0 = 'Method / MCC'
	col_1 = 'ID'
	col_2 = 'OOD'
	col_3 = 'Entire stream'
	cols = [col_0, col_1, col_2, col_3]

	table_system = {cols[0]: [], cols[1]: [], cols[2]: [], cols[3]: []}
	
	ML_pred = readouts_ML[0][indices_experiments[0]]
	labels = readouts_ML[1][indices_experiments[0]]

	if is_pytorch:
		ML_pred = [p.data.cpu().numpy()[0] for p in ML_pred]
		ML_pred = np.asarray(ML_pred)
	
	# for measuring of correct/incorrect reactions divided into ID, OOD data and in total
	y_true_ID = np.where(labels < classes_ID)[0]
	y_true_OOD = np.where(labels >= classes_ID)[0]
	
	table_system[cols[0]].append('Baseline') # baseline (ML alone)
	table_system[cols[1]].append(acc(labels[y_true_ID], ML_pred[y_true_ID]))
	table_system[cols[2]].append(acc(labels[y_true_OOD], ML_pred[y_true_OOD]))
	table_system[cols[3]].append(acc(labels, ML_pred))

	# SM methods
	arr_reaction_SM = readouts_SM_reaction[0]
	arr_reaction_true = readouts_SM_reaction[1]

	for i, name in zip(indices_experiments, names):
		print('name: {}, experiment: {}'.format(name, i))
		print(len(arr_reaction_true), len(arr_reaction_SM))
		
		reaction_true, reaction_pred = arr_reaction_true[i], arr_reaction_SM[i]

		y_true_ID = np.where(reaction_true < classes_ID)[0]
		y_true_OOD = np.where(reaction_true >= classes_ID)[0]

		reaction_pred_ID, reaction_pred_OOD = reaction_pred[y_true_ID], reaction_pred[y_true_OOD]
		reaction_true_ID, reaction_true_OOD = reaction_true[y_true_ID], reaction_true[y_true_OOD]

		# getting some measurements
		table_system[cols[0]].append(name)
		table_system[cols[1]].append(acc(reaction_true_ID, reaction_pred_ID))
		table_system[cols[2]].append(acc(reaction_true_OOD, reaction_pred_OOD))
		table_system[cols[3]].append(acc(reaction_true, reaction_pred))		

	df_results = pd.DataFrame.from_dict(table_system)
	#print(df_results)

	print_dataframe_to_latex(df_results, caption, label, cols, path_for_saving_plots, '{}.tex'.format(label))


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
	
	arr_ml_pred = util.load_artifact('arr_classification_pred.npy', experiments)[0]
	arr_ml_true = util.load_artifact('arr_classification_true.npy', experiments)[0]
	
	# total = ID + ODD
	for y_true, y_pred in zip(arr_ml_true, arr_ml_pred):
		
		listOfResults.append((y_true, y_pred))

	plot_precision_recall_curves(listOfResults)



def print_dataframe_to_latex_B(datasets, df_results, caption, label, cols, path_for_saving_plots, file_name):
	 
	def format_df_to_latex(baseline, row, df_results, col):
	
		proposal = df_results[col][row]
		# formula 1
		delta = round((proposal - baseline) / baseline * 100, 2)
		# formula 2
		#delta = round((proposal / baseline) * 100, 2) - 100

		delta = '(+'+str(delta)+'\%)' if delta > 0 else '(-'+str(delta)+'\%)'
		df_results[col][row] = '{} {}'.format(df_results[col][row], delta)

		return df_results

	for row in range(len(datasets)):
		baseline = df_results[cols[1]][row]
		
		for i in range(2, len(cols)):
			df_results = format_df_to_latex(baseline, row, df_results, cols[i])

	tex_path = os.path.join(path_for_saving_plots, 'tex', file_name)
	
	df_results.to_latex(tex_path, caption=caption, label=label, escape=False, index=False)


def plot1_B(datasets, indices_experiments, classes_ID, readouts_ML, readouts_SM_reaction,
		 names, label, caption, path_for_saving_plots):
	
	def acc(arr_true, arr_pred):
		return round(matthews_corrcoef(arr_true, arr_pred), 2)

	table_system = {}
	cols = ['Data/Method', 'ML']
	
	for name in names:
		cols.append(name)

	for col in cols:
		table_system.update({col: []})

	for dataset in datasets:
		table_system[cols[0]].append(dataset)

		arr_reaction_SM = readouts_SM_reaction[0]
		arr_reaction_true = readouts_SM_reaction[1]

		# for measuring of correct/incorrect reactions when exposed to ID data
		labels = readouts_ML[1][indices_experiments[dataset][0]]
		y_true_ID = np.where(labels < classes_ID[dataset])[0]
		
		# including baseline measurements
		ML_pred = readouts_ML[0][indices_experiments[dataset][0]]

		table_system[cols[1]].append(acc(labels[y_true_ID], ML_pred[y_true_ID]))
		
		for name, i in zip(names, indices_experiments[dataset]):
			# SM methods
			reaction_true, reaction_pred = arr_reaction_true[i], arr_reaction_SM[i]
			y_true_ID = np.where(reaction_true < classes_ID[dataset])[0]

			reaction_pred_ID, reaction_true_ID = reaction_pred[y_true_ID], reaction_true[y_true_ID]

			table_system[name].append(acc(reaction_true_ID, reaction_pred_ID))

	df_results = pd.DataFrame.from_dict(table_system)

	print_dataframe_to_latex_B(datasets, df_results, caption, label, cols, path_for_saving_plots, '{}.tex'.format(label))