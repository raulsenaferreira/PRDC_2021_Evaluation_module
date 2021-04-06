import os
import neptune_config as npte 
from src.plot_functions import pos_neg_stacked_bars
from src import util
from src.Classes.readout import Readout
from sklearn.metrics import confusion_matrix
import numpy as np
import disarray
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt


def print_dataframe_to_latex(df_results, caption, label, path_for_saving_plots, file_name):
	# it should be a for instead
	ind = df_results['MCC'].idxmax()
	df_results['MCC'] = df_results['MCC'].astype(str)
	df_results['MCC'].at[ind] = '\textbf{'+str(df_results['MCC'][ind])+'}'

	ind = df_results['FPR'].idxmin()
	df_results['FPR'] = df_results['FPR'].astype(str)
	df_results['FPR'].at[ind] = '\textbf{'+str(df_results['FPR'][ind])+'}'

	ind = df_results['FNR'].idxmin()
	df_results['FNR'] = df_results['FNR'].astype(str)
	df_results['FNR'].at[ind] = '\textbf{'+str(df_results['FNR'][ind])+'}'

	ind = df_results['Precision'].idxmax()
	df_results['Precision'] = df_results['Precision'].astype(str)
	df_results['Precision'].at[ind] = '\textbf{'+str(df_results['Precision'][ind])+'}'

	ind = df_results['Recall'].idxmax()
	df_results['Recall'] = df_results['Recall'].astype(str)
	df_results['Recall'].at[ind] = '\textbf{'+str(df_results['Recall'][ind])+'}'

	ind = df_results['Micro-F1'].idxmax()
	df_results['Micro-F1'] = df_results['Micro-F1'].astype(str)
	df_results['Micro-F1'].at[ind] = '\textbf{'+str(df_results['Micro-F1'][ind])+'}'

	tex_path = os.path.join(path_for_saving_plots, 'tex', file_name)
	
	df_results.to_latex(tex_path, caption=caption, label=label, escape=False, index=False)


def plot1(indices_experiments, readouts_SM_detection,
		 names, label, caption, path_for_saving_plots):

	table_system = {'Method': [], 'MCC': [], 'FPR': [], 'FNR': [], 'Precision': [], 'Recall': [], 'Micro-F1': []} #dict(Method= [], MCC= [], TP= [], FP= [], TN= [], FN= [], 'Micro-F1': [])
	
	arr_detection_SM = readouts_SM_detection[0]
	arr_detection_true = readouts_SM_detection[1]
	
	arr_cm = [[], [], [], []]
	
	for name, i in zip(names, indices_experiments):
		y_true = arr_detection_true[i]
		y_pred = arr_detection_SM[i]
		#print('y_true', y_true)
		#print('y_pred', y_pred)

		cm_total = confusion_matrix(y_true, y_pred)
		tn = cm_total[0][0]
		fp = cm_total[0][1]
		fn = cm_total[1][0]
		tp = cm_total[1][1]
		arr_cm[0].append(tn)
		arr_cm[1].append(fp)
		arr_cm[2].append(fn)
		arr_cm[3].append(tp)
		
		cr = classification_report(y_true, y_pred, output_dict=True)
		#print(cr)

		df = pd.DataFrame(cm_total)

		table_system['Method'].append(name)
		table_system['MCC'].append(round(matthews_corrcoef(y_true, y_pred), 2))
		table_system['FPR'].append(round(df.da.false_positive_rate.values[0], 2))
		table_system['FNR'].append(round(df.da.false_negative_rate.values[0], 2))
		table_system['Precision'].append(round(df.da.precision.values[0], 2))
		table_system['Recall'].append(round(df.da.recall.values[0], 2))
		table_system['Micro-F1'].append(round(cr['weighted avg']['f1-score'], 2))
	
	df_results = pd.DataFrame.from_dict(table_system)

	print_dataframe_to_latex(df_results, caption, label, path_for_saving_plots, '{}.tex'.format(label))


def plot2(arr_id, path_for_saving_plots):
	
	def plotBoxplot(data, labels):
		fig, ax = plt.subplots()
		#fig.add_subplot(111)
		ax.boxplot(data, labels=labels)
		plt.xticks(rotation=0)

		plt.title('Mathews Correlation Coefficient Boxplot.')
		plt.ylabel("Mathews Correlation Coefficient")
		
		plt.show()

	data, labels = [], []
	project = npte.neptune_init('novelty-detection')

	for name, exp_id in arr_id.items():
		mccs = []
		labels.append(name)
		experiments = project.get_experiments(exp_id)
		
		arr_detection_SM = util.load_artifact('arr_detection_SM.npy', experiments)
		arr_detection_true = util.load_artifact('arr_detection_true.npy', experiments)
		
		arr_cm = [[], [], [], []]
		
		# total = ID + ODD
		for y_true, y_pred in zip(arr_detection_true, arr_detection_SM):
			
			mccs.append(round(matthews_corrcoef(y_true, y_pred), 2))

		data.append(mccs)

	plotBoxplot(data, labels)


def plot3():
	table_system = {'Method': [], 'MCC': [], 'FPR': [], 'FNR': [], 'Precision': [], 'Recall': [], 'Micro-F1': []} #dict(Method= [], MCC= [], TP= [], FP= [], TN= [], FN= [], 'Micro-F1': [])
	
	project = npte.neptune_init(path_for_load_neptune)
	experiments = project.get_experiments(arr_id)
	
	arr_detection_SM = util.load_artifact('arr_detection_SM.npy', experiments)
	arr_detection_true = util.load_artifact('arr_detection_true.npy', experiments)
	
	arr_cm = [[], [], [], []]
	
	for name, y_true, y_pred in zip(names, arr_detection_true, arr_detection_SM):
		
		cm_total = confusion_matrix(y_true, y_pred)
		tn = cm_total[0][0]
		fp = cm_total[0][1]
		fn = cm_total[1][0]
		tp = cm_total[1][1]
		arr_cm[0].append(tn)
		arr_cm[1].append(fp)
		arr_cm[2].append(fn)
		arr_cm[3].append(tp)
		
		cr = classification_report(y_true, y_pred, output_dict=True)
		#print(cr)

		df = pd.DataFrame(cm_total)

		table_system['Method'].append(name)
		table_system['MCC'].append(round(matthews_corrcoef(y_true, y_pred), 2))
		table_system['FPR'].append(round(df.da.false_positive_rate.values[0], 2))
		table_system['FNR'].append(round(df.da.false_negative_rate.values[0], 2))
		table_system['Precision'].append(round(df.da.precision.values[0], 2))
		table_system['Recall'].append(round(df.da.recall.values[0], 2))
		table_system['Micro-F1'].append(round(cr['weighted avg']['f1-score'], 2))
	
	df_results = pd.DataFrame.from_dict(table_system)

	print_dataframe_to_latex(df_results, caption, label, path_for_saving_plots, '{}.tex'.format(label))