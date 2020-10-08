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

	ind = df_results['TP'].idxmax()
	df_results['TP'] = df_results['TP'].astype(str)
	df_results['TP'].at[ind] = '\textbf{'+str(df_results['TP'][ind])+'}'

	ind = df_results['FP'].idxmax()
	df_results['FP'] = df_results['FP'].astype(str)
	df_results['FP'].at[ind] = '\textbf{'+str(df_results['FP'][ind])+'}'

	ind = df_results['TN'].idxmax()
	df_results['TN'] = df_results['TN'].astype(str)
	df_results['TN'].at[ind] = '\textbf{'+str(df_results['TN'][ind])+'}'

	ind = df_results['FN'].idxmax()
	df_results['FN'] = df_results['FN'].astype(str)
	df_results['FN'].at[ind] = '\textbf{'+str(df_results['FN'][ind])+'}'

	ind = df_results['F1_Macro'].idxmax()
	df_results['F1_Macro'] = df_results['F1_Macro'].astype(str)
	df_results['F1_Macro'].at[ind] = '\textbf{'+str(df_results['F1_Macro'][ind])+'}'

	tex_path = os.path.join(path_for_saving_plots, 'tex', file_name)
	
	df_results.to_latex(tex_path, caption=caption, label=label, escape=False, index=False)


def plot(arr_id, names, title, path_for_saving_plots):

	table_ML = {'Architecture': [], 'Accuracy': [], 'F1': [], 'F1-Micro': []}
	table_SM = {'Method': [], 'Detection': [], 'Confidence': [], 'Memory': [], 'Time': []}
	table_system = dict(Experiment= [], MCC= [], TP= [], FP= [], TN= [], FN= [], F1_Macro= []) #'F1_0': [], 'F1_1': [], 
	table_novelty = {'TPR_ID_FPR_OOD': [], 'AUPR_ID': [], 'AUPR_OOD': []}

	project = npte.neptune_init('novelty-detection')
	experiments = project.get_experiments(arr_id)
	
	arr_pnc_id = util.load_artifact('Pos_Neg_Classified_ID.npy', experiments)
	arr_pnl_id = util.load_artifact('Pos_Neg_Labels_ID.npy', experiments)
	arr_pnc_ood = util.load_artifact('Pos_Neg_Classified_OOD.npy', experiments)
	arr_pnl_ood = util.load_artifact('Pos_Neg_Labels_OOD.npy', experiments)
	
	arr_cm = [[], [], [], []]
	
	# total = ID + ODD
	for name, pnl_id, pnl_ood, pnc_id, pnc_ood in zip(names, arr_pnl_id, arr_pnl_ood, arr_pnc_id, arr_pnc_ood):
		
		y_true = np.hstack([pnl_id, pnl_ood])
		y_pred = np.hstack([pnc_id, pnc_ood])
		
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

		table_system['Experiment'].append(name)
		table_system['MCC'].append(round(matthews_corrcoef(y_true, y_pred), 2))
		table_system['TP'].append(round(df.da.true_positive_rate.values[0], 2))
		table_system['FP'].append(round(df.da.false_positive_rate.values[0], 2))
		table_system['TN'].append(round(df.da.true_negative_rate.values[0], 2))
		table_system['FN'].append(round(df.da.false_negative_rate.values[0], 2))
		table_system['F1_Macro'].append(round(cr['macro avg']['f1-score'], 2))
	
	df_results = pd.DataFrame.from_dict(table_system)

	caption = 'Comparing outside-of-the-box-based monitors'
	label = 'table_1'
	print_dataframe_to_latex(df_results, caption, label, path_for_saving_plots, '{}.tex'.format(label))
	
	#fig = plt()
	#df.plot.bar(stacked=True)
	#plt.show()
	file_name = 'fig_1'
	fig_path = os.path.join(path_for_saving_plots, 'img', '{}.pdf'.format(file_name))
	pos_neg_stacked_bars(title, names, arr_cm, fig_path)