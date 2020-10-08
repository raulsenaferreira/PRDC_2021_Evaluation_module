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


def plot(arr_id, names, title, path_for_saving_plots):

	table_ML = {'Architecture': [], 'Accuracy': [], 'F1': []}
	table_SM = {'Method': [], 'Detection': [], 'Confidence': [], 'Memory': [], 'Time:' []}
	table_system = {'Combination': [], 'MCC': [], 'F1': [], 'TP': [], 'FP': [], 'TN': [], 'FN': []}
	table_novelty = {'TPR_ID_FPR_OOD': [], 'AUPR_ID': [], 'AUPR_OOD': []}

	project = npte.neptune_init('novelty-detection')
	experiments = project.get_experiments(arr_id)
	
	arr_pnc_id = util.load_artifact('Pos_Neg_Classified_ID.npy', experiments)
	arr_pnl_id = util.load_artifact('Pos_Neg_Labels_ID.npy', experiments)
	arr_pnc_ood = util.load_artifact('Pos_Neg_Classified_OOD.npy', experiments)
	arr_pnl_ood = util.load_artifact('Pos_Neg_Labels_OOD.npy', experiments)
	#print(arr_pnc_id)
	#print(np.arr_pnc_ood)
	
	arr_cm = []
	
	# total = ID + ODD
	for pnl_id, pnl_ood, pnc_id, pnc_ood in zip(arr_pnl_id, arr_pnl_ood, arr_pnc_id, arr_pnc_ood):
		y_true = np.hstack([pnl_id, pnl_ood])
		y_pred = np.hstack([pnc_id, pnc_ood])
		
		cm_total = confusion_matrix(y_true, y_pred)
		#print(cm_total)

		#df = pd.DataFrame(cm_total)
		#print(df.da.export_metrics(metrics_to_include=['accuracy', 'precision', 'recall', 'f1']))
		#print(matthews_corrcoef(y_true, y_pred))
		print(classification_report(y_true, y_pred))
		cr = classification_report(y_true, y_pred, output_dict=True)

		table = {'Accuracy': [cr['accuracy']],
			'MCC': [matthews_corrcoef(y_true, y_pred)]}
		print(pd.DataFrame.from_dict(table))
		#df = pd.DataFrame(cr)
		#df['MCC'] = matthews_corrcoef(y_true, y_pred)
		#print(df)
		#df.plot.bar(stacked=True)
		#plt.show()
		# access metrics for each class by index
		#print(df.da.sensitivity)

		#arr_cm.append(cm_total)
	
	#pos_neg_stacked_bars(title, names, arr_readouts, path_for_saving_plots)