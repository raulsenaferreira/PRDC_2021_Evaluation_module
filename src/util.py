import os
import numpy as np
import keras
from keras.models import Model


def get_activ_func(model, image, layerIndex):
	inter_output_model = Model(inputs = model.input, outputs = model.get_layer(index=layerIndex).output) #last layer: index 7 or name 'dense'
	return inter_output_model.predict(image)


def act_func(model, X):
	arrWeights = []

	for img in X:
		img = np.asarray([img])
		arrWeights.append(get_activ_func(model, img, layerIndex=-2)[0])

	return arrWeights


def load_artifact(artifact_name, neptune_experiments):	
	result_arr = []
	tmp_path = os.path.join('aux_data', 'temp')
	os.makedirs(tmp_path, exist_ok=True)

	for experiment in neptune_experiments:
		experiment.download_artifact(artifact_name, tmp_path)
		file_path = os.path.join(tmp_path, artifact_name)
		arr = np.load(file_path)
		os.remove(file_path)

		result_arr.append(arr)

	return result_arr


def save_results(experiment, arr_readouts, plot=False):
	print("saving experiments", experiment.name)
	filenames = config_ND.load_file_names()
	csvs_folder_path = os.path.join('src', 'tests', 'results', 'csv', experiment.sub_field, experiment.name)
	img_folder_path = os.path.join('src', 'tests', 'results', 'img', experiment.sub_field, experiment.name)

	metrics.save_results(arr_readouts, csvs_folder_path, filenames, ',')
	
	if plot:
		os.makedirs(img_folder_path, exist_ok=True)
		metrics.plot_pos_neg_rate_stacked_bars(experiment.name, arr_readouts, img_folder_path+'all_images.pdf')