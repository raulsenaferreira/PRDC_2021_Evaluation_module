import os
import numpy as np
import keras
from keras.models import Model
import torch


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
		arr = np.load(file_path, allow_pickle=True)

		os.remove(file_path)

		result_arr.append(arr)

	return result_arr


def load_artifact_2(artifact_name, neptune_experiments):
	result_arr = []
	tmp_path = os.path.join('aux_data', 'temp')
	os.makedirs(tmp_path, exist_ok=True)

	for experiment in neptune_experiments:
		experiment.download_artifact(artifact_name, tmp_path)
		file_path = os.path.join(tmp_path, artifact_name)

		data = []
		while True:
			packet = s.recv(4096)
			if not packet: break
			data.append(packet)
		data_arr = pickle.loads(b"".join(data))
		print (data_arr)
		s.close()


def load_time_info(neptune_experiments, instances, indices_experiments, path_for_saving_plots, dataset, names):
	j = 0
	arr_time = []
	path_to_save = os.path.join(path_for_saving_plots, 'time_results.txt')

	with open(path_to_save, 'a') as file:
		file.write("\nDataset {}:".format(dataset))
		for i in indices_experiments:
			time = neptune_experiments[i].get_numeric_channels_values('ml_time', 'sm_time', 'total_time', 'total_memory')
			
			ml_time = time['ml_time'].values[0]/instances[i]
			sm_time = time['sm_time'].values[0]/instances[i]
			total_time = time['total_time'].values[0]/instances[i]
			ml_time_perc = ml_time / total_time * 100
			sm_time_perc = sm_time / total_time * 100

			file.write("\nTime values per instance for {}:".format(names[j]))
			j+=1
			file.write('\nML time: {} ({}%)'.format(round(ml_time, 5), round(ml_time_perc, 2)))
			file.write('\nSM time: {} ({}%)'.format(round(sm_time, 5), round(sm_time_perc, 2)))
			file.write('\nTotal time: {}\n'.format(round(total_time, 5)))
			#print('ML time per instance: {}'.format(time[i]['total_memory']/instances[i]))
			
			arr_time.append(time)

	return arr_time


def save_results(experiment, arr_readouts, plot=False):
	print("saving experiments", experiment.name)
	filenames = config_ND.load_file_names()
	csvs_folder_path = os.path.join('src', 'tests', 'results', 'csv', experiment.sub_field, experiment.name)
	img_folder_path = os.path.join('src', 'tests', 'results', 'img', experiment.sub_field, experiment.name)

	metrics.save_results(arr_readouts, csvs_folder_path, filenames, ',')
	
	if plot:
		os.makedirs(img_folder_path, exist_ok=True)
		metrics.plot_pos_neg_rate_stacked_bars(experiment.name, arr_readouts, img_folder_path+'all_images.pdf')