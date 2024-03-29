import os
import pandas as pd
import neptune
from src.Classes.readout import Readout
from src.utils import metrics
from src.utils import util
from src.vis import plot_functions
from src.Classes.dataset import Dataset
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
from keras.models import load_model
import seaborn as sns
from PIL import Image
#from src.threats.novelty_detection.methods import image_dist_matching as idm
from src.threats.novelty_detection.methods import kde_based_monitoring as kde_monitor
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from skimage.feature import hog
from skimage import data, exposure
from skimage.color import rgb2gray
#import plotly
import matplotlib
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from src.vis import util as vis


sns.set()


def plot_images(data, labels, num_row, num_col):

	fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
	for i in range(num_row*num_col):
		try:
			ax = axes[i//num_col, i%num_col]
			ax.imshow(np.squeeze(data[i]), cmap='gray')
			ax.set_title('{}'.format(labels[i]))
			ax.set_axis_off()
		except Exception as e:
   			pass 	
		
	plt.tight_layout(pad=3.0)
	plt.show()


def plot_distribution(data):
	#between 32 and 64 dim is enough for keep more than 90% of explained variance of the images:
	#https://openaccess.thecvf.com/content_CVPR_2019/papers/Gong_On_the_Intrinsic_Dimensionality_of_Image_Representations_CVPR_2019_paper.pdf
	isomap = Isomap(n_components = 32) 
	data = isomap.fit_transform(data)

	tsne = TSNE(n_components=2).fit_transform(data)

	# scale and move the coordinates so they fit [0; 1] range
	def scale_to_01_range(x):
		# compute the distribution range
		value_range = (np.max(x) - np.min(x))

		# move the distribution so that it starts from zero
		# by extracting the minimal value from all its values
		starts_from_zero = x - np.min(x)

		# make the distribution fit [0; 1] by dividing by its range
		return starts_from_zero / value_range

	# extract x and y coordinates representing the positions of the images on T-SNE plot
	tx = tsne[:, 0]
	ty = tsne[:, 1]

	tx = scale_to_01_range(tx)
	ty = scale_to_01_range(ty)

	# initialize a matplotlib plot
	fig = plt.figure()
	ax = fig.add_subplot(111)

	# for every class, we'll add a scatter plot separately
	for label in colors_per_class:
		# find the samples of the current class in the data
		indices = [i for i, l in enumerate(labels) if l == label]

		# extract the coordinates of the points of this class only
		current_tx = np.take(tx, indices)
		current_ty = np.take(ty, indices)

		# convert the class color to matplotlib format
		color = np.array(colors_per_class[label], dtype=np.float) / 255

		# add a scatter plot with the corresponding color and label
		ax.scatter(current_tx, current_ty, c=color, label=label)

	# build a legend using the labels we set previously
	ax.legend(loc='best')

	# finally, show the plot
	plt.show()


def act_func(model, X):
	arrWeights = []

	for img in X:
		img = np.asarray([img])
		arrWeights.append(util.get_activ_func(model, img, layerIndex=-2)[0])

	return arrWeights


def visualize_distributions_2(dataset, dataset_name):
	df_subset = {}

	X, y = dataset
	#indices = np.unique(y, return_index=True)[1]
	indices = np.where(y < 20)
	#print(indices)

	#between 32 and 64 dim is enough for keep more than 90% of explained variance of the images:
	#https://openaccess.thecvf.com/content_CVPR_2019/papers/Gong_On_the_Intrinsic_Dimensionality_of_Image_Representations_CVPR_2019_paper.pdf
	isomap = Isomap(n_components = 32) 
	data = isomap.fit_transform(X)

	#tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
	tsne = TSNE(n_components=2).fit_transform(data)

	df_subset['one'] = tsne[:,0]
	df_subset['two'] = tsne[:,1]
	df_subset['y'] = y[indices]

	ax = sns.scatterplot(
		x="one", y="two",
		hue="y",
		palette=sns.color_palette("hls", 20),
		data=df_subset,
		legend="full"#, alpha=0.3,	ax=ax3
	)
	plt.show()


def visualize_distributions(dataset):
	dataset = Dataset(dataset_name)
	#X, y, _, _ = dataset.load_dataset(mode='train')
	#y = np.argmax(y, axis=1) #if using training data

	X, y = dataset.load_dataset(mode='test')
	num_row = 5
	num_col = 9

	#indices = np.unique(y, return_index=True)[1]
	indices = np.where(y < 20)
	#print(indices)
	'''
	#plot_images(X[indices], y[indices], num_row=num_row, num_col=num_col)


	dataset_name = 'BTSC'#'BTSC', GTSRB
	dataset = Dataset(dataset_name)
	X, y, _, _ = dataset.load_dataset(mode='train')
	y = np.argmax(y, axis=1) #if using training data

	num_row = 7
	num_col = 9

	indices = np.unique(y, return_index=True)[1]
	#print(indices)
	#plot_images(X[indices], y[indices], num_row=num_row, num_col=num_col)
	'''

	#path to load the model
	models_folder = os.join.path("src", "bin", "models")
	model_file = models_folder+'leNet_'+dataset_name+'.h5'

	# loading model
	model = load_model(model_file)
	components = 20
	weights = act_func(model, X[indices])
	pca_50 = PCA(n_components=components)
	pca_result_50 = pca_50.fit_transform(weights)
	print('Cumulative explained variation for {} principal components: {}'.format(components, np.sum(pca_50.explained_variance_ratio_)))

	#ploting distributions
	df_subset = {}

	'''
	#using PCA + TSNE...
	tsne = TSNE(n_components=2, verbose=0, perplexity=40, n_iter=300)
	tsne_pca_results = tsne.fit_transform(pca_result_50)
	df_subset['one'] = tsne_pca_results[:,0]
	df_subset['two'] = tsne_pca_results[:,1]

	# or PCA + Isomap
	isomap = Isomap(n_components = 2)
	isomap_pca_results = isomap.fit_transform(pca_result_50)
	df_subset['one'] = isomap_pca_results[:,0]
	df_subset['two'] = isomap_pca_results[:,1]
	'''
	# or just Isomap
	isomap = Isomap(n_components = 2)
	isomap_results = isomap.fit_transform(weights)
	df_subset['one'] = isomap_results[:,0]
	df_subset['two'] = isomap_results[:,1]


	df_subset['y'] = y[indices]
	ax = sns.scatterplot(
		x="one", y="two",
		hue="y",
		palette=sns.color_palette("hls", 20),
		data=df_subset,
		legend="full"#, alpha=0.3,	ax=ax3
	)
	plt.show()


def apply_KDE(dataset, classes_to_monitor, crop=True, grayscale=True):
	X_croped = []
	X_val_croped = []

	X, y, x_val, y_val = dataset.load_dataset(mode='train')
	y = np.argmax(y, axis=1) #if using training data
	y_val = np.argmax(y_val, axis=1) #if using training data

	#X_test, y_test = dataset.load_dataset(mode='test')

	#indices = np.unique(y, return_index=True)[1]
	indices = np.where(y == classes_to_monitor)
	indices_val = np.where(y_val == classes_to_monitor)

	data_train_reshaped = X[indices].flatten().reshape(X[indices].shape[0], -1)

	if crop:
		#center all images
		for img in X[indices]:
			imc = crop_center(img, 12, 12)
			if grayscale:
				X_croped.append(rgb2gray(imc))
			else:
				X_croped.append(imc)

		X_croped = np.asarray(X_croped)
		data_train_reshaped = X_croped.flatten().reshape(X_croped.shape[0], -1)

	# PCA fit for training
	pca = PCA(n_components=20, whiten=False)
	data_train_reshaped = pca.fit_transform(data_train_reshaped)

	# use grid search cross-validation to optimize the bandwidth
	params = {'bandwidth': np.logspace(-1, 1, 20)}
	grid = GridSearchCV(KernelDensity(), params)
	grid.fit(data_train_reshaped)

	print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))

	# use the best estimator to compute the kernel density estimate
	kde = grid.best_estimator_


	'''
	#main
	x_train = X_croped[0]
	#x_train = X[0]
	#x_train = crop_center(x_train,14, 14)
	y_train = y[indices][0]
	train_data_reshaped = x_train.flatten().reshape(1, -1)
	train_data = pca.transform(train_data_reshaped)
	pdfs2 = np.exp(kde.score(train_data))
	print("Image from the same class", pdfs2, y_train)

	x_val = X_val_croped[10]
	#x_val = x_val[10]
	#x_val = crop_center(x_val, 14, 14)
	y_val = y_val[indices_val][10]
	val_data_reshaped = x_val.flatten().reshape(1, -1)
	val_data = pca.transform(val_data_reshaped)
	pdfs1 = np.exp(kde.score(val_data))
	print("Random image", pdfs1, y_val)

	print(pdfs1<pdfs2)
	'''
	# distributions
	pdfs3 = np.exp(kde.score_samples(data_train_reshaped))
	print("printing pdfs: min ({}) and max ({})".format(np.min(pdfs3), np.max(pdfs3)))
	plot_images(X_croped[:50], np.round(pdfs3[:50], 2), 5, 10)

	data_val_reshaped = x_val.flatten().reshape(x_val.shape[0], -1)

	if crop:
		# same trasformation for validation
		for img in x_val[indices_val]:
			imc = crop_center(img, 12, 12)
			if grayscale:
				X_val_croped.append(rgb2gray(imc))
			else:
				X_val_croped.append(imc)

		X_val_croped = np.asarray(X_val_croped)
		data_val_reshaped = X_val_croped.flatten().reshape(X_val_croped.shape[0], -1)
	
	val_data = pca.transform(data_val_reshaped)
	
	pdfs4 = np.exp(kde.score_samples(val_data))
	print("printing pdfs: min ({}) and max ({})".format(np.min(pdfs4), np.max(pdfs4)))
	result = np.where(pdfs4 < np.amin(pdfs3))
	print("index of outliers (min pdfs) at validation data", result)
	plot_images(X_val_croped[result], np.round(pdfs4[result], 2), 5, 10)
	pdfs5 = np.delete(pdfs4, result)
	print("printing pdfs with no outlier: min ({}) and max ({})".format(np.min(pdfs5), np.max(pdfs5)))


def crop_center(img,cropx,cropy):
	y,x = 28, 28
	startx = x//2-(cropx//2)
	starty = y//2-(cropy//2)	
	return img[starty:starty+cropy,startx:startx+cropx]


def hog_test(image):
	#image = data.astronaut()
	fd, hog_image = hog(image, orientations=8, pixels_per_cell=(4, 4),
						cells_per_block=(1, 1), visualize=True, multichannel=True)

	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

	ax1.axis('off')
	ax1.imshow(image, cmap=plt.cm.gray)
	ax1.set_title('Input image')

	# Rescale histogram for better display
	hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

	print(hog_image)

	ax2.axis('off')
	ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
	ax2.set_title('Histogram of Oriented Gradients')
	plt.show()


def prepare_data(model, X, y):
	arrWeights = []
	arr_lbl = []

	for x, lbl in zip(X, y):
		image = np.asarray([x])
		y_pred = np.argmax(model.predict(image))
		
		if y_pred == lbl:
			weights = util.get_activ_func(model, image, layerIndex=-2)[0]
			arrWeights.append(weights)
			arr_lbl.append(lbl)

	return arrWeights, arr_lbl


if __name__ == '__main__':
	
	dataset_name = 'GTSRB'#'BTSC', GTSRB
	classes_to_monitor = 43

	# Get list of experiments
	#experiments = project.get_experiments(id=['PHD-24', 'PHD-25', 'PHD-26'])
	#names = ['oob', 'oob_isomap', 'oob_pca']

	#experiments = project.get_experiments(id=['PHD-24', 'PHD-30', 'PHD-33'])
	#names = ['oob_3_clusters', 'oob_1_cluster', 'oob_1_cluster V2']

	#experiments = ['PHD-30', 'PHD-33', 'PHD-31', 'PHD-34', 'PHD-32', 'PHD-35', 'PHD-37', 'PHD-36']
	#names = ['oob', 'oob V2', 'oob_isomap', 'oob_isomap V2', 'oob_pca', 'oob_pca V2', 'oob_pca_isomap', 'oob_pca_isomap V2']

	#experiments = ['PHD-38', 'PHD-39', 'PHD-40', 'PHD-41', 'PHD-42', 'PHD-43'] #out of distribution experiments using 30% of BTSC dataset
	#names = ['oob 1 cluster', 'oob 3 clusters', 'oob_isomap 1 cluster', 'oob_isomap 3 clusters', 'oob_pca 1 cluster', 'oob_pca 3 clusters']

	#experiments = ['PHD-45', 'PHD-46', 'PHD-47', 'PHD-48', 'PHD-49', 'PHD-50', 'PHD-51', 'PHD-52'] #out of distribution experiments using 100% of BTSC dataset + 30% GTSRB
	#names = ['oob 1 cluster', 'oob 3 clusters', 'oob_isomap 1 cluster', 'oob_isomap 3 clusters', 'oob_pca 1 cluster', 'oob_pca 3 clusters', 'oob_pca_isomap 1 cluster', 'oob_pca_isomap 3 clusters']

	#experiments = ['PHD-45', 'PHD-54']
	#names = ['oob 1', 'oob 2']

	#experiments = ['PHD-45', 'PHD-47', 'PHD-49',
	# 'PHD-61', 'PHD-62', 'PHD-63'] #out of distribution experiments using 100% of BTSC dataset + 30% GTSRB
	#names = ['oob_1_cluster','oob_isomap_1_cluster', 'oob_pca_1_cluster',
	# 'oob_1_cluster_KDE_v2', 'oob_isomap_1_cluster_KDE_v2', 'oob_pca_1_cluster_KDE_v2']

	#experiments = ['PHD-46', 'PHD-48', 'PHD-50',
	# 'PHD-64', 'PHD-65', 'PHD-66'] #out of distribution experiments using 100% of BTSC dataset + 30% GTSRB
	#names = ['oob_3_cluster','oob_isomap_3_cluster', 'oob_pca_3_cluster',
	# 'oob_3_cluster_KDE_v2', 'oob_isomap_3_cluster_KDE_v2', 'oob_pca_3_cluster_KDE_v2']

	#experiments = ['PHD-69', 'PHD-70', 'PHD-71', 'PHD-72', 'PHD-77', 'PHD-79'] # in distribution experiments using GTSRB dataset
	#names = ['knn_2_cluster','knn_3_cluster', 'knn_5_cluster', 'knn_10_cluster', 'Random Forest', 'SGD']
	#title = 'ID=GTSRB; KNN variants & Random Forest & SGD'

	#experiments = ['PHD-73', 'PHD-74', 'PHD-75', 'PHD-76', 'PHD-78', 'PHD-81'] # out of distribution experiments using 100% of BTSC dataset + 30% GTSRB
	#names = ['knn_2_cluster','knn_3_cluster', 'knn_5_cluster', 'knn_10_cluster', 'Random Forest', 'SGD']
	#title = 'ID=GTSRB; OOD=BTSC; KNN variants & Random Forest & SGD'

	#experiments = ['PHD-82', 'PHD-86'] # in distribution experiments using GTSRB dataset
	#names = ['SGD v2','Random Forest v2']
	#title = 'ID=GTSRB; KNN variants & Random Forest & SGD'

	#experiments = ['PHD-83', 'PHD-87'] # out of distribution experiments using 100% of BTSC dataset + 30% GTSRB
	#names = ['SGD v2','Random Forest v2']
	#title = 'ID=GTSRB; OOD=BTSC; KNN variants & Random Forest & SGD'

	#experiments = ['PHD-79', 'PHD-92', 'PHD-90'] #in distribution 
	#names = ['SGD (scale train/ not scale test)','not scaled SGD','scaled SGD']
	#title = 'ID=GTSRB; SGD'

	#experiments = ['PHD-81', 'PHD-93', 'PHD-91'] # out of distribution
	#names = ['SGD (scale train/ not scale test)','not scaled SGD','scaled SGD']
	#title = 'ID=GTSRB; OOD=BTSC; SGD'

	#experiments = ['PHD-93', 'PHD-95', 'PHD-96'] # out of distribution
	#names = ['SGD antigo','SGD (tester modificado)', 'SGD (BTSC x GTSRB mapeado)']
	#title = 'ID=GTSRB; OOD=BTSC; SGD'

	#visualize_experiments(experiments, names, title, classes_to_monitor)
	
	#project = neptune.init('raulsenaferreira/PhD')
	#experiments = project.get_experiments(['PHD-107', 'PHD-108', 'PHD-109'])
	#arr = vis.load_artifact('Pos_Neg_Labels_ID.npy', experiments)
	#print(arr.shape)

	total_instances = 19725
	root_dir = os.path.join('D:','PHD','datasets')
	dataset = Dataset(dataset_name, root_dir)
	X, y, X_val, y_val = dataset.load_dataset(mode='train')
	print("np.shape(X), np.shape(y)", np.shape(X), np.shape(y))
	#img = X[0]
	#print(np.min(img), np.max(img))
	#print("sparsity on RGB", np.count_nonzero(img))
	
	'''
	grayscale = []
	for img in X: 
		grayscale.append(rgb2gray(img))
	
	print(np.min(grayscale), np.max(grayscale))
	print("sparsity on grayscale", np.count_nonzero(grayscale))

	components = 20
	pca_50 = PCA(n_components=components)
	grayscale = np.asarray(grayscale)
	grayscale = grayscale.flatten().reshape(grayscale.shape[0], -1)

	X = X.flatten().reshape(X.shape[0], -1)

	#pca_result_50 = pca_50.fit_transform(grayscale)
	#print('Cumulative explained variation for {} principal components: {}'.format(components, np.sum(pca_50.explained_variance_ratio_)))
	pca_result_50 = pca_50.fit_transform(X)
	print('Cumulative explained variation for {} principal components: {}'.format(components, np.sum(pca_50.explained_variance_ratio_)))
	pca_result_50 = pca_50.fit_transform(X)
	print('Cumulative explained variation for {} principal components: {}'.format(components, np.sum(pca_50.explained_variance_ratio_)))
	'''


	#hog_test(X[30])
	#apply_KDE(dataset, 36, grayscale=False)
	#apply_KDE(dataset, 36, grayscale=True)

	#arr_pdf_cuttoff = kde_monitor.get_pdf_cuttoff(dataset_name, classes_to_monitor)
	#print(arr_pdf_cuttoff)

	'''
	X, y = dataset.load_dataset(mode='test')

	indices = np.where(y == 1)
	image = np.asarray(X[indices][5])
	indices = np.where(y == 41)
	reference = np.asarray(X[indices][5])
	#print(image.shape, reference.shape)
	#idm.plot_diff_images(np.asarray([image]), np.asarray([reference]))
	#idm.histograms(np.asarray([image]), np.asarray([reference]))
	#sim = idm.compare_histograms(image, reference)

	image2 = crop_center(image,14, 14)
	reference2 = crop_center(reference,14, 14)
	#print(image.shape, reference.shape)
	#idm.template_matching(reference, image2)
	#idm.template_matching(image, reference2)
	#idm.plot_diff_images(image, reference)
	#idm.histograms(np.asarray([image]), np.asarray([reference]))
	#sim = idm.compare_histograms(image, reference)

	#visualize_experiments(experiments, names, 'ID=GTSRB; OOD=BTSC', classes_to_monitor)
	'''

	
	#X, y, _, _ = dataset.load_dataset(mode='train')
	#y = np.argmax(y, axis=1) #if using training data

	X, y = dataset.load_dataset(mode='test')
	#img = np.asarray([X[0]])

	#path to load the model
	models_folder = os.path.join("src", "bin", "models")
	model_file = os.path.join(models_folder, 'leNet_'+dataset_name+'.h5')

	# loading model
	model = load_model(model_file)

	
	'''
	plt.matshow(img[0, :, :, :])
	plt.show()
	plt.matshow(arrWeights[ :, :, np.shape()], cmap='viridis')
	plt.show()
	'''
	'''
	color = ('r','g','b')
	for channel,col in enumerate(color):
		histr = cv2.calcHist([img],[channel],None,[256],[0,256])
		plt.plot(histr,color = col)
		plt.xlim([0,256])
	plt.title('Histogram for color scale picture for class {}'.format(y[0]))
	plt.show()

	cv2.destroyAllWindows()
	'''
	
	
	# video or plot of activation functions OK
	#file = 'results\\video\\novelty_detection\\act_func_animation.mp4'
	#plot_functions.run_act_func_animation(model, dataset_name, X, y, 20, -1, 50, file)

	# video or plot of 2D abstraction boxes
	first_nth_classes, layerIndex, steps = 3, -2, 50
	file = 'results\\video\\novelty_detection\\box_abstractions.mp4'
	technique = 'oob'
	monitor_path = 'src\\threats\\novelty_detection\\bin\\monitors\\dnn_internals\\'+dataset_name
	monitor_folder = os.path.join(monitor_path, technique)
	#dim_reduc_obj = 'trained_'
	#plot_functions.run_boxes_analysis(model, dataset_name, technique, X, y, first_nth_classes, layerIndex, steps, file, monitor_folder, dim_reduc_obj=None)
	#'''
	# video or plot of dim reduction + clf trained on act func values
	train_size = int(len(X)*0.5)
	arrWeights, arr_lbl = prepare_data(model, X[:train_size], y[:train_size])
	dim_reduc_obj = PCA(n_components = 2)
	data = dim_reduc_obj.fit_transform(arrWeights)

	clf = KMeans(n_clusters = 3).fit(data, arr_lbl)
	file = os.path.join('results', 'video', 'novelty_detection', 'pca_kmeans_actFunc.mp4')

	plot_functions.plot_single_clf_pca_actFunc_based_analysis(model, dataset_name, clf, X[train_size:],\
	 y[train_size:], first_nth_classes, layerIndex, steps, file, dim_reduc_obj)
	#'''
	'''
	df = pd.DataFrame({'name': ['Raphael', 'Donatello'],
                   'mask': ['red', 'purple'],
                   'weapon': ['sai', 'bo staff']})
	print(df.to_latex(index=False))
	'''


	