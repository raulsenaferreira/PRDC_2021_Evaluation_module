import neptune_config as npte 
from src.plot_functions import visualize_experiments
from src import util


def plot(arr_id):
	project = npte.neptune_init('novelty-detection')
	experiments = project.get_experiments(arr_id)
	arr = util.load_artifact('Pos_Neg_Labels_ID.npy', experiments)
	print(arr.shape)