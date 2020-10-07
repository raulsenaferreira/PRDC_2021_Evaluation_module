import neptune

def neptune_init(threat):
	neptune_root = 'raulsenaferreira/'
	log = 'loading experiments from '

	if threat != 'novelty_detection' and threat != 'adversarial_attack' and threat != 'distributional_shift' and threat != 'anomaly_detection' and threat != 'noise':
		threat='PhD'

	path = neptune_root+'{}'.format(threat)
	print(log+path)

	return neptune.init(path)