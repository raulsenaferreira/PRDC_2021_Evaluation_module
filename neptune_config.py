import neptune

def neptune_init(threat):
	neptune_root = 'raulsenaferreira/'
	log = 'loading experiments from '

	path = neptune_root+'{}'.format(threat)
	print(log+path)

	return neptune.init(path)