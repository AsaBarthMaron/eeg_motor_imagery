'''Filter Bank Common Spatial Pattern (FBCSP) for predicting motor imagery 
from EEG data. 

References:
TODO
'''

import matplotlib.pyplot as plt
import numpy as np
from scipy import io, signal, linalg
from sklearn.metrics import cohen_kappa_score

class FBCSP(object):
	'''Filter Bank Common Spatial Patern (FBCSP) analysis class'''
	def __init__(self, fpath):
		mat_file = io.loadmat(fpath)

		self.data = mat_file['o'][0][0]['data']
		self.marker = np.array(mat_file['o'][0][0]['marker'], dtype='int8')
		self.nS = mat_file['o'][0][0]['nS']
		self.sampFreq = int(mat_file['o'][0][0]['sampFreq'])
		self.tag = mat_file['o'][0][0]['tag']
		# Inherent to experimental structure, trials are always 1 second
		self.trial_len = 1 * self.sampFreq

		# .mat file import format problems, probably there is a better way of handling this
		self.chnames = []
		for name in mat_file['o'][0][0]['chnames']:
			self.chnames.append(name[0][0])

		# Remove ground ('A1', 'A2') and data sync ('X5') channels
		# TODO: is 'X5' sometimes 'X3'?
		for ch in ['A1', 'A2', 'X5']:
			loc = self.chnames.index(ch)
			self.data = np.delete(self.data, loc, axis=1)
			self.chnames.remove(ch)

		self.n_ch = len(self.chnames)

	def filter(self):
		'''Create standard filter bank array of:
		4-8, 8-12, 12-16, 16-20, 20-24, 24-28, 28-32, 32-36, 36-40
		'''

		self.n_filters = 9		# Standard FBCSP filter bank

		# This is the largest feature expansion and uses almost
		# 1GB memory for a 50-55 min session. It would be possible to do 
		# each filter sequentially to minmize memeory load.
		self.X = np.zeros((self.data.shape[0], self.n_ch, self.n_filters))

		# Design filter
		N = 6				# Filter order, from signal.freqz
		rs = 40				# Min. stopband attenuation

		# Passband frequencies (Hz) for filter bank. Am using additional 
		# 1 Hz on either side from published.
		self.wp = np.array([[3, 9] + m for m in np.arange(0,33,4)])

		# Loop over filters and channels and filter data.
		# Published filter is zero phase, but I am just using filtfilt.
		for b in np.arange(self.n_filters):
			# Get passband freqs for this filter
			wp = self.wp[b, ] / (self.sampFreq / 2)
			# Construct filter
			sos = signal.cheby2(N, rs, Wn=wp, btype='bandpass', \
								analog=False, output='sos')
			# Loop over channels, filter
			for ch in np.arange(self.n_ch):
				self.X[0:, ch, b] = signal.sosfiltfilt(sos, self.data[0:, ch])

	def chunk_trials(self):
		'''Break 50-55 min session traces into individual trial epochs'''

		# Find trial start indices. 
		trial_start_inds = np.where(np.diff(self.marker, axis = 0) >= 1)[0] + 1
		# Throw error if # of trial starts and ends are different
		if trial_start_inds.shape != (np.where(np.diff(self.marker, axis = 0) <= -1)[0] + 1).shape:
			raise ValueError('Inconsistent # of trial starts and ends')

		# Get number of training examples (trials)
		self.N = trial_start_inds.shape[0]
		# Allocate X and Y arrays
		X = np.zeros([self.trial_len, self.n_ch, self.N, self.n_filters])
		Y = np.zeros(self.N)
		# Loop over trial start indices and chunk
		for i in np.arange(self.N):
			# Trial start index
			t_i = trial_start_inds[i]
			# Grab epoch / chunk, self.X is filtered unchunked data
			X[0:, 0:, i, 0:] = self.X[t_i:t_i+self.trial_len, 0:, 0:]
			# Grab Y value
			Y[i] = self.marker[t_i]

		# Flip first two X dimensions so X is c x t x N x b  
		# c: n_ch (# channels)
		# t: trial_len (trial length, in samples)
		# N: N (number of training samples/trials)
		# b : n_filters (number of filters)
		self.X = np.swapaxes(X, 0, 1)
		self.Y = Y

	def CSP(self, class_marker):
		'''Perform Common Spatial Pattern (CSP) dimensionality reduction / 
	 	feature extraction. Multi-class extension using OVR. 
	 	TODO: references
	 	class_marker: integer 1-6 indicating y value.
	 	'''

	 	# Calculate trial covariance matrices 
		trial_cov = np.zeros([self.n_ch, self.n_ch, self.N, self.n_filters])
		for b in np.arange(self.n_filters):
			for n in np.arange(self.N):
				trial_cov[0:, 0:, n, b] = np.cov(self.X[0:, 0:, n, b])

		# Allocate arrays for class conditional covariance matrices
		class_cov = np.zeros([self.n_ch, self.n_ch, self.n_filters])
		rest_cov = np.zeros([self.n_ch, self.n_ch, self.n_filters])
		# Average over trials for 'class_marker' and rest 
		class_cov = trial_cov[0:, 0:, self.Y == class_marker, 0:].mean(axis=2)
		rest_cov = trial_cov[0:, 0:, self.Y != class_marker, 0:].mean(axis=2)

		# Solve generalized eigenvalue problem for each Wb
		# Wb: CSP projection matrix for bth band-pass filtered EEG signal
		W = np.zeros([self.n_ch, self.n_ch, self.n_filters])
		l = np.zeros([self.n_ch, self.n_filters]) 	# Eigenvalues
		for b in np.arange(self.n_filters):
			l[0:, b], W[0:, 0:, b] = linalg.eig(class_cov[0:, 0:, b], \
												class_cov[0:, 0:, b] + \
												rest_cov[0:, 0:, b])
			# Sort Wb and lb by eigenvalue, descending
			reorder = np.flip(np.argsort(l[0:, b]))
			l[0:, b] = l[reorder, b]
			W[0:, 0:, b] = W[0:, reorder, b]

		# Select top and bottom 4 eigenvectors (TODO: refernece)
		self.m = 4
		self.Wm = np.concatenate((W[0:, :self.m, 0:], \
							      W[0:, (self.n_ch - self.m):, 0:]), axis=1)

	def CSP_xform(self):
		'''Transform filtered eeg data (X) into CSP coordinates using
		projection matrix Wm. Then extract variance ratio features.
		'''

		# Allocate array for variance ratio features, V
		self.V = np.zeros([self.m * 2, self.N, self.n_filters])
		# Transform X data for each bth band-pass filtered EEG signal
		X_csp = np.zeros([self.m * 2, self.trial_len, self.N, self.n_filters])
		for b in np.arange(self.n_filters):
			# TODO: double check this
			X_csp[0:, 0:, 0:, b] = np.dot(fb.X[0:, 0:, 0:, b].transpose(),\
										  fb.Wm[0:, 0:, b]).transpose()

			# Extract variance ratio features for each trial
			for t in np.arange(self.N):
				x_csp_cov = np.dot(X_csp[0:, 0:, t, b], \
								   X_csp[0:, 0:, t, b].transpose())
				self.V[0:, t, b] = np.log(np.diag(x_csp_cov) / \
										  np.trace(x_csp_cov))

		# V in Ang et al., 2012 is N x (2*m*9)
		self.V = self.V.transpose([1, 0, 2])\
					   .reshape([self.N, 2*self.m*self.n_filters], order='F')

	def MIBIF(self, class_marker):
		'''Mutual Information-based Best Individual Feature (MIBIF) for
		feature selection. 
		Step 1 - Initalize set of features (F)
		Step 2 - Calculate mutual information of each feature f_j with 
				 each class label.
		Step 3 - Select first k features

		TODO: refs. 
		TODO: Original FBCSP papers also
		used Mutual Information-based Rough Set Reduction (MIRSR), but 
		Chin et al., 2009 multi-class FBCSP extension only uses MIBIF. 
		Might consider trying MIRSR later.
		TODO: Confirm that OVR is also the right approach here.
		'''

		# Set up featue vector F
		F = self.V

		# Set one-vs-rest class labels
		Y_ovr = np.zeros(self.Y.shape)
		Y_ovr[self.Y != class_marker] = 0
		Y_ovr[self.Y == class_marker] = 1

		# Step 2 - compute mutual information of each feature with class
		# I(f_j; w) = H(w) - H(w|f_j)
		# f_j: jth feature of F
		# w: class label

		# Calculate class entropy
		_, p_y = np.unique(Y_ovr, return_counts=True)
		p_y = p_y / Y_ovr.shape[0]
		H_y = -(p_y * np.log2(p_y)).sum()

		# Calculate class-conditional feature probabiltiy using kernel
		# density estimation / parzen window.

		# Allocate conditional probability array p(f_ji|y)
		n_features = F.shape[1]
		p_fji_y = np.zeros([n_features, self.N, 2])
		# Allocate marginal probability array p(f_ji)
		p_fji = np.zeros([n_features, self.N])
		# Allocate posterior likelihood array p(y|f_ji)
		p_y_fji = np.zeros([2, n_features, self.N])
		# Loop over features
		for j in np.arange(n_features):
			# Loop over classes
			for y in [0, 1]:
				# Subset of features belonging to given class y
				f_y = F[Y_ovr == y, j]
				# Select kernel bandwidth for conditional p(f_j|y)
				h = bandwidth_selection(f_y)
				# Calculate class conditional probability p(f_ji|y)
				for i in np.arange(self.N):
					f = F[i, j]
					p_fji_y[j, i, y] = kernel_density_estimate(f, f_y, h)
			# Calculate marginal probability p(f_ji), law of total probability
			p_fji[j, 0:] = np.sum(p_fji_y[j, 0:, 0:] * p_y, axis=1)

			# Calculate posterior p(y|f_ji), bayes rule
			p_y_fji[0:, j, 0:] = (p_fji_y[j, 0:, 0:] * p_y).transpose()\
								/ p_fji[j, 0:]
		
		# Calculate feature-conditional class entropy
		# TODO: double check that the 1/N scaling factor is appropriate
		# it was not included in Ang et al., 2012 and that threw me 
		# through a loop for a while.
		H_y_fj = -(p_y_fji * np.log2(p_y_fji)).sum(axis=2).sum(axis=0) \
				 / self.N

		# Calculate (feature) mutual information
		I_j = H_y - H_y_fj

		# Create feature label array for matching CSP feature pairs
		feature_labels = np.zeros([(self.m*2), self.n_filters])
		pattern = np.concatenate([np.arange(self.m), np.arange(self.m-1, -1, -1)])
		for i in np.arange(0, n_features/2, self.m):
			b = int(i/self.m)
			feature_labels[0:, b] = pattern + i
		feature_labels = feature_labels.reshape([self.m*2*self.n_filters], order='F')

		# Sort features by information, get feature labels
		d = 4
		reorder = np.flip(np.argsort(I_j))
		selected_labels = np.unique(feature_labels[reorder[:d]])
		selected_features = np.zeros([self.N, selected_labels.shape[0]*2])
		for j in np.arange(0, selected_features.shape[1], 2):
			l_ind = int(j/2) 
			selected_features[0:, j:j+2] = \
				F[0:, np.where(feature_labels == selected_labels[l_ind])[0]]
		
		self.X_latent = selected_features

def bandwidth_selection(y):
	'''Return smoothing / bandwidth param h, given observations of y'''
	h = ((4 / (3 * np.shape(y)[0])) ** (1/5)) * np.sqrt(np.var(y))
	return h

def gaussian_kernel(y, h):
	'''Return Gaussian smoothed density for y, smoothing param h'''
	K_yh = np.exp(-(y**2) / (2 * (h**2))) / np.sqrt(2 * np.pi)
	return K_yh


def kernel_density_estimate(x, x_train, h):
	'''Return Gaussian kernel density estimate for x, using vector of 
	x_train and smoothing / bandwidth parameter h.
	'''
	# TODO: double check that scaling by h is being done properly
	p = gaussian_kernel(x - x_train, h).sum() / (x_train.shape[0] * h)
	return p
	
def naive_bayes_kde(x_test, x_train, y_train):
	'''Naive bayes classifier using kernel density estimation / parzen
	window for p(x|y). This method is a lazy learner.
	'''

	N, D = x_test.shape
	# Calculate prior (class) probability
	C, p_y = np.unique(y_train, return_counts=True)
	p_y = p_y / y_train.shape[0]

	# Calculate class-conditional feature probability p(x_j|y) using 
	# kernel density estimation / parzen window.
	p_xj_y = np.zeros([x_test.shape[0], D, C.shape[0]])
	# Loop over features
	for j in np.arange(D):
		# Loop over classes
		for y in C.astype(int):
			# Subset of features belonging to given class y
			x_y = x_train[y_train == y, j]
			# Select kernel bandwidth for conditional p(f_j|y)
			h = bandwidth_selection(x_y)
			# Calculate class conditional probability p(f_ji|y)
			for i in np.arange(N):
				p_xj_y[i, j, y] = kernel_density_estimate(x_test[i, j],\
														  x_y, h)

	# Calculate class-conditional probability p(x|w), using naive 
	# assumption of (feature) independence.
	p_x_y = np.prod(p_xj_y, axis=1)

	# Calculate marginal p(x), law of total probability
	p_x = np.sum(p_x_y * p_y, axis=1)

	# Calculate posterior predictive p(w|x), bayes rule
	p_y_x = ((p_x_y * p_y).transpose() / p_x).transpose()

	return p_y_x

def cv_select(N, cv=5):
	'''Select indices for cross validation folds'''

	# Set fold size
	fold_size = int(np.floor(N / cv))

	# Set fold labels
	fold_labels = np.ones((fold_size, cv))
	fold_labels = fold_labels * np.arange(1, cv+1)
	fold_labels = fold_labels.reshape(fold_size * cv)

	# Make sure samples aren't left out if N is not divisible by cv
	n_remaining = N - fold_labels.shape[0]
	remain_labels = np.random.choice(np.arange(1, cv+1), n_remaining)
	fold_labels = np.concatenate([fold_labels, remain_labels])

	# Shuffle labels
	fold_labels = np.random.permutation(fold_labels)
	return fold_labels

def accuracy(y_hat, y_test):
	'''Compute classification accuracy'''

	accuracy = np.sum(y_hat == y_test) / y_hat.shape[0]
	return accuracy

if __name__ == "__main__":
	# Set data file path
	#fpath = '/Volumes/SSD_DATA/kaya_mishchenko_eeg/CLASubjectA1601083StLRHand.mat'
	fpath = '/Volumes/SSD_DATA/kaya_mishchenko_eeg/CLASubjectC1512163StLRHand.mat'

	np.random.seed(0)
	# Initialize FBCSP
	fb = FBCSP(fpath)
	# Filter data, this is done before chunking into trial /sample 
	# structure to avoid creating unnecessary boundary conditions.
	fb.filter()
	# Chunk sessions (~50-55min) into 1s trials (N samples)
	fb.chunk_trials()

	# Allocate samples for cross-validation
	n_folds = 10
	fold_labels = cv_select(fb.N, cv=n_folds)
	# Get set of class labels
	class_set = np.unique(fb.Y)
	# Get fold size. This will be problematic if N is not divisible by 
	# n_folds. TODO: fix this.
	test_size = int(np.floor(fb.N / n_folds))

	# Allocate array for predicted class probabilities p(y|x)
	# axis=1: N
	# axis=2: 2 - # OVR classes
	# axis=3: # of total classes
	p_y_x = np.zeros([test_size, 2, class_set.shape[0]])
	
	# Allocate arrays for accuracy metrics
	a = np.zeros(n_folds)
	kappa = np.zeros(n_folds)

	# Loop over folds
	for fold in np.arange(1, n_folds+1):
		# Split y test data into training and test
		y_test = fb.Y[fold_labels == fold]
		# Loop over classes, this is where the magic of one-versus-rest
		# (OVR) happens. 
		for w in class_set.astype(int):
			# Calculate CSP transformation matrix
			fb.CSP(w)
			# Transform X data into CSP coordinates, and calculate 
			# variance ratio features.
			fb.CSP_xform()
			# Select most informative CSP variance features ('X_latent')
			fb.MIBIF(w)

			# Set OVR sample-class labels 
			Y_ovr = np.zeros(fb.Y.shape)
			Y_ovr[fb.Y != w] = 0
			Y_ovr[fb.Y == w] = 1
			# Split x data into training and test
			x_test = fb.X_latent[fold_labels == fold, 0:]
			x_train = fb.X_latent[fold_labels != fold, 0:]
			y_train = Y_ovr[fold_labels != fold]
			# Classify, returns OVR probabilities for each 'one' in OVR
			p_y_x[0:, 0:, w-1] = naive_bayes_kde(x_test, x_train, y_train)

		# Select class label with highest OVR probability
		y_hat = p_y_x[0:,1,0:].argmax(axis=1) + 1
		# Calculate accuracy and cohen's kappa metrics
		a[fold-1] = accuracy(y_hat, y_test)
		kappa[fold-1] = cohen_kappa_score(y_hat, y_test)

		

