import numpy as np

class VSNLMS:
	def __init__(self, mu, mu_max, mu_min, m0, m1, alpha, delta=0.0):
		"""
		VSNLMS algorithm initialization function
		@:param mu: Initial step adjustment coefficient
		@:param mu_max: Maximum value of the step coefficient (The coefficients are adjusted as they are made
		iterations by the VS-LMS algorithm)
		@:param mu_min: Minimum step coefficient value
		@:param m0_per: Number of gradient values ​​that must change sign for the mu to adjust. This quantity
		is given as a percentage of the value of N
		@:param m1_per: Number of gradient values ​​that must maintain sign for the mu to be adjusted. This quantity
		is given as a percentage of the value of N
		@:param alpha: mu adjustment parameter
		@:param delta: delta of the NLMS algorithm, this is set to ensure that in the event that the denominator is null, it does not
		explode.
		"""
		self.mu = mu
		self.count_mu_values = 0
		self.mu_max = mu_max
		self.mu_min = mu_min
		self.alpha = alpha
		self.m0 = m0
		self.m1 = m1
		self.prev_sign = 0
		self.delta = delta
		self.prev_grad = []
		return

	def calcNewCoef(self, a_n, signal, error):
		"""
		Function that, given some previous coefficients, updates them correspondingly
		:param a_n: Coefficients to update
		:param signal: Input signal vector
		:param error: Error value to update the coefficients
		:return: Vector of updated coefficients formatted the same as a_n
		"""
		grad = np.array(signal) * error
		if len(self.prev_grad) != 0:
			if np.sum(np.sign(grad) != self.prev_grad) > self.m0 and self.mu > self.mu_min:
				self.mu /= self.alpha
			elif np.sum(np.sign(grad) == self.prev_grad) > self.m1 and self.mu < self.mu_max:
				self.mu *= self.alpha
		self.prev_grad = np.sign(grad)
		return a_n + self.mu * grad / (np.dot(signal, signal) + self.delta)  # NLMS

	def getMu(self):
		"""
		Function to obtain the current mu of the algorithm
		:return: current mu.
		"""
		return self.mu
	

class AdaptativeFilter:
	def __init__(self, N, mu=1e-3, mu_max=1.1, mu_min=1e-9, m0_per=0.9, m1_per=0.9, alpha=10, delta=0.0):
		"""
		Adaptive filter initialization function
		@:param N: It is the order of the filter
		@:param mu: Initial step adjustment coefficient
		@:param mu_max: Maximum value of the step coefficient (The coefficients are adjusted as they are made
		iterations by the VS-LMS algorithm)
		@:param mu_min: Minimum step coefficient value
		@:param m0_per: Number of gradient values ​​that must change sign for the mu to adjust. This quantity
		is given as a percentage of the value of N
		@:param m1_per: Number of gradient values ​​that must maintain sign for the mu to be adjusted. This quantity
		is given as a percentage of the value of N
		@:param alpha: mu adjustment parameter
		@:param delta: delta of the NLMS algorithm, this is set to ensure that in the event that the denominator is null, it does not
		explode.
		"""
		self.N = N
		self.w = np.random.randn(N)
		self.mu = mu
		m0 = m0_per * N
		m1 = m1_per * N
		self.vsnlms = VSNLMS(self.mu, mu_max, mu_min, m0, m1, alpha, delta)
		self.inp_signal = list(np.zeros(N))
		self.err = 1
		return

	def fit(self, input, desired):
		"""
		Function that updates the weights of an adaptive filter to match the desired response.
		@:param input: Input vector to the adaptive filter
		@:param desired: Desired signal vector of the adaptive filter.
		"""
		self.inp_signal = list(np.zeros(self.N))
		self.err = 1
		for x, d in tqdm(zip(input, desired)):
			self.inp_signal.append(x)
			self.inp_signal.pop(0)
			self.err = d - np.dot(self.inp_signal, self.w)  # It is defined this way
			self.updateLMS()
		return

	def getFilterParameters(self):
		"""
		Function that returns the parameters of the adaptive filter
		@:return The filter coefficients
		"""
		return np.flip(self.w)

	def updateLMS(self):
		"""
		Function that updates the adaptive filter coefficients, uses the vectors self.inp_signal, sel.w, and the value
		self.err
		"""
		self.w = self.vsnlms.calcNewCoef(self.w, self.inp_signal, self.err)
		return

	def getMu(self):
		"""
		Returns the current Mu of the adaptive filter
		@:return mu Current
		"""
		return self.vsnlms.getMu()

	def applyFilterSame(self, input):
		"""
		Apply the adaptive filter obtained to an input vector
		@:param input: input vector to apply the filter. (IMPORTANT: Does not update self.inp_signal)
		@:return Filter output vector.
		"""
		return signal.convolve(input, np.flip(self.w), mode="same")  # It is not checked that the same mode works

	def applyFilterFull(self, input):
		"""
		Apply the adaptive filter obtained to an input vector
		@:param input: input vector to apply the filter. (IMPORTANT: Does not update self.inp_signal)
		@:return Filter output vector.
		"""
		return signal.convolve(input, np.flip(self.w), mode="full")  # It is not checked that the same mode works

	def resetInput(self):
		"""
		Function that clears the data from self.inp_signal and sets it all to zero
		"""
		self.inp_signal = list(np.zeros(self.N))
		return

	def applyFilterToTap(self, tap):
		"""
		Function that applies the adaptive filter to a single input tap, uses the previous values ​​saved in self.inp_signal
		and update that vector (IMPORTATNE: Update the values ​​of self.inp_signal)
		@:param tap: Value of the input tap to apply the filter
		@:return Output value of adaptive filter
		"""
		self.inp_signal.append(tap)
		self.inp_signal.pop(0)
		temp = self.applyFilterFull(self.inp_signal)
		temp2 = temp[self.N - 1]
		return temp2

	def fitFilterWithErrorTap(self, input_vector, e_tap):
		"""
		Function that updates the values ​​of the adaptive filter coefficients given an error signal tap.
		(IMPORTANT: The signal self.inp_signal must be previously updated to the input tap corresponding to e_tap).
		@:param input_vector: Input vector to the filter to update the coefficients
		@:param e_tap: It is a tap of the measured error signal.
		"""
		self.err = e_tap
		self.w = self.vsnlms.calcNewCoef(self.w, input_vector, self.err)
		return

	def fitFilterWithDesired(self,d_tap):
		"""
		Function that updates the filter coefficients with a current tap of the desired signal
		(IMPORTANT: self.inp_signal must be preloaded with the current input tap).
		:param d_tap: tap of the desired signal
		"""
		self.err = d_tap - np.dot(self.inp_signal, self.w)
		self.updateLMS()
		return

	def setInputVector(self, input):
		"""
		Function that sets the input vector from outside.
		:param input: Input vector to update the adaptive filter
		"""
		if len(input) != self.N:
			raise Exception(f"Filter size does not match {self.N} with the vector input included {len(input)}")
		self.inp_signal = input
		return
	
class Filter:
	def __init__(self,coefs):
		"""
		Function to initialize the filter with a given list of coefficients
		:param coefs: Coefficients of the filter to be initialized must be given as follows:
			[w0,w1,w2,w3,..wN]
		"""
		self.coefs = coefs
		return

	def applyFilter(self, input):
		"""
		Function to apply the filter to a given input signal
		:param input: Input signal vector, with the most recent value located on the right, and the oldest value on
		left.
		:return: Vector of the output signal, filtered.
		"""
		return signal.convolve(input, self.coefs, mode="full")

	def getFilterLen(self):
		"""
		Function that returns the number of filter parameters.
		:return:
		"""
		return len(self.coefs)

	def getCoefs(self):
		"""
		Function that returns a list with the filter coefficients.
		:return: List with the filter coefficients.
		"""
		return self.coefs