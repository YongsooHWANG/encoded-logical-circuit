
import numpy as np


def decode_steane_code(syndrome):
	"""
		steane method 기반 오류 신드롬 계산 및 most likely 오류 확인
	"""
	parity_check_steane = np.array([[1,0,1,0,1,0,1],
							 		[0,1,1,0,0,1,1], 
							 		[0,0,0,1,1,1,1]], dtype=int)

	# qubit permutation 필요
	# 측정된 값을 큐빗 매핑 순서에 따라서 reordering 필요

	syndrome_vector = np.array(syndrome)
	answer = np.matmul(parity_check_steane, syndrome_vector)
	answer %= 2
	
	# decoding : y = Hs' (s'=s+e)

	if np.any(answer):
		qubit_corrupted = None
		for k in range(parity_check_steane.shape[1]):
			if np.array_equal(parity_check_steane[:, k], answer):
				qubit_corrupted = k
				break

		if qubit_corrupted is None:
			raise Exception("Something Wrong..".format(answer))

		return qubit_corrupted

	return


def decode_15hamming_code(syndrome):
	"""
		[[15,7,3]] hamming code 에 대해서, steane method 기반 오류 신드롬 계산 및 most likely 오류 확인
	"""

	parity_check = np.array([[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1],
							 [0,1,1,0,0,1,1,0,0,1,1,0,0,1,1],
							 [0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
							 [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]], dtype=int)

	syndrome_vector = np.array(syndrome)
	answer = np.matmul(parity_check, syndrome_vector)
	answer %= 2

	if np.any(answer):
		qubit_corrupted = None
		for k in range(parity_check.shape[1]):
			if np.array_equal(parity_check[:, k], answer):
				qubit_corrupted = k
				break

		if qubit_corrupted is None:
			raise Exception("Something Wrong..".format(answer))

		return qubit_corrupted

	return