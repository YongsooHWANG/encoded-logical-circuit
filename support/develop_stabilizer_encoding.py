
from icecream import ic
import numpy as np

import checkup_stabilizer as cs

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=100)

def develop_encoding_circuit(stabilizer, logical_x, code_spec):
	"""
		IEEE Circuits and Systems Magazine, First Quarter 2024
		Arijit Mondal and Keshab K. Parhi
	"""
	n, k, r = code_spec[:]

	circuit = []
	# initialize ancilla qubits
	for i in range(n-k):
		circuit.append("PrepZ {}".format(i))

	# apply logical X
	for i in range(k):
		if logical_x[i][n-k+i]:
			for j in range(n):
				if (i + n - k) != j and logical_x[i][j]:
					circuit.append("CNOT {},{}".format(i+n-k, j))

	for i in range(r):
		if stabilizer[i][i+n] == 0:
			circuit.append("H {}".format(i))
		else:
			circuit.append("H {}".format(i))
			circuit.append("S {}".format(i))

		for j in range(n):
			if i!=j:
				if stabilizer[i][j] == 1:
					if stabilizer[i][j+n] == 1:
						circuit.append("CY {},{}".format(i,j))
					else:
						circuit.append("CNOT {},{}".format(i,j))
				else:
					if stabilizer[i][j+n] == 1:
						circuit.append("CZ {},{}".format(i,j))
	
	return circuit


if __name__ == "__main__":
	parity_check_x = np.array([
	[1,0,0,0,	1,0,1,1,	1,0,1,0,1,0,1],
	[0,1,0,0,	0,1,1,0,	1,1,1,0,0,1,1],
	[0,0,1,0,	1,1,1,0,	1,0,0,1,1,1,1],
	[0,0,0,1,	0,0,0,1,	0,1,1,1,1,1,1]])

	parity_check_z = np.array([
	[0,1,1,0,	1,0,0,0,	1,1,1,1,1,0,0],
	[1,0,1,1,	0,1,0,0,	1,1,0,0,1,0,1],
	[1,1,1,1,	0,0,1,0,	0,0,1,0,1,1,0],
	[0,0,0,1,	0,0,0,1,	0,1,1,1,1,1,1]])

	logical_x = np.array([
	[0,0,0,0,	1,1,0,0,	1,0,0,0,0,0,0],
	[0,0,0,0,	1,1,0,1,	0,1,0,0,0,0,0],
	[0,0,0,0,	1,0,1,1,	0,0,1,0,0,0,0],
	[0,0,0,0,	1,0,0,1,	0,0,0,1,0,0,0],
	[0,0,0,0,	1,1,1,1,	0,0,0,0,1,0,0],
	[0,0,0,0,	0,0,1,1,	0,0,0,0,0,1,0],
	[0,0,0,0,	0,1,0,1,	0,0,0,0,0,0,1]])

	stabilizer_matrix = np.block([
		[parity_check_x, np.zeros((4,15), dtype=np.bool)],
		 [np.zeros((4,15), dtype=np.bool), parity_check_z]])

	logical_x = np.array(logical_x)

	# stabilizer, logical X operator, code spec=(n, k, r)
	n = 15
	k = 7
	r = 4
	encoding_circuit = develop_encoding_circuit(stabilizer_matrix, logical_x, (n, k, r))
	ic(encoding_circuit)
	# stabilizer = cs.build_stabilizer_list(encoding_circuit, n)

	# # swap 7&8
	# for j in range(8):
	# 	stabilizer[:, [7,8]] = stabilizer[:, [8,7]]
	# 	stabilizer[:, [7+n, 8+n]] = stabilizer[:, [8+n, 7+n]]

	# 	stabilizer[:, [3,7]] = stabilizer[:, [7,3]]
	# 	stabilizer[:, [3+n, 7+n]] = stabilizer[:, [7+n, 3+n]]

	# 	stabilizer[:, [2,3]] = stabilizer[:, [3,2]]
	# 	stabilizer[:, [2+n, 3+n]] = stabilizer[:, [3+n, 2+n]]


	# ic(stabilizer.astype(int))