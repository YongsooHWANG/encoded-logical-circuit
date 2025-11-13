import os
import simplejson as json
from icecream import ic
import numpy as np

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=100)


def build_stabilizer_list(list_instructions, circuit_width):
	stabilizer = np.zeros((circuit_width, circuit_width*2), dtype=np.bool)

	for inst in list_instructions:
		tokens = inst.split(" ")
		qubits = list(map(int, tokens[1].split(",")))
		
		if tokens[0] == "PrepX":
			stabilizer[qubits[0]][qubits[0]] = True

		elif tokens[0] == "PrepZ":
			stabilizer[qubits[0]][qubits[0] + circuit_width] = True

		elif tokens[0] == "H":
			for j in range(circuit_width):
				stabilizer[j][qubits[0]], stabilizer[j][qubits[0]+circuit_width] =\
					stabilizer[j][qubits[0]+circuit_width], stabilizer[j][qubits[0]]

		elif tokens[0] == "CNOT":
			for j in range(circuit_width):
				if stabilizer[j][qubits[0]]:
					stabilizer[j][qubits[1]] ^= 1

				if stabilizer[j][circuit_width + qubits[1]]:
					stabilizer[j][circuit_width + qubits[0]] ^= 1

	return stabilizer


def build_stabilizer_time_ordered(file):
	raw_data = open(file).read()
	circuit_data = json.loads(raw_data).get("result")
	system_code = circuit_data.get("system_code")

	circuit = {int(idx) : instructions for idx, instructions in system_code.get("circuit").items()}
	initial_mapping = {k: int(v) for k, v in system_code.get("initial_mapping").items()}
	inverse_mapping = {v: k for k, v in initial_mapping.items()}

	ic(circuit)
	ic(initial_mapping)

	list_data_qubits = [k for k in initial_mapping.keys() if "data" in k]
	circuit_width = len(list_data_qubits)

	stabilizer = np.zeros((circuit_width, circuit_width*2), dtype=np.bool)

	circuit_depth = len(circuit.items())
	for k in range(circuit_depth):
		for inst in circuit[k]:
			tokens = inst.split(" ")
			# ic(tokens)

			if any(item in tokens[0] for item in ["Waiting", "Barrier"]): 
				continue

			qubits = list(map(int, tokens[1].split(",")))

			if tokens[0] == "PrepX":
				if "checkup" in inverse_mapping[qubits[0]]: continue
				logical_qubit = int(inverse_mapping[qubits[0]][4:])
				stabilizer[logical_qubit][logical_qubit] = True

			elif tokens[0] == "PrepZ":
				if "checkup" in inverse_mapping[qubits[0]]: continue
				logical_qubit = int(inverse_mapping[qubits[0]][4:])
				stabilizer[logical_qubit][logical_qubit + circuit_width] = True

			elif tokens[0] == "H":
				logical_qubit = int(inverse_mapping[qubits[0]][4:])
				for j in range(circuit_width):
					stabilizer[:, [logical_qubit, logical_qubit+circuit_width]] =\
						stabilizer[:, [logical_qubit+circuit_width, logical_qubit]]

			elif "CNOT" in tokens[0]:
				if "checkup" in inverse_mapping[qubits[0]] or\
					"checkup" in inverse_mapping[qubits[1]]: continue

				logical_ctrl = int(inverse_mapping[qubits[0]][4:])
				logical_trgt = int(inverse_mapping[qubits[1]][4:])

				for i in range(circuit_width):
					if stabilizer[i][logical_ctrl]:
						stabilizer[i][logical_trgt] = np.logical_xor(stabilizer[i][logical_trgt], True)

					if stabilizer[i][logical_trgt + circuit_width]:
						stabilizer[i][logical_ctrl+circuit_width] =\
							np.logical_xor(stabilizer[i][logical_ctrl+circuit_width], True)

			elif "SWAP" in tokens[0]:
				inverse_mapping[qubits[0]], inverse_mapping[qubits[1]] =\
					inverse_mapping[qubits[1]], inverse_mapping[qubits[0]]

	
	print(stabilizer.astype(int))
	
	return stabilizer

		
def display_stabilizer(stabilizer_binary):
	rows, cols = stabilizer_binary.shape

	for i in range(rows):
		for j in range(rows):
			if stabilizer_binary[i][j]:
				if stabilizer_binary[i][j+rows]:
					pauli_operator = "Y"
				else:
					pauli_operator = "X"
			else:
				if stabilizer_binary[i][j+rows]:
					pauli_operator = "Z"
				else:
					pauli_operator = "I"

			print("{:^3}".format(pauli_operator), end="")
		print()


if __name__ == "__main__":
	# golay code

	file = os.path.join("/Users/yongsoo/Desktop/FT_circuit_steane_code/test", "file_encoder_2_(8, 8, 1)_944.json")
	# file = os.path.join("../", "file_data_zero_2_(8, 9)_1525.json")

	stabilizer = build_stabilizer_time_ordered(file)
	display_stabilizer(stabilizer)
	file = os.path.join("/Users/yongsoo/Desktop/FT_circuit_steane_code/test", "DQP-file_encoder_2_(8, 8, 1)_944.json")
	# file = os.path.join("../", "file_data_zero_2_(8, 9)_1525.json")

	stabilizer = build_stabilizer_time_ordered(file)
	display_stabilizer(stabilizer)
	# matrix = [
	# [0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
	# [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
	# [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
	# [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
	# [1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
	# [1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
	# [0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
	# [0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
	# [0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
	# [1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
	# [1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

	# stabilizer = np.array(matrix, dtype=np.bool)
	# A_rref, b_rref, pivots = rref_gf2(stabilizer, None, full_reduce=True)
	# print(A_rref)

	# display_stabilizer(A_rref)
