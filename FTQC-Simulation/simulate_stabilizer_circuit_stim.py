
import os
import sys
import copy
import simplejson as json
import numpy as np
import math
from noise_channel import generate_random_error
from pauli_frame import pauli_vector, pauli_frame_rule
import collections
import itertools
from pprint import pprint
from icecream import ic
from progress.bar import Bar

sys.path.insert(0, "../support")

list_prepare = ["PrepZ", "PrepX"]
list_unitary_1q = ["H", "I", "X"]
list_unitary_2q = ["CNOT", "SWAP", "vvCNOT", "hhCNOT", "vvSWAP", "hhSWAP"]
list_measure = ["MeasZ", "MeasX"]


def inject_pauli_error(circuit_obj, error):
	"""
		inject a pauli error forcibly
		error = {"type": , "qubit": }
	"""
	parsed_gate = getattr(circuit_obj, error.get("type"))
	parsed_gate(error.get("qubit"))

	return circuit_obj



# 논리 큐비트 1개와 2개에 대해서 다시 한 번 정리 필요
def run_circuit_stim(circuit_obj, system_code, **kwargs):
	"""
		run a stabilizer circuit via stim
		simplified noise model (stochastic pauli error model) is applied
		args : stim object, system code (circuit and related information)

	"""
	ic.disable()
	physical_error_rate = kwargs.get("error_rate")
	error_rate = {"unitary1q": physical_error_rate,
				  "unitary2q": physical_error_rate*5,
				  "measure": physical_error_rate*10,
				  "idle": physical_error_rate/10}
				  
	# variance : 0, 1/5, 1/10, 1/20, 1/100
	error_variance = float(0)
	
	global_qubit_mapping = kwargs.get("global_qubit_mapping")
	logical_qubit_indices = kwargs.get("logical_qubit_index")

	limit_error = kwargs.get("limit_error")
	if limit_error is None:
		limit_error = math.inf

	qubit_association = {}
	qubit_mapping = {}

	if logical_qubit_indices is not None:
		if len(logical_qubit_indices) == 1:
			qubit_mapping = {k: global_qubit_mapping[logical_qubit_indices[0]][v]
							for k, v in system_code.get("initial_mapping").items()}

		elif len(logical_qubit_indices) == 2:
			logical_qubit_block_size = kwargs.get("logical_qubit_block_size")
			qubits_arrange = kwargs.get("qubits_arrange")
			# ic(logical_qubit_block_size)

			for k, v in system_code["initial_mapping"].items():
				# 물리 큐비트 인덱스 및 논리 큐비트 인덱스 확인
				row = None
				col = None
				if qubits_arrange in ["vertical", "v"]:
					row = int(v/logical_qubit_block_size["width"])
					col = v%logical_qubit_block_size["width"]

					if row >= logical_qubit_block_size["height"]:
						logical_qubit_index = logical_qubit_indices[1]
					else:
						logical_qubit_index = logical_qubit_indices[0]

					in_physical_qubit = (row%logical_qubit_block_size["height"]) * logical_qubit_block_size["width"] + col	

				else:
					row = int(v/(2*logical_qubit_block_size["width"]))
					col = v%(2*logical_qubit_block_size["width"])

					if col >= logical_qubit_block_size["width"]:
						logical_qubit_index = logical_qubit_indices[1]
					else:
						logical_qubit_index = logical_qubit_indices[0]
					
					in_physical_qubit = (row%logical_qubit_block_size["height"]) * logical_qubit_block_size["width"] +\
						col%logical_qubit_block_size["width"]
				
				corresponding_qubit = global_qubit_mapping[logical_qubit_index][in_physical_qubit]
				qubit_mapping[k] = corresponding_qubit
				
				qubit_association[v] = corresponding_qubit

	else:
		qubit_mapping = system_code.get("initial_mapping")

	
	inverse_mapping = {v: k for k, v in qubit_mapping.items()}

	circuit = system_code.get("circuit")	
	circuit_depth = len(circuit.keys())
	measure_outcomes = {}

	noise_frame = kwargs.get("noise_frame")

	if global_qubit_mapping is not None:
		qchip_size = len(global_qubit_mapping.items()) * len(qubit_mapping.items())
	else:
		qchip_size = len(qubit_mapping.items())

	error_source = []

	if noise_frame is None:
		noise_frame = np.zeros((qchip_size, 2), dtype=int)

	# print(circuit_depth)
	for tdx in range(circuit_depth):
		for inst in circuit[tdx]:
			tokens = inst.split(" ")
			if any(item in tokens for item in ["Barrier-All", "Barrier"]):
				continue

			gate_information = tokens[0].split("-")
			gate = gate_information[-1]

			qubits = list(map(int, tokens[1].split(",")))
			
			# print(inst, qubits)
			if logical_qubit_indices is not None:
				if len(logical_qubit_indices) == 1:
					for idx, qubit in enumerate(qubits):
						qubits[idx] = global_qubit_mapping[logical_qubit_indices[0]][qubits[idx]]

				else:
					for idx, qubit in enumerate(qubits):
						qubits[idx] = qubit_association[qubits[idx]]

			# print(qubits)
			# 여기서 게이트 동작하고, 오류는 나중에 반영
			if gate in list_unitary_2q:
				if gate == "CNOT":
					circuit_obj.cnot(qubits[0], qubits[1])

					# error propagation by CNOT
					# XI -> XX
					# IZ -> ZZ
					if noise_frame[qubits[0]][0]: noise_frame[qubits[1]][0] ^= 1
					if noise_frame[qubits[1]][1]: noise_frame[qubits[0]][1] ^= 1
					
				elif gate == "SWAP":
					circuit_obj.swap(qubits[0], qubits[1])

					noise_frame[qubits[0]], noise_frame[qubits[1]] =\
						noise_frame[qubits[1]], noise_frame[qubits[0]]

				elif gate == "CX":
					circuit_obj.cx(qubits[0], qubits[1])
					# XI -> XX
					# IZ -> ZZ

					if noise_frame[qubits[0]][0]: noise_frame[qubits[1]][0] ^= 1
					if noise_frame[qubits[1]][1]: noise_frame[qubits[0]][1] ^= 1

				elif gate == "CZ":
					circuit_obj.cz(qubits[0], qubits[1])
					# error propagation by CZ
					# XI --> XZ
					# IX --> ZX
					if noise_frame[qubits[0]][0]: noise_frame[qubits[1]][1] ^= 1
					if noise_frame[qubits[1]][0]: noise_frame[qubits[0]][1] ^= 1

			elif gate in list_prepare:
				if gate == "PrepZ":
					circuit_obj.reset_z(qubits[0])
					
					# reset the noise frame by reset the qubit
					noise_frame[qubits[0]][0] = 0
					noise_frame[qubits[0]][1] = 0

				elif gate == "PrepX":
					circuit_obj.reset_x(qubits[0])
				
					# reset the noise frame by reset the qubit
					noise_frame[qubits[0]][0] = 0
					noise_frame[qubits[0]][1] = 0

			elif gate in list_unitary_1q:
				if gate == "H":
					circuit_obj.h(qubits[0])

					noise_frame[qubits[0]][0], noise_frame[qubits[0]][1] =\
						noise_frame[qubits[0]][1], noise_frame[qubits[0]][0]

				elif gate == "X":
					circuit_obj.x(qubits[0])

			elif gate in list_measure:
				if limit_error > len(error_source):
					noise = generate_random_error(1, error_rate=error_rate["measure"], error_variance=error_variance)
				else:
					noise = 'i'

				if noise != 'i':
					parsed_gate = getattr(circuit_obj, noise)
					parsed_gate(qubits[0])
					error_source.append((noise, qubits[0], inverse_mapping[qubits[0]], inst, tdx))

				logical_qubit = inverse_mapping[qubits[0]]	
				
				if gate == "MeasZ":
					measure_outcomes[logical_qubit] = int(circuit_obj.measure(qubits[0]))

				elif gate == "MeasX":
					# stim package 에서 measure in X basis 기능을 제공하지 않음
					# 따라서, 먼저 hadamard 를 수행하고, 이후에 z 축 측정을 수행함
					circuit_obj.h(qubits[0])	
					measure_outcomes[logical_qubit] = int(circuit_obj.measure(qubits[0]))
			
			else: 
				if gate not in ["Waiting", "Idle"]:
					raise Exception(inst, gate)

			# generation quantum error per a location
			if gate in ["CNOT", "vvCNOT", "hhCNOT", "hCNOT"]:
				if limit_error > len(error_source):
					noise = generate_random_error(2, error_rate=error_rate["unitary2q"], error_variance=error_variance)
				else:
					noise = ['i', 'i']

				for i in range(2):
					if noise[i] != 'i':
						parsed_gate = getattr(circuit_obj, noise[i])
						parsed_gate(qubits[i])

						noise_frame[qubits[i]][0] ^= pauli_vector[noise[i]][0]
						noise_frame[qubits[i]][1] ^= pauli_vector[noise[i]][1]

				if noise != ['i', 'i']:
					error_source.extend([(noise[0], qubits[0], inverse_mapping[qubits[0]], inst, tdx),
										(noise[1], qubits[1], inverse_mapping[qubits[1]], inst, tdx)])

			elif gate in list_unitary_1q:
				if limit_error > len(error_source):
					noise = generate_random_error(1, error_rate=error_rate["unitary1q"], error_variance=error_variance)
				else:
					noise = 'i'

				if noise != 'i':
					parsed_gate = getattr(circuit_obj, noise)
					parsed_gate(qubits[0])
					error_source.append((noise, qubits[0], inverse_mapping[qubits[0]], inst, tdx))

					noise_frame[qubits[0]][0] ^= pauli_vector[noise[0]][0]
					noise_frame[qubits[0]][1] ^= pauli_vector[noise[0]][1]

			elif gate in list_prepare:
				if limit_error > len(error_source):
					noise = generate_random_error(1, error_rate=error_rate["measure"], error_variance=error_variance)
				else:
					noise = 'i'

				if noise !='i':
					parsed_gate = getattr(circuit_obj, noise)
					parsed_gate(qubits[0])
					error_source.append((noise, qubits[0], inverse_mapping[qubits[0]], inst, tdx))

					noise_frame[qubits[0]][0] ^= pauli_vector[noise[0]][0]
					noise_frame[qubits[0]][1] ^= pauli_vector[noise[0]][1]

			elif gate in ["Waiting", "Idle"]:
				if limit_error > len(error_source):
					noise = generate_random_error(1, error_rate=error_rate["idle"], error_variance=error_variance)
				else:
					noise = 'i'

				if noise != 'i':
					parsed_gate = getattr(circuit_obj, noise)
					parsed_gate(qubits[0])
					error_source.append((noise, qubits[0], inverse_mapping[qubits[0]], inst, tdx))

					noise_frame[qubits[0]][0] ^= pauli_vector[noise[0]][0]
					noise_frame[qubits[0]][1] ^= pauli_vector[noise[0]][1]

			if gate == "SWAP":
				inverse_mapping[qubits[0]], inverse_mapping[qubits[1]] =\
					inverse_mapping[qubits[1]], inverse_mapping[qubits[0]]

	# return  : stim 객체, 측정 결과, 오류 전파 결과, 오류 소스
	return circuit_obj, measure_outcomes, noise_frame, error_source


# 논리 큐비트 1개와 2개에 대해서 다시 한 번 정리 필요
def run_circuit_stim_physical(circuit_obj, system_code, **kwargs):
	"""
		run a stabilizer circuit via stim
		simplified noise model (stochastic pauli error model) is applied
		args : stim object, system code (circuit and related information)

	"""
	ic.disable()
	physical_error_rate = kwargs.get("error_rate")
	error_rate = {"unitary1q": physical_error_rate,
				  "unitary2q": physical_error_rate*5,
				  "measure": physical_error_rate*10,
				  "idle": physical_error_rate/10}

	# variance : 0, 1/5, 1/10, 1/20, 1/100
	error_variance = float(0)
	
	global_qubit_mapping = kwargs.get("global_qubit_mapping")
	logical_qubit_indices = kwargs.get("logical_qubit_index")

	qubit_association = {}
	qubit_mapping = {}

	if logical_qubit_indices is not None:
		if len(logical_qubit_indices) == 1:
			qubit_mapping = {k: global_qubit_mapping[logical_qubit_indices[0]][v]
							for k, v in system_code.get("initial_mapping").items()}

		elif len(logical_qubit_indices) == 2:
			logical_qubit_block_size = kwargs.get("logical_qubit_block_size")
			qubits_arrange = kwargs.get("qubits_arrange")
			# ic(logical_qubit_block_size)

			for k, v in system_code["initial_mapping"].items():
				# 물리 큐비트 인덱스 및 논리 큐비트 인덱스 확인
				row = None
				col = None
				if qubits_arrange in ["vertical", "v"]:
					row = int(v/logical_qubit_block_size["width"])
					col = v%logical_qubit_block_size["width"]

					if row >= logical_qubit_block_size["height"]:
						logical_qubit_index = logical_qubit_indices[1]
					else:
						logical_qubit_index = logical_qubit_indices[0]

					in_physical_qubit = (row%logical_qubit_block_size["height"]) * logical_qubit_block_size["width"] + col	

				else:
					row = int(v/(2*logical_qubit_block_size["width"]))
					col = v%(2*logical_qubit_block_size["width"])

					if col >= logical_qubit_block_size["width"]:
						logical_qubit_index = logical_qubit_indices[1]
					else:
						logical_qubit_index = logical_qubit_indices[0]
					
					in_physical_qubit = (row%logical_qubit_block_size["height"]) * logical_qubit_block_size["width"] +\
						col%logical_qubit_block_size["width"]
				
				corresponding_qubit = global_qubit_mapping[logical_qubit_index][in_physical_qubit]
				qubit_mapping[k] = corresponding_qubit
				
				qubit_association[v] = corresponding_qubit

	else:
		qubit_mapping = system_code.get("initial_mapping")

	inverse_mapping = {v: k for k, v in qubit_mapping.items()}

	circuit = system_code.get("circuit")	
	circuit_depth = len(circuit.keys())
	measure_outcomes = {}

	noise_frame = kwargs.get("noise_frame")

	if global_qubit_mapping is not None:
		qchip_size = len(global_qubit_mapping.items()) * len(qubit_mapping.items())
	else:
		qchip_size = len(qubit_mapping.items())

	error_source = []

	if noise_frame is None:
		noise_frame = np.zeros((qchip_size, 2), dtype=int)

	for tdx in range(circuit_depth):
		for inst in circuit[tdx]:
			tokens = inst.split(" ")
			if any(item in tokens for item in ["Barrier-All", "Barrier"]):
				continue

			gate_information = tokens[0].split("-")
			gate = gate_information[-1]

			qubits = list(map(int, tokens[1].split(",")))
			
			if logical_qubit_indices is not None:
				if len(logical_qubit_indices) == 1:
					for idx, qubit in enumerate(qubits):
						qubits[idx] = global_qubit_mapping[logical_qubit_indices[0]][qubits[idx]]

				else:
					for idx, qubit in enumerate(qubits):
						qubits[idx] = qubit_association[qubits[idx]]

			# 여기서 게이트 동작하고, 오류는 나중에 반영
			if gate in list_unitary_2q:
				if gate == "CNOT":
					circuit_obj.cnot(qubits[0], qubits[1])

					# error propagation by CNOT
					# XI -> XX
					# IZ -> ZZ
					if noise_frame[qubits[0]][0]: noise_frame[qubits[1]][0] ^= 1
					if noise_frame[qubits[1]][1]: noise_frame[qubits[0]][1] ^= 1
					
				elif gate == "SWAP":
					circuit_obj.swap(qubits[0], qubits[1])

					noise_frame[qubits[0]], noise_frame[qubits[1]] =\
						noise_frame[qubits[1]], noise_frame[qubits[0]]

				elif gate == "CX":
					circuit_obj.cx(qubits[0], qubits[1])
					# XI -> XX
					# IZ -> ZZ

					if noise_frame[qubits[0]][0]: noise_frame[qubits[1]][0] ^= 1
					if noise_frame[qubits[1]][1]: noise_frame[qubits[0]][1] ^= 1

				elif gate == "CZ":
					circuit_obj.cz(qubits[0], qubits[1])
					# error propagation by CZ
					# XI --> XZ
					# IX --> ZX
					if noise_frame[qubits[0]][0]: noise_frame[qubits[1]][1] ^= 1
					if noise_frame[qubits[1]][0]: noise_frame[qubits[0]][1] ^= 1

			elif gate in list_prepare:
				if gate == "PrepZ":
					circuit_obj.reset_z(qubits[0])
					
					# reset the noise frame by reset the qubit
					noise_frame[qubits[0]][0] = 0
					noise_frame[qubits[0]][1] = 0

				elif gate == "PrepX":
					circuit_obj.reset_x(qubits[0])
				
					# reset the noise frame by reset the qubit
					noise_frame[qubits[0]][0] = 0
					noise_frame[qubits[0]][1] = 0

			elif gate in list_unitary_1q:
				if gate == "H":
					circuit_obj.h(qubits[0])

					noise_frame[qubits[0]][0], noise_frame[qubits[0]][1] =\
						noise_frame[qubits[0]][1], noise_frame[qubits[0]][0]

			elif gate in list_measure:
				noise = generate_random_error(1, error_rate=error_rate["measure"], error_variance=error_variance)

				if noise != 'i':
					parsed_gate = getattr(circuit_obj, noise)
					parsed_gate(qubits[0])
					error_source.append((noise, qubits[0], inverse_mapping[qubits[0]], inst, tdx))

				logical_qubit = inverse_mapping[qubits[0]]	
				
				if gate == "MeasZ":
					measure_outcomes[logical_qubit] = int(circuit_obj.measure(qubits[0]))

				elif gate == "MeasX":
					# stim package 에서 measure in X basis 기능을 제공하지 않음
					# 따라서, 먼저 hadamard 를 수행하고, 이후에 z 축 측정을 수행함
					circuit_obj.h(qubits[0])	
					measure_outcomes[logical_qubit] = int(circuit_obj.measure(qubits[0]))


			# generation quantum error per a location
			if gate in list_unitary_2q:
				noise = generate_random_error(2, error_rate=error_rate["unitary2q"], error_variance=error_variance)
				
				for i in range(2):
					if noise[i] != 'i':
						parsed_gate = getattr(circuit_obj, noise[i])
						parsed_gate(qubits[i])

						error_source.append((noise[i], qubits[i], inverse_mapping[qubits[i]], inst, tdx))

						noise_frame[qubits[i]][0] ^= pauli_vector[noise[i]][0]
						noise_frame[qubits[i]][1] ^= pauli_vector[noise[i]][1]

			elif gate in list_unitary_1q:
				noise = generate_random_error(1, error_rate=error_rate["unitary1q"], error_variance=error_variance)
				
				if noise != 'i':
					parsed_gate = getattr(circuit_obj, noise)
					parsed_gate(qubits[0])
					error_source.append((noise, qubits[0], inverse_mapping[qubits[0]], inst, tdx))

					noise_frame[qubits[0]][0] ^= pauli_vector[noise[0]][0]
					noise_frame[qubits[0]][1] ^= pauli_vector[noise[0]][1]

			elif gate in list_prepare:
				noise = generate_random_error(1, error_rate=error_rate["measure"], error_variance=error_variance)
				if noise !='i':
					parsed_gate = getattr(circuit_obj, noise)
					parsed_gate(qubits[0])
					error_source.append((noise, qubits[0], inverse_mapping[qubits[0]], inst, tdx))

					noise_frame[qubits[0]][0] ^= pauli_vector[noise[0]][0]
					noise_frame[qubits[0]][1] ^= pauli_vector[noise[0]][1]

			elif gate in ["Waiting", "Idle"]:
				noise = generate_random_error(1, error_rate=error_rate["idle"], error_variance=error_variance)
				if noise != 'i':
					parsed_gate = getattr(circuit_obj, noise)
					parsed_gate(qubits[0])
					error_source.append((noise, qubits[0], inverse_mapping[qubits[0]], inst, tdx))

					noise_frame[qubits[0]][0] ^= pauli_vector[noise[0]][0]
					noise_frame[qubits[0]][1] ^= pauli_vector[noise[0]][1]

			if gate == "SWAP":
				inverse_mapping[qubits[0]], inverse_mapping[qubits[1]] =\
					inverse_mapping[qubits[1]], inverse_mapping[qubits[0]]

	# return  : stim 객체, 측정 결과, 오류 전파 결과, 오류 소스
	return circuit_obj, measure_outcomes, noise_frame, error_source
	

