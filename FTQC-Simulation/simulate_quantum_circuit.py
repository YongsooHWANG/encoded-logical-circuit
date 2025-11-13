
import os
import sys
import copy
import simplejson as json
import numpy as np
from noise_channel import generate_random_error
from pauli_frame import pauli_vector, pauli_frame_rule
import collections

import itertools
from pprint import pprint
from icecream import ic
from progress.bar import Bar

sys.path.insert(0, "../support")
sys.path.insert(0, "../../SimulateQC")
import qubit_original as qubit
import ftcheckup

list_prepare = ["PrepZ", "PrepX"]
list_unitary_1q = ["H"]
list_unitary_2q = ["CNOT", "SWAP"]
list_measure = ["MeasZ", "MeasX"]

parity_check_steane = np.array([[1,0,1,0,1,0,1],
						 		[0,1,1,0,0,1,1], 
						 		[0,0,0,1,1,1,1]], dtype=int)

pauli_gate = {0 : "i", 1: "x", 2:"y", 3:"z"}

total_iteration = 1000

def decode_steane_method(syndrome):
	"""
		steane method 기반 오류 신드롬 계산 및 most likely 오류 확인
	"""

	# qubit permutation 필요
	# 측정된 값을 큐빗 매핑 순서에 따라서 reordering 필요

	syndrome_vector = np.array(syndrome)
	answer = np.matmul(parity_check_steane, syndrome_vector)
	answer %= 2
	
	# decoding : y = Hs' (s'=s+e)
	# ic(answer, parity_check_steane, syndrome)

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


def display_braket_notation(state_vector, **kwargs):
	"""
		function to display a state vector with braket notation
	"""	
	hightlight = kwargs.get("hightlight")
	flag_hightlight_only = kwargs.get("hightlight_only")

	dimension = state_vector.shape[0]
	size_register = int(np.log2(dimension))

	if hightlight is None:
		for i in range(dimension):
			observable = "".join(list(np.binary_repr(i, size_register)))
			if np.abs(state_vector.item(i)) > 0:
				print("\t{0:>50} | {1} > \t{2}".format(str(state_vector.item(i)), observable,
							np.abs(state_vector.item(i))**2))
	else:
		for i in range(dimension):
			observable = "".join(list(np.binary_repr(i, size_register)))
			if np.abs(state_vector.item(i)) > 0:
				hightlighted_observable = []
				for idx, bit in enumerate(observable):
					# 출력 순서가 반대이니까.. idx 대신, size_qubits-1-idx 
					if (size_register - idx - 1) in hightlight:
						hightlighted_observable.append("\033[95m{}\033[0m".format(bit))
					else:
						if not flag_hightlight_only:
							hightlighted_observable.append("{}".format(bit))

				print("\t{0:>20} | {1} > \t{2}".format(str(state_vector.item(i)), "".join(hightlighted_observable),
							np.abs(state_vector.item(i))**2))


def run_circuit(circuit_obj, system_code, **kwargs):
	"""
		run a stabilizer circuit via stim
	"""
	# ic.disable()
	physical_error_rate = kwargs.get("error_rate")
	error_rate = {"unitary1q": physical_error_rate,
				  "unitary2q": physical_error_rate*5,
				  "measure": physical_error_rate*10,
				  "idle": physical_error_rate/10}

	qubit_mapping = system_code.get("initial_mapping")
	inverse_mapping = {v: k for k, v in qubit_mapping.items()}

	circuit = system_code.get("circuit")
	circuit_depth = len(circuit.keys())
	measure_outcomes = {}

	count_error = 0
	noise_frame = kwargs.get("noise_frame")
	qchip_size = len(qubit_mapping.items())

	if noise_frame is None:
		noise_frame = np.zeros((qchip_size, 2), dtype=int)

	ic(circuit_depth)
	for tdx in range(circuit_depth):
		for inst in circuit[tdx]:
			tokens = inst.split(" ")

			if any(item in tokens for item in ["Barrier-All", "Barrier"]):
				continue

			gate_information = tokens[0].split("-")
			gate = gate_information[-1]

			qubits = list(map(int, tokens[1].split(",")))
			ic(gate, qubits)

			if len(qubits) == 2:
				ic(inverse_mapping[qubits[0]], inverse_mapping[qubits[1]])
			else:
				ic(inverse_mapping[qubits[0]])

			parsed_gate = getattr(circuit_obj, gate)
			
			if gate in list_unitary_2q:
				if gate == "CNOT":
					parsed_gate(qubits)
					# circuit_obj.cnot(qubits[0], qubits[1])
					# error propagation by CNOT
					# XI -> XX
					# IZ -> ZZ
					# if noise_frame[qubits[0]][0]: noise_frame[qubits[1]][0] ^= 1
					# if noise_frame[qubits[1]][1]: noise_frame[qubits[0]][1] ^= 1

				elif gate == "SWAP":
					parsed_gate(qubits)
					# circuit_obj.swap(qubits[0], qubits[1])
					# noise_frame[qubits[0]][0], noise_frame[qubits[0]][1] =\
					# 	noise_frame[qubits[0]][1], noise_frame[qubits[0]][0]

				elif gate == "CX":
					parsed_gate(qubits)
					# circuit_obj.cx(qubits[0], qubits[1])

					# if noise_frame[qubits[0]][0]: noise_frame[qubits[1]][0] ^= 1
					# if noise_frame[qubits[1]][1]: noise_frame[qubits[0]][1] ^= 1

				elif gate == "CZ":
					parsed_gate(qubits)
					# circuit_obj.cz(qubits[0], qubits[1])
					# # error propagation by CZ
					# # XI --> XZ
					# # IX --> ZX
					# if noise_frame[qubits[0]][0]: noise_frame[qubits[1]][1] ^= 1
					# if noise_frame[qubits[1]][0]: noise_frame[qubits[0]][1] ^= 1

			elif gate in list_prepare:
				if gate == "PrepZ":
					parsed_gate(qubits)

				# elif gate == "PrepX":
				# 	circuit_obj.reset_x(qubits[0])

			elif gate in list_unitary_1q:
				if gate == "H":
					parsed_gate(qubits)
					# circuit_obj.h(qubits[0])
					# noise_frame[qubits[0]][0], noise_frame[qubits[0]][1] =\
					# 	noise_frame[qubits[0]][1], noise_frame[qubits[0]][0]

			elif gate in list_measure:
				if gate == "MeasZ":
					data = parsed_gate(qubits)
					# logical_qubit = inverse_mapping[qubits[0]]
					# data = int(circuit_obj.measure(qubits[0]))
					# measure_outcomes[logical_qubit] = data

					# if np.random.random_sample() < error_rate.get("measure"):
					# 	ic(inst, "noise : X on {}".format(qubits))
					# 	count_error+=1
					# 	measure_outcomes[logical_qubit] ^= 1
						
			# # error change by new error
			# if gate in list_unitary_2q:
			# 	noise = generate_random_error(2, error_rate=error_rate["unitary2q"])
			# 	for i in range(2):
			# 		if noise[i] != 'i':
			# 			count_error+=1
			# 			parsed_gate = getattr(circuit_obj, noise[i])
			# 			parsed_gate(qubits[i])

			# 			noise_frame[qubits[i]][0] ^= pauli_vector[noise[i]][0]
			# 			noise_frame[qubits[i]][1] ^= pauli_vector[noise[i]][1]

			# elif gate in list_unitary_1q:
			# 	noise = generate_random_error(1, error_rate=error_rate["unitary1q"])
			# 	if noise != 'i':
			# 		count_error+=1
			# 		parsed_gate = getattr(circuit_obj, noise)
			# 		parsed_gate(qubits[0])

			# 		noise_frame[qubits[0]][0] ^= pauli_vector[noise[0]][0]
			# 		noise_frame[qubits[0]][1] ^= pauli_vector[noise[0]][1]

			# elif gate in list_prepare:
			# 	if gate == "PrepZ" and\
			# 		np.random.random_sample() < error_rate.get("measure"):
			# 			ic("noise : X on {}".format(qubits))
			# 			count_error+=1
			# 			circuit_obj.x(qubits[0])
						
			# 			noise_frame[qubits[0]][0] ^= pauli_vector['x'][0]
			# 			noise_frame[qubits[0]][1] ^= pauli_vector['x'][1]

			# 	elif gate == "PrepX" and\
			# 		np.random.random_sample() < error_rate.get("measure"):
			# 			ic("noise : Z {}".format(qubits))
			# 			count_error+=1
			# 			circuit_obj.z(qubits[0])

			# 			noise_frame[qubits[0]][0] ^= pauli_vector['z'][0]
			# 			noise_frame[qubits[0]][1] ^= pauli_vector['z'][1]						

			# elif gate == "Waiting":
			# 	noise = generate_random_error(1, error_rate=error_rate["idle"])
			# 	if noise != 'i':
			# 		count_error+=1
			# 		parsed_gate = getattr(circuit_obj, noise)
			# 		parsed_gate(qubits[0])

			# 		noise_frame[qubits[0]][0] ^= pauli_vector[noise[0]][0]
			# 		noise_frame[qubits[0]][1] ^= pauli_vector[noise[0]][1]

			
			# if gate not in list_measure + list_prepare:		
			# 	ic(inst, noise, qubits, count_error)

			# if gate == "SWAP":
			# 	inverse_mapping[qubits[0]], inverse_mapping[qubits[1]] =\
			# 		inverse_mapping[qubits[1]], inverse_mapping[qubits[0]]
			ic(measure_outcomes)
	# ic(count_error)
	# ic(noise_frame)
	# propagated_error = []

	# for i in range(qchip_size):
	# 	if noise_frame[i][0]: 
	# 		if noise_frame[i][1]: 
	# 			propagated_error.append('Y')
	# 		else:
	# 			propagated_error.append('X')
	# 	else:
	# 		if noise_frame[i][1]: 
	# 			propagated_error.append('Z')
	# 		else:
	# 			propagated_error.append('I')

	# ic(" ".join(propagated_error))
	# ic.enable()
	return circuit_obj, measure_outcomes, noise_frame
	

def evaluate_logical_error(collection_circuits):
	# 물리적 오류율 기준, 시뮬레이션 반복 통해 논리적 오류율 분석
	distribution = collections.defaultdict(dict)

	total_iteration = 0
	
	ic(collection_circuits)
	for error_factor in [0]:
		break
		fail = 0
		for j in range(total_iteration):
			stim_object = stim.TableauSimulator()

			#########################################
			# noiseless preparation of logical data
			#########################################
			stim_object, result, noise_frame = run_circuit_stim(stim_object,
												collection_circuits.get("Prep-Data-Zero").get("system_code"), error_rate=0)

			#########################################
			# noisy QEC
			#########################################
			# 1. check z (prep. of ancilla zero + check z)
			stim_object, result, noise_frame = run_circuit_stim(stim_object,
												collection_circuits.get("Prep-Anc-Zero").get("system_code"), error_rate=0,
												noise_frame=noise_frame)

			stim_object, result, noise_frame = run_circuit_stim(stim_object, 
												collection_circuits.get("Check_Z").get("system_code"), error_rate=error_factor,
												noise_frame=noise_frame)

			# sort the syndrome vector
			sorted_syndrome = sorted(result.items(), key = lambda item: item[0])
			syndrome_vector = [v[1] for v in sorted_syndrome]
			
			# infer an error from the syndrome vector --> recovery operator
			qubit_corrupted = decode_steane_method(syndrome_vector)
			
			# apply recovery operator : Z gate on qubit_corrupted
			if qubit_corrupted is not None:
				stim_object.z(qubit_corrupted)

			# 2. check x (prep. of anilla plus + check x)
			stim_object, result, noise_frame = run_circuit_stim(stim_object,
												collection_circuits.get("Prep-Anc-Plus").get("system_code"), error_rate=0)

			stim_object,result, noise_frame = run_circuit_stim(stim_object,
												collection_circuits.get("Check_X").get("system_code"), error_rate=error_factor,
												noise_frame=noise_frame)

			# sort the syndrome vector
			sorted_syndrome = sorted(result.items(), key = lambda item: item[0])
			syndrome_vector = [v[1] for v in sorted_syndrome]
			
			# infer an error from the syndrome vector --> recovery operator
			qubit_corrupted = decode_steane_method(syndrome_vector)
			
			# apply recovery operator : Z gate on qubit_corrupted
			if qubit_corrupted is not None:
				stim_object.x(qubit_corrupted)

			#########################################
			# noiseless checkup
			#########################################
			# 1. check z (prep. of ancilla zero + check z)
			stim_object, result, noise_frame = run_circuit_stim(stim_object,
												collection_circuits.get("Prep-Anc-Zero").get("system_code"), error_rate=0,
												noise_frame=noise_frame)

			stim_object, result, noise_frame = run_circuit_stim(stim_object, 
												collection_circuits.get("Check_Z").get("system_code"), error_rate=0,
												noise_frame=noise_frame)

			# sort the syndrome vector
			sorted_syndrome = sorted(result.items(), key = lambda item: item[0])
			syndrome_vector = [v[1] for v in sorted_syndrome]
			
			# infer an error from the syndrome vector --> recovery operator
			qubit_corrupted = decode_steane_method(syndrome_vector)
			
			# apply recovery operator : Z gate on qubit_corrupted
			if qubit_corrupted is not None: 
				fail+=1
				continue
			
			# 2. check x (prep. of anilla plus + check x)
			stim_object, result, noise_frame = run_circuit_stim(stim_object,
												collection_circuits.get("Prep-Anc-Plus").get("system_code"), error_rate=0)

			stim_object,result, noise_frame = run_circuit_stim(stim_object,
												collection_circuits.get("Check_X").get("system_code"), error_rate=error_factor,
												noise_frame=noise_frame)

			# sort the syndrome vector
			sorted_syndrome = sorted(result.items(), key = lambda item: item[0])
			syndrome_vector = [v[1] for v in sorted_syndrome]
			
			# infer an error from the syndrome vector --> recovery operator
			qubit_corrupted = decode_steane_method(syndrome_vector)
			
			# apply recovery operator : Z gate on qubit_corrupted
			if qubit_corrupted is not None: 
				fail+=1
				continue

		distribution[error_factor] = {"rate": 1-float(fail/total_iteration),
									  "failure": fail}

		print("for the physical error rate : {:>5}, the logical error rate : {} (#fails = {})".format(
				error_factor, 1-float(fail/total_iteration), fail))
	

def evaluate_logical_error(collection_circuits):
	
	distribution = collections.defaultdict(float)	
	total_iteration = 1

	for k, v in collection_circuits.items():
		circuit_bandwidth = len(v.get("qchip").get("qubit_connectivity").items())

		
	# for error_factor in [1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]:
	for error_factor in [0]:
		
		fail = 0
		rejected = 0
			
		for j in range(total_iteration):
			circuit_object = qubit.QuantumState(circuit_bandwidth)

			# print(" ============================================================= ")
			# preparation of logical data until the checkup value is 0
			circuit_object, result, noise_frame = run_circuit(circuit_object,
									collection_circuits.get("Initialize-Zero").get("system_code"), error_rate=error_factor)
			
			ic(result)
			# ic(result)
			# if result.get("checkup0") == 1: 
			# 	fail+=1
			# 	rejected+=1
			# 	continue

		# 	# print("After Preparation of Data : ")
		# 	# ic("Stabilizer : ", stim_object.current_inverse_tableau())

		# 	# qubit_mapping = collection_circuits.get("Prep-Data-Zero").get("system_code").get("final_mapping")
		# 	# data_qubits = [v for k, v in qubit_mapping.items() if "data" in k]
			
		# 	# ic("State vector : ")
		# 	# display_braket_notation(stim_object.state_vector(), hightlight=data_qubits, hightlight_only=True)
			
		# 	# preparation of ancilla for check z
		# 	# assumption: noise free ancilla state 
		# 	stim_object, result, noise_frame = run_circuit_stim(stim_object,
		# 						collection_circuits.get("Prep-Anc-Zero").get("system_code"), error_rate=0.0,
		# 																					noise_frame=noise_frame)
			
		# 	# ic(result)
		# 	# print("After Preparation of Zero Ancilla : ")
		# 	# ic("Stabilizer : ", stim_object.current_inverse_tableau())
			
		# 	# qubit_mapping = collection_circuits.get("Prep-Anc-Zero").get("system_code").get("final_mapping")
		# 	# syndrome_qubits = [v for k, v in qubit_mapping.items() if "ancilla" in k]
		# 	# ic(syndrome_qubits)
		# 	# display_braket_notation(stim_object.state_vector(), hightlight=syndrome_qubits)

		# 	stim_object, result, noise_frame = run_circuit_stim(stim_object, 
		# 						collection_circuits.get("Check_Z").get("system_code"), error_rate=0.0, noise_frame=noise_frame)
		# 	# print("After Check for phase flip : ")
		# 	# ic("Stabilizer : ", stim_object.current_inverse_tableau())
		# 	# display_braket_notation(stim_object.state_vector(), hightlight=data_qubits+syndrome_qubits)

		# 	# 신드롬 측정 값 정렬
		# 	sorted_syndrome = sorted(result.items(), key = lambda item: item[0])
		# 	syndrome_vector = [v[1] for v in sorted_syndrome]
		# 	# str_syndrome = "".join([str(v[1]) for v in sorted_syndrome])
			
		# 	# ic(sorted_syndrome)
		# 	# ic(str_syndrome)

		# 	# readout_data[str_syndrome]+=1
			
			
		# 	# syndrome 가지고, 오류 추정 --> recovery operator
		# 	qubit_corrupted = decode_steane_method(syndrome_vector)
		# 	# ic(qubit_corrupted)
		# 	# apply recovery operator : Z gate on qubit_corrupted
		# 	if qubit_corrupted is not None:
		# 		stim_object.z(qubit_corrupted)

		# 	# print(stim_object.current_inverse_tableau())
			
		# 	# checkup process
		# 	# print("Checkup process to see any residual error ")
		# 	stim_object, result, noise_frame = run_circuit_stim(stim_object,
		# 						collection_circuits.get("Prep-Anc-Zero").get("system_code"), error_rate=0.0,
		# 																				noise_frame=noise_frame)
			
		# 	# print("After Preparation of Zero Ancilla : ")
		# 	# ic("Stabilizer : ", stim_object.current_inverse_tableau())
			
		# 	stim_object, result, noise_frame = run_circuit_stim(stim_object, 
		# 						collection_circuits.get("Check_Z").get("system_code"), error_rate=0.0, noise_frame=noise_frame)
			
		# 	# print("Checkup for phase flip")
		# 	# print(stim_object.current_inverse_tableau())
			
		# 	sorted_syndrome = sorted(result.items(), key = lambda item: item[0])
		# 	syndrome_vector = [v[1] for v in sorted_syndrome]

		# 	# syndrome 가지고, 오류 추정 --> recovery operator
		# 	qubit_corrupted = decode_steane_method(syndrome_vector)
			
		# 	# ic("Residual Error After QEC Recovery : ", qubit_corrupted)
		# 	# apply recovery operator : Z gate on qubit_corrupted

		# 	if qubit_corrupted is not None: fail += 1
		# 	# print(" ============================================================= ")
		
		# distribution[error_factor] = {"rate": 1-float(fail/total_iteration),
		# 							  "rejected" : rejected,
		# 							  "failure": fail}

		# print("for the error rate : {:>5}, the yield of preparation : {} (#fails = {}, #rejected = {})".format(
		# 		error_factor, 1-float(fail/total_iteration), fail, rejected))

	# # preparation of ancilla for check x
	# # assumption: noise free ancilla state 
	# stim_object, result = run_circuit_stim(stim_object, 
	# 					collection_circuits.get("Prep-Anc-Plus").get("system_code"), error_rate=0.0)
	
	# print("After Preparation of Zero Ancilla : ")
	# print(stim_object.current_inverse_tableau())

	# print("After Check  : ")
	# stim_object, result = run_circuit_stim(stim_object, 
	# 					collection_circuits.get("Check_X").get("system_code"), error_rate=0)
	# print(stim_object.current_inverse_tableau())
	# ic(result)
	
	# # syndrome 가지고 오류 추정 --> recovery operator
	# qubit_corrupted = decode_steane_method(list(result.values()))

	# # apply recovery operator
	# if qubit_corrupted is not None:
	# 	stim_object.x(qubit_corrupted)
	ic.enable()

def checkup_circuits(collection_circuits):
	for k, v in collection_circuits.items():
		qchip = v.get("qchip")

		analysis = ftcheckup.checkup_fault_tolerance(v.get("system_code"),
						{"width": qchip.get("dimension").get("width"), 
						"height": qchip.get("dimension").get("height")}, 
						write_file=True,
						mapping="ftqc",
						qchip_architecture=qchip.get("architecture"))

		ic(analysis)


if __name__ == "__main__":

	# Ideal HW Connectivity
	collection_circuits_files = {}
	collection_circuits_files[17] = {
		"Initialize-Zero" : "Initialize-Zero.json",
		"Initialize-Plus": "Initialize-Plus.json",
		"SM": "Measure-Syndrome.json",
		"Wait": "Wait.json"
	}


	collection_circuits = collections.defaultdict(dict)
	for arch, files in collection_circuits_files.items():
		for k, v in files.items():
			full_path = os.path.join("./DB-Circuits/Surface-7", v)
			raw_json_data = open(full_path).read()
			circuit_data = json.loads(raw_json_data)

			circuit_data_result = circuit_data.get("result")
			if circuit_data_result is None:
				circuit_data_result = circuit_data
				
			system_code = circuit_data_result.get("system_code")
			system_code["circuit"] = {int(k): v for k, v in system_code["circuit"].items()}
			circuit_data_result["system_code"] = system_code
			collection_circuits[arch][k] = circuit_data_result


		evaluate_logical_error(collection_circuits[17])