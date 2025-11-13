
import os
import sys
import copy
import simplejson as json
import numpy as np
import math
from datetime import datetime, date
from noise_channel import generate_random_error
from pauli_frame import pauli_vector, pauli_frame_rule
import collections
import stim
import itertools
from pprint import pprint
from icecream import ic
from progress.bar import Bar
import multiprocessing

sys.path.insert(0, "../support")

from simulate_stabilizer_circuit_stim import run_circuit_stim, inject_pauli_error, run_circuit_stim_physical
from simulate_steane_QEC import perform_steane_QEC
import send_email
import ftcheckup
import logical_circuits
from utility import display_braket_notation

list_prepare = ["PrepZ", "PrepX"]
list_unitary_1q = ["H", "I", "X"]
list_unitary_2q = ["CNOT", "SWAP", "hhCNOT", "vvCNOT", "vvSWAP", "hhSWAP"]
list_measure = ["MeasZ", "MeasX"]

list_error_rates = [1e-8, 1e-7, 1e-6, 1e-5, ]
total_iteration = 1e+1


def make_extended_qubit_table(logical_layout_size, layout_size, **kwargs):
	'''
		make extended qubit layout of size row x column
		each cell is composed of layout_size
	'''
	rows = logical_layout_size["rows"]
	cols = logical_layout_size["cols"]

	physical_rows = layout_size["height"]
	physical_cols = layout_size["width"]

	physical_qubit_table = collections.defaultdict(lambda: collections.defaultdict(dict))

	extended_row_length = physical_cols * cols

	for logical_index in itertools.product(list(range(rows)), list(range(cols))):
		for physical_index in itertools.product(list(range(physical_rows)), list(range(physical_cols))):
			
			# global index over extended qubit layout
			global_index = (logical_index[0] * physical_rows + physical_index[0]) * extended_row_length +\
				logical_index[1]*physical_cols + physical_index[1]
			
			# local index within a logical qubit block
			local_index = physical_index[0] * physical_cols + physical_index[1]

			physical_qubit_table[logical_index][local_index] = global_index
			
	
	return dict(physical_qubit_table)


def make_logical_qubit_mapping(logical_layout_size, logical_qubits):
	return {i : {"row": int(i/logical_layout_size["cols"]), "col": i%logical_layout_size["cols"]}
					for i in range(logical_qubits)}


def manage_evaluate_logical_circuit(collection_circuits, **kwargs):
	"""
		evaluation of multiple logical qubits circuit
	"""
	number_workers = kwargs.get("number_workers")
	if number_workers is None:
		number_workers = 1
	else:
		number_workers = int(number_workers)

	partial_iterations = math.ceil(total_iteration/number_workers)

	target_circuit = kwargs.get("target")
	list_conns = {}

	for worker_id in range(number_workers):
		parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
		list_conns[worker_id] = {"parent": parent_conn, "child": child_conn}

	
	# list_result = collections.defaultdict(float)
	list_result = {}
	
	# ghz state size : 2 ~ 5
	for i in range(6, 7):
		list_result[i] = collections.defaultdict(lambda: collections.defaultdict(float))

		# parallel jobs 설정
		jobs = [multiprocessing.Process(target=evaluate_logical_circuit,
										args=(collection_circuits, partial_iterations, i, target_circuit, list_conns[worker_id]["child"]))
				for worker_id in range(number_workers)]

		# parallel job 시작
		for job in jobs: job.start()

		flag_job_done = {k: False for k in range(number_workers)}
		list_partial_result = {}

		# 개별 worker 로 부터 결과 수신
		while True:
			for wid in range(number_workers):
				if list_conns[wid]["parent"].poll():
					list_partial_result[wid] = list_conns[wid]["parent"].recv()
					flag_job_done[wid] = True
			if all(list(flag_job_done.values())): break

		for job in jobs: job.join()

		# 결과 취합
		for k, v_list in list_partial_result.items():
			for a, b in v_list.items():
				for er, data in b.items():
					# i: ghz size
					# a: memory duration
					list_result[i][a][er] += data 

		# k: memory duration
		for k, v_list in list_result[i].items():
			for er, data in v_list.items():
				list_result[i][k][er] = (int(data), round(data/(partial_iterations*number_workers), 8))

	return list_result

		

def evaluate_logical_circuit(collection_circuits, partial_iterations, state_size, target_circuit, conn):
	'''
		evaluate the logical error rate of logical bell state

		in logical bell state, the logical error is defined as the decoded qubits are different
		such as logical_qubit[0] = 0 and logical_qubit[1] = 1 or vice versa
	'''
	state_size = 6
	qubits = {"rows": state_size, "cols": state_size, "length": 1}

	if target_circuit == "cnot_test":
		logical_circuit = logical_circuits.cnot_test_circuits[state_size]
	elif target_circuit == "swap_test":
		logical_circuit = logical_circuits.swap_test_circuits[state_size]
	elif target_circuit == "ghz_test":
		logical_circuit = logical_circuits.develop_ghz_circuit(state_size, state_size)
	elif target_circuit == "simple_test":
		logical_circuit = logical_circuits.simple_test_circuits[state_size]
	
	block_size = collection_circuits.get("Prep-Data-Zero").get("qchip").get("dimension")

	total_qubits = qubits["rows"] * qubits["cols"]
	
	# physical_qubit_table = make_extended_qubit_table3D(qubits, block_size)
	physical_qubit_table = make_extended_qubit_table(qubits, block_size)
	logical_qubit_table = make_logical_qubit_mapping(qubits, total_qubits)
	
	distribution =  collections.defaultdict(lambda: collections.defaultdict(float))
	statistics = collections.defaultdict(lambda: collections.defaultdict(int))

	circuit_depth = len(logical_circuit["circuit"].items())
	
	for error_factor in list_error_rates:
		fail = 0
		
		# circuit
		for j in range(partial_iterations):
			stim_object = stim.TableauSimulator()

			parity = {}

			for i in range(circuit_depth):
				for inst in logical_circuit["circuit"][i]:
					tokens = inst.split(" ")
					gate_information = tokens[0].split("-")
					gate = gate_information[-1]
					list_qubits = list(map(int, tokens[1].split(",")))
					
					if gate in list_unitary_2q:
						if gate == "vvCNOT":
							stim_object, result, noise_frame, error_source = run_circuit_stim(stim_object, 
								collection_circuits["vCNOT"]["system_code"], 
								error_rate=error_factor,
								global_qubit_mapping=physical_qubit_table, 
								logical_qubit_index=[(logical_qubit_table[list_qubits[0]]["row"], logical_qubit_table[list_qubits[0]]["col"]),
														(logical_qubit_table[list_qubits[1]]["row"], logical_qubit_table[list_qubits[1]]["col"])],
								logical_qubit_block_size=block_size,
								qubits_arrange="vertical")

						elif gate == "hhCNOT":
							stim_object, result, noise_frame, error_source = run_circuit_stim(stim_object, 
								collection_circuits["hCNOT"]["system_code"],
								error_rate=error_factor,
								global_qubit_mapping=physical_qubit_table, 
								logical_qubit_index=[(logical_qubit_table[list_qubits[0]]["row"], logical_qubit_table[list_qubits[0]]["col"]),
													(logical_qubit_table[list_qubits[1]]["row"], logical_qubit_table[list_qubits[1]]["col"])],
								logical_qubit_block_size=block_size,
								qubits_arrange="horizontal")

						elif gate == "hhSWAP":
							stim_object, result, noise_frame, error_source = run_circuit_stim(stim_object, 
								collection_circuits["hSWAP"]["system_code"],
								error_rate=error_factor,
								global_qubit_mapping=physical_qubit_table, 
								logical_qubit_index=[(logical_qubit_table[list_qubits[0]]["row"], logical_qubit_table[list_qubits[0]]["col"]),
													(logical_qubit_table[list_qubits[1]]["row"], logical_qubit_table[list_qubits[1]]["col"])],
								logical_qubit_block_size=block_size,
								qubits_arrange="horizontal")

						elif gate == "vvSWAP":
							stim_object, result, noise_frame, error_source = run_circuit_stim(stim_object, 
								collection_circuits["vSWAP"]["system_code"],
								error_rate=error_factor,
								global_qubit_mapping=physical_qubit_table, 
								logical_qubit_index=[(logical_qubit_table[list_qubits[0]]["row"], logical_qubit_table[list_qubits[0]]["col"]),
													(logical_qubit_table[list_qubits[1]]["row"], logical_qubit_table[list_qubits[1]]["col"])],
								logical_qubit_block_size=block_size,
								qubits_arrange="vertical")
						else:
							raise Exception("stop : {}".format(inst))

					elif gate in list_unitary_1q: 
						if gate == "H":
							stim_object, result, noise_frame, error_source = run_circuit_stim(stim_object, 
								collection_circuits["H"]["system_code"], 
								error_rate=error_factor,
								global_qubit_mapping=physical_qubit_table, 
								logical_qubit_index=[(logical_qubit_table[list_qubits[0]]["row"], logical_qubit_table[list_qubits[0]]["col"])])
						
						elif gate == "X":
							stim_object, result, noise_frame, error_source = run_circuit_stim(stim_object, 
								collection_circuits["X"]["system_code"], 
								error_rate=error_factor,
								global_qubit_mapping=physical_qubit_table, 
								logical_qubit_index=[(logical_qubit_table[list_qubits[0]]["row"], logical_qubit_table[list_qubits[0]]["col"])])
						
						elif gate == "I":
							stim_object, result, noise_frame, error_source = run_circuit_stim(stim_object, 
								collection_circuits["Identity"]["system_code"], 
								error_rate=error_factor,
								global_qubit_mapping=physical_qubit_table, 
								logical_qubit_index=[(logical_qubit_table[list_qubits[0]]["row"], logical_qubit_table[list_qubits[0]]["col"])])
						else:
							raise Exception("stop: {}".format(inst))

					elif gate in list_prepare:
						if gate == "PrepZ":
							stim_object, result, noise_frame, error_source = run_circuit_stim(stim_object, 
								collection_circuits["Prep-Data-Zero"]["system_code"], 
								error_rate=error_factor,
								global_qubit_mapping=physical_qubit_table, 
								logical_qubit_index=[(logical_qubit_table[list_qubits[0]]["row"], logical_qubit_table[list_qubits[0]]["col"])])

						elif gate == "PrepX": 
							stim_object, result, noise_frame, error_source = run_circuit_stim(stim_object, 
								collection_circuits["Prep-Data-Plus"]["system_code"], 
								error_rate=error_factor,
								global_qubit_mapping=physical_qubit_table, 
								logical_qubit_index=[(logical_qubit_table[list_qubits[0]]["row"], logical_qubit_table[list_qubits[0]]["col"])])

						else:
							raise Exception("stop : {}".format(inst))

					elif gate in list_measure: 
						if gate == "MeasZ":
							stim_object, result, noise_frame, error_source = run_circuit_stim(stim_object, 
								collection_circuits["MeasZ-Data"]["system_code"], 
								error_rate=error_factor,
								global_qubit_mapping=physical_qubit_table, 
								logical_qubit_index=[(logical_qubit_table[list_qubits[0]]["row"], logical_qubit_table[list_qubits[0]]["col"])])
						
						elif gate == "MeasX":
							stim_object, result, noise_frame, error_source = run_circuit_stim(stim_object, 
								collection_circuits["MeasX-Data"]["system_code"], 
								error_rate=error_factor,
								global_qubit_mapping=physical_qubit_table, 
								logical_qubit_index=[(logical_qubit_table[list_qubits[0]]["row"], logical_qubit_table[list_qubits[0]]["col"])])

						else:
							raise Exception("stop : {}".format(inst))

					elif gate in ["Waiting"]:
						stim_object, result, noise_frame, error_source = run_circuit_stim(stim_object, 
							collection_circuits["Identity"]["system_code"], 
							error_rate=error_factor,
							global_qubit_mapping=physical_qubit_table, 
							logical_qubit_index=[(logical_qubit_table[list_qubits[0]]["row"], logical_qubit_table[list_qubits[0]]["col"])])

					else:
						raise Exception("stop : {}".format(inst))
				
				# QEC in batch: 큐비트 레이블링 문제가 있음
				for pair in itertools.product(range(qubits["rows"]), range(qubits["cols"])):
					qubit_index = pair[0] * qubits["cols"] + pair[1]
					perform_steane_QEC(stim_object, error_factor, collection_circuits, noise_frame, 
						global_qubit_mapping=physical_qubit_table, 
						logical_qubit_index=[(logical_qubit_table[qubit_index]["row"], logical_qubit_table[qubit_index]["col"])])

			# 회로 실행 후, 측정을 통해서, 예상 기대값과 일치 여부 확인 : perform the circuit in noise free
			for pair in itertools.product(range(qubits["rows"]), range(qubits["cols"])):
				qubit_index = pair[0] * qubits["cols"] + pair[1]
				stim_object, result, noise_frame, error_source = run_circuit_stim(stim_object, 
					collection_circuits.get("MeasZ-Data").get("system_code"), 
					error_rate=0,
					global_qubit_mapping=physical_qubit_table, 
					logical_qubit_index=[(logical_qubit_table[qubit_index]["row"], logical_qubit_table[qubit_index]["col"])])
				
				# print(result.values())
				parity[(pair[0], pair[1])] = sum(result.values()) % 2
			
			# 두 큐비트에 대한 측정 결과 저장
			

			statistics[error_factor][tuple(parity.values())] += 1
						
	conn.send(dict(statistics))


def manage_evaluate_logical_qubit(collection_circuits, **kwargs):
	"""
		evaluation of logical qubit preparation
	"""
	number_workers = kwargs.get("number_workers")
	
	if number_workers is None:
		number_workers = 1
	else:
		number_workers = int(number_workers)

	partial_iterations = math.ceil(total_iteration / number_workers)
	
	logical_state = kwargs.get("state")

	list_conns = {}	
	
	for worker_id in range(number_workers):
		parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
		list_conns[worker_id] = {"parent": parent_conn, "child": child_conn}

	# 대상 함수 설정
	# evaluate_yield_of_preparation : logical qubit
	# evaluate_physical_qubit : physical qubit
	jobs = [multiprocessing.Process(target=evaluate_yield_of_preparation,
									args=(collection_circuits, partial_iterations, logical_state, list_conns[worker_id]["child"]))
			for worker_id in range(number_workers)]

	for job in jobs: job.start()

	flag_job_done = {k: False for k in range(number_workers)}
	list_partial_result = {}

	# worker 에서 작업 수행 후, 각 worker 로 부터 데이터 수신
	while True:
		for wid in range(number_workers):
			if list_conns[wid]["parent"].poll():
				list_partial_result[wid] = list_conns[wid]["parent"].recv()
				flag_job_done[wid] = True
		if all(list(flag_job_done.values())): break

	for job in jobs: job.join()

	# 각 worker 로 부터 수신한 데이터 정리
	total_data = {}
	for m_time, data in list_partial_result[0].items():
		total_data[m_time] = {}
		for er, performance in data.items():
			total_data[m_time][er] = {"fails": 0, "logical_error_rate": 0, 
									  "trials": 0, "fidelity": 0,
									  "syndrome_error": 0}

	for k, v_list in list_partial_result.items():
		for m_time, data_list in v_list.items():
			for er, data in data_list.items():
				for a, b in data.items():
					total_data[m_time][er][a] += b

	for m_time, data_list in total_data.items():
		for er, data in data_list.items():
			for a, b in data.items():
				total_data[m_time][er][a] = round(b/number_workers, 8)

	return total_data


def evaluate_yield_of_preparation(collection_circuits, partial_iterations, logical_state, conn):
	"""
		logical error 기준 :
			1. checkup after QEC
			2. measure logical data after QEC
	"""
	distribution = collections.defaultdict(lambda: collections.defaultdict(float))
	statistics = collections.defaultdict(lambda: collections.defaultdict(int))

	protocols = {}
	if logical_state == "zero":
		protocols = {"preparation": "Prep-Data-Zero",
					 "measurement": "MeasZ-Data"}
	elif logical_state == "plus":
		protocols = {"preparation": "Prep-Data-Plus",
					 "measurement": "MeasX-Data"}

	list_dominant_failure = collections.defaultdict(list)
	for error_factor in list_error_rates:
		distribution_data = collections.defaultdict(int)

		# error correction failed
		for time_limit in range(1):
			fail = 0

			global_syndrome_error = 0
			dist_checkup = collections.defaultdict(int)
			
			for j in range(partial_iterations):
				stim_object = stim.TableauSimulator()
				
				# preparation of logical data until the checkup value is 0
				trial = 0
				# state preparation
				while True:
					trial+=1
					stim_object, result, noise_frame, error_source = run_circuit_stim(stim_object,
						collection_circuits.get(protocols["preparation"]).get("system_code"), 
						error_rate=error_factor, limit_error=1)

					dist_checkup[result.get("checkup0")]+=1

					# if result.get("checkup0") == 1: 
					# 	list_dominant_failure[error_factor].append(error_source)

					if result.get("checkup0") == 0: break

				
				statistics[error_factor][trial] += 1

	# perform QEC 
				# steane-method QEC 루틴 호출 (stim object, error_factor, 회로 셋, noise frame)
				# error_factor : 0 for noiseless QEC 
				# 				 error_factor for noisy QEC
				syndrome_error = perform_steane_QEC(stim_object, error_factor, collection_circuits, noise_frame)
				global_syndrome_error+=syndrome_error

				# qubit_mapping = collection_circuits.get("Prep-Anc-Plus").get("system_code").get("final_mapping")
				# display_braket_notation(stim_object.state_vector(), hightlight=[v for k, v in qubit_mapping.items()
				# if "data" in k], message="quiscent state")

	# memory during "time_limit" time
				# for i in range(time_limit*100):
				# 	# memory : Identity
				# 	stim_object, result, noise_frame, error_source = run_circuit_stim(stim_object, 
				# 							collection_circuits.get("Identity").get("system_code"), 
				# 							error_rate=error_factor)
				
				# 	# QEC
				# 	perform_steane_QEC(stim_object, 0, collection_circuits, noise_frame)

	# logical measurement : to check the final state in noise free
				stim_object, result, noise_frame, error_source = run_circuit_stim(stim_object,
						collection_circuits.get(protocols["measurement"]).get("system_code"), error_rate=0)
				
				parity = sum(result.values()) % 2
				if parity: fail += 1
			
			average_trials = sum([k*v for k, v in statistics[error_factor].items()]) / partial_iterations

			distribution[time_limit][error_factor] = {"logical_error_rate": float(fail/partial_iterations),
													  "fidelity": 1-float(fail/partial_iterations),
													  "trials" : average_trials,
													  "fails": fail,
													  "syndrome_error": float(global_syndrome_error/(2*partial_iterations))}
	conn.send(dict(distribution))


def checkup_circuits(collection_circuits):
	for k, v in collection_circuits.items():
		qchip = v.get("qchip")
		ic(k, v)
		analysis = ftcheckup.checkup_fault_tolerance(v.get("system_code"),
						{"width": qchip.get("dimension").get("width"), 
						"height": qchip.get("dimension").get("height")}, 
						write_file=True,
						mapping="ftqc",
						qchip_architecture=qchip.get("architecture"))

		ic(analysis)


if __name__ == "__main__":
	import time
	start = time.perf_counter()

	# Ideal HW Connectivity
	collection_circuits_files = {}

	# all-to-all connectivity
	collection_circuits_files[0] = {
		"Prep-Data-Zero" : "PrepZ-Data-Zero-0.json",
		"Prep-Data-Plus" : "PrepZ-Data-Plus-0.json",
		"Prep-Anc-Zero": "PrepZ-Anc-Zero-0.json",
		"Prep-Anc-Plus": "PrepZ-Anc-Plus-0.json",
		"Check_X": "Check-X-0.json",
		"Check_Z": "Check-Z-0.json",
		"MeasZ-Data": "MeasZ-Data-0.json",
		"MeasX-Data": "MeasX-Data-0.json",
		"vCNOT" : "CNOT-0.json",
		"hCNOT" : "CNOT-0.json",
		"hSWAP" : "SWAP-0.json",
		"vSWAP" : "SWAP-0.json",
		"Identity": "Identity-0.json",
		"H" : "Hadamard-0.json",
		"X" : "Pauli-X-0.json"
	}

	# real 2D connectivity lattice
	collection_circuits_files[2] = {
		"Prep-Data-Plus" : "DQP-file_data_plus_2_(5,6)_315.json",
		"Prep-Data-Zero" : "DQP-file_data_zero_2_(5,6)_315.json",
		"Prep-Anc-Plus": "DQP-file_anc_plus_2_(5,6)_315.json",
		"Prep-Anc-Zero": "DQP-file_anc_zero_2_(5,6)_315.json",
		"Check_X": "DQP-file_check-x_2_(5,6)_315.json",
		"Check_Z": "DQP-file_check-z_2_(5,6)_315.json",
		"MeasZ-Data": "MeasZ_data_2.json",
		"MeasX-Data": "MeasX_data_2.json",
		"Identity": "Identity_data_2.json",
		"H" : "Hadamard_data_2.json",
		"X" : "Pauli-X_data_2.json",
		"hCNOT" : "cnot-H_2_5x12_329.json",
		"vCNOT" : "cnot-V_2_10x6_203.json",
		"hSWAP" : "swap-H_2_5x12_329.json",
		"vSWAP" : "swap-V_2_10x6_203.json",	
	}

	# 2d triangular : 2D arch 와 동일한 큐비트 크기
	collection_circuits_files[23] = {
		"Prep-Data-Plus" : "DQP-file_data_plus_23_(5,6)_296.json",
		"Prep-Data-Zero" : "DQP-file_data_zero_23_(5,6)_296.json",
		"Prep-Anc-Plus": "DQP-file_anc_plus_23_(5,6)_296.json",
		"Prep-Anc-Zero": "DQP-file_anc_zero_23_(5,6)_296.json",
		"Check_X": "DQP-file_check-x_23_(5,6)_296.json",
		"Check_Z": "DQP-file_check-z_23_(5,6)_296.json",
		"MeasZ-Data": "MeasZ_data_23.json",
		"MeasX-Data": "MeasX_data_23.json",
		"Identity": "Identity_data_23.json",
		"H" : "Hadamard_data_23.json",
		"X" : "Pauli-X_data_23.json",
		"hCNOT" : "cnot-H_23_5x12_245.json",
		"vCNOT" : "cnot-V_23_10x6_105.json",
		"hSWAP" : "swap-H_23_5x12_245.json",
		"vSWAP" : "swap-V_23_10x6_105.json",
	}

	# 3d cube
	collection_circuits_files[3] = {
		"Prep-Data-Plus" : "DQP-file_data_plus_3_(4,3,3)_251.json",
		"Prep-Data-Zero" : "DQP-file_data_zero_3_(4,3,3)_251.json",
		"Prep-Anc-Plus": "DQP-file_anc_plus_3_(4,3,3)_251.json",
		"Prep-Anc-Zero": "DQP-file_anc_zero_3_(4,3,3)_251.json",
		"Check_X": "DQP-file_check-x_3_(4,3,3)_251.json",
		"Check_Z": "DQP-file_check-z_3_(4,3,3)_251.json",
		"MeasZ-Data": "MeasZ_data_3.json",
		"MeasX-Data": "MeasX_data_3.json",
		"Identity": "Identity_data_3.json",
		"H" : "Hadamard_data_3.json",
		"hCNOT" : "cnot-H_3_4x6x3_105.json",
		"vCNOT" : "cnot-V_3_8x3x3_91.json",
		"hSWAP" : "swap-H_3_4x6x3_105.json",
		"vSWAP" : "swap-V_3_8x3x3_91.json",
	}

	collection_circuits = collections.defaultdict(dict)
	for arch, files in collection_circuits_files.items():
		for k, v in files.items():
			full_path = os.path.join("./DB-Circuits/steane_code_circuits/{}-Arch".format(str(arch)), v)
			raw_json_data = open(full_path).read()
			circuit_data = json.loads(raw_json_data)

			circuit_data_result = circuit_data["result"]
			if circuit_data_result is None:
				circuit_data_result = circuit_data

			system_code = circuit_data_result["system_code"]
			
			system_code["circuit"] = {int(k): v for k, v in system_code["circuit"].items()}
			circuit_data_result["system_code"] = system_code
			collection_circuits[arch][k] = circuit_data_result

	# preparation of logical qubit : 0, 2, 23, 3
	# logical ghz state, swap test: 0, 2, 23
	performance = collections.defaultdict(dict)
	for arch in [2]:
		print("Quantum Chip Architecture : {}".format(arch))
		# checkup_circuits(collection_circuits[arch])
		
		# analyze of single qubit logical error rate 
		# for state in ["plus", "zero"]:
		# 	# evaluate the yield of preparation
		# 	performance[arch][state] = manage_evaluate_logical_qubit(collection_circuits[arch], state=state, number_workers=1)
		
		# multi-qubit logical circuits
		for target_circuit in ["ghz_test"]:
			print("{} test on {}-Arch".format(target_circuit, arch))
			performance[arch][target_circuit] = manage_evaluate_logical_circuit(collection_circuits[arch], 
				target=target_circuit, number_workers=10)
		
		ic(performance[arch])
	
	performance = str(performance)
	today = date.today()

	# adjust the file path & name 
	target_file = os.path.join(".", "file_ghz_states_performance-{}.json".format(today))
	
	with open(target_file, "w") as outfile:
		json.dump(performance, outfile, sort_keys=True, indent=4, separators=(',', ':'))
	
	# file 전달
	send_email.send_mail_with_attachments("email.etri.re.kr", 465, "yhwang@etri.re.kr", "tvqqxhrswca5qag",
		"yhwang@etri.re.kr", "yshwang7026@gmail.com", "simulation results", "logical ghz state simulation result",
		attachments=[target_file])

	end = time.perf_counter()
	print("running time : {} sec".format(end-start))
