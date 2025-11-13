
import os
import sys
import copy
import simplejson as json
import numpy as np
import random
import pandas
from noise_channel import generate_random_error
from pauli_frame import pauli_vector, pauli_frame_rule
from decode_surface17_tomita import manage_decode_surface_tomita
import collections
import multiprocessing
import stim
import itertools
from pprint import pprint
from icecream import ic
from progress.bar import Bar
import time
import math
from pprint import pprint

sys.path.insert(0, "../support")

import ftcheckup

# package for simulating stabilizer circuit with stim package
from simulate_stabilizer_circuit_stim import run_circuit_stim

pauli_gate = {(0, 0) : "i", (1, 0): "x", (0, 1):"y", (1, 1):"z"}

# surface 7 2d lattice 좌표 (x, z) 와 큐비트 인덱스간의 association
# reference : PRR 5, 033019 (2023)
lattice_mapping_d2 = {(0,0): 4, (0,1): 0,
					  (1,0): 2, (1,1):5, (1,2):1,
					  (2,1): 3, (2,2): 6}

# surface 17 2d lattice 좌표 (x, z) 와 큐비트 인덱스간의 association
# reference : PRA 90, 062320
lattice_mapping_d3 = {(0,1):9, (0,2): 2,
				  	  (1,1):1, (1,2): 12, (1,3): 5, (1, 4): 15,
				  	  (2,0):0, (2,1): 11, (2,2): 4, (2,3): 14, (2,4): 8,
				  	  (3,0):10, (3,1): 3, (3,2): 13, (3,3): 7,
				  	  (4,2):6, (4,3): 16}


code_distance = int(sys.argv[1].split("=")[1])
task_job = sys.argv[2].split("=")[1]

if code_distance == 3:
	lattice_mapping = lattice_mapping_d3
	code_name = "Surface-17"

elif code_distance == 2:
	lattice_mapping = lattice_mapping_d2
	code_name = "Surface-7"


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, set):
            return list(obj)  # `set`을 `list`로 변환
        return super().default(obj)


def initialize_qubit_relation(collection_circuits):
	# qubit mapping 읽고,
	# check qubit - data qubit 연결성 확인
	qubit_mapping = collection_circuits.get("SM").get("system_code").get("initial_mapping")
	inverse_mapping = {v: k for k, v in qubit_mapping.items()}

	qubit_connectivity = {int(k): v for k, v in 
		collection_circuits.get("SM").get("qchip").get("qubit_connectivity").items()}

	qubit_relation = collections.defaultdict(list)
	for k, v in qubit_mapping.items():
		if "check" in k:
			qubit_relation[k].extend([inverse_mapping[q] for q in qubit_connectivity[v]])

	return qubit_relation


def manage_surface_code_parallel_simulation(collection_circuits, **kwargs):
	"""
		manage running surface code simulation in parallel
	"""
	task_function = {"evaluate_logical_error":evaluate_logical_error, 
					"find_pattern": find_pattern_error_syndrome}

	number_workers = kwargs.get("number_workers")
	if number_workers is None:
		number_workers = 1
	else:
		number_workers = int(number_workers)

	task = kwargs.get("task")
	if task is None:
		task = "evaluate_logical_error"
		task_
	print(" the number of workers = {}".format(number_workers))

	total_coverage = 100
	worker_coverage = math.ceil(total_coverage / number_workers)


	list_conns = {}
	for worker_id in range(number_workers):
		parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
		list_conns[worker_id] = {"parent": parent_conn, "child": child_conn}

	jobs = [multiprocessing.Process(target=task_function[task],
									args=(collection_circuits, worker_coverage, worker_id, list_conns[worker_id]["child"]))
			for worker_id in range(number_workers)]

	# worker 작업으로 부터.. 결과 데이터 취합
	partial_solution = {}
	flag_job_done = {k: False for k in range(number_workers)}

	for job in jobs: job.start()
	while True:
		for worker_id in range(number_workers):
			if list_conns[worker_id]["parent"].poll():
				partial_solution[worker_id] = list_conns[worker_id]["parent"].recv()
				flag_job_done[worker_id] = True

		if all(list(flag_job_done.values())): break

	for job in jobs: job.join()

	# 작업 종류별로 후처리 수행
	# find_pattern : 데이터 단순 취합 (list append)
	if task == "find_pattern":
		total_solution = collections.defaultdict(list)
	
		for k, v_list in partial_solution.items():
			for syndrome, recovery in v_list.items():
				for r in recovery:
					if r not in total_solution[syndrome]:
						total_solution[syndrome].append(r)
	
	# evaluate_logical_error : 데이터 취합 및 계산 (나누기 by 총 샘플링 횟수)
	elif task == "evaluate_logical_error":
		total_solution = collections.defaultdict(lambda: collections.defaultdict(int))
		for k, v_list in partial_solution.items():
			for error_rate, period in v_list.items():
				for a, b in period.items():
					total_solution[error_rate][a]+=b
		
		for k, v_list in total_solution.items():
			for a, b in v_list.items():
				total_solution[k][a] = b/(number_workers*worker_coverage)
	
	# file id and writing the collected data into the file
	now = time.localtime()
	current_time = "%04d%02d%02d%02d%02d%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)

	file = "surface17_simulation_data-{}-{}.json".format(task, current_time)

	with open(file, "w") as outfile:
		json.dump(total_solution, outfile, sort_keys=True, indent=4, separators=(',', ':'), cls=CustomJSONEncoder)

	print("We have completed writing file {}.".format(file))


def find_pattern_error_syndrome(collection_circuits, worker_coverage, worker_id, conn):
	"""

	"""
	qubit_relation = initialize_qubit_relation(collection_circuits)
	temp_association = collections.defaultdict(list)

	qec_duration = 3

	while len(temp_association) <= worker_coverage:
		flag, ret = run_surface_code_qec(collection_circuits, qubit_relation, qec_duration, error_rate=0.005,)
		
		if flag and ret is not None:
			for k, v_list in ret.items():
				for v in v_list:
					if v not in temp_association[str(k)]:
						temp_association[str(k)].append(v)
	
	# 이단계에서 기대하는 것은 아래와 같이 2단계임. 따라서.. 단계를 높이는 append 보다 extend 가 맞다고 생각됨
	# a : 	[
	#			[(), ()..]
	#			[(), ()..]
	# 		]
	conn.send(temp_association)



def evaluate_logical_error(collection_circuits, worker_coverage, worker_id, conn):
	"""
		worker 별로 물리 오류율에 대해서 특정 iteration 이후에 상태가 유지되는지 확인하는 과정
	"""
	qubit_relation = initialize_qubit_relation(collection_circuits)

	duration = [1, 3, 5, 7, 9]
	error_rates = [0.001, 0.003, 0.005, 0.007]
	
	temp_data = collections.defaultdict(lambda: collections.defaultdict(int))
	for pair in itertools.product(error_rates, duration):
		for idx in range(worker_coverage):
			ret = run_surface_code_qec(collection_circuits, qubit_relation, pair[1], error_rate=pair[0])
		
			if ret[0]:
				temp_data[pair[0]][pair[1]] += 1

	temp_data = dict(temp_data)
	
	conn.send(temp_data)


def run_surface_code_qec(collection_circuits, qubit_relation, total_iteration, **kwargs):
	"""
		single run of surface code quantum error correction circuit
	"""
	error_rate = kwargs.get("error_rate")

	# qubit_relation : the association of check qubits to data qubits
	qubit_relation = initialize_qubit_relation(collection_circuits)

	# qubit mapping
	qubit_mapping = collection_circuits.get("SM").get("system_code").get("initial_mapping")
	
	# stim object for stabilizer circuit simulation
	stim_object = stim.TableauSimulator()

	############################################## 
	# make quiescent state
	##############################################

	# 1) initialize all qubits to be zero state and
	stim_object, result, noise_frame, error_source = run_circuit_stim(stim_object,
						collection_circuits.get("Initialize-Zero").get("system_code"), error_rate=0)

	# 2) make noise-free syndrome measurement
	stim_object, result, noise_frame, error_source = run_circuit_stim(stim_object,
						collection_circuits.get("SM").get("system_code"), error_rate=0,
						noise_frame=noise_frame)

	display_braket_notation(stim_object.state_vector(), hightlight=[v for k, v in qubit_mapping.items()
		if "data" in k], message="quiscent state")

	# quiscent state
	quiscent_state = copy.deepcopy(stim_object.state_vector())

	# reference state for checking the manipulated state later
	reference_state = copy.deepcopy(quiscent_state)

	reference_syndrome = result
	# print("initial reference syndrome : ", reference_syndrome)

	syndrome = {}
	global_error_source = collections.defaultdict(list)

	# pauli frame table : all "i" for data qubits
	pauli_frame_table = {v : 'i' for k, v in qubit_mapping.items() if "data" in k}
	
	table_association = collections.defaultdict(list)
	
	# print("============================================= =============================================")
	
	for i in range(total_iteration):
		# 첫번째 라운드에는 code distance 만큼 신드롬 측정 반복
		# 두번째 부터는 code distance - 1 만큼 반복 (이때, 신드롬 볼륨의 첫번째 layer 는 이전 라운드의 마지막 layer 값으로..)
		# print(" {} - th round.. ".format(i))

		if i == 0:
			# 첫번째 iteration 에서는 syndrome volume 이 empty
			sm_iteration = code_distance
			syndrome_idx = 0
		
		else:
			sm_iteration = code_distance-1
			# 신드롬은 이전 라운드의 마지막 것을 이번 라운드의 첫번째 것으로..
			syndrome.update({0: syndrome[code_distance-1]})
			for d in range(1, code_distance): 
				syndrome.update({d: None})

			syndrome_idx = 1

		# print(" ------------------------------------- ------------------------------------- ")

		if i < total_iteration - 1:
			# conducting the syndrome measurements as much as d rounds
			for sm_idx in range(syndrome_idx, code_distance):
				# print("sm index : ", sm_idx)
				# ic(global_error_source)

				# syndrome measure
				stim_object, readout, noise_frame, error_source = run_circuit_stim(stim_object,
								collection_circuits.get("SM").get("system_code"), error_rate=error_rate,
								noise_frame=noise_frame, skip_qsp=False)

				# display_braket_notation(stim_object.state_vector())
				global_error_source[i].extend(error_source)
				
				# syndrome by making difference between the readouts (previous and current)
				syndrome[sm_idx] = {k: (v+readout.get(k))%2 for k, v in reference_syndrome.items()}
				reference_syndrome = readout

				# display_lattice(noise_frame, syndrome[sm_idx], collection_circuits.get("SM").get("system_code").get("initial_mapping"))
		else:
			# 정해진 QEC 라운드 모두 실행하고 나서, 마지막으로 noise-free QEC 한 번 더..	
			for sm_idx in range(syndrome_idx, code_distance):
				stim_object, readout, noise_frame, error_source = run_circuit_stim(stim_object,
								collection_circuits.get("SM").get("system_code"), error_rate=0,
								noise_frame=noise_frame, skip_qsp=False)

				syndrome[sm_idx] = {k: (v+readout.get(k))%2 for k, v in reference_syndrome.items()}
				reference_syndrome = readout
			
				# display_lattice(noise_frame, syndrome[sm_idx], collection_circuits.get("SM").get("system_code").get("initial_mapping"))


		# 신드롬 볼륨 만들기 of size code_distance x number_checks
		number_checks = collections.defaultdict(int)
		for k in qubit_relation.keys():
			if "checkX" in k: number_checks["X"]+=1
			if "checkZ" in k: number_checks["Z"]+=1

		syndrome_volume = {"X": np.zeros((code_distance, number_checks["X"]), dtype=bool),
						   "Z": np.zeros((code_distance, number_checks["Z"]), dtype=bool)}
		
		for tidx, v_list in syndrome.items():
			for check, v in v_list.items():
				check_type, check_idx = check[-2:]
				syndrome_volume[check_type][tidx][int(check_idx)] = bool(v)

		# decode based on tomita's lookup table
		# option #1 : y correlation
		# option #2 : multi qubit noise instead of 1-qubit noise
		recovery = manage_decode_surface_tomita(syndrome_volume, qubit_relation, 
			flag_y_correlation=False,
			flag_multiqubit_error=True)
		
		# return 되는 recovery form 은 설계한대로 정확함
		# for example, [('readout', 'checkX1'), ('X', 'data0')]

		# update the table association (syndrome : error) from the obtained recovery
		if recovery is not None:
			list_syndrome = [tuple(v.values()) for v in syndrome.values()]
			
			# 신드롬 a 에 대한 recovery 를 추가
			# 동일한 신드롬에 대해서 여러 recovery 존재 가능하므로.. 리스트에 추가
			if recovery not in table_association[tuple(list_syndrome)]:
				table_association[tuple(list_syndrome)].append(recovery)
		
		# 이 단계에서 원하는 형태는 아래와 같음. 2계층
		# [
		# 	[(..), (..)]
		# 	[(..), (..)]
		# ]

		# syndrome 업데이트
		for type in ["X", "Z"]:
			syndrome_volume_size = syndrome_volume[type].shape[:]
			for pair in itertools.product(range(syndrome_volume_size[0]), range(syndrome_volume_size[1])):
				syndrome_value = int(syndrome_volume[type][pair[0]][pair[1]])
				syndrome[pair[0]].update({"check{}{}".format(type, pair[1]): syndrome_value})
			
		# recovery 에 따라 recovery
		# option 1: update pauli frame and physical gate at the end
		# option 2: physical recovery
		if recovery is not None:
			for r in recovery:
				gate, qubit_label = r[:]
				
				if gate in ["Readout", "Next-Window"]: continue

				physical_qubit_index = qubit_mapping[qubit_label]
	
				# recovery option 1: update pauli frame 
				pauli_frame_table[physical_qubit_index] = pauli_frame_rule[
					(pauli_frame_table[physical_qubit_index], gate.lower())][1]
				
				# recovery option 2: physical recovery
				# parsed_gate = getattr(stim_object, gate.lower())
				# parsed_gate(qubit_mapping[qubit_label])
	
				
		# _ = display_braket_notation(stim_object.state_vector(), hightlight=[v for k, v in qubit_mapping.items()
		# 	if "data" in k], get_selection=True)

	# after the entire QEC rounds
	# print("pauli frame table : {}".format(pauli_frame_table))
	
	# print("Before the recovery : ")
	# display_braket_notation(stim_object.state_vector(), hightlight=[v for k, v in qubit_mapping.items()
	# 		if "data" in k])

	# final (before non-Clifford gate operation)
	# QEC recovery based on the pauli frame table
	for k, v in pauli_frame_table.items():
		if v in ["i", "I"]: continue
		parsed_gate = getattr(stim_object, v.lower())
		parsed_gate(k)
	
	# print("after the recovery : ")
	# display_braket_notation(stim_object.state_vector(), hightlight=[v for k, v in qubit_mapping.items()
	# 		if "data" in k])

	# verification by computing the inner product of the data qubits
	# the quantum state is correctly recovered
	# recall the initially prepared state
	# this is only possible with this SW approach
	stim_object2 = stim.TableauSimulator()
	stim_object2.set_state_from_state_vector(list(quiscent_state), endian='little')

	# reset over ancilla qubits to check
	for k in qubit_relation.keys():
		physical_qubit_index = qubit_mapping[k]
		stim_object.reset_z(physical_qubit_index)
		stim_object2.reset_z(physical_qubit_index)

	# ic("recovered state: ")
	# display_braket_notation(stim_object.state_vector(), hightlight=[v for k, v in qubit_mapping.items()
	# 	if "data" in k])

	# ic("prepared state: ")
	# display_braket_notation(stim_object2.state_vector(), hightlight=[v for k, v in qubit_mapping.items()
	# 	if "data" in k])

	recovered_state = stim_object.state_vector()
	prepared_state = stim_object2.state_vector()

	checkup = np.absolute(np.inner(prepared_state, recovered_state))
	
	# the logical state has been well preserved if checkup is 1
	# otherwise, it was broken.
	if checkup == 1: 
		return True, table_association

	else:
		return False, None



def display_lattice(noise, syndrome, qubit_mapping):
	
	inverse_mapping = {v: k for k, v in qubit_mapping.items()}
	lattice = [[None for i in range(5)] for j in range(5)]

	syndrome = dict(syndrome)
	for i in range(5):
		for j in range(5):
			associated_qubit = lattice_mapping.get((i, j))
			if associated_qubit is not None:
				logical_qubit = inverse_mapping[associated_qubit]
				if "check" in logical_qubit:
					lattice[i][j] = (logical_qubit, (-1)**syndrome[logical_qubit], pauli_gate[tuple(noise[associated_qubit])])
				else:
					lattice[i][j] = (logical_qubit, pauli_gate[tuple(noise[associated_qubit])])

	print("-----------------------------------")
	print(pandas.DataFrame(lattice).to_string())
	print("-----------------------------------")


def display_pauli_gate_sequence(binary_vector):
	"""
		display the binary symplectic product as pauli gate
	"""
	vector_size = binary_vector.shape[:]
	str_pauli_sequence = []
	
	for i in range(vector_size[0]):
		if binary_vector[i][0]:
			if binary_vector[i][1]: 
				str_pauli_sequence.extend("Y")
			else:
				str_pauli_sequence.extend("X")
		else:
			if binary_vector[i][1]: 
				str_pauli_sequence.extend("Z")
			else:
				str_pauli_sequence.extend("I")

	return " ".join(str_pauli_sequence)


def display_braket_notation(state_vector, **kwargs):
	"""
		function to display a state vector with braket notation
	"""	
	hightlight = kwargs.get("hightlight")
	flag_hightlight_only = kwargs.get("hightlight_only")
	flag_get_selection = kwargs.get("get_selection")
	msg = kwargs.get("message")

	dimension = state_vector.shape[0]
	size_register = int(np.log2(dimension))

	if msg is not None:
		print(" {} ".format(msg))
	if hightlight is None:
		for i in range(dimension):
			observable = "".join(list(np.binary_repr(i, size_register)))
			if np.abs(state_vector.item(i)) > 0:
				print("\t{0:>50} | {1} > ".format(str(state_vector.item(i)), observable))
				# print("\t{0:>50} | {1} > \t{2}".format(str(state_vector.item(i)), observable,
				# 			np.abs(state_vector.item(i))**2))
	
	elif flag_get_selection:
		interested_states = []
		for i in range(dimension):
			observable = "".join(list(np.binary_repr(i, size_register)))
			if np.abs(state_vector.item(i)) > 0:
				hightlighted_observable = []
				selection = []
				for idx, bit in enumerate(observable):
					# 출력 순서가 반대이니까.. idx 대신, size_qubits-1-idx 
					if (size_register - idx - 1) in hightlight:
						hightlighted_observable.append("\033[95m{}\033[0m".format(bit))
						selection.append(bit)
					else:
						if not flag_hightlight_only:
							hightlighted_observable.append("{}".format(bit))

				amplitude = str(state_vector.item(i))
				observable = "".join(selection)
				interested_states.append((amplitude, observable))
				print("\t{0:>20} | {1} >".format(str(state_vector.item(i)), "".join(hightlighted_observable)))
				# print("\t{0:>20} | {1} > \t{2}".format(str(state_vector.item(i)), "".join(hightlighted_observable),
				# 			np.abs(state_vector.item(i))**2))
		return interested_states

	elif hightlight:
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

				# print("\t{0:>20} | {1} > \t{2}".format(str(state_vector.item(i)), "".join(hightlighted_observable),
				# 			np.abs(state_vector.item(i))**2))
				print("\t{0:>20} | {1} >".format(str(state_vector.item(i)), "".join(hightlighted_observable)))


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

	collection_circuits_files = {}
	collection_circuits_files[17] = {
		"Initialize-Zero" : "Initialize-Zero.json",
		"Initialize-Plus": "Initialize-Plus.json",
		# "SM": "Measure-Syndrome.json",
		"SM": "Measure-Syndrome_improved_CNOT_order.json",
		"Wait": "Wait.json"
	}


	collection_circuits = collections.defaultdict(dict)
	for arch, files in collection_circuits_files.items():
		for k, v in files.items():
			full_path = os.path.join("./DB-Circuits/{}".format(code_name), v)
			raw_json_data = open(full_path).read()
			circuit_data = json.loads(raw_json_data)

			circuit_data_result = circuit_data.get("result")
			if circuit_data_result is None:
				circuit_data_result = circuit_data
				
			system_code = circuit_data_result.get("system_code")
			system_code["circuit"] = {int(k): v for k, v in system_code["circuit"].items()}
			circuit_data_result["system_code"] = system_code
			collection_circuits[arch][k] = circuit_data_result

	# task : what kind of a task this software does
	# 1. find_pattern : find pairs of an error syndrome and an inferred recovery by decoding
	# 2. evaluate_logical_error : evaluate the logical error rate
	
	manage_surface_code_parallel_simulation(collection_circuits[17], 
		task=task_job,
		number_workers=os.cpu_count()-2)
	# evaluate_logical_error(collection_circuits[17])

