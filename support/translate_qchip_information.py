
import os
import sys

sys.path.insert(0, "../support/")
sys.path.insert(0, "")

import math
import simplejson as json
import ftcheckup
import collections
from icecream import ic


def retrieve_gate_error_rates(calibration_data):
	'''
		- IBM QX calibration data 를 토대로, 게이트 별 오류율 확인
		- IBM QX 에서는 U1, U2, U3 게이트에 대한 오류율을 제공함
		- IBM OPEN QASM 논문에 기재된 방식을 따라서, U1, U2, U3 정보를 토대로, standard gate (X, Z, Y, H, .... ) 에 대한
		오류율 꼐산
	'''
	
	gate_error_rates = collections.defaultdict(dict)
	gate_length = collections.defaultdict(dict)

	qchip_data = {item: calibration_data.get(item)
			for item in ["device_name", "qubit_connectivity", "dimension"]}

	# 1-qubit gate : id, reset, rz, x
	for gate in ["id", "rz", "x", "reset"]:
		for key, value in calibration_data["gates"]["1q"][gate].items():
			target_qubit = int(value["qubits"][0])
			gate_error_rates[gate][target_qubit] = value["gate_error"]
			gate_length[gate][target_qubit] = value["gate_length"]
	
	# readout error
	for key, value in calibration_data["qubits"].items():
		target_qubit = int(key)
		gate_error_rates["measure"][target_qubit] = value["readout_error"]
		gate_length["measure"][target_qubit] = value["readout_length"]
		
		if value["unit"] == "ns": 
			gate_length["measure"][target_qubit] *= 10**-9
		else:
			raise Exception("unit error")

	# 2-qubit gate
	for gate in ["cx"]:
		for key, value in calibration_data["gates"]["2q"][gate].items():
			qubits = list(map(int, value["qubits"]))
			gate_error_rates[gate][(qubits[0], qubits[1])] = value["gate_error"]
			gate_length[gate][(qubits[0], qubits[1])] = value["gate_length"]


	qchip_data.update({"gate_error": gate_error_rates,
					   "gate_time": gate_length})

	decoherence = collections.defaultdict(dict)
	for qubit, data in calibration_data["qubits"].items():
		qubit = int(qubit)
		decoherence["T1"][qubit] = data.get("T1")
		decoherence["T2"][qubit] = data.get("T2")
	
	qchip_data.update({"decoherence": decoherence})	

	return qchip_data



def get_device_information(device_file):
	"""
		function to get the information of ibm quantum device specified by user
	"""

	# occasionally, internal error happens.
	# therefore, we iterate the procedure until success

	json_qchip_data = open(device_file).read()
	raw_qchip_data = json.loads(json_qchip_data)

	# IBM QX calibration data 로 부터, standard 게이트 별 오류율 계산
	qchip_performance = retrieve_gate_error_rates(raw_qchip_data)
	
	cnot_error_rate = collections.defaultdict(dict)
	cnot_gate_time = collections.defaultdict(dict)

	for qubits, data in qchip_performance["gate_error"]["cx"].items():
		cnot_error_rate[qubits[0]].update({qubits[1]: data})
	for qubits, data in qchip_performance["gate_time"]["cx"].items():
		cnot_gate_time[qubits[0]].update({qubits[1]: data})

	qchip_data = {}
	# qchip_performance를 mapping 에 활용 가능한 형태로 변경
	for item in ["device_name", "dimension", "qubit_connectivity"]:
		qchip_data[item] = qchip_performance.get(item)

	qchip_data["cnot_gate_time"] = cnot_gate_time
	qchip_data["cnot_error_rate"] = cnot_error_rate

	# qchip_data["calibration_date"] = qchip_performance["date"]
	# qchip_data["device_name"] = qchip_performance
	qchip_data["decoherence"] = qchip_performance.get("decoherence")
	
	qchip_data["measure_time"] = qchip_performance.get("gate_time").get("measure")
	qchip_data["measure_error"] = qchip_performance.get("gate_error").get("measure")

	# pauli twirling -> 1q error rate
	qchip_data["error_rate"] = qchip_performance["gate_error"]["id"]
	qchip_data["gate_time"] = qchip_performance["gate_time"]["id"]

	qchip_data["error_rate"] = {}
	for qubit, t1 in qchip_data["decoherence"]["T1"].items():
		t2 = qchip_data["decoherence"]["T2"][qubit]
		gate_time = qchip_performance["gate_time"]["id"][qubit]

		px = (1 - math.exp(-1*gate_time/t1))/4
		py = px
		pz = (1 - math.exp(-1*gate_time/t2))/2 - px
		qchip_data["error_rate"][int(qubit)] = (px+pz+py)

	# for qchip architecture of multi-modules
	for item in ["inter-module_connection", "ebits"]:
		qchip_data.update({item: raw_qchip_data.get(item)})

	return qchip_data


if __name__ == "__main__":
	directory = "../DB-QChip"
	file_raw_data = "file_qchip_DQC_5x20_raw.json"
	raw_device_file = os.path.join(directory, file_raw_data)

	qchip_data = get_device_information(raw_device_file)
	# ic(qchip_data)

	translated_file = os.path.join(directory, file_raw_data.replace("_raw", ""))
	with open(translated_file, "w") as outfile:
		json.dump(qchip_data, outfile, sort_keys=True, indent=4, separators=(',', ':'))
