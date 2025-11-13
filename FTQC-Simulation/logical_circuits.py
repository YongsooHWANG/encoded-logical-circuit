import sys
sys.path.insert(0, "../../CircuitMapper/library")

import itertools
import collections
from icecream import ic
import formatconversion
import faultylocation

ghz_circuits = {
	2:{
		"initial_mapping":{
			"data0": 0,
			"data1": 1
		},
		"circuit" : {
			0 : ["PrepZ 0", "PrepZ 1"],
			1 : ["H 0", "H-Waiting 1"],
			2 : ["vvCNOT 0,1"]
		}
	},
	3: {
		"initial_mapping": {
			"data0": 0,
			"data1": 1,
			"data2": 2
		},
		"circuit" : {
			0 : ["PrepZ 0", "PrepZ 1", "PrepZ 2"],
			1 : ["H 0", "H-Waiting 1", "H-Waiting 2"],
			2 : ["vvCNOT 0,1", "CNOT-Waiting 2"],
			3 : ["vvCNOT 1,2", "CNOT-Waiting 0"]
		}
	},
	4:{
		"initial_mapping": {
			"data0": 0,
			"data1": 1,
			"data2": 2,
			"data3": 3
		},
		"circuit" : {
			0 : ["PrepZ 0", "PrepZ 1", "PrepZ 2", "PrepZ 3"],
			1 : ["H 0", "H-Waiting 1", "H-Waiting 2", "H-Waiting 3"],
			2 : ["vvCNOT 0,2", "CNOT-Waiting 1", "CNOT-Waiting 3"],
			3 : ["hhCNOT 0,1", "hhCNOT 2,3"]
		}
	}
}

simple_test_circuits = {
	2: {
		"initial_mapping": {
			"data0": 0,
			"data1": 1
		},
		"circuit": {
			0: ["PrepZ 0", "PrepX 1"],
			1: ["X 0", "H-Waiting 1"]
		}
	}
}

swap_test_circuits = {
	2:{
		"initial_mapping": {
			"data0" : 0,
			"data1" : 1
		},
		"circuit": {
			0 : ["PrepZ 0", "PrepZ 1"],
			1 : ["X 0", "X-Waiting 1"],
			2 : ["vvSWAP 0,1"]
		}
	},
	3:{
		"initial_mapping": {
			"data0" : 0,
			"data1" : 1,
			"data2" : 2
		},
		"circuit": {
			0 : ["PrepZ 0", "PrepZ 1", "PrepZ 2"],
			1 : ["X 0", "X-Waiting 1", "X-Waiting 2"],
			2 : ["vvSWAP 0,1", "SWAP-Waiting 2"],
			3 : ["vvSWAP 1,2", "SWAP-Waiting 0"]
		}
	},
	4: {
		"initial_mapping" : {
			"data0" : 0,
			"data1" : 1,
			"data2" : 2,
			"data3" : 3
		},
		"circuit" : {
			0 : ["PrepZ 0", "PrepZ 1", "PrepZ 2", "PrepZ 3"],
			1 : ["X 0", "H-Waiting 1", "H-Waiting 2", "H-Waiting 3"],
			2 : ["vvSWAP 0,1", "SWAP-Waiting 2", "SWAP-Waiting 3"],
			3 : ["vvSWAP 1,2", "SWAP-Waiting 0", "SWAP-Waiting 3"],
			4 : ["vvSWAP 2,3", "SWAP-Waiting 0", "SWAP-Waiting 1"]
		}
	}
}


cnot_test_circuits = {
	2:{
		"initial_mapping": {
			"data0" : 0,
			"data1" : 1
		},
		"circuit": {
			0 : ["PrepX 0", "PrepZ 1"],
			1 : ["X 0", "X-Waiting 1"],
			2 : ["vvCNOT 0,1"]
		}
	},
	3:{
		"initial_mapping": {
			"data0" : 0,
			"data1" : 1,
			"data2" : 2
		},
		"circuit": {
			0 : ["PrepZ 0", "PrepZ 1", "PrepZ 2"],
			1 : ["X 0", "X-Waiting 1", "X-Waiting 2"],
			2 : ["vvCNOT 0,1", "CNOT-Waiting 2"],
			3 : ["vvCNOT 1,2", "CNOT-Waiting 0"]
		}
	},
	4: {
		"initial_mapping" : {
			"data0" : 0,
			"data1" : 1,
			"data2" : 2,
			"data3" : 3
		},
		"circuit" : {
			0 : ["PrepZ 0", "PrepZ 1", "PrepZ 2", "PrepZ 3"],
			1 : ["X 0", "X-Waiting 1", "X-Waiting 2", "X-Waiting 3"],
			2 : ["vvCNOT 0,1", "SWAP-Waiting 2", "SWAP-Waiting 3"],
			3 : ["vvCNOT 1,2", "SWAP-Waiting 0", "SWAP-Waiting 3"],
			4 : ["vvCNOT 2,3", "SWAP-Waiting 0", "SWAP-Waiting 1"]
		}
	}
}


def develop_ghz_circuit(m, n):
	"""
		develop logical GHZ state over 2-dimensional qubit layout of size m x n
	"""
	qubit_array = {}
	qubit_mapping = {}
	qubit_idx = 0
	
	for combination in itertools.product(list(range(m)), list(range(n))):
		qubit_array[(combination[0], combination[1])] = qubit_idx
		qubit_mapping["data{}".format(qubit_idx)] = qubit_idx
		qubit_idx+=1

	list_instructions = []
	# 1단계: 명령어 리스트
	# qubit preparation
	for pair in itertools.product(list(range(m)), list(range(n))):
		if pair[0] == 0 and pair[1] == 0:
			list_instructions.append(["PrepX", "{}".format(qubit_array[(pair[0], pair[1])])])
		else:
			list_instructions.append(["PrepZ", "{}".format(qubit_array[(pair[0], pair[1])])])

	for i in range(0, m-1):
		list_instructions.append(["vvCNOT", "{}".format(qubit_array[(i,0)]), "{}".format(qubit_array[(i+1, 0)])])

	for i in range(m):
		for j in range(n-1):
			list_instructions.append(["hhCNOT", "{}".format(qubit_array[(i,j)]), "{}".format(qubit_array[(i, j+1)])])

	# 2단계 : time ordered form
	time_ordered_form = formatconversion.transform_ordered_syscode(list_instructions, number_qubits=m*n)
	
	# 3단계 : full circuit (with waiting) 
	full_circuit = faultylocation.get_full_circuit(time_ordered_form, qubit_mapping)
	
	return {"circuit": full_circuit, "initial_mapping": qubit_mapping}
	


if __name__ == "__main__":
	develop_ghz_circuit(6, 6)