
import numpy as np
import math
from scipy.stats import norm

import itertools
from icecream import ic

pauli_gate = {0: "i", 1:"x", 2:"y", 3:"z"}


def approximate_decoherence_channel(data, **kwargs):
	"""
		aproximate decoherence channel via pauli twirling
	"""

	px = py = (1-math.exp(-time.get("gate_time")/time.get("t1_time")))/4
	pz = (1-math.exp(-time.get("gate_time")/time.get("t2_time")))/2
	pz -= (1-math.exp(-time.get("gate_time")/time.get("t1_time")))/4

	return {"px": px, "py": py, "pz": pz, "pi": 1 - (px+pz+py)}


def generate_random_error(qubits, **kwargs):

	error_rate = kwargs.get("error_rate")
	variance = kwargs.get("error_variance")
	error_rate = np.random.normal(loc=error_rate, scale=variance, size=1)[0]

	pauli_channel = {}
	pauli_channel["px"] = pauli_channel["py"] = pauli_channel["pz"] = float(error_rate/3)
	pauli_channel["pi"] = 1 - error_rate

	if qubits == 1:
		random_double = np.random.random_sample()

		# x error
		if random_double < pauli_channel["px"]:
			return pauli_gate[1]

		# y error
		elif random_double < pauli_channel["px"] + pauli_channel["py"]:
			return pauli_gate[2]

		# z error
		elif random_double < 1 - pauli_channel["pi"]:
			return pauli_gate[3]

		# no error
		else:
			return pauli_gate[0]


	elif qubits == 2:
		list_error_rates = [pauli_channel["pi"], pauli_channel["px"], pauli_channel["py"], pauli_channel["pz"]]
		# list_error_rates = [pauli_channel["pi"], error_rate]
		accumulated_factor = 0

		error_prob = {}
		for i in list(itertools.product(range(4), range(4))):
			error_prob[i] = {"error": i, 
							 "net" : list_error_rates[i[0]]*list_error_rates[i[1]],
							 "from" : accumulated_factor,
							 "to": list_error_rates[i[0]]*list_error_rates[i[1]] + accumulated_factor}
			accumulated_factor = error_prob[i]["to"]

		random_double = np.random.random_sample()
		for i in error_prob:
			if random_double >= error_prob[i]["from"] and random_double < error_prob[i]["to"]:
				first_qubit_error, second_qubit_error = error_prob[i]["error"][:2]
				break


		return pauli_gate[first_qubit_error], pauli_gate[second_qubit_error]


if __name__ == "__main__":
	import collections
	dist = collections.defaultdict(int)

	for i in range(100000):
		error = generate_random_error(1, error_rate=0.01, error_variance=0)
		dist[error]+=1

	print(dist)