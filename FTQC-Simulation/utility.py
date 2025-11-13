
import numpy as np

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
