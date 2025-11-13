
import os
import sys
import collections
from icecream import ic

from simulate_stabilizer_circuit_stim import run_circuit_stim
from hamming_decoder import decode_steane_code
from utility import display_braket_notation


def perform_steane_QEC(stim_object, error_factor, collection_circuits, noise_frame, **kwargs):
	'''
		perform full QEC (steane method) : check-Z + check-X
	'''
	global_qubit_mapping = kwargs.get("global_qubit_mapping")
	logical_qubit_indices = kwargs.get("logical_qubit_index")

	recovery = None
	recovery_candidates = collections.defaultdict(int)

	syndrome_error = 0

	for it in range(3):
		# check-Z : prepX + check-Z

		# preparation of ancilla for check X : prepZ + check_X (cnot anc to data + measX)
		# assumption: noise free ancilla state 
		while True:
			stim_object, result, noise_frame, error_source = run_circuit_stim(stim_object,
							collection_circuits["Prep-Anc-Plus"]["system_code"], 
							error_rate=error_factor,
							noise_frame=noise_frame,
							global_qubit_mapping=global_qubit_mapping,
							logical_qubit_index=logical_qubit_indices)

			if not result.get("checkup0"): break
		
		# check - X
		stim_object, result, noise_frame, error_source = run_circuit_stim(stim_object, 
							collection_circuits["Check_X"]["system_code"], 
							error_rate=error_factor, 
							noise_frame=noise_frame,
							global_qubit_mapping=global_qubit_mapping,
							logical_qubit_index=logical_qubit_indices)
		
		# 신드롬 측정 값 정렬
		sorted_syndrome = sorted(result.items(), key = lambda item: item[0])
		syndrome_vector = [v[1] for v in sorted_syndrome]

	# error correction
		# syndrome 가지고, 오류 추정 --> recovery operator
		temp_recovery = decode_steane_code(syndrome_vector)
		recovery_candidates[temp_recovery]+=1

	if len(recovery_candidates.items()) > 1: syndrome_error+=1
	recovery = max(recovery_candidates, key=recovery_candidates.get)
	
	if recovery is not None:
		stim_object.x(recovery)
		noise_frame[recovery][0] ^= 1

	recovery_candidates.clear()

	for it in range(3):
		# check-X : prepZ + check-X

		# preparation of ancilla for check z : prepX + cnot data to anc + measZ
		# assumption: noise free ancilla state 
		while True:
			stim_object, result, noise_frame, error_source = run_circuit_stim(stim_object,
							collection_circuits["Prep-Anc-Zero"]["system_code"], 
							error_rate=error_factor,
							noise_frame=noise_frame,
							global_qubit_mapping=global_qubit_mapping, 
							logical_qubit_index=logical_qubit_indices)

			if result.get("checkup0") == 0: break

		# check-Z 
		stim_object, result, noise_frame, error_source = run_circuit_stim(stim_object, 
							collection_circuits["Check_Z"]["system_code"], 
							error_rate=error_factor, 
							noise_frame=noise_frame,
							global_qubit_mapping=global_qubit_mapping,
							logical_qubit_index=logical_qubit_indices)

		# 신드롬 측정 값 정렬
		sorted_syndrome = sorted(result.items(), key = lambda item: item[0])
		syndrome_vector = [v[1] for v in sorted_syndrome]

		temp_recovery = decode_steane_code(syndrome_vector)
		recovery_candidates[temp_recovery]+=1

		# collecting syndrome pattern : 추후 syndrome pattern 으로 부터 노이즈 정보 추출하는 연구 할 때..
		# readout_data[str_syndrome]+=1

	if len(recovery_candidates.items()) > 1: syndrome_error+=1
	recovery = max(recovery_candidates, key=recovery_candidates.get)

	if recovery is not None:
		stim_object.z(recovery)
		noise_frame[recovery][1] ^= 1
	
	return syndrome_error

