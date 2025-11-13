
# surface 17 decoding algorithm
# reference : PRA 90, 062320

import copy
import numpy as np
import random
import collections
import itertools
from pauli_frame import pauli_frame_rule
from icecream import ic

ic.disable()

def manage_decode_surface_tomita(syndrome, qubit_relation, **kwargs):
    """
        function to infer the most like error from a syndrome volume
    """
    global list_function
    list_function = {1: decode_lookup_rule1,
                     2: decode_lookup_rule2,
                     3: decode_lookup_rule3,
                     4: decode_lookup_rule4,}

    # flag for y correlation 
    flag_y_correlation = kwargs.get("flag_y_correlation")
    if flag_y_correlation is None:
        flag_y_correlation = False

    # flag for multiqubit errors
    flag_multiqubit_error = kwargs.get("flag_multiqubit_error")
    if flag_multiqubit_error is None:
        flag_multiqubit_error = False

    if flag_multiqubit_error:
        list_function.update({2: decode_lookup_rule2_updated,
                              4: decode_lookup_rule4_updated,})
    if flag_y_correlation:
        list_function.update({4: decode_lookup_rule4_updated_y_correlated})

    # syndrome volume 에 non-trivial flipped elements 가 하나라도 있으면, decoding 수행
    # 없으면, none return
    if np.count_nonzero(syndrome["X"] == True) or \
        np.count_nonzero(syndrome["Z"] == True):
        # 1-2-3-4-5 와 1-3-2-4-5 두가지 경우를 각각 적용하여 찾은 오류 갯수를 비교하기 위해서,
        # 신드롬 볼륨을 복제하여 사용

        # rule_order 4와 5는 연계해서, 한 번에 처리
        recoveryA = decode_surface_tomita(copy.deepcopy(syndrome["X"]), copy.deepcopy(syndrome["Z"]), 
            qubit_relation, rule_order=[1,2,3,4], flag_y_correlation=flag_y_correlation)

        recoveryB = decode_surface_tomita(syndrome["X"], syndrome["Z"], qubit_relation, 
            rule_order=[1,3,2,4], flag_y_correlation=flag_y_correlation)

        return recoveryA if len(recoveryA) < len(recoveryB) else recoveryB

    else: 
        return None


def decode_surface_tomita(syndromeX, syndromeZ, qubit_relation, rule_order, **kwargs):

    flag_y_correlation = kwargs.get("flag_y_correlation")

    temp_recovery = []
    for i in rule_order:
        temp_recovery, syndromeX, syndromeZ = list_function[i](syndromeX, syndromeZ, qubit_relation, temp_recovery)

    recovery = []
    table_correlation = collections.defaultdict(int)

    if flag_y_correlation:
        for r in temp_recovery:
            if r[0] in ["X", "Z"]:
                if r[1] not in table_correlation.keys():
                    table_correlation[r[1]] = pauli_frame_rule[('i', r[0].lower())][1]
                else:
                    table_correlation[r[1]] = pauli_frame_rule[(table_correlation[r[1]], r[0].lower())][1]
            else:
                recovery.append(r)

        for k, v in table_correlation.items():
            recovery.append((v.upper(), k))

        del temp_recovery
        del table_correlation

    else:
        recovery = temp_recovery

    return recovery



def decode_lookup_rule1(syndromeX, syndromeZ, qubit_relation, recovery):
    """
        동일 신드롬 큐빗상에서 플립이 연속해서 발생하면, 측정 readout 오류일 가능성이 높아서, 신드롬 무시
    """

    rounds, checks = syndromeX.shape[:]

    for pair in itertools.product(range(checks), range(rounds-1)):
        i, j = pair[0:2]

        if syndromeX[j][i] and syndromeX[j+1][i]:
            recovery.append(("Readout", "checkX{}".format(str(i))))
            syndromeX[j][i] = False
            syndromeX[j+1][i] = False

    _, checks = syndromeZ.shape[:]
    for pair in itertools.product(range(checks), range(rounds-1)):
        i, j = pair[0:2]

        if syndromeZ[j][i] and syndromeZ[j+1][i]:
            recovery.append(("Readout", "checkZ{}".format(str(i))))
            syndromeZ[j][i] = False
            syndromeZ[j+1][i] = False

    return recovery, syndromeX, syndromeZ


def decode_lookup_rule2(syndromeX, syndromeZ, qubit_relation, recovery):
    """
        동일 라운드에서 인접한 신드롬 큐빗상에 플립이 발생하면, 가운데 위치한 데이터 큐빗에 오류
    """
    rounds, checks = syndromeX.shape[:]

    for j in range(rounds):
        flipped_check = np.where(syndromeX[j])[0]
        if len(flipped_check) >= 2:
            union = list(set(qubit_relation["checkX{}".format(str(flipped_check[0]))]) \
                    & set(qubit_relation["checkX{}".format(str(flipped_check[1]))]))

            if len(union):
                recovery.append(("Z", union[0]))
                syndromeX[j][flipped_check[0]] = False
                syndromeX[j][flipped_check[1]] = False

        flipped_check = np.where(syndromeZ[j])[0]
        if len(flipped_check) >= 2:
            union = list(set(qubit_relation["checkZ{}".format(str(flipped_check[0]))]) \
                    & set(qubit_relation["checkZ{}".format(str(flipped_check[1]))]))

            if len(union):
                recovery.append(("X", union[0]))
                syndromeZ[j][flipped_check[0]] = False
                syndromeZ[j][flipped_check[1]] = False

    return recovery, syndromeX, syndromeZ


def decode_lookup_rule2_updated(syndromeX, syndromeZ, qubit_relation, recovery):
    """
        동일 라운드에서 인접한 신드롬 큐빗상에 플립이 발생하면, 가운데 위치한 데이터 큐빗에 오류
    """
    rounds, checks = syndromeX.shape[:]

    for i in range(rounds):
        # 현재 라운드에서 flipped syndrome
        flipped_checks = np.where(syndromeX[i])[0]

        # flipped 가 없으면, 다음 라운드로..
        if not len(flipped_checks): continue

        # flipped 신드롬 쌍에 대해서 검사하기 위해서 combinatorial
        list_combination = list(itertools.combinations(flipped_checks, 2))
        combination_checkup = {k : True for k in list_combination}

        while len(flipped_checks) >= 2 and any(combination_checkup.values()):
            for pair in list_combination:
                if pair[0] == pair[1]: continue

                intersection = list(set(qubit_relation["checkX{}".format(str(pair[0]))]) \
                    & set(qubit_relation["checkX{}".format(str(pair[1]))]))

                combination_checkup[pair] = False

                for k in intersection:
                    recovery.append(("Z", k))
                    syndromeX[i][pair[0]] = False
                    syndromeX[i][pair[1]] = False

                    flipped_checks = np.where(syndromeX[i])[0]
                    list_combination = list(itertools.combinations(flipped_checks, 2))

                    break

    for i in range(rounds):
        flipped_checks = np.where(syndromeZ[i])[0]
        if not len(flipped_checks): continue

        list_combination = list(itertools.combinations(flipped_checks, 2))
        combination_checkup = {k : True for k in list_combination}

        while len(flipped_checks) >= 2 and any(combination_checkup.values()):
            for pair in list_combination:
                if pair[0] == pair[1]: continue

                intersection = list(set(qubit_relation["checkZ{}".format(str(pair[0]))]) \
                    & set(qubit_relation["checkZ{}".format(str(pair[1]))]))

                combination_checkup[pair] = False

                for k in intersection:					
                    recovery.append(("X", k))
                    syndromeZ[i][pair[0]] = False
                    syndromeZ[i][pair[1]] = False
                    flipped_checks = np.where(syndromeZ[i])[0]
                    list_combination = list(itertools.combinations(flipped_checks, 2))
                    break

    return recovery, syndromeX, syndromeZ


def decode_lookup_rule3(syndromeX, syndromeZ, qubit_relation, recovery):
	"""
		라운드 r-1 과 r 에 인접한 신드롬 큐비트 상에 신드롬 플립이 발생하는 경우,
		해당 신드롬 큐비트 사이에 위치한 데이터 큐비트에 오류
	"""
	rounds, checks = syndromeX.shape[:]

	flipped_checkX = collections.defaultdict(list)
	flipped_checkZ = collections.defaultdict(list)

	for j in range(rounds):
		flipped_checkX[j].append(np.where(syndromeX[j])[0])
		flipped_checkZ[j].append(np.where(syndromeZ[j])[0])

	# decode syndrome X part
	for i in range(0, rounds-1):
		if not flipped_checkX[i][0].size or not flipped_checkX[i+1][0].size: continue
		for pair in itertools.product(flipped_checkX[i], flipped_checkX[i+1]):
			intersection = list(set(qubit_relation["checkX{}".format(str(pair[0][0]))]) &\
								set(qubit_relation["checkX{}".format(str(pair[1][0]))]))

			for k in intersection:
				recovery.append(("Z", k))
				syndromeX[i][pair[0][0]] = False
				syndromeX[i+1][pair[1][0]] = False

	# decode syndrome Z part
	for i in range(0, rounds-1):
		if not flipped_checkZ[i][0].size or not flipped_checkZ[i+1][0].size: continue
		for pair in itertools.product(flipped_checkZ[i], flipped_checkZ[i+1]):
			intersection = list(set(qubit_relation["checkZ{}".format(str(pair[0][0]))])&\
								set(qubit_relation["checkZ{}".format(str(pair[1][0]))]))

			for k in intersection:
				recovery.append(("X", k))
				syndromeZ[i][pair[0][0]] = False
				syndromeZ[i+1][pair[1][0]] = False

	return recovery, syndromeX, syndromeZ



def decode_lookup_rule4_updated_y_correlated(syndromeX, syndromeZ, qubit_relation, recovery):
	"""
		마지막이 아닌 라운드에서, 신드롬 플립이 발생되었을 때 (데이터 큐빗에 의해서 공유되지 않는..),
		바운더리에 인접한 데이터 큐빗 (두 신드롬 큐빗 사이에 위치하지 않는..) 에 오류 발생
	"""
	# recovery = []
	rounds, checks = syndromeX.shape[:]
	
	for i in range(rounds-1):
		flipped_checks = np.where(syndromeX[i])[0]

		flag_recovery_found = False
		dict_candidates_data_X = {}
		dict_candidates_data_Z = {}

		for j in flipped_checks:
			# flipped syndrome qubit
			flipped_check_qubit = "checkX{}".format(str(j))

			# flipped syndrome qubit 에 인접하고 있는 데이터 큐비트 목록
			list_data_qubits = set(qubit_relation[flipped_check_qubit])

			for k, v in qubit_relation.items():
				if k == flipped_check_qubit: continue
				if "checkX" in k:
					list_data_qubits -= set(qubit_relation[k])

			# recovery 상의 데이터 큐비트와 list_datas 에 교집합이 있으면.. 그걸로..
			list_recovered_qubits = set([k[1] for k in recovery if k != "Next-Window" or k!= "Readout"])

			intersection = list(list_data_qubits & list_recovered_qubits)

			if len(intersection):
				error_data_qubit = intersection[0]
				recovery.append(("Z", error_data_qubit))
				syndromeX[i][j] = False

			else:
				# 아니면.. 랜덤..
				# 아니면.. 일단 데이터 큐빗 목록을 keep 하고..
				# Z stabilizer 결과 확인함

				# flipped check qubit 에 대한 인접한 데이터 큐비트 목록
				list_data_qubits = list(list_data_qubits)
				dict_candidates_data_X[j] = list_data_qubits

		flipped_checks = np.where(syndromeZ[i])[0]
		for j in flipped_checks:
			# flipped syndrome qubit
			flipped_check_qubit = "checkZ{}".format(str(j))

			# flipped syndrome qubit 에 인접하고 있는 데이터 큐비트 목록
			list_data_qubits = set(qubit_relation[flipped_check_qubit])

			for k, v in qubit_relation.items():
				if k == flipped_check_qubit: continue
				if "checkZ" in k:
					list_data_qubits -= set(qubit_relation[k])

			# recovery 상의 데이터 큐비트와 list_datas 에 교집합이 있으면.. 그걸로..
			list_recovered_qubits = set([k[1] for k in recovery if k != "Next-Window" or k!= "Readout"])
			# 아니면.. 랜덤..
			intersection = list(list_data_qubits&list_recovered_qubits)
			
			if len(intersection):
				error_data_qubit = intersection[0]
				recovery.append(("X", error_data_qubit))
				syndromeZ[i][j] = False
			
			else:
				list_data_qubits = list(list_data_qubits)
				dict_candidates_data_Z[j] = list_data_qubits

		list_candidates_data_X = list(itertools.chain.from_iterable(dict_candidates_data_X.values()))
		list_candidates_data_Z = list(itertools.chain.from_iterable(dict_candidates_data_Z.values()))

		intersection = set(list_candidates_data_X) & set(list_candidates_data_Z)

		# list_candidates_data_X(Z) 에 교집합이 있으면 해당 큐비트 선택 --> X a / Z a
		for k in intersection:
			recovery.append(("X", k))
			recovery.append(("Z", k))

			for a, b in dict_candidates_data_X.items():
				if k in b: 
					syndromeX[i][a] = False
					list_candidates_data_X.remove(k)
					dict_candidates_data_X[a].remove(k)
					break

			for a, b in dict_candidates_data_Z.items():
				if k in b: 
					syndromeX[i][a] = False
					list_candidates_data_Z.remove(k)
					dict_candidates_data_Z[a].remove(k)
					break
		
		# 나머지 들에 대해서는 각각 X and Z
		for k in list_candidates_data_X:
			recovery.append(("Z", k))
			for a, b in dict_candidates_data_X.items():
				if k in b: 
					syndromeX[i][a] = False
					list_candidates_data_X.remove(k)
					dict_candidates_data_X[a].remove(k)
					break

		for k in list_candidates_data_Z:
			recovery.append(("X", k))
			for a, b in dict_candidates_data_Z.items():
				if k in b:
					syndromeZ[i][a] = False
					list_candidates_data_Z.remove(k)
					dict_candidates_data_Z[a].remove(k)
					break

	# rule 5
	if len(np.where(syndromeX[rounds-1])[0]) or\
		len(np.where(syndromeZ[rounds-1])[0]):
		recovery.append(("Next-Window", -1))

	return recovery, syndromeX, syndromeZ



def decode_lookup_rule4_updated(syndromeX, syndromeZ, qubit_relation, recovery):
	"""
		마지막이 아닌 라운드에서, 신드롬 플립이 발생되었을 때 (데이터 큐빗에 의해서 공유되지 않는..),
		바운더리에 인접한 데이터 큐빗 (두 신드롬 큐빗 사이에 위치하지 않는..) 에 오류 발생
	"""
	# recovery = []
	rounds, checks = syndromeX.shape[:]
	
	for i in range(rounds-1):
		flipped_checks = np.where(syndromeX[i])[0]

		for j in flipped_checks:
			# flipped syndrome qubit
			flipped_check_qubit = "checkX{}".format(str(j))

			# flipped syndrome qubit 에 인접하고 있는 데이터 큐비트 목록
			list_data_qubits = set(qubit_relation[flipped_check_qubit])

			for k, v in qubit_relation.items():
				if k == flipped_check_qubit: continue
				if "checkX" in k:
					list_data_qubits -= set(qubit_relation[k])

			# recovery 상의 데이터 큐비트와 list_datas 에 교집합이 있으면.. 그걸로..
			list_recovered_qubits = set([k[1] for k in recovery if k != "Next-Window" or k!= "Readout"])

			intersection = list(list_data_qubits & list_recovered_qubits)

			if len(intersection):
				error_data_qubit = intersection[0]

			else:
				# 아니면.. 랜덤..
				list_data_qubits = list(list_data_qubits)
				random_idx = random.randint(0, len(list_data_qubits)-1)
				error_data_qubit = list_data_qubits[random_idx]

			recovery.append(("Z", error_data_qubit))
			syndromeX[i][j] = False

		flipped_checks = np.where(syndromeZ[i])[0]
		for j in flipped_checks:
			# flipped syndrome qubit
			flipped_check_qubit = "checkZ{}".format(str(j))

			# flipped syndrome qubit 에 인접하고 있는 데이터 큐비트 목록
			list_data_qubits = set(qubit_relation[flipped_check_qubit])

			for k, v in qubit_relation.items():
				if k == flipped_check_qubit: continue
				if "checkZ" in k:
					list_data_qubits -= set(qubit_relation[k])

			# recovery 상의 데이터 큐비트와 list_datas 에 교집합이 있으면.. 그걸로..
			list_recovered_qubits = set([k[1] for k in recovery if k != "Next-Window" or k!= "Readout"])
			# 아니면.. 랜덤..
			intersection = list(list_data_qubits&list_recovered_qubits)
			
			if len(intersection):
				error_data_qubit = intersection[0]
			
			else:
				list_data_qubits = list(list_data_qubits)
				random_idx = random.randint(0, len(list_data_qubits)-1)
				error_data_qubit = list_data_qubits[random_idx]

			recovery.append(("X", error_data_qubit))
			syndromeZ[i][j] = False

	# rule 5
	if len(np.where(syndromeX[rounds-1])[0]) or\
		len(np.where(syndromeZ[rounds-1])[0]):
		recovery.append(("Next-Window", -1))

	return recovery, syndromeX, syndromeZ


def decode_lookup_rule4(syndromeX, syndromeZ, qubit_relation, recovery):
	"""
		신드롬 플립이 마지막이 아닌 라운드에서 한 번 발생되었을 때 
		--> 바운더리에 인접한 데이터 큐빗 (두 신드롬 큐빗 사이에 위치 하지 않은..) 에 오류 발생
	"""
	
	rounds, checks = syndromeX.shape[:]
	flipped_checkX = collections.defaultdict(list)
	flipped_checkZ = collections.defaultdict(list)

	for j in range(rounds):
		flipped_checkX[j].extend(np.where(syndromeX[j])[0])
		flipped_checkZ[j].extend(np.where(syndromeZ[j])[0])

	total_Xflips = sum(len(c) for c in flipped_checkX.values())
	total_Zflips = sum(len(c) for c in flipped_checkZ.values())

	flag_skip_included = False
	if total_Xflips == 1:
		# 마지막에 flip 이 없으면.. rule #4
		# 마지막에 있으면, rule #5 --> do nothing
		flipped_round = None
		# for i in range(code_distance):
		for i in range(rounds):
			if len(flipped_checkX[i]) == 1:
				flipped_round = i
				break

		if flipped_round != rounds - 1:
			# flipped syndrome qubit
			flipped_check_qubit = "checkX{}".format(str(flipped_checkX[flipped_round][0]))
			# flipped syndrome qubit에 인접하고 있는 데이터 큐비트 목록
			list_data_qubits = set(qubit_relation[flipped_check_qubit])
			# 다른 checkX 신드롬 큐비트와 연결된 데이터 큐비트는 모두 제외하기..
			for k, v in qubit_relation.items():
				if k == flipped_check_qubit: continue
				if "checkX" in k:
					list_data_qubits -= set(qubit_relation[k])

			list_data_qubits = list(list_data_qubits)
			random_idx = random.randint(0, len(list_data_qubits)-1)
			# 다른 checkX 신드롬 큐비트와 연결된 데이터 큐비트를 모두 제외한 다음..
			# 나머지 큐비트들 중 한 큐비트에 오류가 발생했음
			error_data_qubit = list_data_qubits[random_idx]

			recovery.append(("Z", error_data_qubit))
			syndromeX[flipped_round][flipped_checkX[flipped_round][0]] = False

		else:
			recovery.append(("Next-Window", -1))
			flag_skip_included = True

	if total_Zflips == 1:
		flipped_round = None
		
		# for i in range(code_distance):
		for i in range(rounds):	
			if len(flipped_checkZ[i]) == 1:
				flipped_round = i
				break

		if flipped_round != rounds - 1:
			flipped_check_qubit = "checkZ{}".format(str(flipped_checkZ[flipped_round][0]))
			list_data_qubits = set(qubit_relation[flipped_check_qubit])
			for k, v in qubit_relation.items():
				if k == flipped_check_qubit: continue
				if "checkZ" in k:
					list_data_qubits -= set(qubit_relation[k])

			list_data_qubits = list(list_data_qubits)
			random_idx = random.randint(0, len(list_data_qubits)-1)
			error_data_qubit = list_data_qubits[random_idx]

			recovery.append(("X", error_data_qubit))
			syndromeZ[flipped_round][flipped_checkZ[flipped_round][0]] = False

		else:
			if not flag_skip_included:
				recovery.append(("Next-Window", -1))

	return recovery, syndromeX, syndromeZ


