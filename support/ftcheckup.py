
# -*-coding:utf-8-*-

import error
import copy
import math
import userproperty
import collections
import pandas
import numpy as np
import ast
from icecream import ic

import globalVariable as g
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def checkup_fault_tolerance(system_code, lattice_size, **kwargs):
    '''
        function to investigate the fault tolerance of the circuit
    '''

    # initial mapping
    cloned_qubit_mapping = copy.deepcopy(system_code["initial_mapping"])

    qubit_mapping = system_code["initial_mapping"]
    inverse_mapping = {v: k for k, v in qubit_mapping.items()}

    mapping_type = kwargs.get("mapping")

    qchip_architecture = kwargs.get("qchip_architecture")
    flag_display_waiting = kwargs.get("display_waiting")
    
    layout = [[0 for i in range(lattice_size["width"])] for j in range(lattice_size["height"])]
    
    # qchip_architecture
    # 1 : 1-dimensional linear 
    # 2 : 2-dimensional lattice : rectangular
    # 21 : 
    # 23 : 2-dimensional lattice : triangular (having maximum edge degree)
    if qchip_architecture in [1, 2, 21, 23]:
        for idx, qubit in inverse_mapping.items():
            x_coord = int(idx/lattice_size["width"])
            z_coord = int(idx%lattice_size["width"])

            if "dummy" in qubit:
                layout[x_coord][z_coord] = " "*10
            else:
                layout[x_coord][z_coord] = qubit

            # layout[x_coord][z_coord] = qubit
        
        print("==============================================================================================================")
        print("Initial Mapping: ")
        print("--------------------------------------------------------------------------------------------------------------")
        print(pandas.DataFrame(layout).to_string())
        print("==============================================================================================================")

    system_code["circuit"] = {int(k): v for k, v in system_code["circuit"].items()}
    circuit_depth = max(list(system_code["circuit"].keys())) + 1
    
    qubit_usage_status = {k: bool("data" in k or "ancilla" in k) for k in qubit_mapping.keys()}

    # 큐빗의 waiting 을 모니터링 하기 위해...
    # threshold 에 영향을 주는 waiting (idle status) 을 체크하는 것은.. 큐빗 형태별로 다르다.
    #   data : 프로토콜 내내 data 의 idle 상태는 threshold 에 영향을 주게 됨
    #   ancilla : 프로토콜 내에서 activated 구간에서 idle 일 때만.. threshold 에 영향을 주게 됨
    # 따라서.. 모니터링 큐빗 목록은 초기에는 data 큐빗만 포함되고.. 
    # ancilla 큐빗이 prepared, 즉 activated, 이면, 모니터링 큐빗 목록에 포함됨
    #              measured, 즉 inactivated, 이면 모니터링 큐빗 목록에서 제외됨
    list_monitorning_qubits = [qubit for qubit in qubit_mapping.keys() if "data" in qubit]

    collection_waiting = collections.defaultdict(int)
    max_data_interaction = kwargs.get("allowable_data_interaction")
    count_data_interaction = 0
    
    # gate 동작 길이 순위
    # 
    for idx in range(circuit_depth):
        instructions = system_code["circuit"][idx]

        flag_swap = False
        if idx == 0:
            print("==============================================================================================================")
        
        print("instructions at {}-th index : {}".format(idx, instructions))
        print("--------------------------------------------------------------------------------------------------------------")
        
        # 양자 회로에서 큐빗들의 waiting (idle status) 을 체크하기 위해서
        # 해당 시간에 실제로 동작하는 데이터 및 신드롬 큐빗을 체크.. 
        # dummy 큐빗은.. 오류가 발생해도 그만이므로.. 카운트 하지 않음
        list_qubits_working = []
        list_measure_qubits = []
        list_gates = []

        a = 0
        for inst in instructions:
            tokens = inst.split(" ")
            list_gates.append(tokens[0])

            if "Wait" in tokens[0]: continue

            if tokens[0] in [g.str_gate_prepz, g.str_gate_prepx]:
                physical_qubit = int(tokens[1])
                logical_qubit = inverse_mapping[physical_qubit]
                qubit_usage_status[logical_qubit] = True
                list_monitorning_qubits.append(logical_qubit)
                list_qubits_working.append(logical_qubit)        

                print(" {} {:>3} ({:>8}) : {}".format(tokens[0], physical_qubit, logical_qubit,
                    qubit_usage_status[logical_qubit]))


            elif tokens[0] in [g.str_gate_measz, g.str_gate_measx]:
                physical_qubit = int(tokens[1])
                logical_qubit = inverse_mapping[physical_qubit]
                qubit_usage_status[logical_qubit] = False
                list_qubits_working.append(logical_qubit)

                print(" {} {:>3} ({:>8}) : {}".format(tokens[0], physical_qubit, logical_qubit,
                    qubit_usage_status[logical_qubit]))
                list_measure_qubits.append(logical_qubit)


            elif any(item in tokens[0] for item in [g.str_gate_cnot, g.str_gate_cz, g.str_gate_cy]) and\
                "Wait" not in tokens[0]:

                qubits = list(map(int, tokens[1].split(",")))

                logical_qubit0 = inverse_mapping[qubits[0]]
                logical_qubit1 = inverse_mapping[qubits[1]]
                flag_swap = False

                if "dummy" not in logical_qubit0:
                    list_qubits_working.append(logical_qubit0)
                
                if "dummy" not in logical_qubit1: 
                    list_qubits_working.append(logical_qubit1)

                print(" {} {:>3},{:>3} ({:>8},{:>8}) : {}, {}".format(tokens[0], qubits[0], qubits[1], 
                                                    logical_qubit0, logical_qubit1,
                                                    qubit_usage_status[logical_qubit0],
                                                    qubit_usage_status[logical_qubit1]))

            # SWAP gate case
            elif g.str_gate_swap in tokens[0] and "Wait" not in tokens[0]:
                qubits = list(map(int, tokens[1].split(",")))
                logical_qubit0 = inverse_mapping[qubits[0]]
                logical_qubit1 = inverse_mapping[qubits[1]]

                if "dummy" not in logical_qubit0: 
                    list_qubits_working.append(logical_qubit0)

                if "dummy" not in logical_qubit1: 
                    list_qubits_working.append(logical_qubit1)

                print(" {} {:>3},{:>3} ({:>8},{:>8}) : {}, {}".format(tokens[0], qubits[0], qubits[1], 
                                                            logical_qubit0, logical_qubit1,
                                                            qubit_usage_status[logical_qubit0], 
                                                            qubit_usage_status[logical_qubit1]))

                inverse_mapping[qubits[0]], inverse_mapping[qubits[1]] =\
                    inverse_mapping[qubits[1]], inverse_mapping[qubits[0]]

                flag_swap = True

                if mapping_type == "ftqc":
                    # activated 큐빗간 interaction (SWAP)에 대해서, 오류 발생시킴
                    # 주로 FT circuit mapping 에서 activated qubit 들 간의 swap gate 가 ft condition 을 violation 할 수 있기 때문
                    if qubit_usage_status[logical_qubit0] and qubit_usage_status[logical_qubit1]:
                        count_data_interaction+=1
                        if count_data_interaction > max_data_interaction:
                            raise error.Error("Stop: SWAP between activated qubits")


            # barrier - All : for all qubits
            elif tokens[0] in [g.str_barrier_all]:
                print(" {:>8}".format(tokens[0]))

            # selective barrier for selected qubits
            elif tokens[0] in [g.str_barrier]:
                qubit_list = ast.literal_eval(" ".join(tokens[1:]))
                print(" {:>8} {}".format(tokens[0], qubit_list))

            else:
                physical_qubit = int(tokens[-1])
                logical_qubit = inverse_mapping[physical_qubit]
                list_qubits_working.append(logical_qubit)

                if flag_display_waiting or (not flag_display_waiting and "Wait" not in tokens[0]):
                    print(" {} {:>3} ({:>8}) : {}".format(tokens[0], physical_qubit, logical_qubit,
                        qubit_usage_status[logical_qubit]))

        qubit_mapping = {v: k for k, v in inverse_mapping.items()}

        if flag_swap and qchip_architecture in [1, 2, 21, 23]:
            # 2d array 재 구성
            for idx, qubit in inverse_mapping.items():
                x_coord = int(idx/lattice_size["width"])
                z_coord = int(idx%lattice_size["width"])

                if "dummy" in qubit:
                    layout[x_coord][z_coord] = " "*10
                else:
                    layout[x_coord][z_coord] = qubit

        if qchip_architecture in [1, 2, 21, 23]:
            print("--------------------------------------------------------------------------------------------------------------")
            print(pandas.DataFrame(layout).to_string())
            # df = pandas.DataFrame(layout)
            # fig, ax = plt.subplots(figsize=(12,4))
            # ax.axis('tight')
            # ax.axis('off')

            # the_table = ax.table(cellText=df.values, colLabels=df.columns,loc='center')
            # pp = PdfPages("foo-{}.pdf".format(str(idx)))
            # pp.savefig(fig, bbox_inches='tight')
            # pp.close()

        print("==============================================================================================================")


    system_code["initial_mapping"] = cloned_qubit_mapping


def display_qubit_movements(system_code, lattice_size, **kwargs):
    
    # initial mapping
    qubit_mapping = system_code["initial_mapping"]
    inverse_mapping = {v: k for k, v in qubit_mapping.items()}

    layout = [[0 for i in range(lattice_size["width"])] for j in range(lattice_size["height"])]
    
    for idx, qubit in inverse_mapping.items():
        x_coord = int(idx/lattice_size["width"])
        z_coord = int(idx%lattice_size["width"])

        layout[x_coord][z_coord] = qubit
    
    print(" =====================================================  ")
    print("Initial Mapping: ")
    print(" -----------------------------------------------------  ")
    print(pandas.DataFrame(layout).to_string())
    print(" =====================================================  ")

    circuit_depth = max(list(system_code["circuit"].keys())) + 1
    
    qubit_usage_status = {k: True if "data" in k else False for k in qubit_mapping.keys()}

    max_data_interaction = kwargs.get("allowable_data_interaction")
    ic(max_data_interaction)

    count_data_interaction = 0
    # circuit
    for idx in range(circuit_depth):
        instructions = system_code["circuit"][idx]

        flag_swap = False
        print(" =====================================================  ")
        print("instructions at {}-th index : {}".format(idx, instructions))
        print(" -----------------------------------------------------  ")
        
        for inst in instructions:
            tokens = inst.split(" ")

            if tokens[0] in [g.str_gate_prepz, g.str_gate_prepx]:
                physical_qubit = int(tokens[1])
                logical_qubit = inverse_mapping[physical_qubit]
                
                qubit_usage_status[logical_qubit] = True
                print(" {} {} ({}) -> {}".format(tokens[0], physical_qubit, logical_qubit, qubit_usage_status[logical_qubit]))

                
            elif tokens[0] in [g.str_gate_measz, g.str_gate_measx]:
                physical_qubit = int(tokens[1])
                logical_qubit = inverse_mapping[physical_qubit]
                qubit_usage_status[logical_qubit] = False
                print(" {} {} ({}) -> {}".format(tokens[0], physical_qubit, logical_qubit, qubit_usage_status[logical_qubit]))

            elif tokens[0] in [g.str_gate_cnot, g.str_gate_cz, g.str_gate_cy]:
                qubits = list(map(int, tokens[1].split(",")))
                
                print(" {} {}, {} ({}, {})".format(tokens[0], qubits[0], qubits[1], inverse_mapping[qubits[0]], inverse_mapping[qubits[1]]))

                flag_swap = False

            elif tokens[0] in ["SWAP"]:
                qubits = list(map(int, tokens[1].split(",")))
                logical_qubit0 = inverse_mapping[qubits[0]]
                logical_qubit1 = inverse_mapping[qubits[1]]

                print(" {} {}, {} ({}, {}) -> {} {}".format(tokens[0], qubits[0], qubits[1], logical_qubit0, logical_qubit1,
                                                                    qubit_usage_status[logical_qubit0], qubit_usage_status[logical_qubit1]))

                inverse_mapping[qubits[0]], inverse_mapping[qubits[1]] =\
                    inverse_mapping[qubits[1]], inverse_mapping[qubits[0]]
                
                flag_swap = True

                # activated 큐빗간 interaction (SWAP)에 대해서, 오류 발생시킴
                if qubit_usage_status[logical_qubit0] and qubit_usage_status[logical_qubit1]:
                    count_data_interaction+=1
                    print(" data-type interaction")
                    if count_data_interaction > max_data_interaction:
                        raise error.Error("Stop: SWAP between activated qubits")

            elif tokens[0] in ["Barrier-All"]:
                print(" {} ".format(tokens[0]))
                flag_swap = False

            else:
                qubit = int(tokens[1])
                print(" {} {} ({})".format(tokens[0], qubit, inverse_mapping[qubit]))
                flag_swap = False

        if flag_swap:
            # 2d array 재 구성
            for idx, qubit in inverse_mapping.items():
                x_coord = int(idx/lattice_size["width"])
                z_coord = int(idx%lattice_size["width"])

                layout[x_coord][z_coord] = qubit

        # ic(qubit_usage_status)
        print(" -----------------------------------------------------  ")
        print(pandas.DataFrame(layout).to_string())
        print(" =====================================================  ")


def display_qubit_movements_multi_circuits(system_code, lattice_size, **kwargs):
	
	# initial mapping
	qubit_mapping = system_code["initial_mapping"]
	inverse_mapping = {v: k for k, v in qubit_mapping.items()}

	layout = [[0 for i in range(lattice_size["width"])] for j in range(lattice_size["height"])]
	
	for idx, qubit in inverse_mapping.items():
		x_coord = int(idx/lattice_size["width"])
		z_coord = int(idx%lattice_size["width"])

		layout[x_coord][z_coord] = qubit
	
	print(" =====================================================  ")
	print("Initial Mapping: ")
	print(" -----------------------------------------------------  ")
	print(pandas.DataFrame(layout).to_string())
	print(" =====================================================  ")
	
	for circuit_idx, subcircuit in system_code["circuit"].items():
		circuit_depth = max(list(subcircuit.keys())) + 1

		# circuit
		for idx in range(circuit_depth):
			instructions = subcircuit[idx]

			flag_swap = False
			print(" =====================================================  ")
			print("instructions at {}-th index : {}".format(idx, instructions))
			print(" -----------------------------------------------------  ")
			
			for inst in instructions:
				tokens = inst.split(" ")

				if tokens[0] in [g.str_gate_cnot, g.str_gate_cz, g.str_gate_cy]:
					qubits = list(map(int, tokens[1].split(",")))
					
					print(" {} qubits ({}, {}) -> ({}, {})".format(tokens[0], qubits[0], qubits[1], inverse_mapping[qubits[0]], inverse_mapping[qubits[1]]))

					flag_swap = False

				elif tokens[0] in [g.str_gate_swap]:
					qubits = list(map(int, tokens[1].split(",")))
					
					print(" {} qubits ({}, {}) -> ({}, {})".format(tokens[0], qubits[0], qubits[1], inverse_mapping[qubits[0]], inverse_mapping[qubits[1]]))

					inverse_mapping[qubits[0]], inverse_mapping[qubits[1]] =\
						inverse_mapping[qubits[1]], inverse_mapping[qubits[0]]
					
					flag_swap = True

				else:
					qubit = int(tokens[1])
					print(" {} ({}) -> {}".format(tokens[0], qubit, inverse_mapping[qubit]))
					flag_swap = False

			if flag_swap:
				# 2d array 재 구성
				for idx, qubit in inverse_mapping.items():
					x_coord = int(idx/lattice_size["width"])
					z_coord = int(idx%lattice_size["width"])

					layout[x_coord][z_coord] = qubit

			print(pandas.DataFrame(layout).to_string())
			# pprint(pandas.DataFrame(layout).to_json(orient="table"))
			print(" =====================================================  ")


def display_qubit_mapping(qubit_mapping, layout_size):
    """
        display qubit mapping
    """
    layout = [[0 for i in range(layout_size["width"])] for j in range(layout_size["height"])]

    for key, value in qubit_mapping.items():
        x_coord = int(value/layout_size["width"])
        z_coord = int(value%layout_size["width"])

        layout[x_coord][z_coord] = key

    print("===============================================")
    print(pandas.DataFrame(layout))
    print("===============================================")


def merge_qubit_layout(mapping1, mapping2, direction, layout_size):
    # function merge two blocks in 2D & 3D
    extended_qubit_layout = {}

    if direction in ["vertical", "V"]:
        extended_qubit_layout = mapping1
        for key, value in mapping2.items():
            # x coord : key 값을 평면 넓이로 나누면.. 몇번째 height 에 위치하는지 확인 가능
            # remainder : 해당 평면 내에서의 key 값
            x_coord = int(key/(layout_size["width"] * layout_size["length"]))
            remainder = key % (layout_size["width"] * layout_size["length"])

            z_coord = int(remainder/layout_size["width"])
            y_coord = int(remainder%layout_size["width"])

            # extended index = x * 평면 + z * length + y
            extended_index = (x_coord + layout_size["height"]) * (layout_size["width"]*layout_size["length"])
            extended_index += (z_coord * layout_size["width"]) + y_coord

            extended_qubit_layout[extended_index] = value

 
    elif direction in ["horizon", "H"]:
        for key, value in mapping1.items():
            x_coord = int(key/(layout_size["width"] * layout_size["length"]))
            remainder = key % (layout_size["width"] * layout_size["length"])
            
            z_coord = int(remainder/layout_size["width"])
            y_coord = int(remainder%layout_size["width"])

            # extended index = x * 평면 + z * length + y
            extended_index = x_coord * (layout_size["width"]*layout_size["length"] * 2) 
            extended_index += (z_coord * layout_size["width"]*2) + y_coord
            
            extended_qubit_layout[extended_index] = value

        for key, value in mapping2.items():
            x_coord = int(key/(layout_size["width"] * layout_size["length"]))
            remainder = key % (layout_size["width"] * layout_size["length"])
            
            z_coord = int(remainder/layout_size["width"])
            y_coord = int(remainder%layout_size["width"])

            # extended index = x * 평면 + z * length + y + width
            extended_index = x_coord * (layout_size["width"]*layout_size["length"] * 2) 
            extended_index += (z_coord*layout_size["width"]*2) + y_coord + layout_size["width"]

            extended_qubit_layout[extended_index] = value

    return {v: int(k) for k, v in extended_qubit_layout.items()}

