
import os
import sys

sys.path.insert(0, "../support/")
sys.path.insert(0, "")

import math
import simplejson as json
import ftcheckup
from icecream import ic

path_mother = "/Users/yongsoo/Desktop/FT_circuit_steane_code/test"
# path_mother = "/Users/yongsoo/Desktop/FT_circuit_steane_code/steane_advanced/DQP-arch-2"
# path_mother = "../CHPSimulation/DB-Circuits/golay_code_circuits/3-arch"
# path_mother = "../CHPSimulation/DB-Circuits/steane_code_circuits/Arch-3"
# steane QEC:
# ancilla zero + check-X + ancilla plus + check-Z


target_files = {
	# "b": os.path.join(path_mother, "DQP-file_encoder_2_(8, 8, 1)_944.json"),
	# "a": os.path.join(path_mother, "file_encoder_2_(8, 8, 1)_944.json"),
	# "c": os.path.join(path_mother, "DQP2-file_data_zero_2_(5, 6)_340.json"),
	# "b": os.path.join("../", "file_data_zero_2_(5, 6)_340.json"),
	# "c": os.path.join("", "circuit-regular_graph-ibmhexa-32.json"),
	"c": "circuit-regular_graph-ibmhexa-32.json"
	
}

for k, v in target_files.items():
	raw_data = open(v).read()
	
	try:
		circuit_data = json.loads(raw_data).get("circuit").get("result")
	except:
		circuit_data = json.loads(raw_data)
	

	ftcheckup.checkup_fault_tolerance(circuit_data.get("system_code"), 
						circuit_data["qchip"]["dimension"], 
						write_file=True,
						mapping="ftqc",
						qchip_architecture=3,
						display_waiting=True,
						allowable_data_interaction=math.inf)