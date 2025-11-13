
import os
import sys

sys.path.insert(0, "../support/")
sys.path.insert(0, "")

import simplejson as json
import ftcheckup
from icecream import ic

path_mother = "DB-Circuits/Steane_circuits_data"

# steane QEC:
# ancilla zero + check-X + ancilla plus + check-Z
# target_files = {
# 	"data_zero": os.path.join(path_mother, "DQP-file_data_zero_2_(5, 7)_314.json"),
# 	"anc_zero": os.path.join(path_mother, "DQP-file_anc_zero_2_(5, 7)_314.json"),
# 	"check_X": os.path.join(path_mother, "DQP-file_check-x_2_(5, 7)_314.json"),
# }

target_files = {
	"data_zero": os.path.join(".","file_data_zero_2_(5, 7)_384.json"),
	"anc_zero": os.path.join(".","file_anc_zero_2_(5, 7)_384.json"),
	"check_X": os.path.join(".", "file_check-z_2_(5, 7)_384.json"),
}
# "Prep-Data-Zero" : "DQP-file_data_zero_2_(5, 7)_314.json",
# "Prep-Anc-Plus": "DQP-file_anc_plus_2_(5, 7)_314.json",
# "Prep-Anc-Zero": "DQP-file_anc_zero_2_(5, 7)_314.json",
# "Check_X": "DQP-file_check-x_2_(5, 7)_314.json",
# "Check_Z": "DQP-file_check-z_2_(5, 7)_314.json",
# "hCNOT" : "cnot-H_2_(5, 7)_246.json",
# "vCNOT" : "cnot-V_2_(5, 7)_231.json",
# "hSWAP" : "swap-H_2_(5, 7)_246.json",
# "vSWAP" : "swap-V_2_(5, 7)_231.json",


for k, v in target_files.items():
	ic(k)
	raw_data = open(v).read()
	circuit_data = json.loads(raw_data).get("result")

	ftcheckup.checkup_fault_tolerance(circuit_data.get("system_code"), 
						circuit_data["qchip"]["dimension"], 
						write_file=True,
						mapping="ftqc",
						qchip_architecture=2)