# -*-coding:utf-8-*-

import os
import platform
import socket

path_DB_algorithm = "DB-Algorithms"
path_DB_qasm = "DB-Assembly"
path_DB_buildingblock = "DB-BuildingBlocks"
path_DB_device = "DB-Devices"
path_DB_analysis = "DB-Analysis"
path_DB_syscode = "DB-SystemCode"
path_DB_synthesisoption = "DB-SynthesisCriterion"
path_DB_FTQC = "DB-FTQC"

try:
	hostname = socket.gethostname()
	local_addr_ip = socket.gethostbyname(hostname)
except Exception as e:
	print(e)

# public 합성기 서버(129.254.30.36)에서는 합성기 directory 밖 지정된 위치에 결과물 저장
if platform.system() == "Linux" and local_addr_ip == "129.254.30.223":
	path_DB_jobs = "/media/DATA/DB-Synthesis-Jobs"
else:
	path_DB_jobs = "DB-Jobs"

directory = {"jobs": path_DB_jobs}

# 			"algorithm": path_DB_algorithm,
# 			 "qasm": path_DB_qasm,
# 			 "buildingblock": path_DB_buildingblock,
# 			 "device": path_DB_device,
# 			 "analysis": path_DB_analysis,
# 			 "syscode": path_DB_syscode,
# 			 "synth_option": path_DB_synthesisoption,
# 			 "ftqc": path_DB_FTQC,

try:
	# create directories to store qasm, layout and result files
	for i in directory.values():
		if not os.path.exists(i): os.makedirs(i)

except OSError as e:
	print(e)
