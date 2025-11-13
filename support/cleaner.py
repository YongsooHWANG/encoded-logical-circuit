# -*-coding:utf-8-*-
# last edited date: 2020.01.06

import os
import sys
import path_list
import time

def manage_old_files(day):
	time_interval = 3600 * 24 * int(day)
	while True:
		time.sleep(time_interval)
		delete_files()



def delete_files(**kwargs):
	'''
		특정 디렉토리에서 생성된지 n 일 이상 된 파일들 삭제하는 함수
		arguments: 
					- 없음
					- for loop 에 탐색하고자 하는 디렉토리 추가
	'''
	if "dir" in kwargs:
		list_target_dir = kwargs["dir"]
	else:
		list_target_dir = [path_list.path_DB_jobs]

	print("list_target_dir : ", list_target_dir)
	for target_dir in list_target_dir:
		for item in os.listdir(target_dir):
			print("item", item)
			path_item = os.path.join(target_dir, item)
			print("{} is deleted.. ".format(path_item))

			# remove file
			if os.path.isfile(path_item): os.remove(path_item)

			# remove directory
			else: 
				if len(os.listdir(path_item)):
					import shutil
					shutil.rmtree(path_item)
				else:
					os.rmdir(path_item)