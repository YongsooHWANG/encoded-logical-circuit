
# -*-coding:utf-8-*-
# from __future__ import unicode_literals
# version 1.0



class Error(Exception):
	def __init__(self, msg):
		self.msg = msg

	def __str__(self):
		return self.msg

