# -*-coding:utf-8-*-

try:
	import simplejson as json
except ImportError:
	sys.exit("""You need to install simplejson!
				Install it from http://pypi.python.org/pypi/simplejson
				or run pip install simplejson.""")

# json serializable 함수
def json_property_default(value):
	if isinstance(value, Property):
		return value.get_data()
	raise TypeError("not JSON serializable")

class Property(object):
	def __init__(self, property=None):
		self._information = {}

		if property is not None:
			for i in property.keys():
				self.add_property(i, property[i])

	def add_property(self, property, value):
		self._information[property] = value


	def delete_property(self, property):
		if property in self._information.keys():
			del self._information[property]

			
	def get_property(self, property=None):
		if property is not None:
			return self._information[property]
		else:
			return self._information


	def is_in_property(self, property):
		return property in self._information.keys()


	def update_property(self, property, value):
		if property in self._information.keys():
			self._information[property] = value
		else:
			self._information[property] = value


	def get_property_list(self):
		return self._information.keys()


	def __repr__(self):
		return json.dumps(self._information, sort_keys=True, indent=4, separators=(',', ':'))


	def get_data(self):
		return self._information

	def write2file(self, path_file):
		with open(path_file, 'w') as outfile:
			json.dump(self._information, outfile, sort_keys=True, indent=4, separators=(',', ':'))

