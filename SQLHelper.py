import sqlite3, numpy as np, io

class SQLHelper(object):
	def __init__(self, database):
		# Converts np.array to TEXT when inserting
		sqlite3.register_adapter(np.ndarray, self.adapt_array)

		# Converts TEXT to np.array when selecting
		sqlite3.register_converter("array", self.convert_array)

		self.conn = sqlite3.connect(database, detect_types=sqlite3.PARSE_DECLTYPES)

	def query(self, strQuery, data=list(), many=False):
		resultSet = list()
		try:
			with self.conn:
				self.conn.row_factory = sqlite3.Row
				c = self.conn.cursor()
				if many:
					c.executemany(strQuery, data)
				else:
					c.execute(strQuery, data)
				while True:
					row = c.fetchone()
					if row==None:
						break
					resultSet.append(row)
		except Exception, e:
			print "Error al ejecutar la consulta: "+str(e)
		return resultSet


	def adapt_array(self, arr):
	    """
	    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
	    """
	    out = io.BytesIO()
	    np.save(out, arr)
	    out.seek(0)
	    return sqlite3.Binary(out.read())

	def convert_array(self, text):
	    out = io.BytesIO(text)
	    out.seek(0)
	    return np.load(out)


