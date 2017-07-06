from scipy import optimize

class Trainer(object):
	def __init__(self, N):
		self.N = N
	
	def costFunctionWrapper(self, params, X, y):
		self.N.setParams(params)
		cost = self.N.costFunction(X, y)
		grad = self.N.computeGradients(X,y)
		return cost, grad
	
	def callBackF(self, params):
		self.N.setParams(params)

	def train(self, X, y):
		self.X = X
		self.y = y

		params0 = self.N.getParams()

		options = {'maxiter' : 200, 'disp' : True}
		_res = optimize.minimize(self.costFunctionWrapper, params0, jac = True, method='BFGS', args = (X,y), options=options, callback=self.callBackF)
		self.N.setParams(_res.x)
		self.optimizationResults = _res
