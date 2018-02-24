"""The layer is implemented in a similar way to pytorch's implementation."""
class NonLinearLayer():
	def __init__():
		pass

	def forward(self, x, y, z):
		self.x = x
		self.y = y
		self.z = z
		return x*x + 4*y + z

	def backward(self, e, lr=0.001):
		loss = 2*self.x + 4 + 1
		self.grad = e * loss *  lr

		# the grad will be used by the layer before it.
		return self.grad

	def step(self):
		""" Since it is non-linearity and does not have params, 
		updating is not required.
		"""
		pass
