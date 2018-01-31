import numpy as np

class SimpleLoon:
	x = 0
	y = 0
	z = 0
	Cd = 1	# drag coefficient
	A = 1	# cross-sectional area

	def __init__(self, xi, yi, zi):
		self.x = xi
		self.y = yi
		self.z = zi

	def __str__(self):
		return "x: " + str(self.x) + ", y: " + str(self.y) + ", z: " + str(self.z)

	def travel(self, dx, dy, dz):
		self.x += dx
		self.y += dy
		self.z += dz

	def get_pos(self):
		return self.x, self.y, self.z
