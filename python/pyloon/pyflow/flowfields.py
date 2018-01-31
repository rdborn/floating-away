import numpy as np

class FlowField:
	type = "none"
	
	xdim = 1
	ydim = 1
	zdim = 1
	

	def __init__(self, t, x, y, z):
		self.type = t
		self.xdim = x
		self.ydim = y
		self.zdim = z
		self.magnitude = np.squeeze(np.zeros(x, y, z))
		self.angle_horizontal = np.squeeze(np.zeros(x, y, z))
		self.angle_vertical = np.squeeze(np.zeros(x, y, z))

	def __str__(self):
		return self.type

	def set_flow_super(self, x, y, z, mag, angle_hori, angle_vert):
		

class FlowField1D(FlowField):
	def __init__(self, x):
		FlowField.__init__(self, "1D", x, 1, 1)

	def set_flow(self, x, mag):
		self.set_flow_super(x, 1, 1, mag, 0, 0)

class FlowField2D(FlowField):
	def __init__(self, x, y):
		FlowField.__init__(self, "2D", x, y, 1)
	
	def set_flow(self, x, y, mag, angle_hori):
		self.set_flow_super(x, y, 1, mag, angle_hori, 0)

class FlowField3D(FlowField):
	def __init__(self, x, y, z):
		FlowField.__init__(self, "3D", x, y, z)

	def set_flow(self, x, y, z, mag, angle_hori, angle_vert):
		self.set_flow_super(x, y, z, mag, angle_hori, angle_vert)

	
