import numpy as np
from scattering import *

mesh_file = "mesh/het3D.msh"
permittivity_dict = {10: 1, 20: 11.8, 30: 1}
s = np.array([-1, -2, 0])
p = np.array([2, -1, 0])
k0L = np.pi

problem = IsotropicScattering(mesh_file, permittivity_dict, k0L)
pw = PlaneWave(s, p)

E = problem.solve(pw)

save_field(E, "3D_field.pvd")
phi, FF = problem.get_far_field(E, 40)
#plot_far_field(phi, FF, "isotropic")
