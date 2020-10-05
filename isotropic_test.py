import numpy as np
from scattering import *

mesh_file = "mesh/hexa.msh"
permittivity_dict = {1: 1, 2: 11.8, 3: 1}
s = np.array([1, 2])
p = np.array([-2, 1])
k0L = np.pi

problem = IsotropicScattering(mesh_file, permittivity_dict, k0L)
pw = PlaneWave(s, p)

E = problem.solve(pw)

#phi, FF = problem.get_far_field(E, 40)
#plot_far_field(phi, FF, "isotropic")
