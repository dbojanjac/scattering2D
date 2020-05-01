import numpy as np
np.seterr(all="raise")

from scattering import *

mesh_file = "mesh/hexa.msh"
permittivity_dict = {1: 1, 2: 11.7, 3: 1}
s = np.array([1, 2])
p = np.array([-2, 1])
k0L = np.pi

problem = IsotropicScattering(mesh_file, permittivity_dict, k0L)
pw = PlaneWave(s, p)

E = problem.solve(pw)
save_field(problem.permittivity, "permittivity.pvd")
save_field(E, "isotropic_test.pvd")
