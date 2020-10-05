import numpy as np

from scattering import *

mesh_file = "mesh/homo.msh"
epsilon = [[5.47781441, 0. ], [0., 5.78054277]]
permittivity_dict = {1: epsilon, 2: epsilon, 3: np.identity(2)}
s = np.array([1, 2])
p = np.array([-2, 1])
k0L = np.pi

problem = AnisotropicScattering(mesh_file, permittivity_dict, k0L)
pw = PlaneWave(s, p)

E = problem.solve(pw)

#phi, FF = problem.get_far_field(E, 40)
#plot_far_field(phi, FF, "anisotropic")
