import numpy as np

from scattering import *

mesh_file = "mesh/hexa.msh"
epsilon = ((5.41474, 0), (0, 5.71539))

permittivity_dict = {
        1: epsilon,
        2: epsilon,
        3: ((1, 0), (0, 1))
}
s = np.array([1, 2])
p = np.array([-2, 1])
k0L = np.pi

problem = AnisotropicScattering(mesh_file, permittivity_dict, k0L)
pw = PlaneWave(s, p)

E = problem.solve(pw)

phi, FF = problem.get_far_field(E, 40)
plot_far_field(phi, FF, "anisotropic")
