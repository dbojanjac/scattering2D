import numpy as np
from scattering import *

def run_isotropic_test(mesh_file, s, p):
    dim = s.shape[0]
    permittivity_dict = {1: 1, 2: 11.8, 3: 1}
    k0L = np.pi

    problem = IsotropicScattering(mesh_file, permittivity_dict, k0L)
    pw = PlaneWave(s, p)

    E = problem.solve(pw)

    save_field(E, str(dim) + "D_field.pvd")
    phi, FF = problem.get_far_field(E, 40)
    #plot_far_field(phi, FF, "isotropic")

def run_2d_isotropic_test():
    run_isotropic_test("mesh/hexa.msh", np.array([-1, -2]), np.array([2, -1]))


def run_3d_isotropic_test():
    run_isotropic_test("mesh/het3D.msh", np.array([-1, -2, 0]), np.array([2, -1, 0]))

if __name__ == "__main__":
    run_2d_isotropic_test()
    run_3d_isotropic_test()
