import numpy as np

from scattering import *

def run_anisotropic_test(mesh_file, s, p):
    dim = s.shape[0]
    epsilon = {
            2: [[5.47781441, 0. ], [0., 5.78054277]],
            3: [[5.4844069,  0., 0.], [0., 5.78683953, 0.], [0., 0., 7.33240001]]
        }
    permittivity_dict = {1: epsilon[dim], 2: epsilon[dim], 3: np.identity(dim)}
    k0L = np.pi

    problem = AnisotropicScattering(mesh_file, permittivity_dict, k0L)
    pw = PlaneWave(s, p)

    E = problem.solve(pw)
    save_field(E, str(dim) + "D_anisotropic_field.pvd")

    phi, FF = problem.get_far_field(E, 40)
    #plot_far_field(phi, FF, "anisotropic")

def run_2d_anisotropic_test():
    run_anisotropic_test("mesh/hexa.msh", np.array([-1, -2]), np.array([2, -1]))

def run_3d_anisotropic_test():
    run_anisotropic_test("mesh/het3D.msh", np.array([-1, -2, 0]), np.array([2, -1, 0]))

if __name__ == "__main__":
    run_2d_anisotropic_test()
    run_3d_anisotropic_test()
