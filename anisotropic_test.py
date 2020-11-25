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

if __name__ == "__main__":
    mesh_files = ("mesh/hexa.msh", "mesh/het3D.msh")
    s_vecs = ([-1, -2], [-1, -2, 0])
    p_vecs = ([2, -1], [2, -1, 0])
    for mesh_file, s, p in zip(mesh_files, s_vecs, p_vecs):
        run_anisotropic_test(mesh_file, np.array(s), np.array(p))
