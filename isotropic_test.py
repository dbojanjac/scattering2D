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

if __name__ == "__main__":
    mesh_files = ("mesh/hexa.msh", "mesh/het3D.msh")
    s_vecs = ([-1, -2], [-1, -2, 0])
    p_vecs = ([2, -1], [2, -1, 0])
    for mesh_file, s, p in zip(mesh_files, s_vecs, p_vecs):
        run_isotropic_test(mesh_file, np.array(s), np.array(p))
