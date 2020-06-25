import numpy as np
import csv
from subprocess import run, DEVNULL
from scattering import *

FAR_FIELD_POINTS = 40
LC1 = 2e-2
LC2 = 2e-2
NUM_EXPERIMENTS = 20

def run_command(args):
    print("Running command: {}".format(" ".join(args)))
    run(args, stdout=DEVNULL, check=True)

with open("results1.csv", "w", newline='') as csvfile:
    results = [("Number of elements in cell", "E_field_norm", "FF norm")]
    result_writer = csv.writer(csvfile, delimiter=',')
    for i in range(1, NUM_EXPERIMENTS):
        mesh_file = "mesh/hexa.msh"

        run_command(["gmsh", "-2", "-o", mesh_file, "-setnumber", "n", str(i),
                     "-setnumber", "lc1", LC1, "-setnumber", "lc2", LC2,
                     "mesh/hexa.geo"])

        permittivity_dict = {1: 1, 2: 11.7, 3: 1}
        s = np.array([1, 2])
        p = np.array([-2, 1])
        k0L = np.pi

        print("Isotropic Scattering with permittivity {} and n {}".format(permittivity_dict, i))
        problem = IsotropicScattering(mesh_file, permittivity_dict, k0L)
        pw = PlaneWave(s, p)

        E_isotropic = problem.solve(pw)

        phi, FF_isotropic = problem.get_far_field(E_isotropic, FAR_FIELD_POINTS)

        epsilon = [[5.41474, 0], [0, 5.71539]]

        permittivity_dict = {1: epsilon, 2: epsilon, 3: np.identity(2)}
        print("Anisotropic Scattering with permittivity {} and n {}".format(permittivity_dict, i))
        problem = AnisotropicScattering(problem.mesh, permittivity_dict, k0L)
        E_anisotropic = problem.solve(pw)

        _, FF_anisotropic = problem.get_far_field(E_anisotropic, FAR_FIELD_POINTS)

        E_field_norm = errornorm(E_isotropic, E_anisotropic)
        FF_error = np.linalg.norm(FF_isotropic - FF_anisotropic)
        print("{} {} {}".format(i, E_field_norm, FF_error))
        results.append((i, E_field_norm.real, FF_error))
    result_writer.writerows(results)
