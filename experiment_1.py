import numpy as np
import csv
from subprocess import run, DEVNULL
from scattering import *

FAR_FIELD_POINTS = 120
LC1 = 5e-3
LC2 = 5e-3
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
                     "-setnumber", "lc1", str(LC1), "-setnumber", "lc2", str(LC2),
                     "mesh/hexa.geo"])

        permittivity_dict = {1: 1, 2: 11.8, 3: 1}
        s = np.array([1, 2])
        p = np.array([-2, 1])
        k0L = np.pi

        print("Isotropic Scattering with permittivity {} and n {}".format(permittivity_dict, i))
        problem = IsotropicScattering(mesh_file, permittivity_dict, k0L)
        pw = PlaneWave(s, p)

        E_isotropic = problem.solve(pw)

        phi, FF_isotropic = problem.get_far_field(E_isotropic, FAR_FIELD_POINTS)
        np.save(f"ff_isotropic-{i}.npy", FF_isotropic)

        epsilon = [[5.54637936300038, 0], [0, 5.845776226164643]]

        permittivity_dict = {1: epsilon, 2: epsilon, 3: np.identity(2)}
        print("Anisotropic Scattering with permittivity {} and n {}".format(permittivity_dict, i))
        problem = AnisotropicScattering(problem.mesh, permittivity_dict, k0L)
        E_anisotropic = problem.solve(pw)

        _, FF_anisotropic = problem.get_far_field(E_anisotropic, FAR_FIELD_POINTS)
        np.save(f"ff_anisotropic-{i}.npy", FF_isotropic)

        E_field_norm = errornorm(E_isotropic, E_anisotropic)
        FF_error = np.linalg.norm(FF_isotropic - FF_anisotropic)
        print("{} {} {}".format(i, E_field_norm, FF_error))
        results.append((i, E_field_norm.real, FF_error))
    result_writer.writerows(results)
