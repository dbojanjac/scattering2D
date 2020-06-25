import numpy as np
import csv
from itertools import product
from subprocess import run, DEVNULL
from scattering import *

FAR_FIELD_POINTS = 100
NUM_HEX = 10
FINE=2e-3
COARSE=1e-1
NUM_STEPS=30
mesh_file = "mesh/hexa.msh"
s = np.array([1, 2])
p = np.array([-2, 1])
pw = PlaneWave(s, p)
k0L = np.pi


# Izvedi prvi put control koji se izvodi na Isotropic s gustim meshom.
# Od tog pokretanja zapamti far field.
# Iterativno stvaraj sve manje gust mesh i uspoređuj Anisotropic vs Isotropic.
# Također, uspoređuj Isotropic_control i Anisotropic jer nam to govori koliko
# smo daleko od pravog rješenja. Bilo bi bolje da možemo usporediti i L2 normu
# između 2 rješenja, ali to su funkcije na 2 različita mesha...

def run_command(args):
    print("Running command: {}".format(" ".join(args)))
    run(args, stdout=DEVNULL, check=True)

def create_mesh(i, j):
    run_command(["gmsh", "-2", "-o", mesh_file, "-setnumber", "n", str(NUM_HEX),
                 "-setnumber", "lc1", str(i), "-setnumber", "lc2", str(j),
                 "mesh/hexa.geo"])

def run_isotropic(s, p, k0L):
    permittivity_dict = {1: 1, 2: 11.7, 3: 1}

    print(f"Isotropic Scattering with permittivity {permittivity_dict}")
    problem = IsotropicScattering(mesh_file, permittivity_dict, k0L)
    E_isotropic = problem.solve(pw)

    phi, FF_isotropic = problem.get_far_field(E_isotropic, FAR_FIELD_POINTS)

    return FF_isotropic


create_mesh(FINE, FINE)
FF_isotropic_control = run_isotropic(s, p, k0L)

with open("results2.csv", "w", newline='') as csvfile:
    results = [("lc1", "#cells", "#edges", "#cell_elems", "FF norm")]
    result_writer = csv.writer(csvfile, delimiter=',')
    mesh_linspace = np.linspace(FINE, COARSE, num=NUM_STEPS)
    for i in mesh_linspace:
        if not np.isclose(i, FINE):
            create_mesh(i, i)

        epsilon = [[5.41474, 0], [0, 5.71539]]
        permittivity_dict = {1: epsilon, 2: epsilon, 3: np.identity(2)}
        print(f"Anisotropic Scatteirng with permittivity {permittivity_dict}")
        problem = AnisotropicScattering(mesh_file, permittivity_dict, k0L)
        E_anisotropic = problem.solve(pw)

        _, FF_anisotropic = problem.get_far_field(E_anisotropic, FAR_FIELD_POINTS)

        num_cells = problem.mesh.num_cells()
        num_edges = problem.mesh.num_edges()
        #E_field_norm = errornorm(E_isotropic, E_anisotropic).real
        FF_error = np.linalg.norm(FF_isotropic_control - FF_anisotropic)
        print(f"{i} {num_cells} {num_edges} {FF_error}")
        results.append((i, num_cells, num_edges, FF_error))
    result_writer.writerows(results)
