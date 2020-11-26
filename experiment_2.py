import numpy as np
import csv
from itertools import product
from subprocess import run, DEVNULL
from scattering import *

FAR_FIELD_POINTS = 100
NUM_HEX = 10
FINE=2e-3
COARSE=3e-1
NUM_STEPS=10
HOMO_MESH = "mesh/homo.msh"
HOMO_FILE = "mesh/homo.geo"
HEXA_MESH = "mesh/hexa.msh"
HEXA_FILE = "mesh/hexa.geo"
s = np.array([-1, -2])
p = np.array([2, -1])
pw = PlaneWave(s, p)
k0L = np.pi

def run_command(args):
    print("Running command: {}".format(" ".join(args)))
    run(args, stdout=DEVNULL, check=True)

def create_mesh(i, j, mesh_input, mesh_output):
    run_command(["gmsh", "-2", "-o", mesh_output, "-setnumber", "n", str(NUM_HEX),
                 "-setnumber", "lc1", str(i), "-setnumber", "lc2", str(j),
                 mesh_input])

def create_homo_mesh(i, j=5e-2):
    create_mesh(i, j, HOMO_FILE, HOMO_MESH)

def create_hexa_mesh(i, j=5e-2):
    create_mesh(i, j, HEXA_FILE, HEXA_MESH)

def run_isotropic(s, p, k0L):
    permittivity_dict = {1: 1, 2: 11.8, 3: 1}

    print(f"Isotropic Scattering with permittivity {permittivity_dict}")
    problem = IsotropicScattering(HEXA_MESH, permittivity_dict, k0L)
    num_cells = problem.mesh.num_cells()
    num_edges = problem.mesh.num_edges()
    E_isotropic = problem.solve(pw)

    print(f"control: #num_cells={num_cells}, #num_edges={num_edges}")

    phi, FF_isotropic = problem.get_far_field(E_isotropic, FAR_FIELD_POINTS)

    return FF_isotropic


create_hexa_mesh(FINE)
FF_isotropic_control = run_isotropic(s, p, k0L)

with open("results/results2.csv", "w", newline='') as csvfile:
    results = [("lc1", "#cells", "#edges", "#cell_elems", "abs FF norm", "rel FF norm")]
    result_writer = csv.writer(csvfile, delimiter=',')

    epsilon = [[5.47781441, 0. ], [0., 5.78054277]]
    permittivity_dict = {1: epsilon, 2: np.identity(2)}
    print(f"Anisotropic Scatteirng with permittivity {permittivity_dict}")
    problem = AnisotropicScattering(HOMO_MESH, permittivity_dict, k0L)
    E_anisotropic = problem.solve(pw)

    phi, FF_anisotropic = problem.get_far_field(E_anisotropic, FAR_FIELD_POINTS)

    num_cells = problem.mesh.num_cells()
    num_edges = problem.mesh.num_edges()
    abs_FF_error = np.linalg.norm(FF_isotropic_control - FF_anisotropic)
    rel_FF_error = abs_FF_error/np.linalg.norm(FF_isotropic_control)
    print(f"{num_cells} {num_edges} {abs_FF_error} {rel_FF_error}")
    results.append((num_cells, num_edges, abs_FF_error, rel_FF_error))
    result_writer.writerows(results)
